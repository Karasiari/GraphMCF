from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from scipy.stats import norm

from ..core import GraphMCF

@dataclass
class TimeTestResult:
    """Расширенный результат с таймингами (для профилирования)."""
    graph: GraphMCF
    alpha_target: float
    epsilon: float
    start_time: float
    end_time: float
    iterations_total: int
    alpha_history: List[float]
    edge_counts_history: List[int]
    median_weights_history: List[float]
    timings: Dict[str, Any]  # детальные тайминги

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_target": self.alpha_target,
            "epsilon": self.epsilon,
            "execution_time": float(self.end_time - self.start_time),
            "iterations_total": self.iterations_total,
            "initial_alpha": float(self.alpha_history[0]) if self.alpha_history else None,
            "final_alpha": float(self.alpha_history[-1]) if self.alpha_history else None,
            "alpha_history": [float(x) for x in self.alpha_history],
            "edge_counts_history": [int(x) for x in self.edge_counts_history],
            "median_weights_history": [float(x) for x in self.median_weights_history],
            "timings": self.timings,
        }

class MCFGeneratorWithTimeTest:
    """
    Профилировочная копия MCF-генератора. На каждой итерации меряет:
      - t_alpha: время calculate_alpha() (внутри — разложение на t_prep, t_ld, t_lalpha, t_eig)
      - t_cut:   время generate_cut()
      - t_edit:  время модификации рёбер (remove/upsert)
    Каждые report_every итераций выводит окно агрегатов, плюс возвращает все суммы/окна в result.timings.
    Отдельно замеряется initial_demands.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        demands_weights_distribution: str = "normal",
        demands_median_denominator: int = 1,
        demands_var_denominator: int = 2,
        max_iter: Optional[int] = None,
        report_every: int = 50,
        verbose: bool = True,
    ) -> None:
        self.epsilon = float(epsilon)
        self.dist = demands_weights_distribution
        self.median_div = int(demands_median_denominator)
        self.var_div = int(demands_var_denominator)
        self.max_iter = max_iter
        self.report_every = int(report_every)
        self.verbose = bool(verbose)

    # ---------- утилиты распределения весов ----------
    def _draw_new_weight(self, old_w: float, mode: str, median: int, var: int) -> int:
        lo, hi = 1, max(2 * median, 2)
        xs = np.arange(lo, hi + 1)
        ps = norm.pdf(xs, loc=median, scale=np.sqrt(max(var, 1))); ps /= ps.sum()
        if mode == ">":
            mask = xs > old_w
        elif mode == "<":
            mask = xs < old_w
        else:
            raise ValueError("mode должен быть '>' или '<'")
        if not np.any(mask):
            return int(hi if mode == ">" else lo)
        xs2, ps2 = xs[mask], ps[mask]; ps2 /= ps2.sum()
        return int(np.random.choice(xs2, p=ps2))

    def _split_by_median(self, vec: np.ndarray) -> np.ndarray:
        med = float(np.median(vec))
        side = (vec <= med)
        if side.sum() in (0, len(side)):
            side = (vec < med)
        if side.sum() in (0, len(side)):
            side = np.zeros_like(side, dtype=bool); side[0] = True
        return side

    # ---------- основной запуск ----------
    def generate(
        self,
        graph: GraphMCF,
        alpha_target: float = 0.5,
        analysis_mode: Optional[str] = None,  # оставлено для совместимости API
    ) -> TimeTestResult:

        # --- подготовка параметров распределения ---
        base_w = [d["weight"] for *_ , d in graph.graph.edges(data=True)]
        med_cap = int(round(np.median(base_w))) if base_w else 0
        median_for_weights = max(med_cap // self.median_div, 1)
        var_for_weights = max(med_cap // self.var_div, 1)

        # --- инициализация корреспонденций + таймер ---
        t0 = time.perf_counter()
        graph.generate_initial_demands(
            distribution=self.dist,
            median_weight=median_for_weights,
            var=var_for_weights,
        )
        t1 = time.perf_counter()
        initial_demands_time = t1 - t0

        nodes = list(graph.graph.nodes())
        index = {u: i for i, u in enumerate(nodes)}
        n = len(nodes)
        max_iter = self.max_iter if self.max_iter is not None else 100 * n

        # --- индексная структура рёбер demands-графа ---
        def build_edge_index():
            E_u, E_v, E_w = [], [], []
            eid = {}
            for u, v, d in graph.demands_graph.edges(data=True):
                iu, iv = index[u], index[v]
                if iu > iv: iu, iv = iv, iu
                eid[(iu, iv)] = len(E_w)
                E_u.append(iu); E_v.append(iv); E_w.append(float(d.get("weight", 1.0)))
            E_alive = [True] * len(E_w)
            return (np.asarray(E_u, np.int32),
                    np.asarray(E_v, np.int32),
                    np.asarray(E_w, float),
                    np.asarray(E_alive, bool),
                    eid)

        def masks_for_cut(side: np.ndarray, E_u, E_v, E_alive):
            same = (side[E_u] == side[E_v])
            return (E_alive & same), (E_alive & ~same)

        def pick_idx(mask: np.ndarray, E_w: np.ndarray, mode: str) -> Optional[int]:
            if not np.any(mask): return None
            if mode == "min":
                w = np.where(mask, E_w, np.inf); val = w.min()
                if not np.isfinite(val): return None
                cand = np.flatnonzero(w == val)
            else:
                w = np.where(mask, E_w, -np.inf); val = w.max()
                if not np.isfinite(val): return None
                cand = np.flatnonzero(w == val)
            j = np.random.randint(cand.size)
            return int(candidates := cand[j])

        # операции над demands-графом с таймингом изменения рёбер
        def remove_by_idx(idx, E_u, E_v, E_alive, E_w, timing_acc):
            if not E_alive[idx]:
                return None
            t0 = time.perf_counter()
            iu, iv = int(E_u[idx]), int(E_v[idx])
            old_w = graph.remove_edge_by_indices(iu, iv)
            E_alive[idx] = False
            timing_acc[0] += (time.perf_counter() - t0)
            return old_w

        def upsert(iu: int, iv: int, delta_w: float,
                   E_u, E_v, E_alive, E_w, eid: Dict[Tuple[int, int], int],
                   timing_acc):
            if iu > iv: iu, iv = iv, iu
            t0 = time.perf_counter()
            new_w = graph.upsert_edge_by_indices(iu, iv, delta_w)
            key = (iu, iv)
            if key in eid:
                j = eid[key]
                E_w[j] = new_w
                E_alive[j] = True
                timing_acc[0] += (time.perf_counter() - t0)
                return E_u, E_v, E_alive, E_w
            else:
                j = E_w.size
                eid[key] = j
                E_u = np.append(E_u, iu)
                E_v = np.append(E_v, iv)
                E_w = np.append(E_w, new_w)
                E_alive = np.append(E_alive, True)
                timing_acc[0] += (time.perf_counter() - t0)
                return E_u, E_v, E_alive, E_w

        # --- таймеры по итерациям и структуры таймингов ---
        t_alpha_sum = 0.0
        t_cut_sum = 0.0
        t_edit_sum = 0.0
        window_start_iter = 0

        timings = {
            "initial_demands": initial_demands_time,
            "per_window": [],   # [{"iters":[i0,i1], "t_alpha":..., "t_cut":..., "t_edit":...}, ...]
            "totals": {"t_alpha": 0.0, "t_cut": 0.0, "t_edit": 0.0},
            "alpha_parts_totals": {"t_prep": 0.0, "t_ld": 0.0, "t_lalpha": 0.0, "t_eig": 0.0},
            "alpha_parts_per_window": []  # [{"iters":[i0,i1], "t_prep":..., "t_ld":..., "t_lalpha":..., "t_eig":...}]
        }
        _win_alpha_parts = {"t_prep":0.0,"t_ld":0.0,"t_lalpha":0.0,"t_eig":0.0}

        # --- основной цикл ---
        E_u, E_v, E_w, E_alive, eid = build_edge_index()

        alpha_history: List[float] = []
        edge_counts_history: List[int] = []
        median_weights_history: List[float] = []

        start = time.perf_counter()

        # итерация 0: считаем alpha (разложение по частям)
        a, parts = graph.calculate_alpha_timed()
        dt_alpha = parts["t_prep"] + parts["t_ld"] + parts["t_lalpha"] + parts["t_eig"]
        t_alpha_sum += dt_alpha
        timings["totals"]["t_alpha"] += dt_alpha
        for k in timings["alpha_parts_totals"]:
            timings["alpha_parts_totals"][k] += parts[k]
            _win_alpha_parts[k] += parts[k]

        alpha_history.append(a)
        edge_counts_history.append(graph.demands_graph.number_of_edges())
        w_now = [d["weight"] for *_ , d in graph.demands_graph.edges(data=True)]
        median_weights_history.append(float(np.median(w_now)) if w_now else 0.0)

        it = 0
        while it < max_iter and abs(a - alpha_target) > self.epsilon:
            it += 1

            # --- generate_cut (таймим) ---
            t0 = time.perf_counter()
            if (a - alpha_target) < -self.epsilon:
                v = graph.generate_cut(type="adversarial")  # хотим увеличить alpha
                cut_type = "adversarial"
            else:
                v = graph.generate_cut(type="friendly")     # хотим уменьшить alpha
                cut_type = "friendly"
            dt_cut = time.perf_counter() - t0
            t_cut_sum += dt_cut
            timings["totals"]["t_cut"] += dt_cut

            side = self._split_by_median(np.asarray(v, dtype=float))
            internal_mask, _ = masks_for_cut(side, E_u, E_v, E_alive)

            # --- изменения рёбер (таймим remove/upsert суммарно) ---
            t_edit_acc = [0.0]

            if cut_type == "adversarial":
                j = pick_idx(internal_mask, E_w, "min") or pick_idx(E_alive, E_w, "min")
                if j is None:
                    V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)
                    if V1.size and V2.size:
                        iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
                        E_u, E_v, E_alive, E_w = upsert(iu, iv, 1.0, E_u, E_v, E_alive, E_w, eid, t_edit_acc)
                else:
                    w_old = E_w[j]
                    remove_by_idx(j, E_u, E_v, E_alive, E_w, t_edit_acc)
                    V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)
                    if V1.size and V2.size:
                        iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
                        delta = self._draw_new_weight(w_old, ">", median_for_weights, var_for_weights)
                        E_u, E_v, E_alive, E_w = upsert(iu, iv, float(delta), E_u, E_v, E_alive, E_w, eid, t_edit_acc)
            else:
                j = pick_idx(internal_mask, E_w, "max") or pick_idx(E_alive, E_w, "max")
                if j is None:
                    V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)
                    if V1.size and V2.size:
                        iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
                        E_u, E_v, E_alive, E_w = upsert(iu, iv, 1.0, E_u, E_v, E_alive, E_w, eid, t_edit_acc)
                else:
                    w_old = E_w[j]
                    remove_by_idx(j, E_u, E_v, E_alive, E_w, t_edit_acc)
                    V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)
                    if V1.size and V2.size:
                        iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
                        delta = self._draw_new_weight(w_old, "<", median_for_weights, var_for_weights)
                        E_u, E_v, E_alive, E_w = upsert(iu, iv, float(delta), E_u, E_v, E_alive, E_w, eid, t_edit_acc)

            # учтём время редактирования
            t_edit_sum += t_edit_acc[0]
            timings["totals"]["t_edit"] += t_edit_acc[0]

            # --- пересчёт alpha (таймированный, с разложением) ---
            a, parts = graph.calculate_alpha_timed()
            dt_alpha = parts["t_prep"] + parts["t_ld"] + parts["t_lalpha"] + parts["t_eig"]
            t_alpha_sum += dt_alpha
            timings["totals"]["t_alpha"] += dt_alpha
            for k in timings["alpha_parts_totals"]:
                timings["alpha_parts_totals"][k] += parts[k]
                _win_alpha_parts[k] += parts[k]

            alpha_history.append(a)
            edge_counts_history.append(graph.demands_graph.number_of_edges())
            w_now = [d["weight"] for *_ , d in graph.demands_graph.edges(data=True)]
            median_weights_history.append(float(np.median(w_now)) if w_now else 0.0)

            # --- окно отчёта каждые report_every итераций ---
            if it % self.report_every == 0:
                block = {
                    "iters": [window_start_iter, it],
                    "t_alpha": t_alpha_sum,
                    "t_cut": t_cut_sum,
                    "t_edit": t_edit_sum,
                }
                timings["per_window"].append(block)

                timings["alpha_parts_per_window"].append({
                    "iters": [window_start_iter, it],
                    **_win_alpha_parts
                })

                if self.verbose:
                    print(f"[iters {window_start_iter:>5d}..{it:>5d}]: "
                          f"alpha={t_alpha_sum:.4f}s, cut={t_cut_sum:.4f}s, edit={t_edit_sum:.4f}s | "
                          f"alpha_parts={_win_alpha_parts}")

                window_start_iter = it
                t_alpha_sum = t_cut_sum = t_edit_sum = 0.0
                _win_alpha_parts = {"t_prep":0.0,"t_ld":0.0,"t_lalpha":0.0,"t_eig":0.0}

        # --- финальный «хвост» окна ---
        if (it % self.report_every) != 0:
            block = {
                "iters": [window_start_iter, it],
                "t_alpha": t_alpha_sum,
                "t_cut": t_cut_sum,
                "t_edit": t_edit_sum,
            }
            timings["per_window"].append(block)

            timings["alpha_parts_per_window"].append({
                "iters": [window_start_iter, it],
                **_win_alpha_parts
            })

            if self.verbose:
                print(f"[iters {window_start_iter:>5d}..{it:>5d}]: "
                      f"alpha={t_alpha_sum:.4f}s, cut={t_cut_sum:.4f}s, edit={t_edit_sum:.4f}s | "
                      f"alpha_parts={_win_alpha_parts}")

        end = time.perf_counter()

        return TimeTestResult(
            graph=graph,
            alpha_target=float(alpha_target),
            epsilon=self.epsilon,
            start_time=start,
            end_time=end,
            iterations_total=it,
            alpha_history=[float(x) for x in alpha_history],
            edge_counts_history=[int(x) for x in edge_counts_history],
            median_weights_history=[float(x) for x in median_weights_history],
            timings=timings,
        )
