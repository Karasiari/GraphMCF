from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import time
import math
import numpy as np

from ..core import GraphMCF

@dataclass
class DemandsGenerationResultMulti:
    graph: GraphMCF
    alpha_target: float
    epsilon: float
    start_time: float
    end_time: float
    iterations_total: int
    alpha_history: List[float]
    edge_counts_history: List[int]
    median_weights_history: List[float]
    algo_params: Optional[Dict[str, Any]] = None

    # Доп. поля прикрепляются динамически:
    #   removal_events: List[Dict[str, Any]]
    #   edge_mask_history: List[np.ndarray]
    #   edge_mask_snapshot_iters: List[int]

    def to_dict(self) -> Dict[str, Any]:
        from ..analysis.overall import pack_overall_dict
        return pack_overall_dict(
            self.graph,
            self.alpha_target,
            self.epsilon,
            self.start_time,
            self.end_time,
            self.alpha_history,
            self.edge_counts_history,
            self.median_weights_history,
        )

class MCFGeneratorMultiEdges:
    """
    Модификация базового генератора:
      - в каждой внешней итерации выполняем num_edges суб-операций
      - каждая суб-операция:
          * удаляет ребро с вероятностью p_for_delete_edge (если есть подходящее)
          * добавляет/усиливает ребро с вероятностью p_for_upsert_edge
      - остальная логика (приоритет внутрикластерных, fallback и история) совпадает с оригиналом
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.05,
        # --- контроль initial-генерации demands (ER + веса) ---
        p_ER: float = 0.5,
        distribution: str = "normal",
        median_weight_for_initial: int = 50,
        var_for_initial: int = 100,
        # --- добавляемые рёбра (от capacity) ---
        demands_median_denominator: int = 1,
        demands_var_denominator: int = 2,
        # --- новые параметры модификации ---
        num_edges: Optional[int] = None,          # если None -> ceil(n ** 0.25)
        p_for_delete_edge: float = 1.0,           # "монетка" перед удалением
        p_for_upsert_edge: float = 1.0,           # "монетка" перед добавлением
        # ---
        max_iter: Optional[int] = None,
    ) -> None:
        self.epsilon = float(epsilon)

        # initial-demands
        self.p_ER = float(p_ER)
        self.dist = str(distribution)
        self.median_weight_for_initial = int(median_weight_for_initial)
        self.var_for_initial = int(var_for_initial)

        # добавляемые рёбра: параметры от capacity
        self.median_div = int(demands_median_denominator)
        self.var_div = int(demands_var_denominator)

        # много-рёберные настройки
        self.num_edges_param = None if num_edges is None else int(num_edges)
        self.p_for_delete_edge = float(p_for_delete_edge)
        self.p_for_upsert_edge = float(p_for_upsert_edge)

        self.max_iter = max_iter  # если None → 100*|V|

    # ---------------------- служебные методы (как в base) ---------------------

    def _split_by_median(self, vec: np.ndarray) -> np.ndarray:
        med = float(np.median(vec))
        side = (vec <= med)
        if side.sum() in (0, len(side)):
            side = (vec < med)
        if side.sum() in (0, len(side)):
            side = np.zeros_like(side, dtype=bool); side[0] = True
        return side

    def _draw_new_weight(self, old_w: float, mode: str, median: int, var: int) -> int:
        if self.dist != "normal":
            raise ValueError(f"Unsupported distribution for demands weights: {self.dist}")
        from scipy.stats import norm
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
            return hi if mode == ">" else lo
        xs2, ps2 = xs[mask], ps[mask]; ps2 /= ps2.sum()
        return int(np.random.choice(xs2, p=ps2))

    # ------------------------------- generate ---------------------------------

    def generate(
        self,
        graph: GraphMCF,
        alpha_target: float = 0.5,
        analysis_mode: Optional[str] = None,  # анализ запускается отдельно
    ) -> DemandsGenerationResultMulti:

        # 0) initial demands
        start = time.time()
        graph.generate_initial_demands(
            p=self.p_ER,
            distribution=self.dist,
            median_weight=self.median_weight_for_initial,
            var=self.var_for_initial,
        )

        # 1) параметры для ДОБАВЛЕНИЙ от capacity (как в оригинале)
        base_w = [d["weight"] for *_, d in graph.graph.edges(data=True)]
        med_cap = int(round(np.median(base_w))) if base_w else 0
        median_for_weights = max(med_cap // self.median_div, 1)
        var_for_weights = max(med_cap // self.var_div, 1)

        nodes, index = graph.nodelist_and_index()
        n = len(nodes)
        max_iter = self.max_iter if self.max_iter is not None else 100 * n
        num_edges = self.num_edges_param if self.num_edges_param is not None else int(math.ceil(n ** 0.25))

        # --- компактные структуры demands-графа
        def build_edge_index():
            E_u, E_v, E_w = [], [], []
            eid = {}
            for u, v, d in graph.demands_graph.edges(data=True):
                iu, iv = index[u], index[v]
                if iu > iv:
                    iu, iv = iv, iu
                eid[(iu, iv)] = len(E_w)
                E_u.append(iu); E_v.append(iv); E_w.append(float(d.get("weight", 1.0)))
            m = len(E_w)
            E_alive = np.ones(m, dtype=bool)
            return (np.array(E_u, dtype=np.int32),
                    np.array(E_v, dtype=np.int32),
                    np.array(E_w, dtype=float),
                    E_alive,
                    eid)

        def masks_for_cut(side: np.ndarray, E_u, E_v, E_alive):
            same = (side[E_u] == side[E_v])
            return (E_alive & same), (E_alive & ~same)

        def pick_idx(mask: np.ndarray, E_w: np.ndarray, mode: str) -> Optional[int]:
            if mask.size == 0 or E_w.size == 0:
                return None
            if not np.any(mask):
                return None
            if mode == "min":
                w = np.where(mask, E_w, np.inf); val = w.min()
                if not np.isfinite(val): return None
                cand = np.flatnonzero(w == val)
            else:
                w = np.where(mask, E_w, -np.inf); val = w.max()
                if not np.isfinite(val): return None
                cand = np.flatnonzero(w == val)
            j = np.random.randint(cand.size)
            return int(cand[j])

        # история
        removal_events: List[Dict[str, Any]] = []
        edge_mask_history: List[np.ndarray] = []
        edge_mask_snapshot_iters: List[int] = []
        SNAPSHOT_EVERY = 50

        def snapshot_mask(iter_idx: int, E_mask: np.ndarray):
            edge_mask_history.append(E_mask.copy())
            edge_mask_snapshot_iters.append(iter_idx)

        def remove_by_idx(idx: int, E_u, E_v, E_alive, E_w) -> Optional[Tuple[int, int, float]]:
            if not E_alive[idx]: return None
            iu, iv = int(E_u[idx]), int(E_v[idx])
            old = graph.remove_edge_by_indices(iu, iv)
            E_alive[idx] = False
            return iu, iv, old

        def upsert(iu: int, iv: int, delta_w: float,
                   E_u, E_v, E_alive, E_w, eid: Dict[Tuple[int, int], int]):
            if iu > iv:
                iu, iv = iv, iu
            new_w = graph.upsert_edge_by_indices(iu, iv, delta_w)
            key = (iu, iv)
            if key in eid:
                j = eid[key]
                E_w[j] = new_w
                E_alive[j] = True
                return E_u, E_v, E_alive, E_w
            else:
                j = E_w.size
                eid[key] = j
                E_u = np.append(E_u, iu)
                E_v = np.append(E_v, iv)
                E_w = np.append(E_w, new_w)
                E_alive = np.append(E_alive, True)
                return E_u, E_v, E_alive, E_w

        # индексы и стартовый снимок
        E_u, E_v, E_w, E_alive, eid = build_edge_index()
        snapshot_mask(0, E_alive)

        alpha_hist: List[float] = []
        edges_hist: List[int] = []
        medw_hist: List[float] = []

        it = 0
        for _ in range(max_iter):
            # Метрики на входе внешней итерации
            a = graph.calculate_alpha()
            alpha_hist.append(a)
            edges_hist.append(graph.demands_graph.number_of_edges())
            w_now = [d["weight"] for *_, d in graph.demands_graph.edges(data=True)]
            medw_hist.append(float(np.median(w_now)) if w_now else 0.0)

            diff = a - alpha_target
            if abs(diff) <= self.epsilon:
                break

            # фиксируем ОДИН разрез (side) на всю внешнюю итерацию
            if diff < -self.epsilon:
                # adversarial
                v = graph.generate_cut(type="adversarial")
                side = self._split_by_median(np.asarray(v, dtype=float))
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)

                for _k in range(num_edges):
                    # ВАЖНО: маски считаем внутри суб-итерации (после возможных изменений E_*)
                    mask_internal, _ = masks_for_cut(side, E_u, E_v, E_alive)

                    # УДАЛЕНИЕ (по монетке)
                    w_old_for_add = None
                    if np.random.rand() < self.p_for_delete_edge:
                        j = pick_idx(mask_internal, E_w, "min") or pick_idx(E_alive, E_w, "min")
                        if j is not None:
                            w_old_for_add = E_w[j]
                            removed = remove_by_idx(j, E_u, E_v, E_alive, E_w)
                            if removed is not None:
                                iu, iv, old = removed
                                removal_events.append({
                                    "iter": it + 1,
                                    "cut_type": "adversarial",
                                    "iu": int(iu), "iv": int(iv),
                                    "old_weight": float(old) if old is not None else None,
                                    "was_internal": bool(side[iu] == side[iv]),
                                })

                    # ДОБАВЛЕНИЕ (по монетке)
                    if np.random.rand() < self.p_for_upsert_edge and V1.size and V2.size:
                        iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
                        if w_old_for_add is None:
                            w_old_for_add = float(median_for_weights)
                        delta = self._draw_new_weight(w_old_for_add, ">", median_for_weights, var_for_weights)
                        E_u, E_v, E_alive, E_w = upsert(iu, iv, float(delta), E_u, E_v, E_alive, E_w, eid)

            else:
                # friendly
                v = graph.generate_cut(type="friendly")
                side = self._split_by_median(np.asarray(v, dtype=float))
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)

                for _k in range(num_edges):
                    # ВАЖНО: маски считаем внутри суб-итерации (после возможных изменений E_*)
                    mask_internal, _ = masks_for_cut(side, E_u, E_v, E_alive)

                    # УДАЛЕНИЕ (по монетке)
                    w_old_for_add = None
                    if np.random.rand() < self.p_for_delete_edge:
                        j = pick_idx(mask_internal, E_w, "max") or pick_idx(E_alive, E_w, "max")
                        if j is not None:
                            w_old_for_add = E_w[j]
                            removed = remove_by_idx(j, E_u, E_v, E_alive, E_w)
                            if removed is not None:
                                iu, iv, old = removed
                                removal_events.append({
                                    "iter": it + 1,
                                    "cut_type": "friendly",
                                    "iu": int(iu), "iv": int(iv),
                                    "old_weight": float(old) if old is not None else None,
                                    "was_internal": bool(side[iu] == side[iv]),
                                })

                    # ДОБАВЛЕНИЕ (по монетке)
                    if np.random.rand() < self.p_for_upsert_edge and V1.size and V2.size:
                        iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
                        if w_old_for_add is None:
                            w_old_for_add = float(median_for_weights)
                        delta = self._draw_new_weight(w_old_for_add, "<", median_for_weights, var_for_weights)
                        E_u, E_v, E_alive, E_w = upsert(iu, iv, float(delta), E_u, E_v, E_alive, E_w, eid)

            it += 1
            if it % SNAPSHOT_EVERY == 0:
                snapshot_mask(it, E_alive)

        if (it % SNAPSHOT_EVERY) != 0:
            snapshot_mask(it, E_alive)

        end = time.time()
        res = DemandsGenerationResultMulti(
            graph=graph,
            alpha_target=float(alpha_target),
            epsilon=self.epsilon,
            start_time=start,
            end_time=end,
            iterations_total=len(alpha_hist),
            alpha_history=[float(x) for x in alpha_hist],
            edge_counts_history=[int(x) for x in edges_hist],
            median_weights_history=[float(x) for x in medw_hist],
        )
        # прикладываем историю
        res.removal_events = removal_events
        res.edge_mask_history = edge_mask_history
        res.edge_mask_snapshot_iters = edge_mask_snapshot_iters
        res.algo_params = {
            "variant": "multi_edges",
            "p_ER": self.p_ER,
            "distribution": self.dist,
            "median_weight_for_initial": self.median_weight_for_initial,
            "var_for_initial": self.var_for_initial,
            "demands_median_denominator": self.median_div,
            "demands_var_denominator": self.var_div,
            "num_edges": (self.num_edges_param if self.num_edges_param is not None else int(math.ceil(graph.graph.number_of_nodes() ** 0.25))),
            "p_for_delete_edge": self.p_for_delete_edge,
            "p_for_upsert_edge": self.p_for_upsert_edge,
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
        }
        return res
