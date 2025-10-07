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

    # динамически добавляемые поля:
    #   removal_events: List[Dict[str, Any]]
    #   weight_update_events: List[Dict[str, Any]]
    #   edge_mask_history: List[np.ndarray]
    #   edge_mask_snapshot_iters: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_target": self.alpha_target,
            "epsilon": self.epsilon,
            "execution_time": float(self.end_time - self.start_time),
            "iterations_total": self.iterations_total,
            "edges_final": int(self.graph.demands_graph.number_of_edges()) if self.graph.demands_graph else 0,
        }

class MCFGeneratorMultiEdges:
    """
    Много-рёберная модификация базового генератора.

    Обычный режим (несинхронный):
      • в каждой внешней итерации — num_edges суб-операций;
      • каждая суб-операция по монеткам выполняет update_old + update_new;
      • update_old ∈ {'delete','reduce_weight','change_weight'}
      • update_new ∈ {'upsert','upsert_nonexist'}

    Синхронный режим:
      • если update_type_old == update_type_new == 'replace_weight' —
        запускаем ОСОБУЮ ветку без монеток: в каждой суб-операции списываем дельту
        с выбранного ребра и ровно эту же дельту добавляем на новое ребро.
      • если только один из типов равен 'replace_weight' — печатаем предупреждение и
        НЕ запускаем процесс (возвращаем результат после инициализации).
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
        # --- много-рёберные настройки ---
        num_edges: Optional[int] = None,         # если None -> ceil(n ** 0.25)
        p_for_delete_edge: float = 1.0,          # "монетка" перед update_old (в обычном режиме)
        p_for_upsert_edge: float = 1.0,          # "монетка" перед update_new (в обычном режиме)
        # --- типы операций ---
        update_type_old: str = "delete",         # 'delete'|'reduce_weight'|'change_weight'|'replace_weight'
        update_type_new: str = "upsert",         # 'upsert'|'upsert_nonexist'|'replace_weight'
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

        # типы апдейтов
        self.update_type_old = str(update_type_old).lower().strip()
        self.update_type_new = str(update_type_new).lower().strip()
        self._validate_update_types()

        self.max_iter = max_iter  # если None → 100*|V|

    def _validate_update_types(self) -> None:
        ok_old = {"delete", "reduce_weight", "change_weight", "replace_weight"}
        ok_new = {"upsert", "upsert_nonexist", "replace_weight"}
        if self.update_type_old not in ok_old:
            raise ValueError(f"update_type_old must be one of {ok_old}, got {self.update_type_old!r}")
        if self.update_type_new not in ok_new:
            raise ValueError(f"update_type_new must be one of {ok_new}, got {self.update_type_new!r}")

    # ---------------------- служебные методы ----------------------

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

    # --- низкоуровневые операции над E_* ---

    def _remove_by_idx(self, idx: int, graph: GraphMCF,
                       E_u, E_v, E_alive, E_w) -> Optional[Tuple[int, int, float]]:
        if not E_alive[idx]:
            return None
        iu, iv = int(E_u[idx]), int(E_v[idx])
        old = graph.remove_edge_by_indices(iu, iv)
        E_alive[idx] = False
        return iu, iv, old

    def _upsert_pair(self, iu: int, iv: int, delta_w: float,
                     graph: GraphMCF,
                     E_u, E_v, E_alive, E_w, eid: Dict[Tuple[int, int], int]):
        if iu > iv:
            iu, iv = iv, iu
        new_w = graph.upsert_edge_by_indices(iu, iv, delta_w)
        key = (iu, iv)
        if key in eid:
            j = eid[key]
            E_w[j] = new_w
            E_alive[j] = True
            return E_u, E_v, E_alive, E_w, j, new_w
        else:
            j = E_w.size
            eid[key] = j
            E_u = np.append(E_u, iu)
            E_v = np.append(E_v, iv)
            E_w = np.append(E_w, new_w)
            E_alive = np.append(E_alive, True)
            return E_u, E_v, E_alive, E_w, j, new_w

    # --- истории изменений ---

    @staticmethod
    def _log_removal(removal_events: List[Dict[str, Any]],
                     iter_idx: int, cut_type: str,
                     iu: int, iv: int, old_weight: Optional[float],
                     was_internal: bool) -> None:
        removal_events.append({
            "iter": int(iter_idx),
            "cut_type": str(cut_type),
            "iu": int(iu), "iv": int(iv),
            "old_weight": (float(old_weight) if old_weight is not None else None),
            "was_internal": bool(was_internal),
        })

    @staticmethod
    def _log_weight_update(weight_update_events: List[Dict[str, Any]],
                           iter_idx: int, cut_type: str,
                           action: str, iu: int, iv: int,
                           old_weight: Optional[float],
                           delta: Optional[float],
                           new_weight: Optional[float],
                           was_internal: Optional[bool]) -> None:
        weight_update_events.append({
            "iter": int(iter_idx),
            "cut_type": str(cut_type),
            "action": str(action),   # 'reduce','increase','upsert','upsert_nonexist','replace_reduce','replace_upsert'
            "iu": int(iu), "iv": int(iv),
            "old_weight": (float(old_weight) if old_weight is not None else None),
            "delta": (float(delta) if delta is not None else None),
            "new_weight": (float(new_weight) if new_weight is not None else None),
            "was_internal": (bool(was_internal) if was_internal is not None else None),
        })

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

        # 1) параметры для добавлений от capacity
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
        weight_update_events: List[Dict[str, Any]] = []
        edge_mask_history: List[np.ndarray] = []
        edge_mask_snapshot_iters: List[int] = []
        SNAPSHOT_EVERY = 50

        def snapshot_mask(iter_idx: int, E_mask: np.ndarray):
            edge_mask_history.append(E_mask.copy())
            edge_mask_snapshot_iters.append(iter_idx)

        # индексы и стартовый снимок
        E_u, E_v, E_w, E_alive, eid = build_edge_index()
        snapshot_mask(0, E_alive)

        alpha_hist: List[float] = []
        edges_hist: List[int] = []
        medw_hist: List[float] = []

        # заранее объявим it, чтобы вложенные функции могли ссылаться
        it = 0

        # ----------------- вспомогательные операции для обычного режима -----------------

        def _do_update_old(j: Optional[int], *,
                           cut_type: str,
                           update_type: str,           # 'delete' | 'reduce_weight' | 'change_weight'
                           side_mask: np.ndarray) -> Optional[float]:
            if j is None:
                return None
            nonlocal E_u, E_v, E_alive, E_w

            iu, iv = int(E_u[j]), int(E_v[j])
            was_internal = bool(side_mask[iu] == side_mask[iv])
            old_w = float(E_w[j])

            if update_type == "delete":
                removed = self._remove_by_idx(j, graph, E_u, E_v, E_alive, E_w)
                if removed is not None:
                    iu2, iv2, old = removed
                    self._log_removal(removal_events, it + 1, cut_type, iu2, iv2, old, was_internal)
                    return old_w
                return None

            if update_type == "reduce_weight":
                delta = float(self._draw_new_weight(old_w, "<", median_for_weights, var_for_weights))
                if old_w - delta <= 0:
                    removed = self._remove_by_idx(j, graph, E_u, E_v, E_alive, E_w)
                    if removed is not None:
                        iu2, iv2, old = removed
                        self._log_removal(removal_events, it + 1, cut_type, iu2, iv2, old, was_internal)
                    return old_w
                else:
                     E_u2, E_v2, E_alive2, E_w2, j2, new_w = self._upsert_pair(
                         iu, iv, -delta, graph, E_u, E_v, E_alive, E_w, eid
                     )
                     E_u, E_v, E_alive, E_w = E_u2, E_v2, E_alive2, E_w2
                     self._log_weight_update(weight_update_events, it + 1, cut_type,
                                             "reduce", iu, iv, old_w, -delta, new_w, was_internal)
                     return old_w
                    
            if update_type == "change_weight":
                if cut_type == "friendly":
                    return _do_update_old(j, cut_type=cut_type, update_type="reduce_weight", side_mask=side_mask)
                else:
                    delta = float(self._draw_new_weight(old_w, ">", median_for_weights, var_for_weights))
                    E_u2, E_v2, E_alive2, E_w2, j2, new_w = self._upsert_pair(
                        iu, iv, +delta, graph, E_u, E_v, E_alive, E_w, eid
                    )
                    E_u, E_v, E_alive, E_w = E_u2, E_v2, E_alive2, E_w2
                    self._log_weight_update(weight_update_events, it + 1, cut_type,
                                            "increase", iu, iv, old_w, +delta, new_w, was_internal)
                    return old_w

            return None

        def _do_update_new(*,
                           cut_type: str,
                           sign_mode: str,            # '>' | '<'
                           update_type: str,           # 'upsert' | 'upsert_nonexist'
                           V1: np.ndarray, V2: np.ndarray,
                           w_old_for_add: Optional[float]) -> None:
            if V1.size == 0 or V2.size == 0:
                return
            nonlocal E_u, E_v, E_alive, E_w

            iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
            if iu > iv:
                iu, iv = iv, iu
            key = (iu, iv)

            base_old = float(w_old_for_add) if (w_old_for_add is not None) else float(median_for_weights)
            delta = float(self._draw_new_weight(base_old, sign_mode, median_for_weights, var_for_weights))

            if update_type == "upsert_nonexist" and key in eid and E_alive[eid[key]]:
                self._log_weight_update(weight_update_events, it + 1, cut_type,
                                        "upsert_nonexist", iu, iv,
                                        old_weight=float(E_w[eid[key]]),
                                        delta=None, new_weight=float(E_w[eid[key]]),
                                        was_internal=False)
                return

            E_u2, E_v2, E_alive2, E_w2, j2, new_w = self._upsert_pair(
                iu, iv, +delta, graph, E_u, E_v, E_alive, E_w, eid
            )
            E_u, E_v, E_alive, E_w = E_u2, E_v2, E_alive2, E_w2
            self._log_weight_update(weight_update_events, it + 1, cut_type,
                                    "upsert", iu, iv,
                                    old_weight=None, delta=+delta, new_weight=new_w,
                                    was_internal=False)

        # --------------------------- ВЕТВЛЕНИЕ ПО РЕЖИМУ ---------------------------

        sync_replace = (self.update_type_old == "replace_weight" and
                        self.update_type_new == "replace_weight")
        replace_mismatch = ((self.update_type_old == "replace_weight") ^
                            (self.update_type_new == "replace_weight"))

        # === (A) Конфигурация несинхронная для replace_weight → не запускаем ===
        if replace_mismatch:
            print("[MCFGeneratorMultiEdges] WARN: Для синхронного режима необходимо "
                  "update_type_old == update_type_new == 'replace_weight'. Запуск отменён.")
            a0 = graph.calculate_alpha()
            alpha_hist.append(a0)
            edges_hist.append(graph.demands_graph.number_of_edges())
            w0 = [d["weight"] for *_, d in graph.demands_graph.edges(data=True)]
            medw_hist.append(float(np.median(w0)) if w0 else 0.0)
            end = time.time()
            res = DemandsGenerationResultMulti(
                graph=graph,
                alpha_target=float(alpha_target),
                epsilon=self.epsilon,
                start_time=start,
                end_time=end,
                iterations_total=0,
                alpha_history=[float(x) for x in alpha_hist],
                edge_counts_history=[int(x) for x in edges_hist],
                median_weights_history=[float(x) for x in medw_hist],
                algo_params={
                    "variant": "multi_edges",
                    "mode": "aborted_mismatch_replace_weight",
                    "p_ER": self.p_ER,
                    "distribution": self.dist,
                    "median_weight_for_initial": self.median_weight_for_initial,
                    "var_for_initial": self.var_for_initial,
                    "demands_median_denominator": self.median_div,
                    "demands_var_denominator": self.var_div,
                    "num_edges": (self.num_edges_param if self.num_edges_param is not None else int(math.ceil(n ** 0.25))),
                    "p_for_delete_edge": self.p_for_delete_edge,
                    "p_for_upsert_edge": self.p_for_upsert_edge,
                    "update_type_old": self.update_type_old,
                    "update_type_new": self.update_type_new,
                    "epsilon": self.epsilon,
                    "max_iter": self.max_iter,
                }
            )
            res.removal_events = []
            res.weight_update_events = []
            res.edge_mask_history = edge_mask_history
            res.edge_mask_snapshot_iters = edge_mask_snapshot_iters
            return res

        # === (B) СИНХРОННЫЙ РЕЖИМ: replace_weight без монеток ===
        if sync_replace:
            for _ in range(max_iter):
                a = graph.calculate_alpha()
                alpha_hist.append(a)
                edges_hist.append(graph.demands_graph.number_of_edges())
                w_now = [d["weight"] for *_, d in graph.demands_graph.edges(data=True)]
                medw_hist.append(float(np.median(w_now)) if w_now else 0.0)

                diff = a - alpha_target
                if abs(diff) <= self.epsilon:
                    break

                if diff < -self.epsilon:
                    # adversarial
                    v = graph.generate_cut(type="adversarial")
                    side = self._split_by_median(np.asarray(v, dtype=float))
                    V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)

                    for _k in range(num_edges):
                        mask_internal, _ = masks_for_cut(side, E_u, E_v, E_alive)
                        j_old = pick_idx(mask_internal, E_w, "min") or pick_idx(E_alive, E_w, "min")
                        if j_old is None or V1.size == 0 or V2.size == 0:
                            continue

                        iu_old, iv_old = int(E_u[j_old]), int(E_v[j_old])
                        was_internal = bool(side[iu_old] == side[iv_old])
                        old_w = float(E_w[j_old])
                        delta = float(self._draw_new_weight(old_w, "<", median_for_weights, var_for_weights))
                        if old_w - delta <= 0:
                            removed = self._remove_by_idx(j_old, graph, E_u, E_v, E_alive, E_w)
                            if removed is not None:
                                iu2, iv2, old = removed
                                self._log_removal(removal_events, it + 1, cut_type, iu2, iv2, old,
                                                  was_internal=bool(side[iu2] == side[iv2]))
                        else:
                            E_u2, E_v2, E_alive2, E_w2, j2, new_w_old = self._upsert_pair(
                                iu_old, iv_old, -delta, graph, E_u, E_v, E_alive, E_w, eid
                            )
                            E_u, E_v, E_alive, E_w = E_u2, E_v2, E_alive2, E_w2
                            self._log_weight_update(weight_update_events, it + 1, cut_type,
                                                    "replace_reduce", iu_old, iv_old,
                                                    old_weight=old_w, delta=-delta, new_weight=new_w_old,
                                                    was_internal=bool(side[iu_old] == side[iv_old]))

                        iu_new = int(np.random.choice(V1)); iv_new = int(np.random.choice(V2))
                        E_u2, E_v2, E_alive2, E_w2, j_new, new_w_new = self._upsert_pair(
                            iu_new, iv_new, +delta, graph, E_u, E_v, E_alive, E_w, eid
                        )
                        E_u, E_v, E_alive, E_w = E_u2, E_v2, E_alive2, E_w2
                        self._log_weight_update(weight_update_events, it + 1, "adversarial",
                                                "replace_upsert", iu_new, iv_new,
                                                old_weight=None, delta=+delta, new_weight=new_w_new,
                                                was_internal=False)

                else:
                    # friendly
                    v = graph.generate_cut(type="friendly")
                    side = self._split_by_median(np.asarray(v, dtype=float))
                    V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)

                    for _k in range(num_edges):
                        mask_internal, _ = masks_for_cut(side, E_u, E_v, E_alive)
                        j_old = pick_idx(mask_internal, E_w, "max") or pick_idx(E_alive, E_w, "max")
                        if j_old is None or V1.size == 0 or V2.size == 0:
                            continue

                        iu_old, iv_old = int(E_u[j_old]), int(E_v[j_old])
                        was_internal = bool(side[iu_old] == side[iv_old])
                        old_w = float(E_w[j_old])
                        delta = float(self._draw_new_weight(old_w, "<", median_for_weights, var_for_weights))
                        if old_w - delta <= 0:
                            removed = self._remove_by_idx(j_old, graph, E_u, E_v, E_alive, E_w)
                            if removed is not None:
                                iu2, iv2, old = removed
                                self._log_removal(removal_events, it + 1, cut_type, iu2, iv2, old,
                                                  was_internal=bool(side[iu2] == side[iv2]))
                        else:
                            E_u2, E_v2, E_alive2, E_w2, j2, new_w_old = self._upsert_pair(
                                iu_old, iv_old, -delta, graph, E_u, E_v, E_alive, E_w, eid
                            )
                            E_u, E_v, E_alive, E_w = E_u2, E_v2, E_alive2, E_w2
                            self._log_weight_update(weight_update_events, it + 1, cut_type,
                                                    "replace_reduce", iu_old, iv_old,
                                                    old_weight=old_w, delta=-delta, new_weight=new_w_old,
                                                    was_internal=bool(side[iu_old] == side[iv_old]))

                        iu_new = int(np.random.choice(V1)); iv_new = int(np.random.choice(V2))
                        E_u2, E_v2, E_alive2, E_w2, j_new, new_w_new = self._upsert_pair(
                            iu_new, iv_new, +delta, graph, E_u, E_v, E_alive, E_w, eid
                        )
                        E_u, E_v, E_alive, E_w = E_u2, E_v2, E_alive2, E_w2
                        self._log_weight_update(weight_update_events, it + 1, "friendly",
                                                "replace_upsert", iu_new, iv_new,
                                                old_weight=None, delta=+delta, new_weight=new_w_new,
                                                was_internal=False)

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
                algo_params={
                    "variant": "multi_edges",
                    "mode": "sync_replace_weight",
                    "p_ER": self.p_ER,
                    "distribution": self.dist,
                    "median_weight_for_initial": self.median_weight_for_initial,
                    "var_for_initial": self.var_for_initial,
                    "demands_median_denominator": self.median_div,
                    "demands_var_denominator": self.var_div,
                    "num_edges": (self.num_edges_param if self.num_edges_param is not None else int(math.ceil(n ** 0.25))),
                    "update_type_old": self.update_type_old,
                    "update_type_new": self.update_type_new,
                    "epsilon": self.epsilon,
                    "max_iter": self.max_iter,
                }
            )
            res.removal_events = removal_events
            res.weight_update_events = weight_update_events
            res.edge_mask_history = edge_mask_history
            res.edge_mask_snapshot_iters = edge_mask_snapshot_iters
            return res

        # === (C) Обычный (несинхронный) режим ===
        for _ in range(max_iter):
            a = graph.calculate_alpha()
            alpha_hist.append(a)
            edges_hist.append(graph.demands_graph.number_of_edges())
            w_now = [d["weight"] for *_, d in graph.demands_graph.edges(data=True)]
            medw_hist.append(float(np.median(w_now)) if w_now else 0.0)

            diff = a - alpha_target
            if abs(diff) <= self.epsilon:
                break

            if diff < -self.epsilon:
                # adversarial
                v = graph.generate_cut(type="adversarial")
                side = self._split_by_median(np.asarray(v, dtype=float))
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)

                for _k in range(num_edges):
                    mask_internal, _ = masks_for_cut(side, E_u, E_v, E_alive)

                    # update_old (по монетке)
                    w_old_for_add = None
                    if np.random.rand() < self.p_for_delete_edge:
                        j = pick_idx(mask_internal, E_w, "min") or pick_idx(E_alive, E_w, "min")
                        w_old_for_add = _do_update_old(j,
                                                       cut_type="adversarial",
                                                       update_type=self.update_type_old,
                                                       side_mask=side)
                    # update_new (по монетке)
                    if np.random.rand() < self.p_for_upsert_edge:
                        _do_update_new(cut_type="adversarial",
                                       sign_mode=">",
                                       update_type=self.update_type_new,
                                       V1=V1, V2=V2,
                                       w_old_for_add=w_old_for_add)

            else:
                # friendly
                v = graph.generate_cut(type="friendly")
                side = self._split_by_median(np.asarray(v, dtype=float))
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)

                for _k in range(num_edges):
                    mask_internal, _ = masks_for_cut(side, E_u, E_v, E_alive)

                    w_old_for_add = None
                    if np.random.rand() < self.p_for_delete_edge:
                        j = pick_idx(mask_internal, E_w, "max") or pick_idx(E_alive, E_w, "max")
                        w_old_for_add = _do_update_old(j,
                                                       cut_type="friendly",
                                                       update_type=self.update_type_old,
                                                       side_mask=side)
                    if np.random.rand() < self.p_for_upsert_edge:
                        _do_update_new(cut_type="friendly",
                                       sign_mode="<",
                                       update_type=self.update_type_new,
                                       V1=V1, V2=V2,
                                       w_old_for_add=w_old_for_add)

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
            algo_params={
                "variant": "multi_edges",
                "mode": "async_standard",
                "p_ER": self.p_ER,
                "distribution": self.dist,
                "median_weight_for_initial": self.median_weight_for_initial,
                "var_for_initial": self.var_for_initial,
                "demands_median_denominator": self.median_div,
                "demands_var_denominator": self.var_div,
                "num_edges": (self.num_edges_param if self.num_edges_param is not None else int(math.ceil(n ** 0.25))),
                "p_for_delete_edge": self.p_for_delete_edge,
                "p_for_upsert_edge": self.p_for_upsert_edge,
                "update_type_old": self.update_type_old,
                "update_type_new": self.update_type_new,
                "epsilon": self.epsilon,
                "max_iter": self.max_iter,
            }
        )
        res.removal_events = removal_events
        res.weight_update_events = weight_update_events
        res.edge_mask_history = edge_mask_history
        res.edge_mask_snapshot_iters = edge_mask_snapshot_iters
        return res
        
