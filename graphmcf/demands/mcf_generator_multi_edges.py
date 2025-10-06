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
    Много-рёберная модификация базового генератора.

    В каждой внешней итерации выполняем num_edges суб-операций. Раньше это было строго
    (delete + upsert) по монеткам. Теперь обобщили:

      update_old(type=...):
        - 'delete'          — удалить ребро (как раньше)
        - 'reduce_weight'   — уменьшить вес на случайную дельту (<); если <=0 → удаляем
        - 'change_weight'   — friendly: уменьшить (как reduce_weight); adversarial: увеличить (>)

      update_new(type=...):
        - 'upsert'          — добавить/увеличить вес (как раньше)
        - 'upsert_nonexist' — добавлять только если ребра нет (существующее не трогаем)

    Вероятности p_for_delete_edge / p_for_upsert_edge остаются: управляют фактом вызова
    соответствующих обновлений. Типы обновлений задаются гиперпараметрами:
      - update_type_old: 'delete'|'reduce_weight'|'change_weight'  (default 'delete')
      - update_type_new: 'upsert'|'upsert_nonexist'               (default 'upsert')

    Истории:
      - удаление — в removal_events (для совместимости с анализом),
      - все изменения веса/добавления — в weight_update_events.
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
        p_for_delete_edge: float = 1.0,          # "монетка" перед update_old
        p_for_upsert_edge: float = 1.0,          # "монетка" перед update_new
        # --- новые гиперпараметры типов операций ---
        update_type_old: str = "delete",         # 'delete'|'reduce_weight'|'change_weight'
        update_type_new: str = "upsert",         # 'upsert'|'upsert_nonexist'
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
        ok_old = {"delete", "reduce_weight", "change_weight"}
        ok_new = {"upsert", "upsert_nonexist"}
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
            "action": str(action),                 # 'reduce','increase','upsert','upsert_nonexist'
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

        # ----------------- обобщённые операции -----------------

        def update_old(j: Optional[int], *,
                       cut_type: str,
                       update_type: str,           # 'delete' | 'reduce_weight' | 'change_weight'
                       side_mask: np.ndarray) -> Optional[float]:
            """
            Возвращает old_w (чтобы использовать при update_new) или None.
            Может менять E_* (добавление/удаление), поэтому делаем rebinding через nonlocal.
            """
            if j is None:
                return None

            nonlocal E_u, E_v, E_alive, E_w  # будем перепривязывать при изменении размеров массивов

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
                # уменьшить на дельту (<)
                delta = float(self._draw_new_weight(old_w, "<", median_for_weights, var_for_weights))
                E_u2, E_v2, E_alive2, E_w2, j2, new_w = self._upsert_pair(
                    iu, iv, -delta, graph, E_u, E_v, E_alive, E_w, eid
                )
                # перепривязка массивов (на случай добавления)
                E_u, E_v, E_alive, E_w = E_u2, E_v2, E_alive2, E_w2
                if new_w <= 0:
                    E_alive[j] = False
                    self._log_removal(removal_events, it + 1, cut_type, iu, iv, old_w, was_internal)
                else:
                    self._log_weight_update(weight_update_events, it + 1, cut_type,
                                            "reduce", iu, iv, old_w, -delta, new_w, was_internal)
                return old_w

            if update_type == "change_weight":
                if cut_type == "friendly":
                    # friendly → уменьшение
                    return update_old(j, cut_type=cut_type, update_type="reduce_weight", side_mask=side_mask)
                else:
                    # adversarial → увеличение
                    delta = float(self._draw_new_weight(old_w, ">", median_for_weights, var_for_weights))
                    E_u2, E_v2, E_alive2, E_w2, j2, new_w = self._upsert_pair(
                        iu, iv, +delta, graph, E_u, E_v, E_alive, E_w, eid
                    )
                    E_u, E_v, E_alive, E_w = E_u2, E_v2, E_alive2, E_w2
                    self._log_weight_update(weight_update_events, it + 1, cut_type,
                                            "increase", iu, iv, old_w, +delta, new_w, was_internal)
                    return old_w

            # неизвестный тип — ничего не делаем
            return None

        def update_new(*,
                       cut_type: str,
                       sign_mode: str,            # '>' | '<'  (адверсариал/френдли)
                       update_type: str,           # 'upsert' | 'upsert_nonexist'
                       V1: np.ndarray, V2: np.ndarray,
                       w_old_for_add: Optional[float]) -> None:
            """
            Добавление/усиление нового ребра между V1 и V2.
            При 'upsert_nonexist' существующее ребро не меняем.
            Может добавлять новые рёбра → rebinding через nonlocal.
            """
            if V1.size == 0 or V2.size == 0:
                return

            nonlocal E_u, E_v, E_alive, E_w  # на случай добавления новой записи

            iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
            if iu > iv:
                iu, iv = iv, iu
            key = (iu, iv)

            base_old = float(w_old_for_add) if (w_old_for_add is not None) else float(median_for_weights)
            delta = float(self._draw_new_weight(base_old, sign_mode, median_for_weights, var_for_weights))

            if update_type == "upsert_nonexist" and key in eid and E_alive[eid[key]]:
                # ребро уже есть — пропуск (логируем как факт)
                self._log_weight_update(weight_update_events, it + 1, cut_type,
                                        "upsert_nonexist", iu, iv,
                                        old_weight=float(E_w[eid[key]]),
                                        delta=None, new_weight=float(E_w[eid[key]]),
                                        was_internal=False)
                return

            # обычный upsert
            E_u2, E_v2, E_alive2, E_w2, j2, new_w = self._upsert_pair(
                iu, iv, +delta, graph, E_u, E_v, E_alive, E_w, eid
            )
            E_u, E_v, E_alive, E_w = E_u2, E_v2, E_alive2, E_w2
            self._log_weight_update(weight_update_events, it + 1, cut_type,
                                    "upsert", iu, iv,
                                    old_weight=None, delta=+delta, new_weight=new_w,
                                    was_internal=False)

        # --------------------------- основной цикл ---------------------------

        it = 0
        for _ in range(max_iter):
            # метрики на входе внешней итерации
            a = graph.calculate_alpha()
            alpha_hist.append(a)
            edges_hist.append(graph.demands_graph.number_of_edges())
            w_now = [d["weight"] for *_, d in graph.demands_graph.edges(data=True)]
            medw_hist.append(float(np.median(w_now)) if w_now else 0.0)

            diff = a - alpha_target
            if abs(diff) <= self.epsilon:
                break

            # ОДИН разрез на внешнюю итерацию
            if diff < -self.epsilon:
                # adversarial
                v = graph.generate_cut(type="adversarial")
                side = self._split_by_median(np.asarray(v, dtype=float))
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)

                for _k in range(num_edges):
                    mask_internal, _mask_cross = masks_for_cut(side, E_u, E_v, E_alive)

                    # update_old (по монетке)
                    w_old_for_add = None
                    if np.random.rand() < self.p_for_delete_edge:
                        j = pick_idx(mask_internal, E_w, "min") or pick_idx(E_alive, E_w, "min")
                        w_old_for_add = update_old(j,
                                                   cut_type="adversarial",
                                                   update_type=self.update_type_old,
                                                   side_mask=side)

                    # update_new (по монетке)
                    if np.random.rand() < self.p_for_upsert_edge:
                        update_new(cut_type="adversarial",
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
                    mask_internal, _mask_cross = masks_for_cut(side, E_u, E_v, E_alive)

                    # update_old (по монетке)
                    w_old_for_add = None
                    if np.random.rand() < self.p_for_delete_edge:
                        j = pick_idx(mask_internal, E_w, "max") or pick_idx(E_alive, E_w, "max")
                        w_old_for_add = update_old(j,
                                                   cut_type="friendly",
                                                   update_type=self.update_type_old,
                                                   side_mask=side)

                    # update_new (по монетке)
                    if np.random.rand() < self.p_for_upsert_edge:
                        update_new(cut_type="friendly",
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
        )
        # прикладываем историю
        res.removal_events = removal_events
        res.weight_update_events = weight_update_events
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
            "update_type_old": self.update_type_old,
            "update_type_new": self.update_type_new,
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
        }
        return res
