from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import time
import math
import numpy as np
import networkx as nx

from ..core import GraphMCF

@dataclass
class DemandsGenerationResultMultiGraph:
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

    # динамически прикладываемые поля:
    #   removal_events: List[Dict[str, Any]]        # удаления АГРЕГИРОВАННЫХ рёбер (когда суммарный вес стал 0)
    #   weight_update_events: List[Dict[str, Any]]  # "multiremove" и "multiadd" событий
    #   edge_mask_history: List[np.ndarray]         # снимки маски живых АГРЕГИРОВАННЫХ рёбер
    #   edge_mask_snapshot_iters: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_target": self.alpha_target,
            "epsilon": self.epsilon,
            "execution_time": float(self.end_time - self.start_time),
            "iterations_total": self.iterations_total,
            "edges_final": int(self.graph.demands_graph.number_of_edges()) if self.graph.demands_graph else 0,
        }

class MCFGeneratorMultiGraph:
    """
    Генератор с мультирёбрами:
      • работаем с двумя синхронизированными структурами:
          - graph.demands_multigraph : nx.MultiGraph с отдельными мультирёбрами (каждое со своим весом)
          - graph.demands_graph      : nx.Graph, где вес = сумма мульти-весов по паре вершин
      • на каждом внешнем шаге считаем alpha и выбираем тип разреза (friendly/adversarial);
      • внутри внешней итерации выполняем num_edges «перекидываний»:
          1) выбрать мультиребро в demands_multigraph «под» агрегированным ребром,
             выбранным по старым правилам (min/max среди внутрикластерных; при отсутствии — среди всех);
          2) удалить выбранное мультиребро (в мультиграфе) и вычесть его вес из агрегированного графа
             (если суммарный вес пары стал ≤0 — удалить агрегированное ребро физически);
          3) добавить новое мультиребро с тем же весом между случайными вершинами из разных долей разреза
             и прибавить этот вес в агрегированном графе.

    Монеток здесь НЕТ — скорость регулируем числом подшагов num_edges.

    История и вычисление alpha ведутся по demands_graph, как и прежде.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.05,
        # --- initial multidemands (ER + веса для мультиграфа) ---
        p_ER: float = 0.5,
        distribution: str = "normal",
        median_weight_for_initial: int = 50,
        var_for_initial: int = 25,
        multi_max: int = 25,
        # --- скорость сходимости ---
        num_edges: Optional[int] = None,   # если None -> ceil(n ** 0.25)
        # ---
        max_iter: Optional[int] = None,
    ) -> None:
        self.epsilon = float(epsilon)

        # параметры первичной генерации multi-demands
        self.p_ER = float(p_ER)
        self.dist = str(distribution)
        self.median_weight_for_initial = int(median_weight_for_initial)
        self.var_for_initial = int(var_for_initial)
        self.multi_max = int(multi_max)

        self.num_edges_param = None if num_edges is None else int(num_edges)
        self.max_iter = max_iter  # если None → 100*|V|

    # ------------------------------------------------------------------ utils
    def _split_by_median(self, vec: np.ndarray) -> np.ndarray:
        med = float(np.median(vec))
        side = (vec <= med)
        if side.sum() in (0, len(side)):
            side = (vec < med)
        if side.sum() in (0, len(side)):
            side = np.zeros_like(side, dtype=bool); side[0] = True
        return side

    # ------------------------------- generate --------------------------------
    def generate(
        self,
        graph: GraphMCF,
        alpha_target: float = 0.5,
        analysis_mode: Optional[str] = None,  # анализ запускается отдельно
    ) -> DemandsGenerationResultMultiGraph:

        # 0) первичная генерация мультирёбер + агрегирования
        start = time.time()
        # ОЖИДАЕМ наличие метода, инициализирующего оба графа:
        #   - graph.demands_multigraph : nx.MultiGraph
        #   - graph.demands_graph      : nx.Graph (агрегат по весам)
        graph.generate_initial_multidemands(
            p=self.p_ER,
            distribution=self.dist,
            median_weight=self.median_weight_for_initial,
            var=self.var_for_initial,
            multi_max=self.multi_max
        )
        assert isinstance(graph.demands_multigraph, nx.MultiGraph), "demands_multigraph должен быть nx.MultiGraph"
        assert isinstance(graph.demands_graph, nx.Graph), "demands_graph должен быть nx.Graph"

        # 1) подготовим компактные структуры для АГРЕГИРОВАННОГО графа
        nodes, index = graph.nodelist_and_index()
        n = len(nodes)
        max_iter = self.max_iter if self.max_iter is not None else 100 * n
        num_edges = self.num_edges_param if self.num_edges_param is not None else int(math.ceil(n ** 0.25))

        def build_agg_index():
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

        # 2) подготовим индекс для МУЛЬТИграфа
        def build_multi_index(eid: Dict[Tuple[int, int], int]):
            EM_u, EM_v, EM_w, EM_key = [], [], [], []
            EM_alive: List[bool] = []
            # сопоставления:
            multi_to_agg: List[int] = []         # mid -> j (индекс агрег. ребра)
            agg_to_multi: Dict[int, List[int]] = {}  # j -> [mid,...]
            meid: Dict[Tuple[int, int, int], int] = {}

            for u, v, key, d in graph.demands_multigraph.edges(keys=True, data=True):
                iu, iv = index[u], index[v]
                if iu > iv:
                    iu, iv = iv, iu
                w = float(d.get("weight", 1.0))
                j = eid[(iu, iv)]  # должен существовать, т.к. агрегат — сумма мульти-весов
                mid = len(EM_w)

                EM_u.append(iu); EM_v.append(iv); EM_w.append(w); EM_key.append(int(key))
                EM_alive.append(True)
                multi_to_agg.append(j)
                agg_to_multi.setdefault(j, []).append(mid)
                meid[(iu, iv, int(key))] = mid

            return (np.array(EM_u, dtype=np.int32),
                    np.array(EM_v, dtype=np.int32),
                    np.array(EM_w, dtype=float),
                    np.array(EM_key, dtype=np.int32),
                    np.array(EM_alive, dtype=bool),
                    multi_to_agg, agg_to_multi, meid)

        def masks_for_cut(side: np.ndarray, E_u, E_v, E_alive):
            same = (side[E_u] == side[E_v])
            return (E_alive & same), (E_alive & ~same)

        def pick_idx(mask: np.ndarray, E_w: np.ndarray, mode: str) -> Optional[int]:
            if mask.size == 0 or E_w.size == 0 or not np.any(mask):
                return None
            if mode == "min":
                w = np.where(mask, E_w, np.inf); val = np.min(w)
                if not np.isfinite(val): return None
                cand = np.flatnonzero(w == val)
            else:
                w = np.where(mask, E_w, -np.inf); val = np.max(w)
                if not np.isfinite(val): return None
                cand = np.flatnonzero(w == val)
            j = np.random.randint(cand.size)
            return int(cand[j])

        # --- построить индексы
        E_u, E_v, E_w, E_alive, eid = build_agg_index()
        (EM_u, EM_v, EM_w, EM_key, EM_alive,
         multi_to_agg, agg_to_multi, meid) = build_multi_index(eid)

        # --- истории
        removal_events: List[Dict[str, Any]] = []
        weight_update_events: List[Dict[str, Any]] = []
        edge_mask_history: List[np.ndarray] = []
        edge_mask_snapshot_iters: List[int] = []
        SNAPSHOT_EVERY = 50

        def snapshot_mask(iter_idx: int, E_mask: np.ndarray):
            edge_mask_history.append(E_mask.copy())
            edge_mask_snapshot_iters.append(iter_idx)

        snapshot_mask(0, E_alive)

        # ------------ низкоуровневые операции с мульти/агрегированным графом ------------

        def _remove_multiedge_by_mid(mid: int) -> Tuple[int, int, float]:
            """
            Удаляет мультиребро mid из demands_multigraph и корректно синхронизирует агрегат:
              - если old_sum - w <= 0 → физически удаляет агрегированное ребро (без upsert(-w));
              - иначе → уменьшает его вес на w через upsert(-w).
            Возвращает (iu, iv, w_removed).
            """
            nonlocal E_u, E_v, E_w, E_alive
            nonlocal EM_u, EM_v, EM_w, EM_key, EM_alive
            nonlocal multi_to_agg, agg_to_multi, meid

            iu, iv, key, w = int(EM_u[mid]), int(EM_v[mid]), int(EM_key[mid]), float(EM_w[mid])
            if not EM_alive[mid]:
                return iu, iv, 0.0

            # 1) удалить мультиребро из мультиграфа
            u_node, v_node = nodes[iu], nodes[iv]
            graph.demands_multigraph.remove_edge(u_node, v_node, key=key)
            EM_alive[mid] = False

            # 2) синхронизировать агрегированный demands_graph
            j = multi_to_agg[mid]          # индекс агрегированного ребра
            old_sum = float(E_w[j])        # текущая сумма весов по паре (iu, iv)
            new_est = old_sum - w

            if new_est <= 0:
                # Полное удаление агрегированного ребра: НИКАКОГО upsert(-w)
                _ = graph.remove_edge_by_indices(iu, iv)
                E_alive[j] = False
                E_w[j] = 0.0
            else:
                # Частичное уменьшение — безопасно через upsert(-w)
                new_w = graph.upsert_edge_by_indices(iu, iv, -w)
                E_w[j] = float(new_w)
                E_alive[j] = True

            return iu, iv, w

        def _append_multiedge(iu: int, iv: int, w: float) -> int:
            """
            Создаёт мультиребро весом w между iu,iv в мультиграфе, возвращает mid.
            Обновляет агрегированный demands_graph (+w) и Е_* / EM_* структуры.
            """
            nonlocal E_u, E_v, E_w, E_alive
            nonlocal EM_u, EM_v, EM_w, EM_key, EM_alive
            nonlocal multi_to_agg, agg_to_multi, meid

            if iu > iv:
                iu, iv = iv, iu
            u_node, v_node = nodes[iu], nodes[iv]

            # сохраним существующие ключи до добавления
            existing_keys = set()
            if graph.demands_multigraph.has_edge(u_node, v_node):
                existing_keys = set(graph.demands_multigraph[u_node][v_node].keys())

            graph.demands_multigraph.add_edge(u_node, v_node, weight=float(w))
            # определим добавленный key
            new_keys = set(graph.demands_multigraph[u_node][v_node].keys())
            new_key_candidates = list(new_keys - existing_keys)
            if not new_key_candidates:
                # fallback: возьмём любой ключ (не должно случаться)
                new_key = max(new_keys) if new_keys else 0
            else:
                new_key = int(new_key_candidates[0])

            # обновим агрегированный граф
            new_sum_w = graph.upsert_edge_by_indices(iu, iv, +float(w))

            # обновим индексы агрегированного графа
            key = (iu, iv)
            if key in eid:
                j = eid[key]
                E_w[j] = new_sum_w
                E_alive[j] = True
            else:
                # новое агрегированное ребро → добавить запись
                j = E_w.size
                eid[key] = j
                E_u = np.append(E_u, iu)
                E_v = np.append(E_v, iv)
                E_w = np.append(E_w, new_sum_w)
                E_alive = np.append(E_alive, True)

            # зарегистрировать мультиребро
            mid = EM_w.size
            EM_u = np.append(EM_u, iu)
            EM_v = np.append(EM_v, iv)
            EM_w = np.append(EM_w, float(w))
            EM_key = np.append(EM_key, int(new_key))
            EM_alive = np.append(EM_alive, True)
            multi_to_agg.append(j)
            agg_to_multi.setdefault(j, []).append(mid)
            meid[(iu, iv, int(new_key))] = mid
            return mid

        def _pick_multiedge_under_agg(j: int) -> Optional[int]:
            """Случайно выбирает живое мультиребро, лежащее под агрегированным ребром j."""
            mids = agg_to_multi.get(j, [])
            if not mids:
                return None
            alive = [mid for mid in mids if EM_alive[mid]]
            if not alive:
                return None
            return int(np.random.choice(alive))

        # --------------------------- основной цикл ---------------------------

        alpha_hist: List[float] = []
        edges_hist: List[int] = []
        medw_hist: List[float] = []

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

            # один разрез на внешнюю итерацию
            if diff < -self.epsilon:
                # adversarial: тянем внутреннее ребро с МИН суммарным весом
                v = graph.generate_cut(type="adversarial")
                side = self._split_by_median(np.asarray(v, dtype=float))
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)

                for _k in range(num_edges):
                    mask_internal, _mask_cross = masks_for_cut(side, E_u, E_v, E_alive)

                    # шаг 1: выбрать агрег. ребро j, затем мультиребро под ним
                    j = pick_idx(mask_internal, E_w, mode="min") or pick_idx(E_alive, E_w, mode="min")
                    if j is None:
                        continue
                    mid = _pick_multiedge_under_agg(j)
                    if mid is None:
                        # редкий случай: нет живых мульти-ребер под агрегатным (несогласованность)
                        # fallback: пропустим подшаг
                        continue

                    # шаг 2: удалить выбранное мультиребро (и уменьшить агрегат)
                    iu_old, iv_old, w_removed = _remove_multiedge_by_mid(mid)
                    # логирование «мульти-удаления»
                    weight_update_events.append({
                        "iter": it + 1, "cut_type": "adversarial",
                        "action": "multiremove",
                        "iu": int(iu_old), "iv": int(iv_old),
                        "delta": -float(w_removed),
                    })
                    # если агрегатное ребро исчезло — логируем удаление
                    j_old = multi_to_agg[mid]
                    if not E_alive[j_old]:
                        was_internal = bool(side[iu_old] == side[iv_old])
                        removal_events.append({
                            "iter": it + 1, "cut_type": "adversarial",
                            "iu": int(iu_old), "iv": int(iv_old),
                            "old_weight": float(w_removed),  # логируем вес снятого мультиребра
                            "was_internal": was_internal,
                        })

                    # шаг 3: добавить мультиребро с тем же весом между долями разреза
                    if V1.size and V2.size and w_removed > 0:
                        iu_new = int(np.random.choice(V1)); iv_new = int(np.random.choice(V2))
                        _ = _append_multiedge(iu_new, iv_new, w_removed)
                        weight_update_events.append({
                            "iter": it + 1, "cut_type": "adversarial",
                            "action": "multiadd",
                            "iu": int(iu_new), "iv": int(iv_new),
                            "delta": +float(w_removed),
                        })

            else:
                # friendly: тянем внутреннее ребро с МАКС суммарным весом
                v = graph.generate_cut(type="friendly")
                side = self._split_by_median(np.asarray(v, dtype=float))
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)

                for _k in range(num_edges):
                    mask_internal, _mask_cross = masks_for_cut(side, E_u, E_v, E_alive)

                    j = pick_idx(mask_internal, E_w, mode="max") or pick_idx(E_alive, E_w, mode="max")
                    if j is None:
                        continue
                    mid = _pick_multiedge_under_agg(j)
                    if mid is None:
                        continue

                    iu_old, iv_old, w_removed = _remove_multiedge_by_mid(mid)
                    weight_update_events.append({
                        "iter": it + 1, "cut_type": "friendly",
                        "action": "multiremove",
                        "iu": int(iu_old), "iv": int(iv_old),
                        "delta": -float(w_removed),
                    })
                    j_old = multi_to_agg[mid]
                    if not E_alive[j_old]:
                        was_internal = bool(side[iu_old] == side[iv_old])
                        removal_events.append({
                            "iter": it + 1, "cut_type": "friendly",
                            "iu": int(iu_old), "iv": int(iv_old),
                            "old_weight": float(w_removed),
                            "was_internal": was_internal,
                        })

                    if V1.size and V2.size and w_removed > 0:
                        iu_new = int(np.random.choice(V1)); iv_new = int(np.random.choice(V2))
                        _ = _append_multiedge(iu_new, iv_new, w_removed)
                        weight_update_events.append({
                            "iter": it + 1, "cut_type": "friendly",
                            "action": "multiadd",
                            "iu": int(iu_new), "iv": int(iv_new),
                            "delta": +float(w_removed),
                        })

            it += 1
            if it % SNAPSHOT_EVERY == 0:
                snapshot_mask(it, E_alive)

        if (it % SNAPSHOT_EVERY) != 0:
            snapshot_mask(it, E_alive)

        end = time.time()
        res = DemandsGenerationResultMultiGraph(
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
                "variant": "multigraph",
                "p_ER": self.p_ER,
                "distribution": self.dist,
                "median_weight_for_initial": self.median_weight_for_initial,
                "var_for_initial": self.var_for_initial,
                "multi_max": self.multi_max,
                "num_edges": (self.num_edges_param if self.num_edges_param is not None else int(math.ceil(n ** 0.25))),
                "epsilon": self.epsilon,
                "max_iter": self.max_iter,
            },
        )
        # истории
        res.removal_events = removal_events
        res.weight_update_events = weight_update_events
        res.edge_mask_history = edge_mask_history
        res.edge_mask_snapshot_iters = edge_mask_snapshot_iters
        return res
