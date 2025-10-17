from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import time
import math
import numpy as np
import networkx as nx

from ..core import GraphMCF

@dataclass
class DemandsGenerationResultMultiGraphSA:
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

class MCFGeneratorMultiGraphSA:
    """
    Мультиграф + имитация отжига (Simulated Annealing на уровне *батча* из num_edges перекидываний).

    Ход итерации:
      1) считаем alpha_old и решаем тип разреза (friendly/adversarial) по знаку (alpha_old - alpha_target);
      2) формируем ПРОПОЗАЛ: num_edges перекидываний (remove один мультиребро под выбранным агрегированным
         ребром + add мультиребро той же массы между кластерами);
      3) временно применяем весь батч, считаем alpha_new;
      4) если |alpha_new - target| < |alpha_old - target| → принять безусловно,
         иначе принять с вероятностью exp(-t * |alpha_target - alpha_new|), где t > 0;
      5) если батч принят — логируем события (multiremove/multiadd и, при необходимости, удаление агрег. ребра)
         и оставляем изменения; иначе — откатываем полностью (без логов).
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.05,
        # --- initial multidemands (ER / deterministic) ---
        p_ER: float = 0.5,
        distribution: str = "normal",
        median_weight_for_initial: int = 50,
        var_for_initial: int = 25,
        multi_max: int = 25,
        initial_generation: str = "ER",   # 'ER' | 'deterministic'
        demands_sum: float = 1000.0,      # для deterministic
        # --- скорость сходимости ---
        num_edges: Optional[int] = None,  # если None -> ceil(n ** 0.25)
        max_iter: Optional[int] = None,
        # --- SA-параметр ---
        t: float = 1.0,                   # температура отжига
    ) -> None:
        self.epsilon = float(epsilon)

        # init multi-demands params
        self.p_ER = float(p_ER)
        self.dist = str(distribution)
        self.median_weight_for_initial = int(median_weight_for_initial)
        self.var_for_initial = int(var_for_initial)
        self.multi_max = int(multi_max)
        self.initial_generation = str(initial_generation)
        self.demands_sum = float(demands_sum)

        self.num_edges_param = None if num_edges is None else int(num_edges)
        self.max_iter = max_iter

        self.t = float(t)

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
    ) -> DemandsGenerationResultMultiGraphSA:

        # 0) первичная генерация мультирёбер + агрегирования
        start = time.time()
        if self.initial_generation == "ER":
            graph.generate_initial_multidemands(
                p=self.p_ER,
                distribution=self.dist,
                median_weight=self.median_weight_for_initial,
                var=self.var_for_initial,
                multi_max=self.multi_max,
            )
        elif self.initial_generation == "deterministic":
            graph.generate_deterministic_initial_multidemands(
                distribution=self.dist,
                median_weight=self.median_weight_for_initial,
                var=self.var_for_initial,
                multi_max=self.multi_max,
                demands_sum=self.demands_sum,
            )
        else:
            raise ValueError("initial_generation must be 'ER' or 'deterministic'")

        assert isinstance(graph.demands_multigraph, nx.MultiGraph)
        assert isinstance(graph.demands_graph, nx.Graph)

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

        def build_multi_index(eid: Dict[Tuple[int, int], int]):
            EM_u, EM_v, EM_w, EM_key = [], [], [], []
            EM_alive: List[bool] = []
            multi_to_agg: List[int] = []
            agg_to_multi: Dict[int, List[int]] = {}
            meid: Dict[Tuple[int, int, int], int] = {}

            for u, v, key, d in graph.demands_multigraph.edges(keys=True, data=True):
                iu, iv = index[u], index[v]
                if iu > iv:
                    iu, iv = iv, iu
                w = float(d.get("weight", 1.0))
                j = eid[(iu, iv)]
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

        # --- индексы
        E_u, E_v, E_w, E_alive, eid = build_agg_index()
        (EM_u, EM_v, EM_w, EM_key, EM_alive,
         multi_to_agg, agg_to_multi, meid) = build_multi_index(eid)

        # --- истории (пишем только ПОСЛЕ принятия батча)
        removal_events: List[Dict[str, Any]] = []
        weight_update_events: List[Dict[str, Any]] = []
        edge_mask_history: List[np.ndarray] = []
        edge_mask_snapshot_iters: List[int] = []
        SNAPSHOT_EVERY = 50

        def snapshot_mask(iter_idx: int, E_mask: np.ndarray):
            edge_mask_history.append(E_mask.copy())
            edge_mask_snapshot_iters.append(iter_idx)

        snapshot_mask(0, E_alive)

        # ---------- низкоуровневые операции ----------
        def _remove_multiedge_by_mid(mid: int) -> Tuple[int, int, float]:
            """Удаляет мультиребро mid и обновляет агрегат (физически удаляя его, если масса стала ≤0)."""
            nonlocal E_u, E_v, E_w, E_alive
            nonlocal EM_u, EM_v, EM_w, EM_key, EM_alive
            nonlocal multi_to_agg, agg_to_multi, meid

            iu, iv, key, w = int(EM_u[mid]), int(EM_v[mid]), int(EM_key[mid]), float(EM_w[mid])
            if not EM_alive[mid]:
                return iu, iv, 0.0

            u_node, v_node = nodes[iu], nodes[iv]
            graph.demands_multigraph.remove_edge(u_node, v_node, key=key)
            EM_alive[mid] = False

            j = multi_to_agg[mid]
            old_sum = float(E_w[j])
            new_est = old_sum - w

            if new_est <= 0:
                _ = graph.remove_edge_by_indices(iu, iv)
                E_alive[j] = False
                E_w[j] = 0.0
            else:
                new_w = graph.upsert_edge_by_indices(iu, iv, -w)
                E_w[j] = float(new_w)
                E_alive[j] = True

            return iu, iv, w

        def _append_multiedge(iu: int, iv: int, w: float) -> int:
            """Добавляет мультиребро w на (iu,iv), обновляет агрегат, возвращает mid."""
            nonlocal E_u, E_v, E_w, E_alive
            nonlocal EM_u, EM_v, EM_w, EM_key, EM_alive
            nonlocal multi_to_agg, agg_to_multi, meid

            if iu > iv:
                iu, iv = iv, iu
            u_node, v_node = nodes[iu], nodes[iv]

            existing_keys = set()
            if graph.demands_multigraph.has_edge(u_node, v_node):
                existing_keys = set(graph.demands_multigraph[u_node][v_node].keys())

            graph.demands_multigraph.add_edge(u_node, v_node, weight=float(w))

            new_keys = set(graph.demands_multigraph[u_node][v_node].keys())
            new_key_candidates = list(new_keys - existing_keys)
            new_key = int(new_key_candidates[0]) if new_key_candidates else (max(new_keys) if new_keys else 0)

            new_sum_w = graph.upsert_edge_by_indices(iu, iv, +float(w))

            key = (iu, iv)
            if key in eid:
                j = eid[key]
                E_w[j] = new_sum_w
                E_alive[j] = True
            else:
                j = E_w.size
                eid[key] = j
                E_u = np.append(E_u, iu)
                E_v = np.append(E_v, iv)
                E_w = np.append(E_w, new_sum_w)
                E_alive = np.append(E_alive, True)

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

        def _readd_multiedge_exact(iu: int, iv: int, key: int, w: float, j_old: int, mid_old: int) -> None:
            """
            ВОССТАНОВЛЕНИЕ удалённого мультиребра с исходным key и весом w на (iu,iv).
            Не создаёт новый mid: реанимирует старый (EM_alive[mid_old] = True).
            """
            nonlocal E_u, E_v, E_w, E_alive
            nonlocal EM_u, EM_v, EM_w, EM_key, EM_alive

            u_node, v_node = nodes[iu], nodes[iv]
            graph.demands_multigraph.add_edge(u_node, v_node, key=int(key), weight=float(w))
            # восстановить агрегат
            new_sum = graph.upsert_edge_by_indices(iu, iv, +float(w))
            E_w[j_old] = float(new_sum)
            E_alive[j_old] = True
            # восстановить запись о мульти-ребре
            EM_alive[mid_old] = True  # остальные EM_* значения уже содержат (iu,iv,key,w)

        # --------------------------- основной цикл (с SA) ---------------------------
        alpha_hist: List[float] = []
        edges_hist: List[int] = []
        medw_hist: List[float] = []

        it = 0
        for _ in range(max_iter):
            # метрики на входе внешней итерации
            if graph.alpha is None:
                a_old = graph.calculate_alpha()
            else:
                a_old = graph.alpha
            alpha_hist.append(a_old)
            edges_hist.append(graph.demands_graph.number_of_edges())
            w_now = [d["weight"] for *_, d in graph.demands_graph.edges(data=True)]
            medw_hist.append(float(np.median(w_now)) if w_now else 0.0)

            diff_old = a_old - alpha_target
            if abs(diff_old) <= self.epsilon:
                break

            # разрез на итерацию
            if diff_old < -self.epsilon:
                cut_type = "adversarial"
                v = graph.generate_cut(type="adversarial")
                side = self._split_by_median(np.asarray(v, dtype=float))
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)
                pick_mode = ("min", ">")   # remove min internal; add across with ">"
            else:
                cut_type = "friendly"
                v = graph.generate_cut(type="friendly")
                side = self._split_by_median(np.asarray(v, dtype=float))
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)
                pick_mode = ("max", "<")   # remove max internal; add across with "<"

            # === СФОРМИРОВАТЬ БАТЧ-ПРЕДЛОЖЕНИЕ ===
            # Сохраняем всё, что нужно для возможного отката.
            proposal_applied: List[Dict[str, Any]] = []
            feasible_ops = 0

            for _k in range(num_edges):
                mask_internal, _mask_cross = masks_for_cut(side, E_u, E_v, E_alive)
                j = pick_idx(mask_internal, E_w, pick_mode[0]) or pick_idx(E_alive, E_w, pick_mode[0])
                if j is None or V1.size == 0 or V2.size == 0:
                    continue
                # мультиребро под агрегатом
                mids = agg_to_multi.get(j, [])
                mids_alive = [m for m in mids if EM_alive[m]]
                if not mids_alive:
                    continue
                mid = int(np.random.choice(mids_alive))

                # 1) удалить выбранное мультиребро (и обновить агрегат)
                iu_old, iv_old, w_removed = _remove_multiedge_by_mid(mid)
                j_old = multi_to_agg[mid]
                was_internal = bool(side[iu_old] == side[iv_old])

                # 2) добавить мультиребро той же массы между долями
                iu_new = int(np.random.choice(V1)); iv_new = int(np.random.choice(V2))
                mid_new = _append_multiedge(iu_new, iv_new, w_removed)
                j_new = multi_to_agg[mid_new]

                proposal_applied.append({
                    "mid_old": int(mid),
                    "iu_old": int(iu_old), "iv_old": int(iv_old),
                    "key_old": int(EM_key[mid]),
                    "w_removed": float(w_removed),
                    "j_old": int(j_old),
                    "was_internal": bool(was_internal),

                    "mid_new": int(mid_new),
                    "iu_new": int(iu_new), "iv_new": int(iv_new),
                    "j_new": int(j_new),
                })
                feasible_ops += 1

            if feasible_ops == 0:
                # Нечего предлагать — считаем это холостой шаг
                it += 1
                if it % SNAPSHOT_EVERY == 0:
                    snapshot_mask(it, E_alive)
                continue

            # === ПРОВЕРКА ПРИЁМКИ (SA) ===
            a_new = graph.calculate_alpha()
            diff_new = a_new - alpha_target

            improved = (abs(diff_new) < abs(diff_old))
            accept = improved or (np.random.rand() < np.exp(-self.t * abs(diff_new)))

            if accept:
                # --- ЛОГИРУЕМ события из принятого батча ---
                for rec in proposal_applied:
                    # multiremove
                    weight_update_events.append({
                        "iter": it + 1, "cut_type": cut_type,
                        "action": "multiremove",
                        "iu": rec["iu_old"], "iv": rec["iv_old"],
                        "delta": -rec["w_removed"],
                    })
                    # если агрегатное ребро исчезло после удаления этого мультиребра — лог удаления
                    if not E_alive[rec["j_old"]]:
                        removal_events.append({
                            "iter": it + 1, "cut_type": cut_type,
                            "iu": rec["iu_old"], "iv": rec["iv_old"],
                            "old_weight": rec["w_removed"],
                            "was_internal": rec["was_internal"],
                        })
                    # multiadd
                    weight_update_events.append({
                        "iter": it + 1, "cut_type": cut_type,
                        "action": "multiadd",
                        "iu": rec["iu_new"], "iv": rec["iv_new"],
                        "delta": +rec["w_removed"],
                    })
                # оставляем изменения как есть
            else:
                # возвращаем актуальное значение alpha
                graph.alpha = a_old
                # --- ОТКАТ в обратном порядке ---
                for rec in reversed(proposal_applied):
                    # 1) убрать добавленное мультиребро (это обратный шаг к _append_multiedge)
                    _ = _remove_multiedge_by_mid(rec["mid_new"])
                    # 2) вернуть исходное мультиребро (точно с тем же key и весом) и агрегат
                    _readd_multiedge_exact(
                        rec["iu_old"], rec["iv_old"], rec["key_old"], rec["w_removed"],
                        rec["j_old"], rec["mid_old"]
                    )
                # ничего не логируем

            it += 1
            if it % SNAPSHOT_EVERY == 0:
                snapshot_mask(it, E_alive)

        if (it % SNAPSHOT_EVERY) != 0:
            snapshot_mask(it, E_alive)

        end = time.time()
        res = DemandsGenerationResultMultiGraphSA(
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
                "variant": "multigraph_sa",
                "p_ER": self.p_ER,
                "distribution": self.dist,
                "median_weight_for_initial": self.median_weight_for_initial,
                "var_for_initial": self.var_for_initial,
                "multi_max": self.multi_max,
                "initial_generation": self.initial_generation,
                "demands_sum": self.demands_sum,
                "num_edges": (self.num_edges_param if self.num_edges_param is not None else int(math.ceil(n ** 0.25))),
                "epsilon": self.epsilon,
                "t": self.t,
                "max_iter": self.max_iter,
            },
        )
        # истории
        res.removal_events = removal_events
        res.weight_update_events = weight_update_events
        res.edge_mask_history = edge_mask_history
        res.edge_mask_snapshot_iters = edge_mask_snapshot_iters
        return res
