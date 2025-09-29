"""
Алгоритм генерации корреспонденций с контролем alpha.
Хранит результат в DemandsGenerationResult; работает поверх GraphMCF.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import time
import numpy as np

from ..core import GraphMCF

@dataclass
class DemandsGenerationResult:
    graph: GraphMCF
    alpha_target: float
    epsilon: float
    start_time: float
    end_time: float
    iterations_total: int
    alpha_history: List[float]
    edge_counts_history: List[int]
    median_weights_history: List[float]

    def to_dict(self) -> Dict[str, Any]:
        from ..analysis.overall import pack_overall_dict
        # Упаковка в универсальный dict в стиле overall
        return pack_overall_dict(self.graph, self.alpha_target, self.epsilon,
                                 self.start_time, self.end_time,
                                 self.alpha_history, self.edge_counts_history, self.median_weights_history)

class MCFGenerator:
    def __init__(
        self,
        epsilon: float = 0.05,
        demands_weights_distribution: str = "normal",
        demands_median_denominator: int = 1,
        demands_var_denominator: int = 2,
        max_iter: Optional[int] = None,
    ) -> None:
        self.epsilon = float(epsilon)
        self.dist = demands_weights_distribution
        self.median_div = int(demands_median_denominator)
        self.var_div = int(demands_var_denominator)
        self.max_iter = max_iter  # если None -> 100*|V|

    def _split_by_median(self, vec: np.ndarray) -> np.ndarray:
        med = float(np.median(vec))
        side = (vec <= med)
        if side.sum() in (0, len(side)):
            side = (vec < med)
        if side.sum() in (0, len(side)):
            side = np.zeros_like(side, dtype=bool); side[0] = True
        return side

    def _draw_new_weight(self, old_w: float, mode: str, median: int, var: int) -> int:
        # дискретная усечённая нормаль 1..2*median
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

    def generate(
        self,
        graph: GraphMCF,
        alpha_target: float = 0.5,
        analysis_mode: Optional[str] = None,
    ) -> DemandsGenerationResult:
        # начальные веса распределим относительно медианы capacity
        base_w = [d["weight"] for *_ , d in graph.graph.edges(data=True)]
        med_cap = int(round(np.median(base_w))) if base_w else 0
        median_for_weights = max(med_cap // self.median_div, 1)
        var_for_weights = max(med_cap // self.var_div, 1)

        start = time.time()

        # инициализируем demands и кэш Лапласиана
        graph.generate_initial_demands(
            distribution=self.dist,
            median_weight=median_for_weights,
            var=var_for_weights,
        )

        nodes, index = graph.nodelist_and_index()
        n = len(nodes)
        max_iter = self.max_iter if self.max_iter is not None else 100 * n

        # плоские массивы рёбер demands-графа (для быстрых масок)
        def build_edge_index():
            E_u, E_v, E_w = [], [], []
            eid = {}
            for u, v, d in graph.demands_graph.edges(data=True):
                iu, iv = index[u], index[v]
                if iu > iv: iu, iv = iv, iu
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
            return int(cand[j])

        def remove_by_idx(idx: int, E_u, E_v, E_alive, E_w):
            if not E_alive[idx]: return None
            iu, iv = int(E_u[idx]), int(E_v[idx])
            old = graph.remove_edge_by_indices(iu, iv)
            E_alive[idx] = False
            return old
            
        def upsert(iu: int, iv: int, delta_w: float,
                   E_u, E_v, E_alive, E_w, eid: Dict[Tuple[int,int], int]):
            nonlocal E_u, E_v, E_w, E_alive  # ← добавили nonlocal, чтобы можно было перепривязать массивы
               
            if iu > iv:
                iu, iv = iv, iu

            new_w = graph.upsert_edge_by_indices(iu, iv, delta_w)
            key = (iu, iv)

            if key in eid:
                j = eid[key]
                E_w[j] = new_w
                E_alive[j] = True
            else:
                j = E_w.size                       # индекс нового элемента — старая длина
                eid[key] = j
                # пере-создаем массивы с добавленным значением (НЕ через срез!)
                E_u = np.append(E_u, iu)
                E_v = np.append(E_v, iv)
                E_w = np.append(E_w, new_w)
                E_alive = np.append(E_alive, True)
        
        E_u, E_v, E_w, E_alive, eid = build_edge_index()

        alpha_hist: List[float] = []
        edges_hist: List[int] = []
        medw_hist: List[float] = []

        for _ in range(max_iter):
            a = graph.calculate_alpha()
            alpha_hist.append(a)
            edges_hist.append(graph.demands_graph.number_of_edges())
            weights_now = [d["weight"] for *_ , d in graph.demands_graph.edges(data=True)]
            medw_hist.append(float(np.median(weights_now)) if weights_now else 0.0)

            diff = a - alpha_target
            if abs(diff) <= self.epsilon:
                break

            if diff < -self.epsilon:
                # нужно повысить alpha -> adversarial
                v = graph.generate_cut(type="adversarial")
                side = self._split_by_median(np.asarray(v, dtype=float))
                mask_int, _ = masks_for_cut(side, E_u, E_v, E_alive)
                j = pick_idx(mask_int, E_w, "min") or pick_idx(E_alive, E_w, "min")
                if j is None:
                    # добавим любое кросс-ребро
                    V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)
                    if V1.size and V2.size:
                        iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
                        upsert(iu, iv, 1.0, E_u, E_v, E_alive, E_w, eid)
                    continue
                w_old = E_w[j]
                remove_by_idx(j, E_u, E_v, E_alive, E_w)
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)
                if V1.size and V2.size:
                    iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
                    delta = self._draw_new_weight(w_old, ">", median_for_weights, var_for_weights)
                    upsert(iu, iv, float(delta), E_u, E_v, E_alive, E_w, eid)
            else:
                # нужно понизить alpha -> friendly
                v = graph.generate_cut(type="friendly")
                side = self._split_by_median(np.asarray(v, dtype=float))
                mask_int, _ = masks_for_cut(side, E_u, E_v, E_alive)
                j = pick_idx(mask_int, E_w, "max") or pick_idx(E_alive, E_w, "max")
                if j is None:
                    V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)
                    if V1.size and V2.size:
                        iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
                        upsert(iu, iv, 1.0, E_u, E_v, E_alive, E_w, eid)
                    continue
                w_old = E_w[j]
                remove_by_idx(j, E_u, E_v, E_alive, E_w)
                V1, V2 = np.flatnonzero(side), np.flatnonzero(~side)
                if V1.size and V2.size:
                    iu = int(np.random.choice(V1)); iv = int(np.random.choice(V2))
                    delta = self._draw_new_weight(w_old, "<", median_for_weights, var_for_weights)
                    upsert(iu, iv, float(delta), E_u, E_v, E_alive, E_w, eid)

        end = time.time()
        res = DemandsGenerationResult(
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

        # По запросу — запустить анализ прямо отсюда (оставлено опциональным)
        if analysis_mode == "simple":
            from ..analysis.simple import analyze_simple
            analyze_simple(graph, res.alpha_target, res.epsilon, res.start_time, res.end_time,
                           res.alpha_history, res.edge_counts_history, res.median_weights_history)
        return res
