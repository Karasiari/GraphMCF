from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import networkx as nx

from ..core import GraphMCF

@dataclass
class DemandsGenerationResultGravity:
    """
    Результат запуска гравитационного генератора.
    """
    graph: GraphMCF
    beta: float
    intensity: int
    centrality: str
    start_time: float
    end_time: float
    alpha_history: List[float]
    edge_counts_history: List[int]
    median_weights_history: List[float]
    algo_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "intensity": self.intensity,
            "centrality": self.centrality,
            "execution_time": float(self.end_time - self.start_time),
            "edges_count": int(self.graph.demands_graph.number_of_edges()) if self.graph.demands_graph else 0,
        }

class GravitationalGenerator:
    """
    Генерация demands-графа по гравитационной / антигравитационной модели.

    Шаги:
      • массы вершин m_i по centrality
      • меры на парах: G_ij ∝ m_i m_j; A_ij — шаблон (по умолчанию 1/(m_i m_j))
      • нормируем меры → вероятности p⁺, p⁻
      • выбор подмножества рёбер и способ генерации по edge_mode:
          - 'static_top'      : union-топ с обработкой β=0/1 и шифлом ties
          - 'static_betascore': топ по (1-β)·p⁺ + β·p⁻; затем маскирование и раздельная перенормировка
          - 'dynamic'         : как 'static_top' для выбора, далее распределяем ОБЩЕЕ intensity по рёбрам
                                через лестничный закон (dyn_max, dyn_law∈{linear,exponential}, dyn_k)
      • веса: M⁺ ~ Binomial(n⁺_j, p⁺_j), M⁻ ~ Binomial(n⁻_j, p⁻_j)
        (в статике n_j = intensity; в dynamic — по распределению)
      • итог: round((1−β) M⁺ + β M⁻)
    """

    def __init__(
        self,
        *,
        beta: float = 0.5,
        intensity: int = 100,
        centrality: str = "degree",
        # PageRank (если centrality="pagerank"):
        pagerank_alpha: float = 0.85,
        pagerank_tol: float = 1e-8,
        pagerank_max_iter: int = 100,
        # Доля сильных пар (0<edge_perc≤1)
        edge_perc: float = 1.0,
        # Режим отбора/генерации
        edge_mode: str = "static_top",   # 'static_top' | 'static_betascore' | 'dynamic'
        # Параметры для 'dynamic'
        dyn_max: Optional[float] = None,
        dyn_law: Optional[str] = None,   # None|'linear'|'exponential'
        dyn_k: Optional[float] = None,
    ) -> None:
        assert 0.0 <= beta <= 1.0, "beta ∈ [0,1]"
        assert intensity >= 0, "intensity должен быть неотрицательным"
        assert 0.0 < edge_perc <= 1.0, "edge_perc должен быть в (0, 1]"
        self.beta = float(beta)
        self.intensity = int(intensity)
        self.centrality = str(centrality)
        self.edge_perc = float(edge_perc)
        self.edge_mode = str(edge_mode).lower().strip()

        # pagerank params
        self.pagerank_alpha = float(pagerank_alpha)
        self.pagerank_tol = float(pagerank_tol)
        self.pagerank_max_iter = int(pagerank_max_iter)

        # dynamic params
        self.dyn_max = None if dyn_max is None else float(dyn_max)
        self.dyn_law = None if dyn_law is None else str(dyn_law).lower().strip()
        self.dyn_k = None if dyn_k is None else float(dyn_k)

        if self.edge_mode == "dynamic":
            assert self.dyn_max is not None and self.dyn_max > 0.0, "dynamic: dyn_max>0 обязателен"
            assert self.dyn_law in ("linear", "exponential"), "dynamic: dyn_law ∈ {'linear','exponential'}"
            if self.dyn_law == "linear":
                assert self.dyn_k is not None and self.dyn_k > 0.0, "dynamic: dyn_k>0 обязателен для linear"
            else:
                assert self.dyn_k is not None and 0.0 < self.dyn_k < 1.0, "dynamic: 0<dyn_k<1 обязателен для exponential"

    # ---------------------------------------------------------------------
    def generate(self, graph: GraphMCF) -> DemandsGenerationResultGravity:
        import time
        start = time.time()

        nodes, _ = graph.nodelist_and_index()
        m = self._compute_masses(graph, nodes, self.centrality)   # shape (n,)
        n = m.size
        iu, iv = self._pair_index(n)                               # shape (M,)

        # меры → вероятности
        g_measures = self._measures_gravity(m, iu, iv)             # shape (M,)
        a_measures = self._measures_antigravity(m, iu, iv)         # shape (M,)
        g_probs = self._normalize_to_probs(g_measures)
        a_probs = self._normalize_to_probs(a_measures)

        # === выбор подмножества пар согласно edge_mode (и edge_perc) ===
        mask = np.ones_like(g_probs, dtype=bool)
        if self.edge_perc < 1.0 and g_probs.size:
            M = g_probs.size
            K = max(1, int(np.ceil(self.edge_perc * M)))
            beta, tol = float(self.beta), 1e-12

            if self.edge_mode == "static_betascore":
                score = (1.0 - beta) * g_probs + beta * a_probs
                top_idx = self._top_k_with_tie_shuffle(score, K)
                mask = np.zeros(M, dtype=bool); mask[top_idx] = True

            else:
                # 'static_top' и 'dynamic' — union-логика с корректной обработкой краёв β
                if beta <= tol:
                    top_idx = self._top_k_with_tie_shuffle(g_probs, K)
                    mask = np.zeros(M, dtype=bool); mask[top_idx] = True
                elif beta >= 1.0 - tol:
                    top_idx = self._top_k_with_tie_shuffle(a_probs, K)
                    mask = np.zeros(M, dtype=bool); mask[top_idx] = True
                else:
                    k_half = max(1, K // 2)
                    top_g = self._top_k_with_tie_shuffle(g_probs, k_half)
                    top_a = self._top_k_with_tie_shuffle(a_probs, k_half)
                    mask = np.zeros(M, dtype=bool)
                    mask[top_g] = True
                    mask[top_a] = True  # объединение

        # применяем маску и перенормируем вероятности внутри выбранного множества
        g_probs = self._mask_and_renorm(g_probs, mask)
        a_probs = self._mask_and_renorm(a_probs, mask)

        # === генерация весов ===
        if self.edge_mode == "dynamic":
            # Распределяем ОБЩЕЕ self.intensity по рёбрам отдельно для каждой модели:
            # строим «лестницу» s_j (max → law) по убыванию p_j, НОРМИРУЕМ её в w_j = s_j/sum(s),
            # затем trials n_j = round(intensity * w_j) с сохранением суммы.
            n_plus  = self._allocate_trials_dynamic(g_probs)   # сумма n_plus == self.intensity
            n_minus = self._allocate_trials_dynamic(a_probs)   # сумма n_minus == self.intensity
            M_plus_w  = np.random.binomial(n_plus,  g_probs).astype(np.int32)
            M_minus_w = np.random.binomial(n_minus, a_probs).astype(np.int32)
        else:
            # статические режимы: одинаковое n для всех (вне топа p=0 → нулевые вклады)
            if self.intensity > 0 and g_probs.size:
                M_plus_w  = np.random.binomial(self.intensity, g_probs).astype(np.int32)
                M_minus_w = np.random.binomial(self.intensity, a_probs).astype(np.int32)
            else:
                M_plus_w  = np.zeros_like(g_probs, dtype=np.int32)
                M_minus_w = np.zeros_like(a_probs, dtype=np.int32)

        # итоговая смесь
        final_w = np.rint((1.0 - self.beta) * M_plus_w + self.beta * M_minus_w).astype(np.int64)

        # собрать demands_graph с исходными метками узлов
        Gd = nx.Graph()
        Gd.add_nodes_from(nodes)
        for j in range(final_w.size):
            w = int(final_w[j])
            if w > 0:
                u = nodes[int(iu[j])]
                v = nodes[int(iv[j])]
                Gd.add_edge(u, v, weight=w)

        graph.demands_graph = Gd
        if hasattr(graph, "_recompute_demands_laplacian"):
            graph._recompute_demands_laplacian()

        algo_params = {
            "variant": "gravity",
            "beta": self.beta,
            "intensity": self.intensity,
            "centrality": self.centrality,
            "edge_perc": self.edge_perc,
            "edge_mode": self.edge_mode,
            "pagerank_alpha": self.pagerank_alpha,
            "pagerank_tol": self.pagerank_tol,
            "pagerank_max_iter": self.pagerank_max_iter,
            "dyn_max": self.dyn_max,
            "dyn_law": self.dyn_law,
            "dyn_k": self.dyn_k,
        }

        end = time.time()
        med_w = float(np.median([d["weight"] for *_, d in Gd.edges(data=True)]) if Gd.number_of_edges() else 0.0)
        res = DemandsGenerationResultGravity(
            graph=graph,
            beta=self.beta,
            intensity=self.intensity,
            centrality=self.centrality,
            start_time=start,
            end_time=end,
            alpha_history=[],  # нет итеративной динамики
            edge_counts_history=[Gd.number_of_edges()],
            median_weights_history=[med_w],
            algo_params=algo_params,
        )
        return res

    # ---------------------------------------------------------------------
    # ХУКИ / вспомогательные методы
    # ---------------------------------------------------------------------
    def _compute_masses(self, graph: GraphMCF, nodes: List[Any], centrality: str) -> np.ndarray:
        """
        centrality: 'degree' | 'closeness' | 'harmonic_closeness' | 'pagerank'
        Политика eps: НЕ клиппим, если все значения > 0; иначе только нули поднимаем до eps.
        """
        G = graph.graph
        key = str(centrality).lower().strip()
        eps = 1e-12

        if key == "degree":
            deg = dict(G.degree())
            masses = np.array([float(deg.get(u, 0.0)) for u in nodes], dtype=float)

        elif key == "closeness":
            try:
                c = nx.closeness_centrality(G, wf_improved=True)
            except TypeError:
                c = nx.closeness_centrality(G)
            masses = np.array([float(c.get(u, 0.0)) for u in nodes], dtype=float)

        elif key in ("harmonic_closeness", "harmonic"):
            c = nx.harmonic_centrality(G)
            masses = np.array([float(c.get(u, 0.0)) for u in nodes], dtype=float)

        elif key == "pagerank":
            pr = nx.pagerank(G, alpha=self.pagerank_alpha, tol=self.pagerank_tol, max_iter=self.pagerank_max_iter)
            masses = np.array([float(pr.get(u, 0.0)) for u in nodes], dtype=float)

        else:
            # fallback: степень
            deg = dict(G.degree())
            masses = np.array([float(deg.get(u, 0.0)) for u in nodes], dtype=float)

        if np.any(masses <= 0.0):
            m = masses.copy()
            m[m <= 0.0] = eps
            return m
        return masses

    def _pair_index(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        iu, iv = np.triu_indices(n, k=1)
        return iu.astype(np.int32), iv.astype(np.int32)

    def _measures_gravity(self, m: np.ndarray, iu: np.ndarray, iv: np.ndarray) -> np.ndarray:
        return (m[iu] * m[iv]).astype(float)

    def _measures_antigravity(self, m: np.ndarray, iu: np.ndarray, iv: np.ndarray) -> np.ndarray:
        # TODO: заменить на согласованную формулу при необходимости
        eps = 0
        prod = (m[iu] * m[iv]).astype(float)
        return 1.0 / (prod + eps)

    def _normalize_to_probs(self, measures: np.ndarray) -> np.ndarray:
        measures = np.maximum(0.0, measures.astype(float))
        s = measures.sum()
        if s <= 0.0:
            return np.full_like(measures, 1.0 / max(1, measures.size), dtype=float)
        return measures / s

    def _mask_and_renorm(self, probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Обнуляет вероятности вне mask и нормирует оставшиеся к сумме 1 (если есть положительные)."""
        out = probs.copy()
        out[~mask] = 0.0
        s = out.sum()
        if s > 0.0:
            out /= s
        return out

    # ---------- TOP-K с честной обработкой ties ----------
    def _top_k_with_tie_shuffle(self, scores: np.ndarray, K: int) -> np.ndarray:
        """
        Возвращает индексы top-K по scores с равномерным добором из «порогового» блока равных значений.
        """
        K = int(K)
        if K <= 0 or scores.size == 0:
            return np.empty(0, dtype=np.int64)

        # быстро находим порог
        part_idx = np.argpartition(scores, -K)[-K:]
        threshold = scores[part_idx].min()

        # индексы строго выше порога
        above = np.flatnonzero(scores > threshold)

        if above.size >= K:
            choose = np.random.choice(above, size=K, replace=False)
            return np.sort(choose.astype(np.int64))

        # добираем из ties на пороге (случайно)
        need = K - above.size
        ties = np.flatnonzero(scores == threshold)
        shuffle = np.random.permutation(ties)[:need]
        out = np.concatenate([above, shuffle])
        return np.sort(out.astype(np.int64))

    # ---------- Dynamic: распределение количества бросков по «лестнице» ----------
    def _allocate_trials_dynamic(self, probs: np.ndarray) -> np.ndarray:
        """
        Строит «ступенчатые» веса на рёбра по убыванию probs (ties перемешиваются),
        затем НОРМИРУЕТ их и превращает в целые числа испытаний:
            n_j = round( self.intensity * (s_j / sum_k s_k) )
        так, чтобы sum_j n_j == self.intensity.
        """
        M = probs.size
        if M == 0 or self.intensity <= 0:
            return np.zeros(M, dtype=np.int64)

        # Стабильная сортировка по убыванию + равномерный шифл внутри блоков равных значений
        order = np.argsort(-probs, kind="mergesort")
        vals = probs[order]
        blocks = []
        start = 0
        for i in range(1, M + 1):
            if i == M or vals[i] < vals[i - 1]:
                block = order[start:i]
                block = np.random.permutation(block)  # ties → случайный порядок
                blocks.append(block)
                start = i
        order = np.concatenate(blocks)

        # Лестница s_j: начинаем с dyn_max, далее применяем закон на строгом убывании p
        s = np.zeros(M, dtype=float)
        current = float(self.dyn_max)
        prev_val = np.inf
        for idx in order:
            p = probs[idx]
            if p <= 0.0:
                s[idx] = 0.0
                continue
            if p < prev_val:
                if self.dyn_law == "linear":
                    current = max(0.0, current - float(self.dyn_k))
                else:  # 'exponential'
                    current = current * float(self.dyn_k)
            s[idx] = current
            prev_val = p

        total = s.sum()
        if total <= 0.0:
            return np.zeros(M, dtype=np.int64)

        # НОРМИРОВКА: w = s / sum(s), trials = intensity * w (с честным добором по дробным частям)
        w = s / total
        raw = w * float(self.intensity)

        n_floor = np.floor(raw).astype(np.int64)
        remainder = int(self.intensity - n_floor.sum())
        if remainder > 0:
            frac = raw - n_floor
            add_idx = np.argsort(-frac)[:remainder]
            n_floor[add_idx] += 1
        return n_floor
