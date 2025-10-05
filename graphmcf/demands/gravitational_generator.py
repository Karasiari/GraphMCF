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
    По аналогии с остальными генераторами, храним ключевые метрики.
    """
    graph: GraphMCF
    beta: float
    intensity: int
    centrality: str
    start_time: float
    end_time: float
    # для единообразия с остальными анализами оставим эти поля (они тут не используются внутри):
    alpha_history: List[float]
    edge_counts_history: List[int]
    median_weights_history: List[float]
    # параметры алгоритма (для логов/сводок)
    algo_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        # Совместимая упаковка — минимум полей
        return {
            "beta": self.beta,
            "intensity": self.intensity,
            "centrality": self.centrality,
            "execution_time": float(self.end_time - self.start_time),
            "edges_count": int(self.graph.demands_graph.number_of_edges()) if self.graph.demands_graph else 0,
        }

class GravitationalGenerator:
    """
    Генератор корреспонденций на основе гравитационной и антигравитационной модели.

    Идея:
      • Сначала считаем «массы» вершин m_i (по centrality).
      • Гравитационные меры:     G_ij  ∝  f_gravity(m_i, m_j)       (по умолчанию: m_i * m_j)
      • Антигравитационные меры: A_ij  ∝  f_antigravity(m_i, m_j)    (шаблон; временный fallback ниже)
      • Нормируем меры → вероятности на парах (i<j).
      • В каждом из двух миров (M+, M-) генерируем веса рёбер ~ Binomial(intensity, p_ij).
      • Итог: round( (1-β) * M+ + β * M- ).

    ВАЖНО: формулы для антигравитации и набор centrality пока шаблонные.
    Здесь оставлены расширяемые «хуки»:
       - _compute_masses(...)
       - _measures_gravity(...)
       - _measures_antigravity(...)
    """

    def __init__(self,
                 beta: float = 0.5,
                 intensity: int = 100,
                 centrality: str = "degree",
                 pagerank_alpha: float = 0.85,
                 pagerank_tol: float = 1e-8,
                 pagerank_max_iter: int = 100) -> None:
        assert 0.0 <= beta <= 1.0, "beta ∈ [0,1]"
        assert intensity >= 0, "intensity должен быть неотрицательным"
        self.beta = float(beta)
        self.intensity = int(intensity)
        self.centrality = str(centrality)

        self.pagerank_alpha = float(pagerank_alpha)
        self.pagerank_tol = float(pagerank_tol)
        self.pagerank_max_iter = int(pagerank_max_iter)

    # ---------------------------------------------------------------------
    # Публичный запуск
    # ---------------------------------------------------------------------
    def generate(self, graph: GraphMCF) -> DemandsGenerationResultGravity:
        """
        Генерирует self.demands_graph у `graph` по гравитационной/антигравитационной модели.
        """
        import time
        start = time.time()

        # 1) массы вершин (centrality)
        nodes, index = graph.nodelist_and_index()
        m = self._compute_masses(graph, nodes, self.centrality)   # shape (n,)
        n = m.size

        # 2) пары (i<j)
        iu, iv = self._pair_index(n)                               # np.int32 arrays of shape (M,)
        # 3) меры (не нормированные) для двух моделей
        g_measures = self._measures_gravity(m, iu, iv)             # shape (M,)
        a_measures = self._measures_antigravity(m, iu, iv)         # shape (M,)

        # 4) нормировка в вероятности
        g_probs = self._normalize_to_probs(g_measures)
        a_probs = self._normalize_to_probs(a_measures)

        # 5) порождение весов рёбер: Binomial(intensity, p_ij)
        #   это эквивалент "intensity раз бросаем монетки для каждой пары"
        if self.intensity > 0:
            M_plus_w  = np.random.binomial(self.intensity, g_probs).astype(np.int32)
            M_minus_w = np.random.binomial(self.intensity, a_probs).astype(np.int32)
        else:
            M_plus_w  = np.zeros_like(g_probs, dtype=np.int32)
            M_minus_w = np.zeros_like(a_probs, dtype=np.int32)

        # 6) взвешенная сумма и округление
        final_w = np.rint((1.0 - self.beta) * M_plus_w + self.beta * M_minus_w).astype(np.int64)

        # 7) собрать demands_graph
        Gd = nx.Graph()
        Gd.add_nodes_from(nodes)
        for j in range(final_w.size):
            w = int(final_w[j])
            if w > 0:
                Gd.add_edge(int(iu[j]), int(iv[j]), weight=w)

        # сохранить
        graph.demands_graph = Gd
        # сохранить параметры
        algo_params = {
            "variant": "gravity",
            "beta": self.beta,
            "intensity": self.intensity,
            "centrality": self.centrality,
        }

        end = time.time()
        # упаковка результата (истории alpha тут нет; кладём пустые)
        res = DemandsGenerationResultGravity(
            graph=graph,
            beta=self.beta,
            intensity=self.intensity,
            centrality=self.centrality,
            start_time=start,
            end_time=end,
            alpha_history=[],
            edge_counts_history=[Gd.number_of_edges()],
            median_weights_history=[float(np.median([d["weight"] for *_, d in Gd.edges(data=True)]) if Gd.number_of_edges() else 0.0)],
            algo_params=algo_params,
        )
        return res

    # ---------------------------------------------------------------------
    # ХУКИ / вспомогательные методы
    # ---------------------------------------------------------------------
    def _compute_masses(self, graph: GraphMCF, nodes: list[int], centrality: str) -> np.ndarray:
        """
        Возвращает вектор масс m_i для вершин в порядке `nodes`.

        Поддерживаемые варианты:
          - 'degree'               : невзвешенная степень на capacity-графе graph.graph
          - 'closeness'            : closeness centrality (wf_improved=True при наличии)
          - 'harmonic_closeness'   : harmonic centrality (устойчив к несвязности)
          - 'pagerank'             : PageRank с параметрами (alpha, tol, max_iter)

        Политика eps:
           • НЕ клиппим, если все значения строго > 0.
           • Если встречаются нули (разреженность/несвязность), ТОЛЬКО нули поднимаем до eps,
            остальные значения не трогаем (для сохранения точности).
        """
        import networkx as nx
        import numpy as np

        G = graph.graph
        ckey = str(centrality).lower().strip()
        n = len(nodes)
        eps = 1e-12  # применяется только к нулям, если они есть

        if ckey == "degree":
            deg = dict(G.degree())
            masses = np.array([float(deg.get(u, 0.0)) for u in nodes], dtype=float)

        elif ckey == "closeness":
            try:
                c = nx.closeness_centrality(G, wf_improved=True)
            except TypeError:
                c = nx.closeness_centrality(G)
            masses = np.array([float(c.get(u, 0.0)) for u in nodes], dtype=float)

        elif ckey in ("harmonic_closeness", "harmonic"):
            c = nx.harmonic_centrality(G)
            masses = np.array([float(c.get(u, 0.0)) for u in nodes], dtype=float)

        elif ckey == "pagerank":
            # PageRank > 0 для всех вершин при alpha<1 (за счёт телепортации)
            pr = nx.pagerank(G,
                              alpha=self.pagerank_alpha,
                              tol=self.pagerank_tol,
                              max_iter=self.pagerank_max_iter)
            masses = np.array([float(pr.get(u, 0.0)) for u in nodes], dtype=float)

        else:
            # безопасный fallback: degree
            deg = dict(G.degree())
            masses = np.array([float(deg.get(u, 0.0)) for u in nodes], dtype=float)

        # «умный eps»: только если встречаются нули — поднимем именно их
        if np.any(masses <= 0.0):
            mask_zero = (masses <= 0.0)
            masses = masses.copy()
            masses[mask_zero] = eps

        return masses

    def _pair_index(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает массивы индексов пар (i<j) длины M = n*(n-1)/2
        """
        iu, iv = np.triu_indices(n, k=1)
        return iu.astype(np.int32), iv.astype(np.int32)

    def _measures_gravity(self, m: np.ndarray, iu: np.ndarray, iv: np.ndarray) -> np.ndarray:
        """
        ШАБЛОН гравитационных мер:
            G_ij ∝ m_i * m_j
        """
        return (m[iu] * m[iv]).astype(float)

    def _measures_antigravity(self, m: np.ndarray, iu: np.ndarray, iv: np.ndarray) -> np.ndarray:
        """
        ШАБЛОН антигравитационных мер.
        TODO: заменить на согласованную формулу, когда ты её определишь.

        Временный безопасный fallback (чтобы всё работало из коробки):
          • используем обратные массы:  A_ij ∝ 1 / ((m_i * m_j) + eps)
          • это противопоставляет высокоцентральным парам — низкоцентральные усиливаются
        """
        eps = 1e-9
        prod = (m[iu] * m[iv]).astype(float)
        return 1.0 / (prod + eps)

    def _normalize_to_probs(self, measures: np.ndarray) -> np.ndarray:
        """
        Нормирует неотрицательные меры в вероятности. Если сумма == 0 (все меры нули),
        возвращает равномерные вероятности.
        """
        measures = np.maximum(0.0, measures.astype(float))
        s = measures.sum()
        if s <= 0.0:
            # равномерно по парам
            return np.full_like(measures, 1.0 / measures.size, dtype=float)
        return measures / s
