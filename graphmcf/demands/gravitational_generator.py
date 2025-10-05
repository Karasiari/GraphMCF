from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import networkx as nx

from ..core import GraphMCF

@dataclass
class DemandsGenerationResultGravity:
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
    Генерация demands-графа по гравитационной / антигравитационной модели:
      • считаем массы вершин m_i по заданной centrality
      • меры на парах: G_ij ∝ m_i m_j; A_ij — шаблон (по умолчанию ~ 1/(m_i m_j))
      • нормируем меры → вероятности p_ij^+ и p_ij^-
      • (НОВОЕ) оставляем только топ-K пар по вероятностям (объединение топов по обеим моделям),
        для остальных p=0; внутри топа — перенормировка в сумме к 1
      • веса: M^+_ij ~ Binomial(intensity, p_ij^+), M^-_ij ~ Binomial(intensity, p_ij^-)
      • итог: round((1−β) M^+ + β M^-)
    """

    def __init__(
        self,
        beta: float = 0.5,
        intensity: int = 100,
        centrality: str = "degree",
        # PageRank (если centrality="pagerank"):
        pagerank_alpha: float = 0.85,
        pagerank_tol: float = 1e-8,
        pagerank_max_iter: int = 100,
        # Новое: доля сильных пар, которые участвуют в генерации (0<edge_perc≤1)
        edge_perc: float = 1.0,
    ) -> None:
        assert 0.0 <= beta <= 1.0, "beta ∈ [0,1]"
        assert intensity >= 0, "intensity должен быть неотрицательным"
        assert 0.0 < edge_perc <= 1.0, "edge_perc должен быть в (0, 1]"
        self.beta = float(beta)
        self.intensity = int(intensity)
        self.centrality = str(centrality)
        self.edge_perc = float(edge_perc)

        # pagerank params
        self.pagerank_alpha = float(pagerank_alpha)
        self.pagerank_tol = float(pagerank_tol)
        self.pagerank_max_iter = int(pagerank_max_iter)

    # ---------------------------------------------------------------------
    def generate(self, graph: GraphMCF) -> DemandsGenerationResultGravity:
        import time
        start = time.time()

        nodes, _index = graph.nodelist_and_index()
        m = self._compute_masses(graph, nodes, self.centrality)   # shape (n,)
        n = m.size
        iu, iv = self._pair_index(n)                               # shape (M,)

        g_measures = self._measures_gravity(m, iu, iv)             # shape (M,)
        a_measures = self._measures_antigravity(m, iu, iv)         # shape (M,)

        g_probs = self._normalize_to_probs(g_measures)
        a_probs = self._normalize_to_probs(a_measures)

        # === (НОВОЕ) отбор топ-K пар ===
        if self.edge_perc < 1.0 and g_probs.size:
            M = g_probs.size
            K = max(1, int(np.ceil(self.edge_perc * M)))

            beta = float(self.beta)
            tol = 1e-12
            if beta <= tol:
                top_idx = np.argpartition(g_probs, -K)[-K:]
                mask = np.zeros(M, dtype=bool)
                mask[top_idx] = True
            elif beta >= 1.0 - tol:
                top_idx = np.argpartition(a_probs, -K)[-K:]
                mask = np.zeros(M, dtype=bool)
                mask[top_idx] = True
            else:
                # топы по каждой модели:
                top_g_idx = np.argpartition(g_probs, -(K//2))[-(K//2):]
                top_a_idx = np.argpartition(a_probs, -(K//2))[-(K//2):]
                mask = np.zeros(M, dtype=bool)
                mask[top_g_idx] = True
                mask[top_a_idx] = True  # объединение сильных по любой модели

            # зануляем вне топа и перенормируем внутри
            g_probs = self._mask_and_renorm(g_probs, mask)
            a_probs = self._mask_and_renorm(a_probs, mask)

        # веса по биному (интенсивность испытаний)
        if self.intensity > 0 and g_probs.size:
            M_plus_w  = np.random.binomial(self.intensity, g_probs).astype(np.int32)
            M_minus_w = np.random.binomial(self.intensity, a_probs).astype(np.int32)
        else:
            M_plus_w  = np.zeros_like(g_probs, dtype=np.int32)
            M_minus_w = np.zeros_like(a_probs, dtype=np.int32)

        # взвешенная сумма и округление
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

        # параметры
        algo_params = {
            "variant": "gravity",
            "beta": self.beta,
            "intensity": self.intensity,
            "centrality": self.centrality,
            "edge_perc": self.edge_perc,
            "pagerank_alpha": self.pagerank_alpha,
            "pagerank_tol": self.pagerank_tol,
            "pagerank_max_iter": self.pagerank_max_iter,
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
            alpha_history=[],
            edge_counts_history=[Gd.number_of_edges()],
            median_weights_history=[med_w],
            algo_params=algo_params,
        )
        return res

    # ---------------------------------------------------------------------
    # ХУКИ
    # ---------------------------------------------------------------------
    def _compute_masses(self, graph: GraphMCF, nodes: List[Any], centrality: str) -> np.ndarray:
        """
        Поддержка centrality: 'degree', 'closeness', 'harmonic_closeness', 'pagerank'.
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
        eps = 0.0
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
