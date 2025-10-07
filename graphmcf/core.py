"""
Базовый объект GraphMCF: загрузка графа из матрицы смежности, визуализация,
генерация cut-векторов, расчёт alpha (с кэшем Лапласиана demands-графа),
и хелперы для инкрементального обновления demands-графа.
"""
from __future__ import annotations
import copy
import random
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any, Tuple, List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigsh

from .data import (
    compute_laplacian_matrix,
    update_laplacian_on_edge_add,
    update_laplacian_on_edge_weight_update,
    update_laplacian_on_edge_remove,
)

class GraphMCF:
    def __init__(self, adjacency_matrix: np.ndarray) -> None:
        self.adjacency_matrix = np.array(adjacency_matrix, dtype=float)
        self._validate_adjacency_matrix()
        self.graph = self._create_networkx_graph()
        self.n = self.graph.number_of_nodes()

        # demands-граф и его Лапласиан (кэшируются и инкрементально обновляются)
        self.initial_demands_graph: Optional[nx.Graph] = None
        self.demands_graph: Optional[nx.Graph] = None
        self.demands_laplacian: Optional[np.ndarray] = None

        # кэши для расчёта alpha / cut
        self.graph_pinv_sqrt: Optional[np.ndarray] = None
        self.graph_spec: Optional[Dict[str, Any]] = None

        # результаты анализа последнего запуска (если делали simple-анализ)
        self.analysis_results: Optional[Dict[str, Any]] = None

    # ---------- базовая подготовка ----------
    def _validate_adjacency_matrix(self) -> None:
        A = self.adjacency_matrix
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Матрица смежности должна быть квадратной")
        if not np.allclose(A, A.T):
            raise ValueError("Матрица смежности должна быть симметричной (неориентированный граф)")
        if (A < 0).any():
            raise ValueError("Веса рёбер должны быть неотрицательными")

    def _create_networkx_graph(self) -> nx.Graph:
        A = self.adjacency_matrix
        n = A.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                w = A[i, j]
                if w:
                    G.add_edge(i, j, weight=float(w))
        # берём крупнейшую компоненту
        comp = max(nx.connected_components(G), key=len)
        Gc = G.subgraph(comp).copy()
        # перенумеруем матрицу под компоненту
        nodes = sorted(Gc.nodes())
        idx = {u: i for i, u in enumerate(nodes)}
        A2 = np.zeros((len(nodes), len(nodes)))
        for u, v, d in Gc.edges(data=True):
            i, j = idx[u], idx[v]
            A2[i, j] = A2[j, i] = d["weight"]
        self.adjacency_matrix = A2
        return Gc

    # ---------- визуализация ----------
    def visualise(self, title="Исходный граф", node_size=300, font_size=10) -> None:
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(9, 7))
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size, node_color="#4C79DA", alpha=0.9)
        nx.draw_networkx_edges(self.graph, pos, edge_color="#888", alpha=0.8)
        edge_labels = {(u, v): f"{d['weight']:.0f}" for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=font_size)
        plt.title(title); plt.axis("off"); plt.tight_layout(); plt.show()

    def visualise_with_demands(
        self,
        node_size: int = 110,
        font_size: int = 9,
        figsize=(14, 6),
        demand_edge_width_range=(1.5, 6.0),
        node_color="dimgray",
        base_edge_color="gray",
        demand_edge_cmap="viridis",
        edge_alpha=0.9,
        colorbar_label="Вес корреспонденции",
    ) -> None:
        if not isinstance(self.demands_graph, nx.Graph):
            raise AttributeError("self.demands_graph не задан")
        DG = self.demands_graph
        pos = nx.spring_layout(self.graph, seed=42)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)

        # слева — базовый граф
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size, node_color=node_color, alpha=0.95, ax=axL)
        nx.draw_networkx_edges(self.graph, pos, edge_color=base_edge_color, width=1.5, alpha=edge_alpha, ax=axL)
        labels = {(u, v): f"{d.get('weight', 0):.0f}" for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, font_size=font_size, ax=axL)
        axL.set_title("Исходный граф"); axL.axis("off")

        # справа — demands
        nx.draw_networkx_nodes(DG, pos, node_size=node_size, node_color=node_color, alpha=0.95, ax=axR)
        edgelist = list(DG.edges(data=True))
        uv = [(u, v) for u, v, _ in edgelist]
        W = np.array([float(d.get("weight", 1.0)) for _, _, d in edgelist]) if edgelist else np.array([])
        # ширины
        if W.size:
            w_min, w_max = float(W.min()), float(W.max())
            lo, hi = demand_edge_width_range
            widths = [0.5 * (lo + hi)] * len(W) if np.isclose(w_min, w_max) else list(lo + (W - w_min) * (hi - lo) / (w_max - w_min))
        else:
            widths = []
        # цвета
        cmap = mpl.cm.get_cmap(demand_edge_cmap)
        vmin, vmax = (float(W.min()), float(W.max())) if W.size else (0.0, 1.0)
        nx.draw_networkx_edges(DG, pos, edgelist=uv, width=widths, edge_color=W, edge_cmap=cmap,
                               edge_vmin=vmin, edge_vmax=vmax, alpha=edge_alpha, ax=axR)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cbar = plt.colorbar(sm, ax=axR); cbar.set_label(colorbar_label, fontsize=font_size)
        axR.set_title("Граф корреспонденций"); axR.axis("off")
        plt.tight_layout(); plt.show()

    # ---------- demands: init + alpha + cut ----------
    def generate_initial_demands(
        self,
        p: float = 0.5,
        distribution: str = "normal",
        median_weight: int = 50,
        var: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        if distribution != "normal":
            raise ValueError("Пока поддерживается только distribution='normal'")
        n = self.graph.number_of_nodes()
        base_nodes = list(self.graph.nodes())
        G_rand = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
        # случайная пермутация отображения вершин
        perm = list(range(n)); random.shuffle(perm)
        mapping = {i: base_nodes[perm[i]] for i in G_rand.nodes()}
        G_rand = nx.relabel_nodes(G_rand, mapping)
        # веса ~ нормальным образом (дискретно)
        import numpy as np
        from scipy.stats import norm as _norm
        def draw():
            mu, sigma = median_weight, np.sqrt(var)
            lo, hi = 1, 2 * median_weight
            xs = np.arange(lo, hi + 1)
            ps = _norm.pdf(xs, loc=mu, scale=sigma); ps /= ps.sum()
            return int(np.random.choice(xs, p=ps))
        Gd = nx.Graph(); Gd.add_nodes_from(base_nodes)
        for u, v in G_rand.edges():
            Gd.add_edge(u, v, weight=draw())
        self.initial_demands_graph = copy.deepcopy(Gd)
        self.demands_graph = Gd
        self.demands_laplacian = compute_laplacian_matrix(Gd, nodelist=base_nodes)

    def generate_initial_multidemands(
        self,
        *,
        p: float = 0.5,
        distribution: str = "normal",
        median_weight: int = 50,
        var: int = 100,
        multi_max: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        if distribution != "normal":
            raise ValueError("Пока поддерживается только distribution='normal'")
        n = self.graph.number_of_nodes()
        nodes = list(self.graph.nodes())        
        G_rand = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
        # случайная пермутация отображения вершин
        perm = list(range(n)); random.shuffle(perm)
        mapping = {i: nodes[perm[i]] for i in G_rand.nodes()}
        G_rand = nx.relabel_nodes(G_rand, mapping)
        # веса ~ нормальным образом (дискретно)
        import numpy as np
        from scipy.stats import norm as _norm
        def draw():
            mu, sigma = median_weight, np.sqrt(var)
            lo, hi = 1, 2 * median_weight
            xs = np.arange(lo, hi + 1)
            ps = _norm.pdf(xs, loc=mu, scale=sigma); ps /= ps.sum()
            return int(np.random.choice(xs, p=ps))
        Gd = nx.Graph(); Gd.add_nodes_from(nodes)
        for u, v in G_rand.edges():
            Gd.add_edge(u, v, weight=draw())
        self.initial_demands_graph = copy.deepcopy(Gd)
        self.demands_graph = Gd

        self.demands_multigraph = nx.MultiGraph()
        self.demands_multigraph.add_nodes_from(nodes)
        multi_max = int(max(1, multi_max))
        n = len(nodes)
        if n >= 2:
            for i in range(n):
                u = nodes[i]
                for j in range(i + 1, n):
                    v = nodes[j]
                    if np.random.rand() <= float(p):
                        w = int(np.random.choice(xs, p=ps))
                        k = int(np.random.randint(1, multi_max + 1))
                        per = int(round(w / k))

                        if per <= 0:
                            # НЕ дробим: одно мультиребро исходного веса w
                            self.demands_multigraph.add_edge(u, v, weight=float(w))
                            agg_w = float(w)
                        else:
                            # дробим на k мультирёбер веса per
                            for _ in range(k):
                                self.demands_multigraph.add_edge(u, v, weight=float(per))
                                agg_w = float(k * per)

                        self.demands_graph.add_edge(u, v, weight=agg_w)

        # сохранить копии начального состояния
        self.initial_demands_multigraph = self.demands_multigraph.copy()
        self.demands_laplacian = compute_laplacian_matrix(self.demands_graph, nodelist=nodes)

    def _ensure_graph_pinv_sqrt(self) -> np.ndarray:
        if self.graph_pinv_sqrt is None:
            nodelist = list(self.graph.nodes())
            Lg = nx.laplacian_matrix(self.graph, nodelist=nodelist, weight="weight").astype(float).toarray()
            Lg_pinv = np.linalg.pinv(Lg)
            self.graph_pinv_sqrt = fractional_matrix_power(Lg_pinv, 0.5)
        return self.graph_pinv_sqrt

    def calculate_alpha(self) -> float:
        if self.demands_graph is None:
            raise AttributeError("demands_graph не задан")
        Ld = self.demands_laplacian
        if Ld is None:
            Ld = compute_laplacian_matrix(self.demands_graph, nodelist=list(self.graph.nodes()))
            self.demands_laplacian = Ld
        Lg_inv_sqrt = self._ensure_graph_pinv_sqrt()
        L_alpha = Lg_inv_sqrt @ Ld @ Lg_inv_sqrt
        # наибольший собственный
        eig, _ = eigsh(L_alpha, k=1, which="LA")
        lam_max = float(eig[0]) if eig.size else 0.0
        #lam_max = float(np.linalg.eigvalsh(L_alpha)[-1]) if L_alpha.size else 0.0
        tr = float(np.trace(L_alpha))
        return lam_max / tr if tr != 0.0 else float("inf")

    def calculate_alpha_timed(self) -> (float, dict):
        import time
        import numpy as np
        import networkx as nx

        timings = {"t_prep": 0.0, "t_ld": 0.0, "t_lalpha": 0.0, "t_eig": 0.0}

        if self.graph_pinv_sqrt is None:
            t0 = time.perf_counter()
            nodelist = list(self.graph.nodes())
            Lg = nx.laplacian_matrix(self.graph, nodelist=nodelist, weight="weight").astype(float).toarray()
            Lg_pinv = np.linalg.pinv(Lg)
            from scipy.linalg import fractional_matrix_power
            self.graph_pinv_sqrt = fractional_matrix_power(Lg_pinv, 0.5)
            timings["t_prep"] = time.perf_counter() - t0

        S = self.graph_pinv_sqrt
        
        t0 = time.perf_counter()
        if self.demands_laplacian is None:
            nodelist = list(self.graph.nodes())
            self.demands_laplacian = nx.laplacian_matrix(self.demands_graph, nodelist=nodelist, weight="weight").astype(float).toarray()
        Ld = self.demands_laplacian
        timings["t_ld"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        SLd = S @ Ld
        L_alpha = SLd @ S
        timings["t_lalpha"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        eig, _ = eigsh(L_alpha, k=1, which="LA")
        lam_max = float(eig[0]) if eig.size else 0.0
        timings["t_eig"] = time.perf_counter() - t0

        tr = float(np.trace(L_alpha))
        alpha = lam_max / tr if tr != 0.0 else float("inf")
        return alpha, timings

    def generate_cut(self, type: str = "friendly", rng_seed: Optional[int] = None) -> np.ndarray:
        if type not in {"friendly", "adversarial"}:
            raise ValueError("type должен быть 'friendly' или 'adversarial'")
        if self.graph_spec is None:
            nodelist = list(self.graph.nodes())
            L = nx.laplacian_matrix(self.graph, nodelist=nodelist, weight="weight").astype(float).toarray()
            vals, vecs = np.linalg.eigh(L)
            eps = 1e-12
            start = 1 if vals[0] <= eps else 0
            self.graph_spec = {"eigvals": vals[start:], "eigvecs": vecs[:, start:]}

        vecs = self.graph_spec["eigvecs"]
        m = vecs.shape[1]
        if m <= 1:
            v = vecs[:, 0] if m == 1 else np.ones(self.n)
            return v / (np.linalg.norm(v) or 1.0)
        mid = m // 2
        lower, upper = vecs[:, :mid], vecs[:, mid:] if m % 2 == 0 else vecs[:, mid + 1:]
        part = upper if type == "friendly" else lower
        rng = np.random.default_rng(rng_seed)
        coeffs = np.abs(rng.normal(size=part.shape[1]))
        v = part @ coeffs
        nrm = np.linalg.norm(v)
        return v / (nrm or 1.0)

    # ---------- публичные хелперы для генератора ----------
    def nodelist_and_index(self) -> Tuple[List[int], Dict[int, int]]:
        nodes = list(self.graph.nodes())
        return nodes, {u: i for i, u in enumerate(nodes)}

    def remove_edge_by_indices(self, iu: int, iv: int) -> Optional[float]:
        """Удаляет ребро (по индексам узлов) из demands-графа и обновляет Лапласиан. Возвращает старый вес или None."""
        if self.demands_graph is None:
            return None
        nodes, _ = self.nodelist_and_index()
        u, v = nodes[iu], nodes[iv]
        if not self.demands_graph.has_edge(u, v):
            return None
        w = float(self.demands_graph[u][v]["weight"])
        self.demands_graph.remove_edge(u, v)
        update_laplacian_on_edge_remove(self.demands_laplacian, iu, iv, w)
        return w

    def upsert_edge_by_indices(self, iu: int, iv: int, delta_w: float) -> float:
        """
        Добавляет новое ребро или увеличивает вес существующего (на delta_w>0) в demands-графе.
        Возвращает новый итоговый вес ребра.
        """
        if iu > iv:
            iu, iv = iv, iu
        nodes, _ = self.nodelist_and_index()
        u, v = nodes[iu], nodes[iv]
        if self.demands_graph is None:
            self.demands_graph = nx.Graph(); self.demands_graph.add_nodes_from(nodes)
            self.demands_laplacian = np.zeros((len(nodes), len(nodes)))
        if self.demands_graph.has_edge(u, v):
            old = float(self.demands_graph[u][v]["weight"])
            new = old + float(delta_w)
            self.demands_graph[u][v]["weight"] = new
            update_laplacian_on_edge_weight_update(self.demands_laplacian, iu, iv, old, new)
            return new
        else:
            w = float(delta_w)
            self.demands_graph.add_edge(u, v, weight=w)
            update_laplacian_on_edge_add(self.demands_laplacian, iu, iv, w)
            return w

    # ---------- удобный раннер: запуск алгоритма как метода ----------
    def run_mcf(self, **kwargs):
        """Удобная обёртка — запустить MCF-генератор как метод этого объекта."""
        from .demands.mcf_generator import MCFGenerator
        gen = MCFGenerator(**kwargs.pop("generator_kwargs", {}))
        return gen.generate(graph=self, **kwargs)
