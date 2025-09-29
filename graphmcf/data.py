"""
Утилиты обновления Лапласиана demands-графа (инкрементально).
"""
import numpy as np
import networkx as nx

def compute_laplacian_matrix(graph: nx.Graph, nodelist=None) -> np.ndarray:
    mat = nx.laplacian_matrix(graph, nodelist=nodelist, weight="weight")
    return mat.astype(float).toarray()

def update_laplacian_on_edge_add(L: np.ndarray, i: int, j: int, w: float) -> None:
    L[i, i] += w; L[j, j] += w
    L[i, j] = L[j, i] = -w

def update_laplacian_on_edge_weight_update(L: np.ndarray, i: int, j: int, old_w: float, new_w: float) -> None:
    delta = new_w - old_w
    L[i, i] += delta; L[j, j] += delta
    L[i, j] = L[j, i] = -new_w

def update_laplacian_on_edge_remove(L: np.ndarray, i: int, j: int, w: float) -> None:
    L[i, i] -= w; L[j, j] -= w
    L[i, j] = L[j, i] = 0.0
