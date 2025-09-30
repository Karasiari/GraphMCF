from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pack_overall_dict(graph,
                      alpha_target: float,
                      epsilon: float,
                      start_time: float,
                      end_time: float,
                      alpha_history: Iterable[float],
                      edge_counts_history: Iterable[int],
                      median_weights_history: Iterable[float]) -> Dict[str, Any]:
    """
    Базовая «заморозка» результатов одиночного прогона для сводного анализа.
    (Оставлено совместимым со старыми вызовами.)
    """
    a = np.array(alpha_history, dtype=float)
    dif = a - alpha_target
    abs_dif = np.abs(dif)

    converged = bool((abs_dif <= epsilon).any()) if a.size else False
    conv_iter = int(np.argmax(abs_dif <= epsilon)) if converged else len(a)

    bad_adv = 0
    bad_frd = 0
    for i in range(1, len(a)):
        if dif[i-1] < -epsilon and dif[i] < dif[i-1]:
            bad_adv += 1
        if dif[i-1] >  epsilon and dif[i] > dif[i-1]:
            bad_frd += 1
    bad_total = bad_adv + bad_frd

    nmax = 100 * graph.graph.number_of_nodes()

    info_init: Dict[str, Any] = {}
    info_final: Dict[str, Any] = {}
    if graph.initial_demands_graph is not None and graph.demands_graph is not None:
        wi = [d["weight"] for *_ , d in graph.initial_demands_graph.edges(data=True)]
        wf = [d["weight"] for *_ , d in graph.demands_graph.edges(data=True)]
        info_init = {
            "edges_count": int(graph.initial_demands_graph.number_of_edges()),
            "median_weight": float(np.median(wi)) if wi else 0.0,
        }
        info_final = {
            "edges_count": int(graph.demands_graph.number_of_edges()),
            "median_weight": float(np.median(wf)) if wf else 0.0,
        }

    return {
        "alpha_target": float(alpha_target),
        "epsilon": float(epsilon),
        "execution_time": float(end_time - start_time),
        "iterations_total": int(len(alpha_history)),
        "iterations_convergence": int(conv_iter),
        "converged": converged,
        "convergence_ratio": float(conv_iter / nmax if nmax else 0.0),
        "initial_alpha": float(a[0]) if a.size else None,
        "final_alpha": float(a[-1]) if a.size else None,
        "bad_adjustments_total": int(bad_total),
        "bad_adjustments_adversarial": int(bad_adv),
        "bad_adjustments_friendly": int(bad_frd),
        "bad_ratio_total": float(bad_total / max(1, len(a) - 1)),
        "bad_ratio_adversarial": float(bad_adv / max(1, len(a) - 1)),
        "bad_ratio_friendly": float(bad_frd / max(1, len(a) - 1)),
        "initial_graph": info_init,
        "final_graph": info_final,
        "alpha_history": [float(x) for x in alpha_history],
        "edge_counts_history": [int(x) for x in edge_counts_history],
        "median_weights_history": [float(x) for x in median_weights_history],
    }

# ---------------------------------------------------------------------------

def _align_masks(m1: np.ndarray, m2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Выравнивает две булевы маски по длине (короткую дополняет False)."""
    m1 = np.asarray(m1, dtype=bool)
    m2 = np.asarray(m2, dtype=bool)
    L = max(m1.size, m2.size)
    if m1.size < L:
        t = np.zeros(L, dtype=bool); t[:m1.size] = m1; m1 = t
    if m2.size < L:
        t = np.zeros(L, dtype=bool); t[:m2.size] = m2; m2 = t
    return m1, m2

def compute_overlap_ratio_mean(edge_mask_history: Optional[List[np.ndarray]]) -> Optional[float]:
    """
    Возвращает среднюю долю общих рёбер между соседними снимками:
        mean_k |E_k ∩ E_{k-1}| / |E_k|
    Если снимков < 2, вернёт None.
    """
    if not edge_mask_history or len(edge_mask_history) < 2:
        return None
    vals: List[float] = []
    for k in range(1, len(edge_mask_history)):
        prev_m, curr_m = _align_masks(edge_mask_history[k-1], edge_mask_history[k])
        common = np.count_nonzero(prev_m & curr_m)
        curr_sz = max(1, int(np.count_nonzero(curr_m)))
        vals.append(float(common) / float(curr_sz))
    return float(np.mean(vals)) if vals else None

def compute_internal_removal_ratio(removal_events: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    """
    Доля внутрикластерных среди всех УДАЛЁННЫХ рёбер.
    Если нет удалений — вернёт None.
    """
    if not removal_events:
        return None
    tot = len(removal_events)
    intr = sum(1 for ev in removal_events if bool(ev.get("was_internal", False)))
    return (intr / tot) if tot > 0 else None

# ---------------------------------------------------------------------------

def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Превращает список замороженных словарей (с метаданными) в DataFrame,
    заполняя все необходимые производные метрики для графиков.
    """
    rows = []
    for r in records:
        init_edges = r.get("initial_graph", {}).get("edges_count", None)
        final_edges = r.get("final_graph", {}).get("edges_count", None)
        init_med = r.get("initial_graph", {}).get("median_weight", None)
        final_med = r.get("final_graph", {}).get("median_weight", None)

        edges_ratio = (final_edges / init_edges) if (init_edges not in (None, 0) and final_edges is not None) else np.nan
        median_ratio = (final_med / init_med) if (init_med not in (None, 0.0) and final_med is not None) else np.nan

        rows.append({
            "graph_id": r.get("graph_id"),
            "n_nodes": r.get("n_nodes"),
            "alpha_target": r.get("alpha_target"),
            "epsilon": r.get("epsilon"),
            "converged": bool(r.get("converged", False)),
            "bad_ratio_total": r.get("bad_ratio_total"),
            "edges_ratio": edges_ratio,
            "median_ratio": median_ratio,
            "internal_removed_ratio": r.get("internal_removed_ratio"),
            "mean_overlap_ratio": r.get("mean_overlap_ratio"),
            "execution_time": r.get("execution_time"),
            "initial_alpha": r.get("initial_alpha"),
            "final_alpha": r.get("final_alpha"),
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------

def plot_overall_summary(df: pd.DataFrame) -> None:
    """
    Рисует 6 графиков (пп. 2–7), как в задаче:
      2) доля плохих подгонов vs alpha_target (цвет — сходимость)
      3) финальные/начальные рёбра vs alpha_target (цвет — сходимость)
      4) медиана весов фин./нач. vs alpha_target (цвет — сходимость)
      5) доля внутрикластерных удалений vs alpha_target
      6) средняя доля общих рёбер (по снимкам) vs alpha_target
      7) время работы vs alpha_target
    """
    if df.empty:
        print("Нет данных для отрисовки.")
        return

    # палитра по сходимости
    colors = np.where(df["converged"].values, "green", "red")

    plt.figure(figsize=(16, 12))

    # 2) bad_ratio_total
    ax = plt.subplot(2, 3, 1)
    ax.scatter(df["alpha_target"], df["bad_ratio_total"], c=colors, s=30)
    ax.set_title("Доля плохих подгонов vs alpha_target")
    ax.set_xlabel("alpha_target")
    ax.set_ylabel("доля плохих")
    ax.grid(alpha=.3)

    # 3) edges_ratio
    ax = plt.subplot(2, 3, 2)
    ax.scatter(df["alpha_target"], df["edges_ratio"], c=colors, s=30)
    ax.set_title("|E_final| / |E_initial| vs alpha_target")
    ax.set_xlabel("alpha_target")
    ax.set_ylabel("отношение числа рёбер")
    ax.grid(alpha=.3)

    # 4) median_ratio
    ax = plt.subplot(2, 3, 3)
    ax.scatter(df["alpha_target"], df["median_ratio"], c=colors, s=30)
    ax.set_title("median_w(final) / median_w(initial) vs alpha_target")
    ax.set_xlabel("alpha_target")
    ax.set_ylabel("отношение медиан весов")
    ax.grid(alpha=.3)

    # 5) internal_removed_ratio
    ax = plt.subplot(2, 3, 4)
    y = df["internal_removed_ratio"].astype(float)
    ax.scatter(df["alpha_target"], y, c=colors, s=30)
    ax.set_title("Доля внутрикластерных среди удалённых ребер")
    ax.set_xlabel("alpha_target")
    ax.set_ylabel("доля внутрикластерных удалений")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)

    # 6) mean_overlap_ratio
    ax = plt.subplot(2, 3, 5)
    y = df["mean_overlap_ratio"].astype(float)
    ax.scatter(df["alpha_target"], y, c=colors, s=30)
    ax.set_title("Средняя доля общих рёбер (снимки /50 ит.)")
    ax.set_xlabel("alpha_target")
    ax.set_ylabel("mean |∩| / |E_k|")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)

    # 7) execution_time
    ax = plt.subplot(2, 3, 6)
    ax.scatter(df["alpha_target"], df["execution_time"], c=colors, s=30)
    ax.set_title("Время работы vs alpha_target")
    ax.set_xlabel("alpha_target")
    ax.set_ylabel("секунды")
    ax.grid(alpha=.3)

    plt.tight_layout()
    plt.show()

def print_overall_header(df: pd.DataFrame) -> None:
    """
    Печатает «общую сводку» (п.1):
      - количество вершин по графам
      - выбранный epsilon (если один; иначе множество)
      - среднее initial_alpha по всем прогоном
      - число сошедшихся запусков из всех
    """
    if df.empty:
        print("Сводка: данных нет.")
        return

    # n_nodes по графам
    groups = df.groupby("graph_id")["n_nodes"].max().to_dict()
    nodes_line = ", ".join([f"g{gid}: {n}" for gid, n in sorted(groups.items())])

    epsilons = sorted(set(df["epsilon"].dropna().astype(float).tolist()))
    eps_line = (f"{epsilons[0]}" if len(epsilons) == 1 else f"{epsilons}")

    init_alpha_vals = df["initial_alpha"].dropna().astype(float)
    init_alpha_mean = init_alpha_vals.mean() if not init_alpha_vals.empty else float("nan")

    conv_count = int(df["converged"].sum())
    total = int(len(df))

    print("=== ОБЩАЯ СВОДКА (batch) ===")
    print(f"Вершины по графам: {nodes_line}")
    print(f"epsilon: {eps_line}")
    print(f"Среднее initial_alpha: {init_alpha_mean:.4f}")
    print(f"Сошедшихся запусков: {conv_count} из {total} ({conv_count/total:.2%})")

def analyze_overall_batch(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Удобный помощник: превращает записи в DataFrame, печатает сводку (п.1)
    и рисует все 6 графиков (пп.2–7). Возвращает DataFrame.
    """
    df = records_to_dataframe(records)
    print_overall_header(df)
    plot_overall_summary(df)
    return df
