import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def analyze_simple(graph, alpha_target, epsilon, start_time, end_time,
                   alpha_history, edge_counts_history, median_weights_history):
    print("=== АНАЛИЗ (single run) ===")
    T = end_time - start_time
    print(f"Время: {T:.2f} c, итераций: {len(alpha_history)}")
    if alpha_history:
        print(f"alpha_target={alpha_target}, final_alpha={alpha_history[-1]:.4f}")
    # сводка «плохих шагов»
    a = np.array(alpha_history, dtype=float)
    dif = a - alpha_target
    bad = 0
    for i in range(1, len(a)):
        if dif[i-1] < -epsilon and dif[i] < dif[i-1]: bad += 1
        if dif[i-1] >  epsilon and dif[i] > dif[i-1]: bad += 1
    print(f"Плохих шагов: {bad}")

    # графики
    plt.figure(figsize=(14, 5))
    plt.subplot(1,3,1)
    plt.plot(a, lw=2); plt.axhline(alpha_target, ls="--", c="r")
    plt.fill_between(range(len(a)), alpha_target-epsilon, alpha_target+epsilon, color="r", alpha=0.1)
    plt.title("alpha by iter"); plt.grid(alpha=.3)

    plt.subplot(1,3,2)
    plt.plot(edge_counts_history, lw=2); plt.title("#edges (demands)"); plt.grid(alpha=.3)

    plt.subplot(1,3,3)
    plt.plot(median_weights_history, lw=2); plt.title("median weight (demands)"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.show()

    # короткая сводка начального/итогового графов корреспонденций
    if graph.initial_demands_graph and graph.demands_graph:
        Gi, Gf = graph.initial_demands_graph, graph.demands_graph
        wi = [d["weight"] for *_ , d in Gi.edges(data=True)]
        wf = [d["weight"] for *_ , d in Gf.edges(data=True)]
        df = pd.DataFrame({
            "метрика": ["edges", "median_w", "mean_w", "sum_w"],
            "initial": [Gi.number_of_edges(), np.median(wi) if wi else 0, np.mean(wi) if wi else 0, sum(wi) if wi else 0],
            "final"  : [Gf.number_of_edges(), np.median(wf) if wf else 0, np.mean(wf) if wf else 0, sum(wf) if wf else 0],
        })
        print(df.to_string(index=False))
