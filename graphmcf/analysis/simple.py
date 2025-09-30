import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def analyze_simple(graph, alpha_target, epsilon, start_time, end_time,
                   alpha_history, edge_counts_history, median_weights_history):
    print("=== АНАЛИЗ (single run) ===")
    T = end_time - start_time
    a = np.array(alpha_history, dtype=float)
    n_steps = max(0, len(a) - 1)
    dif = a - float(alpha_target)

    # базовые метрики
    n_nodes = graph.graph.number_of_nodes() if hasattr(graph, "graph") else None
    max_iter = 100 * n_nodes if n_nodes is not None else None
    converged = bool((np.abs(dif) <= float(epsilon)).any()) if a.size else False
    init_alpha = float(a[0]) if a.size else None
    final_alpha = float(a[-1]) if a.size else None

    # классификация шагов
    bad_total = 0
    bad_adv = 0
    bad_frd = 0
    good_idx = []
    bad_idx = []

    for i in range(1, len(a)):
        prev, cur = dif[i-1], dif[i]
        if prev < -epsilon:  # ожидался adversarial (нужно увеличить alpha)
            if cur < prev:  # alpha ещё уменьшилась → плохо
                bad_total += 1; bad_adv += 1; bad_idx.append(i)
            else:           # alpha выросла или застыла → хорошо
                good_idx.append(i)
        elif prev > epsilon:  # ожидался friendly (нужно уменьшить alpha)
            if cur > prev:   # alpha ещё выросла → плохо
                bad_total += 1; bad_frd += 1; bad_idx.append(i)
            else:            # alpha уменьшилась или застыла → хорошо
                good_idx.append(i)

    bad_ratio = round(bad_total / n_steps, 2) if n_steps else 0.0
    bad_adv_ratio = round(bad_adv / n_steps, 2) if n_steps else 0.0
    bad_frd_ratio = round(bad_frd / n_steps, 2) if n_steps else 0.0

    # печать расширенной сводки
    print(f"Время выполнения: {T:.2f} c")
    if max_iter is not None:
        used_share = (len(alpha_history) / max_iter) if max_iter else 0.0
        print(f"Итераций: {len(alpha_history)} "
              f"(доля от максимума {max_iter} = {used_share:.2%})")
    else:
        print(f"Итераций: {len(alpha_history)}")

    if a.size:
        print(f"alpha_target={alpha_target}, epsilon={epsilon}")
        print(f"initial_alpha={init_alpha:.4f}, final_alpha={final_alpha:.4f}, "
              f"converged={converged}")

    print(f"Плохие шаги всего: {bad_total} (доля {bad_ratio:.2f})")
    print(f"  — из них adversarial-плохих: {bad_adv} (доля {bad_adv_ratio:.2f})")
    print(f"  — из них friendly-плохих:   {bad_frd} (доля {bad_frd_ratio:.2f})")

    # графики
    plt.figure(figsize=(14, 5))

    # 1) alpha by iter + подсветка плохих/хороших
    plt.subplot(1, 3, 1)
    x = np.arange(len(a))
    plt.plot(x, a, lw=2)
    plt.axhline(alpha_target, ls="--", c="r")
    plt.fill_between(x, alpha_target - epsilon, alpha_target + epsilon, color="r", alpha=0.1)

    if good_idx:
        plt.scatter(good_idx, a[good_idx], s=18, c="green", label="хорошие")
    if bad_idx:
        plt.scatter(bad_idx, a[bad_idx], s=18, c="red", label="плохие")

    plt.title("alpha by iter")
    plt.grid(alpha=.3)
    plt.legend(loc="best", fontsize=9)

    # 2) #edges (demands)
    plt.subplot(1, 3, 2)
    plt.plot(edge_counts_history, lw=2)
    plt.title("#edges (demands)")
    plt.grid(alpha=.3)

    # 3) median weight (demands)
    plt.subplot(1, 3, 3)
    plt.plot(median_weights_history, lw=2)
    plt.title("median weight (demands)")
    plt.grid(alpha=.3)

    plt.tight_layout()
    plt.show()

    # короткая сводка (initial vs final) — округление до целых
    if getattr(graph, "initial_demands_graph", None) and getattr(graph, "demands_graph", None):
        Gi, Gf = graph.initial_demands_graph, graph.demands_graph
        wi = [d["weight"] for *_ , d in Gi.edges(data=True)]
        wf = [d["weight"] for *_ , d in Gf.edges(data=True)]

        def rint(x):
            return int(np.rint(x)) if isinstance(x, (int, float, np.floating)) else int(x)

        df = pd.DataFrame({
            "метрика": ["edges", "median_w", "mean_w", "sum_w"],
            "initial": [
                rint(Gi.number_of_edges()),
                rint(np.median(wi) if wi else 0),
                rint(np.mean(wi) if wi else 0),
                rint(sum(wi) if wi else 0),
            ],
            "final": [
                rint(Gf.number_of_edges()),
                rint(np.median(wf) if wf else 0),
                rint(np.mean(wf) if wf else 0),
                rint(sum(wf) if wf else 0),
            ],
        })
        print(df.to_string(index=False))
