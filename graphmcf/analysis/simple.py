import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def analyze_simple(graph, alpha_target, epsilon, start_time, end_time,
                   alpha_history, edge_counts_history, median_weights_history,
                   edge_mask_history, edge_mask_snapshot_iters, removal_events,
                   algo_params: dict | None = None,):
    print("=== АНАЛИЗ (single run) ===")
    T = end_time - start_time
    a = np.array(alpha_history, dtype=float)
    n_steps = max(0, len(a) - 1)
    dif = a - float(alpha_target)

    if algo_params is not None:
      ne = algo_params.get("num_edges", None)
      pdel = algo_params.get("p_for_delete_edge", None)
      pins = algo_params.get("p_for_upsert_edge", None)
      if algo_params.get("variant", "simple") == "multi_edges": 
        print(f"num_edges={ne}, p_for_delete_edge={pdel}, p_for_upsert_edge={pins}")
                     
    # базовые метрики
    n_nodes = graph.graph.number_of_nodes() if hasattr(graph, "graph") else None
    max_iter = 100 * n_nodes if n_nodes is not None else None
    converged = bool((np.abs(dif) <= float(epsilon)).any()) if a.size else False
    init_alpha = float(a[0]) if a.size else None
    final_alpha = float(a[-1]) if a.size else None

    # классификация шагов (плохие/хорошие)
    bad_total = 0
    bad_adv = 0
    bad_frd = 0
    good_idx = []
    bad_idx = []

    for i in range(1, len(a)):
        prev, cur = dif[i-1], dif[i]
        if prev < -epsilon:  # ожидался adversarial (нужно увеличить alpha)
            if cur < prev:   # alpha снизилась ещё → плохо
                bad_total += 1; bad_adv += 1; bad_idx.append(i)
            else:            # выросла или не изменилась → ок
                good_idx.append(i)
        elif prev > epsilon: # ожидался friendly (нужно уменьшить alpha)
            if cur > prev:   # alpha выросла ещё → плохо
                bad_total += 1; bad_frd += 1; bad_idx.append(i)
            else:            # снизилась или не изменилась → ок
                good_idx.append(i)

    bad_ratio = round(bad_total / n_steps, 2) if n_steps else 0.0
    bad_adv_ratio = round(bad_adv / n_steps, 2) if n_steps else 0.0
    bad_frd_ratio = round(bad_frd / n_steps, 2) if n_steps else 0.0

    # печать расширенной сводки
    print(f"Время выполнения: {T:.2f} c")
    if max_iter is not None:
        used_share = (len(alpha_history) / max_iter) if max_iter else 0.0
        print(f"Итераций: {len(alpha_history)} (доля от максимума {max_iter} = {used_share:.2%})")
    else:
        print(f"Итераций: {len(alpha_history)}")

    if a.size:
        print(f"alpha_target={alpha_target}, epsilon={epsilon}")
        print(f"initial_alpha={init_alpha:.4f}, final_alpha={final_alpha:.4f}, converged={converged}")

    print(f"Плохие шаги всего: {bad_total} (доля {bad_ratio:.2f})")
    print(f"  — adversarial-плохих: {bad_adv} (доля {bad_adv_ratio:.2f})")
    print(f"  — friendly-плохих:   {bad_frd} (доля {bad_frd_ratio:.2f})")

    # базовые графики
    plt.figure(figsize=(14, 5))
    # 1) alpha by iter (+ подсветка)
    plt.subplot(1, 3, 1)
    x = np.arange(len(a))
    plt.plot(x, a, lw=2)
    plt.axhline(alpha_target, ls="--", c="r")
    plt.fill_between(x, alpha_target - epsilon, alpha_target + epsilon, color="r", alpha=0.1)
    if good_idx: plt.scatter(good_idx, a[good_idx], s=18, c="green", label="хорошие")
    if bad_idx:  plt.scatter(bad_idx,  a[bad_idx],  s=18, c="red",   label="плохие")
    plt.title("alpha by iter"); plt.grid(alpha=.3); plt.legend(loc="best", fontsize=9)

    # 2) #edges (demands)
    plt.subplot(1, 3, 2)
    plt.plot(edge_counts_history, lw=2)
    plt.title("#edges (demands)"); plt.grid(alpha=.3)

    # 3) median weight (demands)
    plt.subplot(1, 3, 3)
    plt.plot(median_weights_history, lw=2)
    plt.title("median weight (demands)"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.show()

    # ===== Доп. графики стабильности =====

    def _align_masks(m1: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Выравнивает две булевы маски по длине (короткую дополняет False)."""
        m1 = np.asarray(m1, dtype=bool)
        m2 = np.asarray(m2, dtype=bool)
        L = max(m1.size, m2.size)
        if m1.size < L:
            tmp = np.zeros(L, dtype=bool); tmp[:m1.size] = m1; m1 = tmp
        if m2.size < L:
            tmp = np.zeros(L, dtype=bool); tmp[:m2.size] = m2; m2 = tmp
        return m1, m2

    # 1) |E_k ∩ E_{k-1}|
    if edge_mask_history and edge_mask_snapshot_iters and len(edge_mask_history) >= 2:
        overlap_counts = []
        snap_x = []
        for k in range(1, len(edge_mask_history)):
            prev_mask, curr_mask = _align_masks(edge_mask_history[k-1], edge_mask_history[k])
            common = int(np.count_nonzero(prev_mask & curr_mask))
            overlap_counts.append(common)
            snap_x.append(int(edge_mask_snapshot_iters[k]))
        plt.figure(figsize=(10,4))
        plt.plot(snap_x, overlap_counts, lw=2)
        plt.title("Стабильность рёбер: |E_k ∩ E_{k-1}|")
        plt.xlabel("итерация (снимки раз в 50)"); plt.ylabel("кол-во общих рёбер")
        plt.grid(alpha=.3); plt.show()

    # 2) |E_k ∩ E_{k-1}| / |E_k|
    if edge_mask_history and edge_mask_snapshot_iters and len(edge_mask_history) >= 2:
        overlap_ratio = []
        snap_x = []
        for k in range(1, len(edge_mask_history)):
            prev_mask, curr_mask = _align_masks(edge_mask_history[k-1], edge_mask_history[k])
            common = np.count_nonzero(prev_mask & curr_mask)
            curr_sz = max(1, int(np.count_nonzero(curr_mask)))  # защита от деления на 0
            overlap_ratio.append(float(common) / float(curr_sz))
            snap_x.append(int(edge_mask_snapshot_iters[k]))
        plt.figure(figsize=(10,4))
        plt.plot(snap_x, overlap_ratio, lw=2)
        plt.title("Стабильность (норм.): |E_k ∩ E_{k-1}| / |E_k|")
        plt.xlabel("итерация (снимки раз в 50)"); plt.ylabel("доля общих рёбер")
        plt.ylim(0, 1); plt.grid(alpha=.3); plt.show()

    # 3) Доля внутрикластерных удалений по окнам из 10 итераций
    if removal_events:
        events = removal_events
        buckets_internal = {}
        buckets_total = {}
        for ev in events:
            it_ev = int(ev.get("iter", 0))
            b = (it_ev - 1) // 10 if it_ev > 0 else 0
            buckets_total[b] = buckets_total.get(b, 0) + 1
            if bool(ev.get("was_internal", False)):
                buckets_internal[b] = buckets_internal.get(b, 0) + 1

        bs = sorted(set(buckets_total.keys()) | set(buckets_internal.keys()))
        xs = [(b + 1) * 10 for b in bs]  # верхняя граница окна
        ys = []
        for b in bs:
            tot = buckets_total.get(b, 0)
            intr = buckets_internal.get(b, 0)
            ys.append((intr / tot) if tot > 0 else np.nan)

        plt.figure(figsize=(10,4))
        plt.plot(xs, ys, lw=2)
        plt.title("Доля внутрикластерных удалённых рёбер (окна по 10 итераций)")
        plt.xlabel("итерация (верхняя граница окна)")
        plt.ylabel("доля внутрикластерных удалений")
        plt.ylim(0, 1); plt.grid(alpha=.3); plt.show()

    # короткая сводка (initial vs final) — округление до целых
    if getattr(graph, "initial_demands_graph", None) and getattr(graph, "demands_graph", None):
        Gi, Gf = graph.initial_demands_graph, graph.demands_graph
        wi = [d["weight"] for *_, d in Gi.edges(data=True)]
        wf = [d["weight"] for *_, d in Gf.edges(data=True)]

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
