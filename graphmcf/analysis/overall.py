import numpy as np

def pack_overall_dict(graph, alpha_target, epsilon, start_time, end_time,
                      alpha_history, edge_counts_history, median_weights_history):
    a = np.array(alpha_history, dtype=float)
    dif = a - alpha_target
    abs_dif = np.abs(dif)
    converged = bool((abs_dif <= epsilon).any()) if a.size else False
    conv_iter = int(np.argmax(abs_dif <= epsilon)) if converged else len(a)
    bad_adv = bad_frd = 0
    for i in range(1, len(a)):
        if dif[i-1] < -epsilon and dif[i] < dif[i-1]: bad_adv += 1
        if dif[i-1] >  epsilon and dif[i] > dif[i-1]: bad_frd += 1
    bad_total = bad_adv + bad_frd
    nmax = 100 * graph.graph.number_of_nodes()
    info_init = {}
    info_final = {}
    if graph.initial_demands_graph is not None and graph.demands_graph is not None:
        wi = [d["weight"] for *_ , d in graph.initial_demands_graph.edges(data=True)]
        wf = [d["weight"] for *_ , d in graph.demands_graph.edges(data=True)]
        info_init = {
            "edges_count": graph.initial_demands_graph.number_of_edges(),
            "median_weight": float(np.median(wi)) if wi else 0.0,
        }
        info_final = {
            "edges_count": graph.demands_graph.number_of_edges(),
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
        "bad_ratio_total": float(bad_total / max(1, len(a)-1)),
        "bad_ratio_adversarial": float(bad_adv / max(1, len(a)-1)),
        "bad_ratio_friendly": float(bad_frd / max(1, len(a)-1)),
        "initial_graph": info_init,
        "final_graph": info_final,
        "alpha_history": [float(x) for x in alpha_history],
        "edge_counts_history": [int(x) for x in edge_counts_history],
        "median_weights_history": [float(x) for x in median_weights_history],
    }

def analyze_overall(*args, **kwargs):
    # оставлено для обратной совместимости, если захочешь вызывать как раньше
    return pack_overall_dict(*args, **kwargs)
