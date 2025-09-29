import numpy as np
from graphmcf import GraphMCF
from graphmcf.demands import MCFGenerator
from graphmcf.analysis import analyze_simple

A = np.array([
    [0,5,0,0,0,0],
    [5,0,3,0,0,0],
    [0,3,0,4,0,0],
    [0,0,4,0,6,0],
    [0,0,0,6,0,2],
    [0,0,0,0,2,0],
], dtype=float)

g = GraphMCF(A)
g.visualise("Исходный граф")

gen = MCFGenerator(epsilon=0.05)
res = gen.generate(graph=g, alpha_target=0.5, analysis_mode=None)

analyze_simple(g, alpha_target=0.5, epsilon=0.05,
               start_time=res.start_time, end_time=res.end_time,
               alpha_history=res.alpha_history,
               edge_counts_history=res.edge_counts_history,
               median_weights_history=res.median_weights_history)

g.visualise_with_demands()
