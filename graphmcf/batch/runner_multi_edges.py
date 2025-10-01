from __future__ import annotations
from typing import Iterable, List, Union, Optional, Dict, Any, Tuple
import numpy as np

from ..core import GraphMCF
from ..demands import MCFGeneratorMultiEdges, DemandsGenerationResultMulti
from ..analysis.overall import (
    pack_overall_dict,
    compute_internal_removed_ratio_windowed,
    compute_overlap_ratio_mean,
    analyze_overall_for_graph,
)

class GraphMCFBatchMultiEdges:
    """
    Пакетные прогоны по коллекции графов для модификации multi-edges.
    Отличие от GraphMCFBatch — по умолчанию использует MCFGeneratorMultiEdges.
    Остальная логика, метрики и графики совпадают.

    Все параметры генератора передаются через **gen_kwargs, например:
        num_edges=None (=> ceil(n**0.25)),
        p_for_delete_edge=1.0,
        p_for_upsert_edge=1.0,
        p_ER, distribution, median_weight_for_initial, var_for_initial,
        demands_median_denominator, demands_var_denominator, epsilon, max_iter
    """

    def __init__(
        self,
        graphs: Iterable[Union[np.ndarray, GraphMCF]],
        graph_names: Optional[Iterable[str]] = None
    ) -> None:
        self.graphs: List[GraphMCF] = []
        for g in graphs:
            self.graphs.append(g if isinstance(g, GraphMCF) else GraphMCF(g))

        n = len(self.graphs)
        names_list = list(graph_names) if graph_names is not None else []
        if len(names_list) < n:
            names_list = names_list + [f"g{i}" for i in range(len(names_list), n)]
        self.graph_names: List[str] = names_list[:n]

    def _run_single_graph(
        self,
        g: GraphMCF,
        gid: int,
        gname: Optional[str],
        alpha_values: Iterable[float],
        generator: Optional[MCFGeneratorMultiEdges] = None,
        **gen_kwargs,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        gen = generator or MCFGeneratorMultiEdges(**gen_kwargs)
        recs: List[Dict[str, Any]] = []

        for a in alpha_values:
            res: DemandsGenerationResultMulti = gen.generate(graph=g, alpha_target=float(a), analysis_mode=None)

            base = pack_overall_dict(
                graph=g,
                alpha_target=float(a),
                epsilon=float(gen.epsilon),
                start_time=res.start_time,
                end_time=res.end_time,
                alpha_history=res.alpha_history,
                edge_counts_history=res.edge_counts_history,
                median_weights_history=res.median_weights_history,
            )

            base["graph_id"] = int(gid)
            base["graph_name"] = str(gname) if gname is not None else f"g{gid}"
            base["n_nodes"] = int(g.graph.number_of_nodes())

            base["internal_removed_ratio"] = compute_internal_removed_ratio_windowed(
                getattr(res, "removal_events", None),
                window=20
            )
            base["mean_overlap_ratio"] = compute_overlap_ratio_mean(
                getattr(res, "edge_mask_history", None)
            )

            # num_edges для сводки
            algo_params = getattr(res, "algo_params", None)
            if isinstance(algo_params, dict):
                ne = algo_params.get("num_edges", None)
                base["num_edges"] = int(ne) if (ne is not None) else None

            recs.append(base)

        df_graph = analyze_overall_for_graph(recs, graph_id=gid, graph_name=gname)
        return recs, {"graph_id": gid, "graph_name": gname, "df": df_graph}

    def run_mcf_over_per_graph(
        self,
        alpha_values: Iterable[float],
        generator: Optional[MCFGeneratorMultiEdges] = None,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """
        Идём по графам и для каждого строим сводку/графики по батчу alpha_target.
        По умолчанию используем MCFGeneratorMultiEdges.
        """
        all_records: List[Dict[str, Any]] = []
        per_graph_df: Dict[int, Any] = {}

        for gid, g in enumerate(self.graphs):
            gname = self.graph_names[gid] if self.graph_names and gid < len(self.graphs) else f"g{gid}"
            recs, info = self._run_single_graph(
                g=g, gid=gid, gname=gname, alpha_values=alpha_values,
                generator=generator, **gen_kwargs
            )
            all_records.extend(recs)
            per_graph_df[int(info["graph_id"])] = info["df"]

        return {"all_records": all_records, "per_graph_df": per_graph_df}
