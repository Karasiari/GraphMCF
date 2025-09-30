from __future__ import annotations
from typing import Iterable, List, Union, Optional, Dict, Any, Tuple
import numpy as np

from ..core import GraphMCF
from ..demands import MCFGenerator, DemandsGenerationResult
from ..analysis.overall import (
    pack_overall_dict,
    compute_internal_removal_ratio,
    compute_overlap_ratio_mean,
    analyze_overall_for_graph,
)

class GraphMCFBatch:
    """
    Пакетные прогоны по коллекции графов.
    Можно передать список имён графов. Если имён меньше/нет — сгенерируем g0, g1, ...
    По каждому графу:
      - запускает все alpha_target,
      - собирает records,
      - сразу выводит сводку/графики по батчу (одному графу),
      - возвращает словарь с таблицами по каждому графу + список всех records.
    """

    def __init__(self,
                 graphs: Iterable[Union[np.ndarray, GraphMCF]],
                 graph_names: Optional[Iterable[str]] = None) -> None:
        # нормализуем графы
        self.graphs: List[GraphMCF] = []
        for g in graphs:
            if isinstance(g, GraphMCF):
                self.graphs.append(g)
            else:
                self.graphs.append(GraphMCF(g))

        # нормализуем имена
        n = len(self.graphs)
        names_list = list(graph_names) if graph_names is not None else []
        if len(names_list) < n:
            # дозаполним дефолтами
            names_list = names_list + [f"g{i}" for i in range(len(names_list), n)]
        self.graph_names: List[str] = names_list[:n]

    def _run_single_graph(
        self,
        g: GraphMCF,
        gid: int,
        gname: Optional[str],
        alpha_values: Iterable[float],
        generator: Optional[MCFGenerator] = None,
        **gen_kwargs,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        gen = generator or MCFGenerator(**gen_kwargs)
        recs: List[Dict[str, Any]] = []
        for a in alpha_values:
            res: DemandsGenerationResult = gen.generate(graph=g, alpha_target=float(a), analysis_mode=None)

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

            base["internal_removed_ratio"] = compute_internal_removal_ratio(
                getattr(res, "removal_events", None)
            )
            base["mean_overlap_ratio"] = compute_overlap_ratio_mean(
                getattr(res, "edge_mask_history", None)
            )

            recs.append(base)

        # отрисуем и выведем сводку для ЭТОГО графа (с именем)
        df_graph = analyze_overall_for_graph(recs, graph_id=gid, graph_name=gname)
        return recs, {"graph_id": gid, "graph_name": gname, "df": df_graph}

    def run_mcf_over_per_graph(
        self,
        alpha_values: Iterable[float],
        generator: Optional[MCFGenerator] = None,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """
        Главный метод: идёт по графам и для каждого строит сводку/графики по батчу alpha_target.
        Возвращает словарь:
          {
            "all_records": List[Dict[str, Any]],
            "per_graph_df": Dict[int, pd.DataFrame]
          }
        """
        all_records: List[Dict[str, Any]] = []
        per_graph_df: Dict[int, Any] = {}

        for gid, g in enumerate(self.graphs):
            gname = self.graph_names[gid] if self.graph_names and gid < len(self.graph_names) else f"g{gid}"
            recs, info = self._run_single_graph(
                g=g, gid=gid, gname=gname, alpha_values=alpha_values,
                generator=generator, **gen_kwargs
            )
            all_records.extend(recs)
            per_graph_df[int(info["graph_id"])] = info["df"]

        return {"all_records": all_records, "per_graph_df": per_graph_df}
