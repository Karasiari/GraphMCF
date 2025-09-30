from __future__ import annotations
from typing import Iterable, List, Union, Optional, Dict, Any
import numpy as np

from ..core import GraphMCF
from ..demands import MCFGenerator, DemandsGenerationResult
from ..analysis.overall import pack_overall_dict, compute_internal_removal_ratio, compute_overlap_ratio_mean

class GraphMCFBatch:
    """
    Утилита для пакетных прогонов генераторов по выборке графов.
    Важно: мы «замораживаем» метрики сразу после каждого прогона, чтобы
    последующие запуски на том же объекте графа не перезаписали статистики.
    """

    def __init__(self, graphs: Iterable[Union[np.ndarray, GraphMCF]]) -> None:
        self.graphs: List[GraphMCF] = []
        for g in graphs:
            if isinstance(g, GraphMCF):
                self.graphs.append(g)
            else:
                self.graphs.append(GraphMCF(g))

    def run_mcf_over(
        self,
        alpha_values: Iterable[float],
        generator: Optional[MCFGenerator] = None,
        **gen_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Прогоняет MCFGenerator по всем графам и всем alpha_target.
        Возвращает список «замороженных» словарей записей (records),
        пригодных для анализа overall.
        """
        gen = generator or MCFGenerator(**gen_kwargs)

        records: List[Dict[str, Any]] = []
        for gid, g in enumerate(self.graphs):
            for a in alpha_values:
                res: DemandsGenerationResult = gen.generate(graph=g, alpha_target=float(a), analysis_mode=None)

                # «Заморозим» базовые метрики (исп. текущее состояние g сразу после прогона)
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

                # Добавим метаданные и расширенные показатели
                base["graph_id"] = int(gid)
                base["n_nodes"] = int(g.graph.number_of_nodes())

                # Доля внутрикластерных среди удалённых
                base["internal_removed_ratio"] = compute_internal_removal_ratio(
                    getattr(res, "removal_events", None)
                )

                # Средняя доля общих рёбер между снимками (раз в 50 итераций)
                base["mean_overlap_ratio"] = compute_overlap_ratio_mean(
                    getattr(res, "edge_mask_history", None)
                )

                records.append(base)

        return records
