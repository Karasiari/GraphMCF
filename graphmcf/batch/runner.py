from typing import Iterable, List, Union, Optional
import numpy as np
from ..core import GraphMCF
from ..demands import MCFGenerator, DemandsGenerationResult

class GraphMCFBatch:
    """
    Мини-утилита: хранит выборку графов и прогоняет один/несколько генераторов по ней.
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
    ) -> List[DemandsGenerationResult]:
        gen = generator or MCFGenerator(**gen_kwargs)
        results: List[DemandsGenerationResult] = []
        for g in self.graphs:
            for a in alpha_values:
                res = gen.generate(graph=g, alpha_target=a, analysis_mode="overall")
                results.append(res)
        return results
