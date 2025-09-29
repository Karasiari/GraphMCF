import numpy as np
from graphmcf.batch import GraphMCFBatch

mats = [
    np.array([[0,4,0,0],[4,0,2,0],[0,2,0,3],[0,0,3,0]], float),
    np.array([[0,1,1,0],[1,0,2,2],[1,2,0,2],[0,2,2,0]], float),
]
batch = GraphMCFBatch(mats)
results = batch.run_mcf_over(alpha_values=[0.3,0.5,0.7])
# свести в табличку:
import pandas as pd
df = pd.DataFrame([r.to_dict() for r in results])
print(df[["alpha_target","iterations_total","final_alpha","converged"]])
