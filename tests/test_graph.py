import numpy as np
import pandas as pd
from psx_banking.graph import rolling_correlation_matrix,correlation_adjacency

def test_graph_uses_only_prior_months():
    data=pd.DataFrame({"UBL":[.01,.02,-.01],"HBL":[.02,.01,-.02]},
      index=pd.to_datetime(["2024-01-01","2024-02-01","2024-03-01"]))
    assert rolling_correlation_matrix(data,"2024-03-01",window=2).shape==(2,2)

def test_adjacency_self_loops():
    corr=pd.DataFrame([[1,.8],[.8,1]],columns=["UBL","HBL"])
    assert np.all(np.diag(correlation_adjacency(corr,threshold=.5))==1)
