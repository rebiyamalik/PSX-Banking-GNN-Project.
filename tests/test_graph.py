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


def test_future_data_does_not_change_past_graph():
    dates = pd.to_datetime([
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
        "2024-04-01",
    ])

    returns = pd.DataFrame({
        "UBL": [0.01, 0.02, -0.01, 0.50],
        "HBL": [0.02, 0.01, -0.02, -0.50],
    }, index=dates)

    original = rolling_correlation_matrix(
        returns, "2024-03-01", window=2
    )

    returns.loc["2024-04-01"] = [-0.90, 0.90]

    updated = rolling_correlation_matrix(
        returns, "2024-03-01", window=2
    )

    pd.testing.assert_frame_equal(original, updated)
