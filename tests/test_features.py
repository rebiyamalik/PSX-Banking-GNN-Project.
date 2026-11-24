import pandas as pd
from psx_banking.features import daily_to_monthly,add_next_month_targets

def sample():
    return pd.DataFrame({"date":pd.to_datetime(["2024-01-02","2024-01-03","2024-02-01","2024-02-02","2024-03-01","2024-03-04"]),
      "open":[10,11,12,13,14,15],"high":[11,12,13,14,15,16],"low":[9,10,11,12,13,14],
      "close":[10,11,12,13,14,15],"volume":[100,110,120,130,140,150],"symbol":["UBL"]*6})

def test_monthly_aggregation():
    m=daily_to_monthly(sample())
    assert len(m)==3
    assert m.month.dt.strftime("%Y-%m").tolist()==["2024-01","2024-02","2024-03"]

def test_next_month_targets():
    x=add_next_month_targets(daily_to_monthly(sample()))
    assert pd.notna(x.loc[0,"target_next_month_return"])
    assert pd.isna(x.loc[2,"target_next_month_return"])
