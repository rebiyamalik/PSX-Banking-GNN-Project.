"""Chronological validation and regression metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error

def expanding_month_splits(months,min_train_months=12):
    ordered=sorted(pd.to_datetime(pd.Series(months).dropna().unique()))
    if len(ordered)<=min_train_months: raise ValueError("Not enough months for initial training window")
    for i in range(min_train_months,len(ordered)): yield ordered[:i],ordered[i]

def regression_metrics(actual,predicted):
    a=np.asarray(actual,float); p=np.asarray(predicted,float); ok=np.isfinite(a)&np.isfinite(p)
    if not ok.any(): raise ValueError("No finite actual/predicted pairs")
    return {"mae":float(mean_absolute_error(a[ok],p[ok])),"rmse":float(np.sqrt(mean_squared_error(a[ok],p[ok]))),"n":int(ok.sum())}
