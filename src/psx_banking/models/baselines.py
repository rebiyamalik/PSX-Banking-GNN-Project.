"""Simple baselines to evaluate before deep learning."""
from sklearn.ensemble import RandomForestRegressor

def persistence_prediction(training_values):
    values=training_values.dropna()
    if values.empty: raise ValueError("No training values available")
    return float(values.iloc[-1])

def make_random_forest(random_state=42):
    return RandomForestRegressor(n_estimators=300,max_depth=3,min_samples_leaf=3,
        max_features=0.8,random_state=random_state,n_jobs=-1)
