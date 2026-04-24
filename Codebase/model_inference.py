import joblib
import pandas as pd
import numpy as np

def load_model(filepath="model.joblib"):
    return joblib.load(filepath)

def get_all_predictions(model, X: pd.DataFrame):
    """Extracts predictions for hybrid model and isolated baselines."""
    # Hybrid Model Output (Calibrated)
    hybrid_scores = model.predict_risk(X)
    if isinstance(hybrid_scores, (pd.DataFrame, pd.Series)):
        hybrid_scores = hybrid_scores.values.flatten()
        
    # Standardize input for baselines
    X_scaled = model.scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Baseline: LightGBM Only (Raw)
    lgbm_scores = model.lgbm.predict_proba(X_scaled_df)[:, 1]
    
    # Baseline: Isolation Forest Only (Raw)
    if_raw = -model.iforest.score_samples(X_scaled_df)
    if_scores = model.if_scaler.transform(if_raw.reshape(-1, 1)).flatten()
    
    return hybrid_scores, lgbm_scores, if_scores