import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, 
    balanced_accuracy_score, brier_score_loss, roc_auc_score, average_precision_score
)
from constants import CRITICAL, HIGH, MEDIUM

def generate_result_df(y_true, scores):
    """Centralizes band calculation so it is identical across metrics and plots."""
    df = pd.DataFrame({"true_label": y_true, "score": scores})
    
    def get_band(s):
        if s >= CRITICAL: return "Critical"
        elif s >= HIGH: return "High"
        elif s >= MEDIUM: return "Medium"
        else: return "Low"
        
    df["band"] = df["score"].apply(get_band)
    return df

def calculate_ml_metrics(y_true, scores, threshold=MEDIUM):
    # 🔥 FIX: ensure valid probability range to prevent Brier Score crashes on baseline outliers
    scores = np.clip(scores, 0.0, 1.0)
    
    preds = (scores >= threshold).astype(int)
    
    metrics = {
        "Threshold_Used": threshold,
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1_Score": f1_score(y_true, preds, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, preds),
        "Brier_Score": brier_score_loss(y_true, scores),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores)
    }
    
    cm = confusion_matrix(y_true, preds)
    return metrics, cm

def band_class_distribution(df_results):
    """Generates the strong composition table showing risk bands vs actual labels."""
    result = df_results.groupby(["band", "true_label"]).size().unstack(fill_value=0)
    result = result.reindex(["Low", "Medium", "High", "Critical"]).fillna(0).astype(int)
    result.to_csv("outputs/metrics/band_class_distribution.csv")
    return result

def compare_baselines(y_true, hybrid_scores, lgbm_scores, if_scores, threshold=MEDIUM):
    models = {
        "Hybrid (Calibrated)": hybrid_scores,
        "LightGBM Only (Raw)": lgbm_scores,
        "IsolationForest Only (Raw)": if_scores
    }
    
    results = []
    for name, s in models.items():
        m, _ = calculate_ml_metrics(y_true, s, threshold)
        m["Model"] = name
        results.append(m)
        
    df = pd.DataFrame(results).set_index("Model")
    df.to_csv("outputs/metrics/baseline_comparison.csv")
    return df