import os
import numpy as np
import pandas as pd

# Enforce global reproducibility
np.random.seed(42)

from data_loader import load_test_data
from model_inference import load_model, get_all_predictions
from metrics import calculate_ml_metrics, compare_baselines, generate_result_df, band_class_distribution
from plots import (apply_style, plot_confusion_matrix, plot_pr_and_roc, 
                   plot_risk_distribution, plot_band_distribution, 
                   plot_reliability_diagram, plot_f1_vs_threshold, plot_cdf)
from evaluation import evaluate_trust_dynamics, evaluate_trust_repair_trace, simulate_stateful_security, plot_policy_distribution
from benchmarks import benchmark_inference

def main():
    print("[*] Creating output directories...")
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    
    apply_style()

    print("[*] Loading Model and Data...")
    model = load_model("model.joblib")
    X, y = load_test_data("UNSW_NB15_testing-set.csv")
    
    print("[*] Running Inferences...")
    hybrid_scores, lgbm_scores, if_scores = get_all_predictions(model, X)
    
    print(f"[*] Risk Stats: Min={np.min(hybrid_scores):.4f}, Mean={np.mean(hybrid_scores):.4f}, Max={np.max(hybrid_scores):.4f}")
    
    print("[*] Generating Result DataFrame & Global Bands...")
    df_results = generate_result_df(y, hybrid_scores)
    df_results.to_csv("outputs/metrics/full_results.csv", index=False)
    
    print("\n[*] Band Distribution (Normalized):")
    band_order = ["Low", "Medium", "High", "Critical"]
    norm_counts = df_results["band"].value_counts(normalize=True).reindex(band_order, fill_value=0)
    print(norm_counts)
    
    abs_counts = df_results["band"].value_counts().reindex(band_order, fill_value=0)
    abs_counts.to_frame("count").to_csv("outputs/metrics/band_counts.csv")
    
    print("\n[*] Generating ML Metrics & Baselines...")
    metrics, cm = calculate_ml_metrics(y, hybrid_scores)
    pr_auc = metrics["PR_AUC"]
    
    print(f"[*] Core ML Evaluation -> PR-AUC: {pr_auc:.4f}")
    
    pd.DataFrame([metrics]).to_csv("outputs/metrics/core_ml_metrics.csv", index=False)
    band_class_distribution(df_results)
    compare_baselines(y, hybrid_scores, lgbm_scores, if_scores)
    
    print("[*] Generating Plots...")
    plot_confusion_matrix(cm)
    plot_pr_and_roc(y, hybrid_scores, pr_auc)
    plot_risk_distribution(hybrid_scores)
    plot_band_distribution(hybrid_scores)
    plot_reliability_diagram(y, hybrid_scores)
    plot_f1_vs_threshold(y, hybrid_scores)
    plot_cdf(hybrid_scores)
    
    print("[*] Simulating Trust, Policy, and Stateful Logic...")
    plot_policy_distribution(df_results)
    evaluate_trust_dynamics()
    evaluate_trust_repair_trace()
    simulate_stateful_security()
    
    print("[*] Running Benchmarks...")
    benchmark_inference(model, X)
    
    print("\n[+] All evaluation artifacts generated successfully in ./outputs/")

if __name__ == "__main__":
    main()