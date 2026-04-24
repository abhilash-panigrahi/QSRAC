import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, f1_score
from sklearn.calibration import calibration_curve
from constants import CRITICAL, HIGH, MEDIUM

BANDS = {"Critical": CRITICAL, "High": HIGH, "Medium": MEDIUM, "Low": 0.0}
COLORS = {"Critical": "#d73027", "High": "#fc8d59", "Medium": "#fee090", "Low": "#91bfdb"}

def apply_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (8, 5), 'figure.dpi': 300})

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.ylabel('True Label')
    plt.xlabel(f'Predicted Label (Threshold ≥ {MEDIUM:.4f})')
    plt.title('Confusion Matrix (Hybrid Calibrated)')
    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.close()

def plot_pr_and_roc(y_true, scores, pr_auc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    p, r, _ = precision_recall_curve(y_true, scores)
    ax1.plot(r, p, color='blue', lw=2, label="Hybrid (Calibrated)")
    ax1.set_title(f"Precision-Recall Curve (AUC={pr_auc:.3f})")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.legend()
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label="Hybrid (Calibrated)")
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("outputs/plots/pr_roc_curves.png")
    plt.close()

def plot_risk_distribution(scores):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(scores, fill=True, color="gray", alpha=0.5, bw_adjust=0.5)
    
    for band, thresh in BANDS.items():
        if band != "Low":
            plt.axvline(x=thresh, color=COLORS[band], linestyle='--', lw=2, label=f"{band} (≥{thresh:.4f})")
            
    plt.title("Calibrated Risk Score Distribution (Hybrid Calibrated)")
    plt.xlabel("Risk Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/risk_distribution_kde.png")
    plt.close()

def plot_band_distribution(scores):
    categories = []
    for s in scores:
        if s >= CRITICAL: categories.append("Critical")
        elif s >= HIGH: categories.append("High")
        elif s >= MEDIUM: categories.append("Medium")
        else: categories.append("Low")
        
    counts = pd.Series(categories).value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=[COLORS[k] for k in counts.index])
    plt.title("Sample Count per Risk Band (Hybrid Calibrated)")
    plt.ylabel("Number of Requests")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')
        
    plt.tight_layout()
    plt.savefig("outputs/plots/band_distribution_bar.png")
    plt.close()

def plot_reliability_diagram(y_true, scores):
    prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', lw=2, label="Hybrid (Calibrated)")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfectly Calibrated")
    plt.title("Reliability Diagram (Calibration)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/reliability_diagram.png")
    plt.close()

def plot_f1_vs_threshold(y_true, scores):
    thresholds = np.linspace(0.1, 0.99, 100)
    f1_scores = [f1_score(y_true, (scores >= t).astype(int), zero_division=0) for t in thresholds]
    
    plt.figure()
    plt.plot(thresholds, f1_scores, lw=2, color='green', label="F1 Score")
    plt.axvline(x=MEDIUM, color='red', linestyle='--', label=f"Chosen Cutoff ({MEDIUM:.4f})")
    plt.title("F1 Score vs Threshold (Hybrid Calibrated)")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/f1_vs_threshold.png")
    plt.close()

def plot_cdf(scores):
    sorted_scores = np.sort(scores)
    yvals = np.arange(len(sorted_scores)) / float(len(sorted_scores))
    
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_scores, yvals, color='purple', lw=2)
    plt.axvline(x=MEDIUM, color=COLORS["Medium"], linestyle='--', label="Medium")
    plt.axvline(x=HIGH, color=COLORS["High"], linestyle='--', label="High")
    plt.axvline(x=CRITICAL, color=COLORS["Critical"], linestyle='--', label="Critical")
    
    plt.title("CDF of Calibrated Risk Scores (Hybrid Calibrated)")
    plt.xlabel("Risk Score")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/risk_cdf.png")
    plt.close()