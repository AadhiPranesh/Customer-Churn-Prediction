"""
Model Evaluation and Metrics Module for FreshBasket Customer Churn Prediction.
Computes PR-AUC, ROC-AUC, F1, Recall, Precision, Accuracy, Confusion Matrices,
and generates evaluation curves and comparison plots.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve
)

from src.config import FIGURES_DIR

METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"
CM_DIR = METRICS_DIR / "confusion_matrices"


def evaluate_model(
    model_name: str,
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluate a fitted model pipeline on test data using a specified decision threshold.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Predict probabilities
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X_test)[:, 1]
    else:
        probas = pipeline.predict(X_test).astype(float)
        
    preds = (probas >= threshold).astype(int)
    
    # Metrics
    pr_auc = float(average_precision_score(y_test, probas))
    
    # Guard ROC-AUC against constant probability outputs
    if len(np.unique(probas)) > 1:
        roc_auc = float(roc_auc_score(y_test, probas))
    else:
        roc_auc = 0.50

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Save individual confusion matrix CSV
    clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Active (0)", "Actual Churned (1)"],
        columns=["Pred Active (0)", "Pred Churned (1)"]
    )
    cm_df.to_csv(CM_DIR / f"{clean_name}_confusion_matrix.csv")
    
    return {
        "Model": model_name,
        "Threshold": round(threshold, 3),
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc_auc, 4),
        "F1-Score": round(f1, 4),
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "Accuracy": round(acc, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "probabilities": probas,
        "predictions": preds
    }


def compare_all_models(
    models_dict: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run evaluation across all models and export model_comparison.csv.
    """
    results_list = []
    eval_cache = {}
    
    for name, m_info in models_dict.items():
        pipe = m_info["pipeline"]
        thresh = m_info.get("optimal_threshold", 0.50)
        eval_res = evaluate_model(name, pipe, X_test, y_test, threshold=thresh)
        eval_cache[name] = eval_res
        
        # Row for DataFrame
        row = {k: v for k, v in eval_res.items() if k not in ["probabilities", "predictions"]}
        results_list.append(row)
        
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    
    # Export to CSV
    csv_path = METRICS_DIR / "model_comparison.csv"
    df_results.to_csv(csv_path, index=False)
    
    return df_results, eval_cache


def plot_model_comparison(df_results: pd.DataFrame) -> Path:
    """Generate side-by-side bar chart of model performance metrics."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    metrics = ["PR-AUC", "ROC-AUC", "F1-Score", "Recall", "Accuracy"]
    df_plot = df_results.set_index("Model")[metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.plot(kind="bar", ax=ax, width=0.8, colormap="viridis", edgecolor="black", alpha=0.9)
    ax.set_title("Model Benchmarking on Cohort Validation Set", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right", fontsize=11)
    ax.legend(title="Metric", loc="upper left", bbox_to_anchor=(1.01, 1))
    
    # Add horizontal baseline line for PR-AUC (prevalence = 0.1302)
    baseline_prauc = df_results[df_results["Model"] == "Majority Class Baseline"]["PR-AUC"].values[0]
    ax.axhline(baseline_prauc, color="red", linestyle="--", linewidth=1.2, label=f"Random Chance PR-AUC ({baseline_prauc:.2f})")
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_precision_recall_curves(eval_cache: Dict[str, Any], y_test: pd.Series) -> Path:
    """Generate publication-grade Precision-Recall curves comparing all models."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    colors = {
        "Majority Class Baseline": "#7f8c8d",
        "Heuristic Recency Rule": "#e67e22",
        "Logistic Regression": "#3498db",
        "Random Forest": "#27ae60",
        "HistGradientBoosting": "#8e44ad"
    }
    
    for name, res in eval_cache.items():
        probas = res["probabilities"]
        prec, rec, _ = precision_recall_curve(y_test, probas)
        ap = res["PR-AUC"]
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {ap:.3f})", color=colors.get(name, "#2c3e50"), linewidth=2.2)
        
    # Baseline chance
    prevalence = float(y_test.mean())
    ax.axhline(prevalence, color="black", linestyle="--", linewidth=1.2, label=f"No-Skill Baseline ({prevalence:.3f})")
    
    ax.set_title("Precision-Recall Curves Across Candidate Models", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "precision_recall_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path
