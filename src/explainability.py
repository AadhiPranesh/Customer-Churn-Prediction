"""
Explainability and Business Insights Module for FreshBasket Customer Churn Prediction.
Calculates tree feature importances, permutation importance on held-out data,
and compares predicted churn risks across demographic and cohort segments.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from src.config import FIGURES_DIR, TARGET_COL

METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"


def compute_feature_importances(
    rf_pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute both MDI (tree feature importance) and Permutation Feature Importance on test data.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Permutation Importance
    perm = permutation_importance(
        rf_pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="average_precision",
        n_jobs=1
    )
    
    df_perm = pd.DataFrame({
        "Feature": feature_names,
        "Permutation_Importance_Mean": perm.importances_mean,
        "Permutation_Importance_Std": perm.importances_std
    }).sort_values(by="Permutation_Importance_Mean", ascending=False).reset_index(drop=True)
    
    # 2. Tree-based Feature Importance (MDI)
    # Extract transformed feature names from ColumnTransformer
    prep = rf_pipeline.named_steps["prep"]
    clf = rf_pipeline.named_steps["clf"]
    
    try:
        transformed_names = prep.get_feature_names_out()
    except Exception:
        transformed_names = [f"feat_{i}" for i in range(len(clf.feature_importances_))]
        
    df_tree = pd.DataFrame({
        "Transformed_Feature": transformed_names,
        "Tree_Importance": clf.feature_importances_
    }).sort_values(by="Tree_Importance", ascending=False).reset_index(drop=True)

    df_perm.to_csv(METRICS_DIR / "permutation_feature_importance.csv", index=False)
    df_tree.to_csv(METRICS_DIR / "tree_feature_importance.csv", index=False)

    return df_perm, df_tree


def plot_feature_importance(df_perm: pd.DataFrame, top_n: int = 15) -> Path:
    """Generate high-resolution chart of top permutation feature importances."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    df_top = df_perm.head(top_n).sort_values(by="Permutation_Importance_Mean", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        df_top["Feature"],
        df_top["Permutation_Importance_Mean"],
        xerr=df_top["Permutation_Importance_Std"],
        color="#2980b9",
        edgecolor="black",
        alpha=0.85,
        capsize=4
    )
    
    ax.set_title(f"Top {top_n} Predictive Churn Drivers\n(Permutation Feature Importance on Held-Out Validation Cohort)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in Precision-Recall AUC (PR-AUC)", fontsize=11)
    ax.set_ylabel("Engineered Feature", fontsize=11)
    
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.annotate(f"{w:.4f}",
                        xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(8, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9, fontweight="bold")
                        
    plt.tight_layout()
    out_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def analyze_segment_risks(
    df_full: pd.DataFrame,
    rf_pipeline: Any,
    feature_cols: list
) -> pd.DataFrame:
    """
    Generate customer-level predicted churn probabilities and compare actual vs predicted
    churn risk across membership tiers, regional cities, signup cohorts, and history flags.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    df_eval = df_full.copy()
    df_eval["PREDICTED_CHURN_PROB"] = rf_pipeline.predict_proba(df_eval[feature_cols])[:, 1]
    
    records = []
    
    def add_segment_summary(dimension: str, group_col: str):
        grouped = df_eval.groupby(group_col).agg(
            Customer_Count=("CUSTOMER_ID", "count"),
            Actual_Churn_Rate=(TARGET_COL, "mean"),
            Predicted_Churn_Risk=("PREDICTED_CHURN_PROB", "mean")
        ).reset_index()
        for _, row in grouped.iterrows():
            records.append({
                "Segment_Dimension": dimension,
                "Segment_Value": str(row[group_col]),
                "Customer_Count": int(row["Customer_Count"]),
                "Actual_Churn_Rate": round(float(row["Actual_Churn_Rate"]) * 100, 2),
                "Predicted_Churn_Risk": round(float(row["Predicted_Churn_Risk"]) * 100, 2),
                "Risk_Calibration_Diff": round((float(row["Predicted_Churn_Risk"]) - float(row["Actual_Churn_Rate"])) * 100, 2)
            })

    # Tiers
    add_segment_summary("Membership Tier", "MEMBERSHIP_TIER")
    # Cities
    add_segment_summary("City Market", "CITY")
    # Cohorts
    add_segment_summary("Signup Year", "signup_cohort_year")
    # History Flag
    add_segment_summary("Insufficient History Flag", "INSUFFICIENT_HISTORY_FLAG")

    df_segments = pd.DataFrame(records)
    csv_path = METRICS_DIR / "segment_risk_analysis.csv"
    df_segments.to_csv(csv_path, index=False)

    return df_segments
