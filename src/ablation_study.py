"""
Ablation Study Module for FreshBasket Customer Churn Prediction.
Quantifies the incremental value of behavioral, transaction, engagement,
and support friction features over static demographics.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation import evaluate_model
from src.config import RANDOM_STATE

METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"


def run_ablation_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Execute feature ablation across:
    Group A: Demographics and membership only.
    Group B: Demographics + historical transactions & spend.
    Group C: Full feature set (+ digital engagement & support friction).
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    cat_cols = ["GENDER", "CITY", "MEMBERSHIP_TIER", "HOME_STORE_PRICE_TIER", "PREFERRED_CATEGORY"]
    
    # Group A: Demographics & Membership Only
    group_a_num = [
        "AGE",
        "AGE_IS_MISSING",
        "GENDER_IS_MISSING",
        "MARKETING_OPT_IN",
        "PREFERRED_STORE_ID",
        "customer_tenure_months",
        "signup_cohort_year"
    ]
    group_a_cols = cat_cols + group_a_num

    # Group B: Demographics + Transactions & Monetary Activity
    group_b_num = group_a_num + [
        "total_spend_15m",
        "avg_monthly_spend_15m",
        "std_monthly_spend_15m",
        "max_monthly_spend_15m",
        "total_txns_15m",
        "avg_monthly_txns_15m",
        "active_months_count_15m",
        "zero_trans_months_count_15m",
        "avg_basket_value_mean",
        "avg_basket_value_max",
        "avg_distinct_categories",
        "max_distinct_categories",
        "total_spend_l3m",
        "total_txns_l3m",
        "active_months_l3m",
        "total_spend_l6m",
        "total_txns_l6m",
        "recency_months",
        "spend_trend_3m_vs_prior",
        "txns_trend_3m_vs_prior",
        "purchased_in_last_3m_flag",
        "active_months_ratio",
        "months_in_activity_window",
        "zero_activity_in_window_flag"
    ]
    group_b_cols = cat_cols + group_b_num

    # Group C: Full Permitted Feature Set (+ Engagement & Support)
    group_c_num = group_b_num + [
        "avg_promo_txn_pct_15m",
        "total_coupons_15m",
        "total_coupons_l3m",
        "coupon_redemption_rate",
        "total_app_sessions_15m",
        "avg_monthly_app_sessions_15m",
        "total_app_sessions_l3m",
        "total_app_sessions_l6m",
        "app_trend_3m_vs_prior",
        "total_emails_opened_15m",
        "avg_monthly_emails_opened_15m",
        "total_emails_opened_l3m",
        "total_emails_opened_l6m",
        "total_support_tickets_15m",
        "total_support_tickets_l3m",
        "total_support_tickets_l6m",
        "support_ticket_trend_l3m",
        "total_complaints_15m",
        "total_complaints_l3m",
        "total_complaints_l6m",
        "had_recent_complaint_flag"
    ]
    group_c_cols = cat_cols + group_c_num

    feature_groups = {
        "Group A: Demographics Only": (group_a_cols, group_a_num),
        "Group B: Demographics + Transactions": (group_b_cols, group_b_num),
        "Group C: Full Permitted Features": (group_c_cols, group_c_num)
    }

    y_train = train_df["CHURNED"]
    y_test = test_df["CHURNED"]

    ablation_results = []
    
    for group_name, (cols, num_c) in feature_groups.items():
        # Build Pipeline
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_c)
        ])
        
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=9,
            min_samples_split=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        )
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(train_df[cols], y_train)
        
        # Evaluate
        eval_metrics = evaluate_model(group_name, pipe, test_df[cols], y_test, threshold=0.50)
        
        row = {
            "Feature Group": group_name,
            "Total Features": len(cols),
            "PR-AUC": eval_metrics["PR-AUC"],
            "ROC-AUC": eval_metrics["ROC-AUC"],
            "F1-Score": eval_metrics["F1-Score"],
            "Recall": eval_metrics["Recall"],
            "Precision": eval_metrics["Precision"],
            "Accuracy": eval_metrics["Accuracy"]
        }
        ablation_results.append(row)

    df_ablation = pd.DataFrame(ablation_results)
    
    # Calculate Uplift vs Group A
    base_prauc = df_ablation.loc[df_ablation["Feature Group"] == "Group A: Demographics Only", "PR-AUC"].values[0]
    base_f1 = df_ablation.loc[df_ablation["Feature Group"] == "Group A: Demographics Only", "F1-Score"].values[0]
    
    df_ablation["PR-AUC Uplift vs A"] = (df_ablation["PR-AUC"] - base_prauc).round(4)
    df_ablation["F1 Uplift vs A"] = (df_ablation["F1-Score"] - base_f1).round(4)

    # Export to CSV
    csv_path = METRICS_DIR / "ablation_comparison.csv"
    df_ablation.to_csv(csv_path, index=False)

    return df_ablation
