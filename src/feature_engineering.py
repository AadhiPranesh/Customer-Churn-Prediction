"""
Feature Engineering Module for FreshBasket Customer Churn Prediction.
Strictly enforces leakage-free feature engineering using historical monthly activity
up to March 31, 2024 only.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

from src.config import (
    FEATURE_CUTOFF_DATE,
    PROCESSED_DATA_DIR,
    FINAL_DATA_DIR,
    TARGET_COL,
    ID_COL
)


def load_processed_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cleaned tables from data/processed/."""
    profile_path = PROCESSED_DATA_DIR / "cleaned_customer_profile.parquet"
    activity_path = PROCESSED_DATA_DIR / "cleaned_monthly_activity.parquet"
    churn_path = PROCESSED_DATA_DIR / "cleaned_churn_label.parquet"
    
    if not profile_path.exists() or not activity_path.exists() or not churn_path.exists():
        raise FileNotFoundError("Processed datasets missing. Run Phase 2 data cleaning first!")
        
    df_profile = pd.read_parquet(profile_path)
    df_activity = pd.read_parquet(activity_path)
    df_churn = pd.read_parquet(churn_path)
    
    return df_profile, df_activity, df_churn


def engineer_customer_features(
    df_profile: pd.DataFrame,
    df_activity: pd.DataFrame,
    df_churn: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Engineer leakage-safe customer-level features:
    1. Filter activity strictly <= FEATURE_CUTOFF_DATE (2024-03-31).
    2. Compute recency, frequency, spend trends, digital engagement, and support friction.
    3. Merge with Customer Profile and Churn Labels.
    4. Handle edge cases (e.g. customers with 0 activity in window).
    5. Validate dataset integrity.
    """
    audit = {}
    cutoff_dt = pd.to_datetime(FEATURE_CUTOFF_DATE)
    audit["feature_cutoff_date"] = str(cutoff_dt.date())
    
    # 1. Filter activity strictly to predictive window
    act_pred = df_activity[df_activity["MONTH"] <= cutoff_dt].copy()
    act_pred = act_pred.sort_values([ID_COL, "MONTH"])
    audit["predictive_activity_rows"] = len(act_pred)
    audit["predictive_activity_months"] = sorted(act_pred["MONTH"].dt.strftime("%Y-%m").unique())
    audit["total_predictive_months"] = len(audit["predictive_activity_months"])
    
    # Sub-windows for velocity and trends
    # Last 3 months: Jan 2024 - Mar 2024
    m_l3m_start = pd.to_datetime("2024-01-01")
    # Last 6 months: Oct 2023 - Mar 2024
    m_l6m_start = pd.to_datetime("2023-10-01")
    
    act_l3m = act_pred[act_pred["MONTH"] >= m_l3m_start]
    act_l6m = act_pred[act_pred["MONTH"] >= m_l6m_start]
    
    # 2. Aggregations across entire 15-month predictive window
    agg_15m = act_pred.groupby(ID_COL).agg(
        total_spend_15m=("TOTAL_SPEND", "sum"),
        avg_monthly_spend_15m=("TOTAL_SPEND", "mean"),
        std_monthly_spend_15m=("TOTAL_SPEND", "std"),
        max_monthly_spend_15m=("TOTAL_SPEND", "max"),
        total_txns_15m=("TRANSACTIONS", "sum"),
        avg_monthly_txns_15m=("TRANSACTIONS", "mean"),
        active_months_count_15m=("TRANSACTIONS", lambda s: (s > 0).sum()),
        zero_trans_months_count_15m=("TRANSACTIONS", lambda s: (s == 0).sum()),
        avg_basket_value_mean=("AVG_BASKET_VALUE", "mean"),
        avg_basket_value_max=("AVG_BASKET_VALUE", "max"),
        avg_distinct_categories=("DISTINCT_CATEGORIES", "mean"),
        max_distinct_categories=("DISTINCT_CATEGORIES", "max"),
        avg_promo_txn_pct_15m=("PROMO_TXN_PCT", "mean"),
        total_coupons_15m=("COUPONS_REDEEMED", "sum"),
        total_app_sessions_15m=("APP_SESSIONS", "sum"),
        avg_monthly_app_sessions_15m=("APP_SESSIONS", "mean"),
        total_emails_opened_15m=("EMAILS_OPENED", "sum"),
        avg_monthly_emails_opened_15m=("EMAILS_OPENED", "mean"),
        total_support_tickets_15m=("SUPPORT_TICKETS", "sum"),
        total_complaints_15m=("COMPLAINT_FLAG", "sum"),
        months_in_activity_window=("MONTH", "count")
    ).reset_index()
    
    # 3. Aggregations for Recent Windows (Last 3 Months & Last 6 Months)
    agg_l3m = act_l3m.groupby(ID_COL).agg(
        total_spend_l3m=("TOTAL_SPEND", "sum"),
        total_txns_l3m=("TRANSACTIONS", "sum"),
        active_months_l3m=("TRANSACTIONS", lambda s: (s > 0).sum()),
        total_app_sessions_l3m=("APP_SESSIONS", "sum"),
        total_emails_opened_l3m=("EMAILS_OPENED", "sum"),
        total_coupons_l3m=("COUPONS_REDEEMED", "sum"),
        total_support_tickets_l3m=("SUPPORT_TICKETS", "sum"),
        total_complaints_l3m=("COMPLAINT_FLAG", "sum")
    ).reset_index()
    
    agg_l6m = act_l6m.groupby(ID_COL).agg(
        total_spend_l6m=("TOTAL_SPEND", "sum"),
        total_txns_l6m=("TRANSACTIONS", "sum"),
        total_app_sessions_l6m=("APP_SESSIONS", "sum"),
        total_emails_opened_l6m=("EMAILS_OPENED", "sum"),
        total_support_tickets_l6m=("SUPPORT_TICKETS", "sum"),
        total_complaints_l6m=("COMPLAINT_FLAG", "sum")
    ).reset_index()
    
    # 4. Purchase Recency
    active_months_only = act_pred[act_pred["TRANSACTIONS"] > 0]
    last_purchase_series = active_months_only.groupby(ID_COL)["MONTH"].max().reset_index()
    last_purchase_series.rename(columns={"MONTH": "last_active_month_dt"}, inplace=True)
    
    # Calculate months since last purchase relative to cutoff (2024-03-31)
    last_purchase_series["recency_months"] = (
        (cutoff_dt.year - last_purchase_series["last_active_month_dt"].dt.year) * 12 +
        (cutoff_dt.month - last_purchase_series["last_active_month_dt"].dt.month)
    )
    
    # Merge activity aggregations
    df_act_feats = agg_15m.merge(agg_l3m, on=ID_COL, how="left")
    df_act_feats = df_act_feats.merge(agg_l6m, on=ID_COL, how="left")
    df_act_feats = df_act_feats.merge(last_purchase_series[[ID_COL, "recency_months"]], on=ID_COL, how="left")
    
    # Impute NAs for customers missing L3M/L6M activity rows
    l3m_fill_cols = [c for c in agg_l3m.columns if c != ID_COL]
    l6m_fill_cols = [c for c in agg_l6m.columns if c != ID_COL]
    df_act_feats[l3m_fill_cols] = df_act_feats[l3m_fill_cols].fillna(0)
    df_act_feats[l6m_fill_cols] = df_act_feats[l6m_fill_cols].fillna(0)
    df_act_feats["std_monthly_spend_15m"] = df_act_feats["std_monthly_spend_15m"].fillna(0)
    
    # Customers with no purchases in window receive sentinel recency (15 months = window length)
    df_act_feats["recency_months"] = df_act_feats["recency_months"].fillna(15.0)
    
    # 5. Trends & Ratios
    # Velocity: L3M vs Prior 3M (Oct-Dec 2023)
    prior_3m_spend = (df_act_feats["total_spend_l6m"] - df_act_feats["total_spend_l3m"]).clip(lower=0) / 3.0
    recent_3m_spend = df_act_feats["total_spend_l3m"] / 3.0
    df_act_feats["spend_trend_3m_vs_prior"] = recent_3m_spend / (prior_3m_spend + 1.0)
    
    prior_3m_txns = (df_act_feats["total_txns_l6m"] - df_act_feats["total_txns_l3m"]).clip(lower=0) / 3.0
    recent_3m_txns = df_act_feats["total_txns_l3m"] / 3.0
    df_act_feats["txns_trend_3m_vs_prior"] = recent_3m_txns / (prior_3m_txns + 0.1)
    
    prior_3m_app = (df_act_feats["total_app_sessions_l6m"] - df_act_feats["total_app_sessions_l3m"]).clip(lower=0) / 3.0
    recent_3m_app = df_act_feats["total_app_sessions_l3m"] / 3.0
    df_act_feats["app_trend_3m_vs_prior"] = recent_3m_app / (prior_3m_app + 0.1)
    
    prior_3m_support = (df_act_feats["total_support_tickets_l6m"] - df_act_feats["total_support_tickets_l3m"]).clip(lower=0)
    df_act_feats["support_ticket_trend_l3m"] = df_act_feats["total_support_tickets_l3m"] - prior_3m_support
    
    # Engagement ratios & flags
    df_act_feats["purchased_in_last_3m_flag"] = (df_act_feats["active_months_l3m"] > 0).astype(int)
    df_act_feats["had_recent_complaint_flag"] = (df_act_feats["total_complaints_l3m"] > 0).astype(int)
    df_act_feats["coupon_redemption_rate"] = df_act_feats["total_coupons_15m"] / (df_act_feats["total_txns_15m"] + 1e-4)
    df_act_feats["active_months_ratio"] = df_act_feats["active_months_count_15m"] / df_act_feats["months_in_activity_window"]
    
    # 6. Join with Profile and Churn Labels
    df_final = df_profile.merge(df_act_feats, on=ID_COL, how="left")
    
    # Include churn labels and history flags
    churn_cols = [ID_COL, "INSUFFICIENT_HISTORY_FLAG", "MONTHS_OBSERVED", "NEVER_PURCHASED_FLAG", TARGET_COL]
    df_final = df_final.merge(df_churn[churn_cols], on=ID_COL, how="left")
    
    # 7. Handle Customers with No Activity Records in Predictive Window (e.g. newly signed up or zero-record)
    no_act_mask = df_final["total_txns_15m"].isna()
    audit["customers_without_activity_records"] = int(no_act_mask.sum())
    
    # Add explicit indicator
    df_final["zero_activity_in_window_flag"] = no_act_mask.astype(int)
    
    # Impute numeric activity features with 0 and recency with 15
    num_act_cols = df_act_feats.select_dtypes(include=[np.number]).columns.drop(ID_COL)
    for col in num_act_cols:
        df_final[col] = df_final[col].fillna(0)
    df_final.loc[no_act_mask, "recency_months"] = 15.0
    
    # 8. Customer Tenure & Cohort Encodings
    df_final["customer_tenure_months"] = (
        (cutoff_dt.year - df_final["SIGNUP_DATE"].dt.year) * 12 +
        (cutoff_dt.month - df_final["SIGNUP_DATE"].dt.month)
    )
    df_final["signup_cohort_year"] = df_final["SIGNUP_DATE"].dt.year
    df_final["signup_cohort_quarter"] = df_final["SIGNUP_DATE"].dt.to_period("Q").astype(str)
    
    # 9. Final Validation
    audit["total_final_customers"] = len(df_final)
    audit["unique_customers"] = df_final[ID_COL].nunique()
    audit["total_columns"] = df_final.shape[1]
    audit["null_values_total"] = int(df_final.isna().sum().sum())
    audit["target_distribution"] = df_final[TARGET_COL].value_counts().to_dict()
    
    # Sanity checks
    assert audit["total_final_customers"] == 2600, f"Expected 2600 customers, got {len(df_final)}"
    assert audit["unique_customers"] == 2600, "Duplicate customer IDs detected!"
    assert audit["null_values_total"] == 0, f"Detected {audit['null_values_total']} null values in final dataset!"
    assert set(df_final[TARGET_COL].unique()) == {0, 1}, "Target column must be binary {0, 1}!"
    
    return df_final, audit


def save_modeling_dataset(df_final: pd.DataFrame) -> Tuple[Path, Path]:
    """Save the final modeling dataset to Parquet and CSV in data/final/."""
    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    parquet_path = FINAL_DATA_DIR / "customer_churn_modeling_dataset.parquet"
    csv_path = FINAL_DATA_DIR / "customer_churn_modeling_dataset.csv"
    
    df_final.to_parquet(parquet_path, index=False)
    df_final.to_csv(csv_path, index=False)
    
    return parquet_path, csv_path
