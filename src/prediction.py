"""
Customer Prediction Module for FreshBasket Customer Churn Prediction.
Generates operational customer-level churn probabilities, risk buckets,
and individualized interpretable risk drivers using the champion model.
"""

from pathlib import Path
from typing import Tuple
import pickle
import numpy as np
import pandas as pd

from src.config import FINAL_DATA_DIR, PREDICTIONS_DIR, ID_COL
from src.model_training import get_feature_columns

MODEL_PATH = Path(__file__).resolve().parent.parent / "outputs" / "models" / "random_forest.pkl"
PREDICTIONS_FILE = PREDICTIONS_DIR / "customer_churn_predictions.csv"


def assign_risk_drivers(row: pd.Series) -> Tuple[str, str]:
    """
    Determine primary and secondary interpretable risk drivers for a customer
    based on their individual behavioral metrics.
    """
    drivers = []
    
    # 1. Dormancy / Recency
    if row.get("recency_months", 0) >= 3:
        drivers.append("High Inactivity (No purchase in 90+ days)")
    elif row.get("purchased_in_last_3m_flag", 1) == 0:
        drivers.append("No Purchases in Recent Quarter (Q1 2024)")
        
    # 2. Spend Velocity Drop
    if row.get("spend_trend_3m_vs_prior", 1.0) < 0.40:
        drivers.append("Severe Spend Drop (>60% reduction vs prior)")
    elif row.get("txns_trend_3m_vs_prior", 1.0) < 0.40:
        drivers.append("Transaction Frequency Contraction")
        
    # 3. Support Friction & Complaints
    if row.get("had_recent_complaint_flag", 0) == 1:
        drivers.append("Recent Formal Support Complaint")
    elif row.get("total_support_tickets_15m", 0) >= 3:
        drivers.append("Recurring Customer Service Friction (3+ tickets)")
        
    # 4. Digital Engagement Deceleration
    if row.get("total_app_sessions_l3m", 0) == 0 and row.get("total_app_sessions_15m", 0) > 0:
        drivers.append("Mobile App Abandonment (0 sessions in Q1 2024)")
    elif row.get("app_trend_3m_vs_prior", 1.0) < 0.50:
        drivers.append("Declining Mobile App Logins")
        
    # 5. Marketing Disengagement
    if row.get("total_emails_opened_l3m", 0) == 0 and row.get("total_emails_opened_15m", 0) > 0:
        drivers.append("Marketing Email Disengagement (0 opens in Q1 2024)")
        
    # 6. Low Promo / Coupon Usage
    if row.get("total_coupons_l3m", 0) == 0 and row.get("total_coupons_15m", 0) > 0:
        drivers.append("Coupon Redemption Cessation")
        
    # 7. Insufficient History Fallback
    if row.get("INSUFFICIENT_HISTORY_FLAG", 0) == 1:
        drivers.append("Sparse Baseline History (<4 months tenure)")
        
    # Default fallbacks
    if not drivers:
        drivers.append("Stable Activity Pattern")
    if len(drivers) < 2:
        drivers.append("Routine Account Maintenance")
        
    return drivers[0], drivers[1]


def generate_customer_predictions(
    threshold: float = 0.330
) -> pd.DataFrame:
    """
    Score all 2,600 loyalty customers using the champion Random Forest pipeline,
    assign risk tiers and human-interpretable drivers, and save predictions.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data and model
    dataset_path = FINAL_DATA_DIR / "customer_churn_modeling_dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Final modeling dataset not found: {dataset_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Champion model pipeline not found: {MODEL_PATH}")
        
    df = pd.read_parquet(dataset_path)
    all_feature_cols, _, _ = get_feature_columns(df)
    
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
        
    # 2. Score probabilities
    probabilities = pipeline.predict_proba(df[all_feature_cols])[:, 1]
    
    # 3. Build operational prediction table
    df_pred = pd.DataFrame()
    df_pred[ID_COL] = df[ID_COL]
    df_pred["CHURN_PROBABILITY"] = probabilities.round(4)
    df_pred["PREDICTED_CHURN_LABEL"] = (df_pred["CHURN_PROBABILITY"] >= threshold).astype(int)
    
    # Risk Buckets
    conditions = [
        df_pred["CHURN_PROBABILITY"] < 0.30,
        (df_pred["CHURN_PROBABILITY"] >= 0.30) & (df_pred["CHURN_PROBABILITY"] < 0.60),
        df_pred["CHURN_PROBABILITY"] >= 0.60
    ]
    choices = ["Low Risk", "Medium Risk", "High Risk"]
    df_pred["RISK_BUCKET"] = np.select(conditions, choices, default="Low Risk")
    
    # 4. Interpretability: Assign primary and secondary drivers
    primary_drivers = []
    secondary_drivers = []
    for _, row in df.iterrows():
        p, s = assign_risk_drivers(row)
        primary_drivers.append(p)
        secondary_drivers.append(s)
        
    df_pred["PRIMARY_RISK_DRIVER"] = primary_drivers
    df_pred["SECONDARY_RISK_DRIVER"] = secondary_drivers
    
    # 5. Metadata & Operational Attributes
    df_pred["MEMBERSHIP_TIER"] = df["MEMBERSHIP_TIER"]
    df_pred["CITY"] = df["CITY"]
    df_pred["TENURE_MONTHS"] = df["customer_tenure_months"]
    df_pred["INSUFFICIENT_HISTORY_FLAG"] = df["INSUFFICIENT_HISTORY_FLAG"]
    df_pred["PREDICTION_RELIABILITY"] = np.where(
        df["INSUFFICIENT_HISTORY_FLAG"] == 1,
        "Cautious (Tenure < 4 Months)",
        "Standard (Mature History)"
    )
    
    # Strictly verify NO ground truth target label in output
    assert "CHURNED" not in df_pred.columns, "Target leakage! CHURNED must not appear in operational predictions."
    assert len(df_pred) == 2600, f"Expected 2600 rows, got {len(df_pred)}"
    assert df_pred[ID_COL].nunique() == 2600, "Duplicate customer IDs detected in prediction file."
    assert (df_pred["CHURN_PROBABILITY"] >= 0.0).all() and (df_pred["CHURN_PROBABILITY"] <= 1.0).all(), "Invalid probabilities!"
    
    # Sort by risk descending
    df_pred = df_pred.sort_values(by="CHURN_PROBABILITY", ascending=False).reset_index(drop=True)
    
    # Save CSV
    df_pred.to_csv(PREDICTIONS_FILE, index=False)
    
    return df_pred
