"""
Data cleaning and quality audit module for FreshBasket Customer Churn Prediction.
Performs deterministic, documented cleaning across all three tables.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import (
    FEATURE_CUTOFF_DATE,
    LABEL_WINDOW_START,
    LABEL_WINDOW_END,
    PROCESSED_DATA_DIR
)


def clean_customer_profile(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the customer profile table:
    1. Remove duplicate records.
    2. Normalize text casing and whitespace in CITY, MEMBERSHIP_TIER, GENDER.
    3. Impute missing HOME_STORE_PRICE_TIER deterministically from PREFERRED_STORE_ID.
    4. Impute missing AGE with median and add AGE_IS_MISSING indicator.
    5. Fill missing GENDER with 'Unknown' and add GENDER_IS_MISSING indicator.
    6. Convert SIGNUP_DATE to datetime.
    """
    audit = {}
    audit["raw_row_count"] = len(df_raw)
    audit["raw_unique_customers"] = df_raw["CUSTOMER_ID"].nunique()
    
    # Check duplicates
    dups_count = df_raw.duplicated().sum()
    audit["exact_duplicates_removed"] = int(dups_count)
    df = df_raw.drop_duplicates().copy()
    
    audit["post_dedup_row_count"] = len(df)
    
    # Audit raw missingness
    audit["missing_before"] = df.isna().sum().to_dict()
    
    # 1. Normalize text fields
    audit["raw_unique_cities"] = df["CITY"].nunique()
    df["CITY"] = df["CITY"].astype(str).str.strip().str.title()
    audit["clean_unique_cities"] = df["CITY"].nunique()
    
    audit["raw_tiers"] = df["MEMBERSHIP_TIER"].value_counts().to_dict()
    df["MEMBERSHIP_TIER"] = df["MEMBERSHIP_TIER"].astype(str).str.strip().str.title()
    audit["clean_tiers"] = df["MEMBERSHIP_TIER"].value_counts().to_dict()
    
    # Normalize GENDER
    gender_missing = df["GENDER"].isna() | (df["GENDER"].astype(str).str.strip() == "")
    audit["gender_missing_count"] = int(gender_missing.sum())
    df["GENDER_IS_MISSING"] = gender_missing.astype(int)
    df["GENDER"] = df["GENDER"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["GENDER"] == "", "GENDER"] = "Unknown"
    
    # 2. Impute HOME_STORE_PRICE_TIER deterministically from PREFERRED_STORE_ID
    price_tier_missing = df["HOME_STORE_PRICE_TIER"].isna() | (df["HOME_STORE_PRICE_TIER"].astype(str).str.strip() == "")
    audit["price_tier_missing_count"] = int(price_tier_missing.sum())
    
    store_to_tier = (
        df[~price_tier_missing]
        .groupby("PREFERRED_STORE_ID")["HOME_STORE_PRICE_TIER"]
        .first()
        .to_dict()
    )
    df["HOME_STORE_PRICE_TIER"] = df["HOME_STORE_PRICE_TIER"].fillna(
        df["PREFERRED_STORE_ID"].map(store_to_tier)
    )
    df["HOME_STORE_PRICE_TIER"] = df["HOME_STORE_PRICE_TIER"].astype(str).str.strip().str.title()
    audit["price_tier_unresolved_after_mapping"] = int(df["HOME_STORE_PRICE_TIER"].isna().sum())
    
    # 3. Impute AGE with median and flag missingness
    age_missing = df["AGE"].isna()
    audit["age_missing_count"] = int(age_missing.sum())
    median_age = float(df["AGE"].median())
    audit["median_age_imputed"] = median_age
    df["AGE_IS_MISSING"] = age_missing.astype(int)
    df["AGE"] = df["AGE"].fillna(median_age).astype(int)
    
    # 4. Parse Dates
    df["SIGNUP_DATE"] = pd.to_datetime(df["SIGNUP_DATE"])
    
    # Final missingness
    audit["missing_after"] = df.isna().sum().to_dict()
    
    return df, audit


def clean_monthly_activity(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the monthly activity table:
    1. Investigate and fix negative spend (data entry sign error; abs(TOTAL_SPEND)).
    2. Remove duplicate (CUSTOMER_ID, MONTH) pairs.
    3. Reconstruct missing TOTAL_SPEND when TRANSACTIONS > 0 but TOTAL_SPEND == 0.
    4. Preserve inactive zero-transaction months (vital churn signal).
    5. Convert MONTH to datetime.
    6. Add IS_PREDICTIVE_WINDOW flag (MONTH <= 2024-03-31) to prevent leakage.
    """
    audit = {}
    audit["raw_row_count"] = len(df_raw)
    audit["raw_unique_customers"] = df_raw["CUSTOMER_ID"].nunique()
    
    df = df_raw.copy()
    
    # 1. Investigate and fix negative spend
    neg_spend_mask = df["TOTAL_SPEND"] < 0
    audit["negative_spend_count"] = int(neg_spend_mask.sum())
    audit["negative_spend_samples"] = df.loc[neg_spend_mask, ["CUSTOMER_ID", "TOTAL_SPEND", "AVG_BASKET_VALUE", "TRANSACTIONS"]].head(5).to_dict(orient="records")
    
    # Take absolute value to correct data-entry sign error
    df["TOTAL_SPEND"] = df["TOTAL_SPEND"].abs()
    
    # 2. Deduplication on (CUSTOMER_ID, MONTH)
    dups_subset = df.duplicated(subset=["CUSTOMER_ID", "MONTH"], keep="first")
    audit["activity_duplicates_removed"] = int(dups_subset.sum())
    df = df.drop_duplicates(subset=["CUSTOMER_ID", "MONTH"], keep="first").copy()
    audit["post_dedup_row_count"] = len(df)
    
    # 3. Handle zero spend with positive transactions
    zero_spend_pos_txns = (df["TOTAL_SPEND"] == 0) & (df["TRANSACTIONS"] > 0) & (df["AVG_BASKET_VALUE"] > 0)
    audit["zero_spend_positive_txns_count"] = int(zero_spend_pos_txns.sum())
    # Reconstruct true spend
    df.loc[zero_spend_pos_txns, "TOTAL_SPEND"] = (
        df.loc[zero_spend_pos_txns, "TRANSACTIONS"] * df.loc[zero_spend_pos_txns, "AVG_BASKET_VALUE"]
    ).round(2)
    
    # 4. Audit pure zero activity months (preserved)
    zero_both = (df["TOTAL_SPEND"] == 0) & (df["TRANSACTIONS"] == 0)
    audit["zero_spend_zero_txns_preserved_count"] = int(zero_both.sum())
    
    # 5. Date conversion
    df["MONTH"] = pd.to_datetime(df["MONTH"])
    
    # 6. Leakage Guard Flag
    cutoff_dt = pd.to_datetime(FEATURE_CUTOFF_DATE)
    df["IS_PREDICTIVE_WINDOW"] = df["MONTH"] <= cutoff_dt
    audit["records_in_predictive_window"] = int(df["IS_PREDICTIVE_WINDOW"].sum())
    audit["records_in_label_window"] = int((~df["IS_PREDICTIVE_WINDOW"]).sum())
    
    return df, audit


def clean_churn_label(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the churn label table:
    1. Remove duplicate customer rows.
    2. Convert observation dates to datetime.
    3. Document and flag missing LAST_PURCHASE_DATE (customers who never purchased).
    4. Audit target distribution and insufficient history flag.
    """
    audit = {}
    audit["raw_row_count"] = len(df_raw)
    audit["raw_unique_customers"] = df_raw["CUSTOMER_ID"].nunique()
    
    # Deduplication
    dups = df_raw.duplicated().sum()
    audit["exact_duplicates_removed"] = int(dups)
    df = df_raw.drop_duplicates().copy()
    audit["post_dedup_row_count"] = len(df)
    
    # Dates
    df["OBSERVATION_END_DATE"] = pd.to_datetime(df["OBSERVATION_END_DATE"])
    df["LAST_PURCHASE_DATE"] = pd.to_datetime(df["LAST_PURCHASE_DATE"])
    
    # Missing last purchase date
    missing_lp = df["LAST_PURCHASE_DATE"].isna()
    audit["missing_last_purchase_date_count"] = int(missing_lp.sum())
    df["NEVER_PURCHASED_FLAG"] = missing_lp.astype(int)
    
    # Target distribution
    audit["churn_distribution"] = df["CHURNED"].value_counts().to_dict()
    audit["churn_rate_pct"] = float((df["CHURNED"].mean()) * 100)
    
    # Insufficient history flag
    audit["insufficient_history_distribution"] = df["INSUFFICIENT_HISTORY_FLAG"].value_counts().to_dict()
    
    return df, audit


def run_data_validation_checks(
    df_profile: pd.DataFrame,
    df_activity: pd.DataFrame,
    df_churn: pd.DataFrame
) -> Dict[str, bool]:
    """
    Perform sanity checks and assertions to guarantee data cleanliness and integrity.
    """
    checks = {}
    
    # 1. Profile uniqueness
    checks["profile_customer_id_unique"] = bool(df_profile["CUSTOMER_ID"].is_unique)
    checks["profile_no_null_customer_id"] = bool(df_profile["CUSTOMER_ID"].isna().sum() == 0)
    checks["profile_no_null_price_tier"] = bool(df_profile["HOME_STORE_PRICE_TIER"].isna().sum() == 0)
    checks["profile_cities_standardized_to_10"] = bool(df_profile["CITY"].nunique() == 10)
    checks["profile_tiers_standardized_to_3"] = bool(set(df_profile["MEMBERSHIP_TIER"].unique()) == {"Silver", "Gold", "Platinum"})
    
    # 2. Activity checks
    checks["activity_no_duplicate_customer_month"] = bool(not df_activity.duplicated(subset=["CUSTOMER_ID", "MONTH"]).any())
    checks["activity_no_negative_spend"] = bool((df_activity["TOTAL_SPEND"] < 0).sum() == 0)
    checks["activity_all_months_18"] = bool(df_activity["MONTH"].nunique() == 18)
    checks["activity_predictive_window_flag_present"] = bool("IS_PREDICTIVE_WINDOW" in df_activity.columns)
    
    # 3. Churn label checks
    checks["churn_customer_id_unique"] = bool(df_churn["CUSTOMER_ID"].is_unique)
    checks["churn_target_binary_only"] = bool(set(df_churn["CHURNED"].unique()).issubset({0, 1}))
    checks["churn_has_2600_customers"] = bool(len(df_churn) == 2600)
    
    # 4. Cross-table consistency
    checks["profile_and_churn_identical_customers"] = bool(
        set(df_profile["CUSTOMER_ID"]) == set(df_churn["CUSTOMER_ID"])
    )
    checks["all_activity_customers_in_profile"] = bool(
        set(df_activity["CUSTOMER_ID"]).issubset(set(df_profile["CUSTOMER_ID"]))
    )
    
    all_passed = all(checks.values())
    checks["ALL_CHECKS_PASSED"] = all_passed
    return checks
