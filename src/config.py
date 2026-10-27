"""
Configuration and constants for the FreshBasket Customer Churn Prediction project.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DATA_DIR = PROJECT_ROOT / "data" / "final"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

# Excel Sheet Names
SHEET_PROFILE = "fb Customer Profile"
SHEET_ACTIVITY = "fb Monthly Activity"
SHEET_CHURN = "fb Churn Label"
SHEET_GLOSSARY = "Glossary"
SHEET_DATA_DICT = "Data Dictionary"

# Observation & Leakage Control Dates
OBSERVATION_START_DATE = "2023-01-01"
FEATURE_CUTOFF_DATE = "2024-03-31"  # Leakage boundary: only data <= 2024-03-31 used for predictive features
LABEL_WINDOW_START = "2024-04-01"   # Churn outcome window start
LABEL_WINDOW_END = "2024-06-30"     # Churn outcome window end

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Target Variable
TARGET_COL = "CHURNED"
ID_COL = "CUSTOMER_ID"
