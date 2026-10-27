"""
Data loader module for FreshBasket Customer Churn Prediction.
Loads raw sheets from Excel while handling the decorative banner row safely.
"""

from pathlib import Path
import pandas as pd
from typing import Dict, Tuple
from src.config import (
    RAW_DATA_PATH,
    SHEET_PROFILE,
    SHEET_ACTIVITY,
    SHEET_CHURN,
    SHEET_GLOSSARY,
    SHEET_DATA_DICT
)


def load_raw_table(sheet_name: str, skiprows: int = 1) -> pd.DataFrame:
    """
    Safely load a specific worksheet from the FreshBasket Excel workbook.
    
    Parameters
    ----------
    sheet_name : str
        Name of worksheet to load.
    skiprows : int, default=1
        Number of rows to skip. Row 1 is a decorative banner; Row 2 has headers.
        
    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    if not Path(RAW_DATA_PATH).exists():
        raise FileNotFoundError(f"Raw dataset not found at: {RAW_DATA_PATH}")
        
    df = pd.read_excel(
        RAW_DATA_PATH,
        sheet_name=sheet_name,
        skiprows=skiprows,
        engine="openpyxl"
    )
    return df


def load_all_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three primary modeling tables: Customer Profile, Monthly Activity, and Churn Label.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        df_profile, df_activity, df_churn
    """
    df_profile = load_raw_table(SHEET_PROFILE, skiprows=1)
    df_activity = load_raw_table(SHEET_ACTIVITY, skiprows=1)
    df_churn = load_raw_table(SHEET_CHURN, skiprows=1)
    return df_profile, df_activity, df_churn


def load_metadata_tables() -> Dict[str, pd.DataFrame]:
    """
    Load Glossary and Data Dictionary reference sheets.
    """
    df_glossary = pd.read_excel(RAW_DATA_PATH, sheet_name=SHEET_GLOSSARY, header=None, engine="openpyxl")
    df_dict = pd.read_excel(RAW_DATA_PATH, sheet_name=SHEET_DATA_DICT, skiprows=1, engine="openpyxl")
    return {
        "glossary": df_glossary,
        "data_dictionary": df_dict
    }
