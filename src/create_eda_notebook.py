"""
Script to create the formal Jupyter Notebook: notebooks/01_eda_and_feature_engineering.ipynb
Generates standard Jupyter Notebook format (v4) with explanatory markdown, code, and visualizations.
"""

from pathlib import Path
import json

NOTEBOOKS_DIR = Path(r"a:\Customer Churn Prediction\notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "01_eda_and_feature_engineering.ipynb"


def build_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# FreshBasket Retail Loyalty Churn Prediction\n",
                "## Phase 3: Exploratory Data Analysis & Leakage-Safe Feature Engineering\n",
                "\n",
                "**Business Objective:** Identify at-risk loyalty members before they churn, diagnose behavioural and support drivers, and prepare a leakage-free customer-level dataset for predictive modelling.\n",
                "\n",
                "### Key Rules Enforced:\n",
                "- **Prediction Target:** `CHURNED` (1 = zero purchases during April–June 2024 despite prior history).\n",
                "- **Leakage Guard:** All behavioural, monetary, engagement, and support features are computed strictly using data up to **March 31, 2024**.\n",
                "- **Population:** Exactly 2,600 unique loyalty members."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Set root directory\n",
                "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.insert(0, str(project_root))\n",
                "\n",
                "from src.config import PROCESSED_DATA_DIR, FINAL_DATA_DIR, FIGURES_DIR\n",
                "from src.feature_engineering import load_processed_tables, engineer_customer_features, save_modeling_dataset\n",
                "from src.generate_eda_charts import generate_all_eda_charts\n",
                "\n",
                "print('Environment initialized successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Processed Clean Data\n",
                "We load the verified clean tables generated in Phase 2 from `data/processed/`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_profile, df_activity, df_churn = load_processed_tables()\n",
                "\n",
                "print(f'Customer Profile: {df_profile.shape}')\n",
                "print(f'Monthly Activity: {df_activity.shape}')\n",
                "print(f'Churn Label:      {df_churn.shape}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Churn Target Distribution & Class Imbalance Analysis\n",
                "We analyze the ground truth churn outcome across all 2,600 unique members."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "churn_counts = df_churn['CHURNED'].value_counts()\n",
                "churn_pct = df_churn['CHURNED'].value_counts(normalize=True) * 100\n",
                "\n",
                "summary_churn = pd.DataFrame({'Count': churn_counts, 'Percentage': churn_pct.round(2)})\n",
                "print(summary_churn)\n",
                "\n",
                "# Visualize distribution\n",
                "plt.figure(figsize=(6, 4))\n",
                "bars = plt.bar(['Active (0)', 'Churned (1)'], churn_counts, color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)\n",
                "plt.title('Overall Churn Class Balance (2,600 Members)', fontweight='bold')\n",
                "plt.ylabel('Customer Count')\n",
                "for bar in bars:\n",
                "    h = bar.get_height()\n",
                "    plt.annotate(f'{h:,} ({h/len(df_churn)*100:.1f}%)', xy=(bar.get_x() + bar.get_width() / 2, h),\n",
                "                 xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Segment Analysis: Churn by Tier, City, and Signup Cohort\n",
                "We examine whether static member demographics explain churn risk."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_base = df_profile.merge(df_churn, on='CUSTOMER_ID')\n",
                "\n",
                "# By Tier\n",
                "tier_churn = df_base.groupby('MEMBERSHIP_TIER')['CHURNED'].agg(['count', 'mean'])\n",
                "tier_churn['churn_pct'] = (tier_churn['mean'] * 100).round(2)\n",
                "print('--- Churn by Tier ---')\n",
                "print(tier_churn)\n",
                "\n",
                "# By City\n",
                "city_churn = df_base.groupby('CITY')['CHURNED'].agg(['count', 'mean'])\n",
                "city_churn['churn_pct'] = (city_churn['mean'] * 100).round(2)\n",
                "print('\\n--- Churn by City ---')\n",
                "print(city_churn.sort_values(by='churn_pct', ascending=False))\n",
                "\n",
                "# By Signup Year\n",
                "df_base['SIGNUP_YEAR'] = df_base['SIGNUP_DATE'].dt.year\n",
                "year_churn = df_base.groupby('SIGNUP_YEAR')['CHURNED'].agg(['count', 'mean'])\n",
                "year_churn['churn_pct'] = (year_churn['mean'] * 100).round(2)\n",
                "print('\\n--- Churn by Signup Year ---')\n",
                "print(year_churn)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Leakage-Safe Feature Engineering\n",
                "We execute our leakage-safe feature engineering pipeline strictly on activity through **March 31, 2024**.\n",
                "- **Recency:** Months since last shopping transaction relative to cutoff.\n",
                "- **Velocity / Trends:** Recent 3 months (Jan-Mar 2024) vs prior 3 months (Oct-Dec 2023) spend, transactions, and app sessions.\n",
                "- **Digital Engagement:** App session volume and marketing email opens.\n",
                "- **Support Friction:** Support tickets and formal complaints."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_model, audit_info = engineer_customer_features(df_profile, df_activity, df_churn)\n",
                "print(f'Final Customer Modeling Table Shape: {df_model.shape}')\n",
                "print(f'Total Null Values: {df_model.isna().sum().sum()}')\n",
                "print(f'Customer IDs Unique: {df_model[\"CUSTOMER_ID\"].is_unique}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Key Driver & Correlation Analysis\n",
                "We examine the linear correlation between engineered behavioral drivers and the churn target."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "num_cols = df_model.select_dtypes(include=[np.number]).columns.drop(['CUSTOMER_ID', 'CHURNED', 'PREFERRED_STORE_ID'])\n",
                "corrs = df_model[num_cols].apply(lambda s: s.corr(df_model['CHURNED'])).sort_values()\n",
                "\n",
                "print('--- Top 10 Protective Factors (Negative Correlation with Churn) ---')\n",
                "print(corrs.head(10))\n",
                "\n",
                "print('\\n--- Top 10 Churn Risk Factors (Positive Correlation with Churn) ---')\n",
                "print(corrs.tail(10))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Save Final Modeling Dataset\n",
                "We persist the final modeling dataset to `data/final/customer_churn_modeling_dataset.parquet` and `.csv`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "parquet_path, csv_path = save_modeling_dataset(df_model)\n",
                "print(f'Saved Parquet dataset: {parquet_path}')\n",
                "print(f'Saved CSV dataset:     {csv_path}')"
            ]
        }
    ]

    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.7"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
        
    print(f"Created notebook at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
