# FreshBasket Loyalty Churn Prediction

A machine learning solution for predicting and diagnosing customer churn within the FreshBasket Retail loyalty membership program (Silver, Gold, and Platinum tiers).

---

## 1. Project Overview & Business Objective
FreshBasket Retail operates a customer loyalty program across 10 metropolitan markets and 40 retail store locations. Customer retention is critical: blanket promotional mailings are costly and inefficient. 

The objective of this project is to:
1. Predict individual customer churn risk for the next observation period.
2. Uncover the behavioural, digital engagement, and customer service drivers of churn.
3. Compare churn risk across member tiers, cities, and signup cohorts.
4. Simulate retention intervention scenarios (e.g. proactive outreach, targeted coupons).

---

## 2. Dataset Architecture & Leakage Boundary
The dataset spans 18 months of customer activity (January 2023 – June 2024) across 2,600 loyalty members:
- **`fb Customer Profile`**: Member demographics, signup date, preferred store and category.
- **`fb Monthly Activity`**: Monthly time series (spend, transaction frequency, app sessions, marketing emails, support tickets, complaints).
- **`fb Churn Label`**: Ground truth churn outcome evaluated as of `2024-06-30`.

> [!IMPORTANT]
> **Churn Definition & Leakage Guard:**  
> A member is labeled **`CHURNED = 1`** if they recorded **zero purchase transactions in the final 3 months (April–June 2024)** despite prior purchase history.  
> To strictly prevent data leakage, **all predictive features are computed solely from data on or before March 31, 2024**. Activity from April–June 2024 is strictly isolated as the label observation window.

---

## 3. Project Structure
```text
Customer Churn Prediction/
├── data/
│   ├── raw/                  # FreshBasket_Loyalty_Churn_Dataset.xlsx (untouched original)
│   ├── processed/            # Cleaned tables (Parquet & CSV for Profile, Activity, Churn Label)
│   └── final/                # Single merged customer-level modeling dataset (2,600 x 64)
├── notebooks/
│   └── 01_eda_and_feature_engineering.ipynb  # Interactive EDA & feature validation notebook
├── src/                      # Modular production-grade Python package
│   ├── __init__.py
│   ├── config.py             # Paths, feature cutoff dates (2024-03-31), seed=42
│   ├── data_loader.py        # Safe Excel reader skipping decorative banner rows
│   ├── data_cleaning.py      # Cleaning logic, text normalization, anomaly resolution
│   ├── feature_engineering.py # 57 leakage-free features across 15 historical months
│   ├── generate_eda_charts.py# Publication-quality exploratory visualization scripts
│   ├── create_eda_notebook.py# Programmatic Jupyter notebook generator
│   ├── model_training.py     # 5-fold CV threshold optimization & model pipelines
│   ├── evaluation.py         # PR-AUC, ROC-AUC, F1, Precision, Recall, Confusion Matrices
│   ├── ablation_study.py     # 3-group feature ablation evaluation
│   ├── explainability.py     # Permutation importance & segment risk calibration
│   ├── prediction.py         # Operational inference, risk bucketing & risk driver extraction
│   └── retention_simulation.py # Financial ROI models, sensitivity matrices & charts
├── outputs/
│   ├── figures/              # 13 publication-quality visualizations (PNG)
│   ├── metrics/              # Benchmark comparisons, ablation results, confusion matrices
│   ├── models/               # Serialized scikit-learn models (joblib/pickle)
│   ├── predictions/          # customer_churn_predictions.csv (operational scoring)
│   └── reports/              # Comprehensive technical reports & presentation deliverables
│       ├── final_assessment_report.md
│       ├── presentation_deck_and_speaker_notes.md
│       ├── final_quality_checklist.md
│       ├── data_quality_report.md
│       ├── phase3_eda_and_feature_report.md
│       ├── phase4_modeling_report.md
│       └── phase5_retention_and_predictions_report.md
├── run_cleaning.py           # Phase 2 pipeline runner
├── run_phase3.py             # Phase 3 feature engineering & EDA runner
├── run_phase4.py             # Phase 4 modeling & ablation study runner
├── run_phase5.py             # Phase 5 predictions & retention simulation runner
├── requirements.txt          # Python dependencies
└── README.md                 # Complete project documentation
```

---

## 4. Setup & Execution

### Prerequisites
- Python 3.10+ (tested on Python 3.13.7 64-bit Windows)

### Installation
```bash
pip install -r requirements.txt
```

### Reproducing All Phases
Each phase can be executed independently using its verified pipeline runner:

```bash
# Phase 2: Data Cleaning & Quality Audit
python run_cleaning.py

# Phase 3: Leakage-Safe Feature Engineering & EDA
python run_phase3.py

# Phase 4: Model Training, Cohort Benchmarking & Ablation Study
python run_phase4.py

# Phase 5: Operational Predictions & Retention Intervention Simulation
python run_phase5.py
```

---

## 5. Key Deliverables & Documentation

1. **Final Assessment Technical Report:**  
   [`outputs/reports/final_assessment_report.md`](file:///a:/Customer%20Churn%20Prediction/outputs/reports/final_assessment_report.md)  
   Comprehensive 13-section report detailing problem framing, data quality audits, EDA discoveries, leakage guards, model benchmarks, feature ablation, probability calibration, retention economics, and A/B testing protocols.

2. **Executive Presentation Deck & Speaker Notes:**  
   [`outputs/reports/presentation_deck_and_speaker_notes.md`](file:///a:/Customer%20Churn%20Prediction/outputs/reports/presentation_deck_and_speaker_notes.md)  
   12 professional slides formatted for leadership review, equipped with verbatim 12–15 minute speaker notes and defensive Q&A preparation.

3. **Final Quality & Compliance Checklist:**  
   [`outputs/reports/final_quality_checklist.md`](file:///a:/Customer%20Churn%20Prediction/outputs/reports/final_quality_checklist.md)  
   16-point compliance verification matrix confirming zero target leakage, reproducible seeds, and reconciled metrics.

4. **Customer Predictions & Risk Drivers:**  
   [`outputs/predictions/customer_churn_predictions.csv`](file:///a:/Customer%20Churn%20Prediction/outputs/predictions/customer_churn_predictions.csv)  
   Operational scoring for all 2,600 members with calibrated probabilities, threshold-optimized risk buckets, and top 2 actionable risk drivers.
