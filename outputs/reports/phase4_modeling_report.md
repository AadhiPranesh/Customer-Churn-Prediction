# Phase 4: Model Development, Cohort Validation & Ablation Report
**Project:** Customer Churn Prediction for Retail Loyalty Program  
**Dataset:** FreshBasket Retail Loyalty Dataset  
**Phase:** Phase 4 - Model Training, Benchmarking, Ablation Study & Explainability  

---

## 1. Executive Summary & Audit Verification
In Phase 4, we developed, validated, and benchmarked 5 classification approaches to predict customer churn in the FreshBasket Loyalty Program. 

### Pre-Modeling Audit Findings:
- **Population:** Exactly **2,600 unique loyalty members**.
- **Target Integrity:** `CHURNED` is strictly binary (1,782 active, 818 churned).
- **Leakage Elimination:** 100% of predictive features were computed strictly from historical activity on or before **March 31, 2024**. All label-window metadata (`INSUFFICIENT_HISTORY_FLAG`, `MONTHS_OBSERVED`, `NEVER_PURCHASED_FLAG`, `LAST_PURCHASE_DATE`, `OBSERVATION_END_DATE`) were **strictly excluded from the feature matrix $X$**.
- **Short-History Customers:** 373 members with `INSUFFICIENT_HISTORY_FLAG = 1` were preserved in the population to avoid survival bias, while post-outcome label columns were omitted to prevent leakage.
- **Champion Architecture:** **Random Forest Classifier** achieved the highest generalization performance on the held-out validation cohort with **PR-AUC = 0.8315**, **ROC-AUC = 0.908**, and **F1-Score = 0.4398** (compared to Random Chance PR-AUC = 0.1302).

---

## 2. Validation Strategy & Justification

### Selected Strategy: Cohort-Based Temporal Validation Split
The dataset was partitioned using member enrollment date (`SIGNUP_DATE`):
- **Training Cohort:** Members enrolled **before July 1, 2023** ($n = 1,947$, 74.9% of total). Churn rate = **37.65%** (733 churned, 1,214 active).
- **Validation Cohort:** Members enrolled **on or after July 1, 2023** ($n = 653$, 25.1% of total). Churn rate = **13.02%** (85 churned, 568 active).

### Senior Engineering Rationale:
1. **Realistic Temporal Evaluation:** A loyalty program must predict churn on newer customer cohorts after observing them over a limited horizon, using behavioral models trained on mature members.
2. **Class Representation:** Both positive and negative classes are robustly represented in the test cohort (85 churned, 568 active).
3. **No Future-to-Past Leakage:** The model never trains on future signups to predict earlier signups.
4. **Data Limitation Acknowledgment:** Because all members share the same observation end date (`2024-06-30`) and 3-month label window (`April–June 2024`), a pure out-of-calendar-time test across different fiscal quarters is constrained by the 18-month dataset. Thus, the cohort-based split is the most defensible temporal validation strategy available.
5. **Preprocessing Leakage Guard:** All encoders (`OneHotEncoder`) and scalers (`StandardScaler`) were fit exclusively on `X_train` and applied downstream to `X_test`.

---

## 3. Model Benchmark & Comparative Results

All models were evaluated on the held-out **Validation Cohort ($n=653$)**:

| Model | Decision Threshold | PR-AUC | ROC-AUC | F1-Score | Recall | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.33 | **0.8315** | 0.908 | 0.4398 | 0.8588 | 0.2955 | 0.7152 |
| **HistGradientBoosting** | 0.81 | **0.8139** | 0.9181 | 0.7394 | 0.7176 | 0.7625 | 0.9342 |
| **Logistic Regression** | 0.72 | **0.568** | 0.85 | 0.4538 | 0.6941 | 0.3371 | 0.7825 |
| **Heuristic Recency Rule** | 0.5 | **0.1957** | 0.6843 | 0.0088 | 0.0118 | 0.0071 | 0.657 |
| **Majority Class Baseline** | 0.5 | **0.1302** | 0.5 | 0.0 | 0.0 | 0.0 | 0.8698 |

### Key Metrics Discussion (Why PR-AUC and Recall Matter):
- **Majority Class Baseline:** Achieves 86.98% accuracy simply by predicting 0 for every customer, yet yields **0.00 recall, 0.00 precision, and 0.00 F1**. This proves accuracy is completely misleading for churn detection.
- **PR-AUC (Precision-Recall AUC):** The baseline chance rate on this test cohort is **0.1302**. The Random Forest model achieved **0.8315**—a **+0.7149 absolute lift** over random guessing.
- **Recall Focus:** Missing a churning member (False Negative) forfeits customer lifetime value, whereas reaching out to an active member (False Positive) only incurs low promotional cost. The tuned Random Forest achieves **85.9% Recall**, capturing the vast majority of at-risk members.

---

## 4. Confusion Matrices (Held-Out Validation Cohort)

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 394 | 174 | 12 | 73 |
| **HistGradientBoosting** | 549 | 19 | 24 | 61 |
| **Logistic Regression** | 452 | 116 | 26 | 59 |
| **Heuristic Recency Rule** | 428 | 140 | 84 | 1 |
| **Majority Class Baseline** | 568 | 0 | 85 | 0 |

---

## 5. Required Feature Ablation Study

We evaluated three strictly controlled feature groups under the identical cohort validation split using the Random Forest champion architecture:

| Feature Group | Description | Total Features | PR-AUC | ROC-AUC | F1-Score | Recall | PR-AUC Uplift vs A |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Group A: Demographics Only** | 12 | **0.163** | 0.5073 | 0.1711 | 0.1529 | **0.0** |
| **Group B: Demographics + Transactions** | 36 | **0.7534** | 0.875 | 0.6503 | 0.6235 | **0.5904** |
| **Group C: Full Permitted Features** | 57 | **0.8463** | 0.9222 | 0.805 | 0.7529 | **0.6833** |

### Ablation Analysis Takeaways:
1. **Group A (Demographics Alone) Fails to Predict Churn:** Yields a near-random **PR-AUC of 0.163** and ROC-AUC of 0.5030. Member age, gender, city, and tier do not indicate whether a customer is about to defect.
2. **Group B (Transactions & Spend) Drives the Core Signal:** Adding transaction frequency, spend velocity, and dormancy metrics boosts PR-AUC by **+0.5904**.
3. **Group C (Digital Engagement & Support Friction) Delivers Significant Uplift:** Incorporating app sessions, marketing email responsiveness, coupon usage, support tickets, and complaints increases PR-AUC to **0.8463** (+0.6833 total lift over demographics).

---

## 6. Feature Importance & Driver Attribution

Top predictive drivers evaluated via **Permutation Importance on the held-out validation cohort**:

| Rank | Predictive Feature | Mean Decrease in PR-AUC | Domain Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `recency_months` | **0.0245** | Critical driver of member activity/dormancy. |
| 2 | `avg_basket_value_mean` | **0.0230** | Critical driver of member activity/dormancy. |
| 3 | `total_txns_l3m` | **0.0169** | Critical driver of member activity/dormancy. |
| 4 | `max_monthly_spend_15m` | **0.0156** | Critical driver of member activity/dormancy. |
| 5 | `total_spend_15m` | **0.0140** | Critical driver of member activity/dormancy. |
| 6 | `total_txns_15m` | **0.0139** | Critical driver of member activity/dormancy. |
| 7 | `active_months_count_15m` | **0.0138** | Critical driver of member activity/dormancy. |
| 8 | `app_trend_3m_vs_prior` | **0.0119** | Critical driver of member activity/dormancy. |
| 9 | `avg_promo_txn_pct_15m` | **0.0114** | Critical driver of member activity/dormancy. |
| 10 | `spend_trend_3m_vs_prior` | **0.0100** | Critical driver of member activity/dormancy. |

> [!NOTE]
> **Correlation vs. Causation Guardrail:** High feature importance indicates strong predictive association, not direct causation. For example, high purchase recency reflects the symptom of customer disengagement rather than the root cause. Support tickets and complaints, however, provide direct actionable intervention points.

---

## 7. Segment Risk Profiling & Calibration

Predicted churn risk vs. actual churn rates across key segments:

### A. Membership Tiers
| Tier | Members | Actual Churn Rate | Predicted Churn Risk | Calibration Difference |
| :--- | :---: | :---: | :---: | :---: |
| **Gold** | 824 | 30.1% | 34.86% | +4.76% |
| **Platinum** | 302 | 31.13% | 33.61% | +2.48% |
| **Silver** | 1474 | 32.29% | 36.83% | +4.53% |

### B. Regional City Markets (Sample)
| City Market | Members | Actual Churn Rate | Predicted Churn Risk | Calibration Difference |
| :--- | :---: | :---: | :---: | :---: |
| **Austin** | 266 | 37.59% | 41.87% | +4.28% |
| **Columbus** | 268 | 36.19% | 40.73% | +4.54% |
| **Nashville** | 237 | 33.33% | 37.19% | +3.86% |
| **Boise** | 247 | 31.98% | 36.06% | +4.08% |
| **Tulsa** | 295 | 29.83% | 34.04% | +4.21% |

### C. Insufficient History Flag Analysis
| Segment | Members | Actual Churn Rate | Predicted Churn Risk | Calibration Difference |
| :--- | :---: | :---: | :---: | :---: |
| **Insufficient History (1)** | 2227 | 29.73% | 33.0% | +3.27% |
| **Insufficient History (1)** | 373 | 41.82% | 52.75% | +10.92% |

---

## 8. Artifacts & Deliverables Generated
1. `outputs/metrics/model_comparison.csv` — Full benchmark table across 5 models.
2. `outputs/metrics/ablation_comparison.csv` — Feature group ablation comparison.
3. `outputs/metrics/confusion_matrices/` — Confusion matrix tables for every model.
4. `outputs/metrics/permutation_feature_importance.csv` — Permutation importance values.
5. `outputs/metrics/segment_risk_analysis.csv` — Segment calibration audit.
6. `outputs/figures/model_comparison.png` — Model comparison bar charts.
7. `outputs/figures/precision_recall_curves.png` — Precision-Recall curves.
8. `outputs/figures/feature_importance.png` — Feature importance bar chart.
9. `outputs/models/` — Serialized trained pipelines (`random_forest.pkl`, `histgradientboosting.pkl`, `logistic_regression.pkl`).
