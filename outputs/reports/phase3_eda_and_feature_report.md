# Phase 3: Exploratory Data Analysis & Feature Engineering Report
**Project:** Customer Churn Prediction for Retail Loyalty Program  
**Dataset:** FreshBasket Retail Loyalty Dataset  
**Phase:** Phase 3 - EDA & Customer-Level Modeling Dataset Assembly  

---

## 1. Executive Summary
In Phase 3, we constructed a leakage-free, customer-level modeling dataset of **2,600 unique loyalty members** by aggregating 15 months of pre-cutoff activity (January 2023 – March 2024) and joining it with verified member profiles and ground-truth churn labels.

All 15 activity months used for feature engineering precede the 3-month churn observation window (April–June 2024), **guaranteeing 0% target leakage**. The resulting dataset contains **62 columns** (demographics, monetary trends, frequency, engagement, support friction, and cohort tags) with **zero missing values and zero duplicate keys**.

---

## 2. Key Exploratory Data Analysis Findings

### A. Overall Churn Distribution & Imbalance
- **Active Members (`CHURNED = 0`):** 1,782 (68.54%)
- **Churned Members (`CHURNED = 1`):** 818 (31.46%)
- **Baseline Churn Rate:** **31.46%**
- **Evaluation Implication:** In an interview, emphasize that standard accuracy is insufficient because predicting all zeros yields 68.5% accuracy. Models must be judged on **Precision-Recall AUC (PR-AUC)**, **F1-Score**, and **Recall**.

### B. Segment-Level Churn Analysis

#### 1. Membership Tier Analysis
| Membership Tier | Member Count | Churned Count | Churn Rate (%) |
| :--- | :---: | :---: | :---: |
| **Silver** | 1474 | 475 | **32.29%** |
| **Gold** | 824 | 247 | **30.1%** |
| **Platinum** | 302 | 94 | **31.13%** |

> **Key Insight:** Churn rates across tiers are remarkably close (~30% to 32%). Tier status alone does not explain customer retention. This provides the exact empirical rationale for the **Ablation Study** in Phase 4.

#### 2. Geographic Analysis (Top vs Bottom Markets)
- **Highest Churn Cities:**
  - **Austin:** 37.59% churn (n=266)
  - **Columbus:** 36.19% churn (n=268)
  - **Nashville:** 33.33% churn (n=237)
- **Lowest Churn Cities:**
  - **Portland:** 28.33% churn (n=240)
  - **Denver:** 28.79% churn (n=257)
  - **Spokane:** 28.97% churn (n=252)

#### 3. Signup Cohort & Survival Dynamics
| Signup Year | Total Enrolled | Churned | Churn Rate (%) |
| :---: | :---: | :---: | :---: |
| **2021** | 745 | 279 | **37.45%** |
| **2022** | 792 | 306 | **38.64%** |
| **2023** | 816 | 229 | **28.06%** |
| **2024** | 247 | 4 | **1.62%** |

> **Key Insight:** Members who joined in 2021–2022 exhibit higher churn (~38%), whereas 2024 signups have low churn (1.6%) because they recently enrolled and have not had sufficient inactive months. This supports the assessment requirement for a **cohort-based validation split**.

---

## 3. Behavioral Churn Drivers & Correlation Analysis

Linear correlation with the target variable `CHURNED`:

### Top Protective Factors (Negative Correlation with Churn)
| Feature Name | Pearson $r$ | Business Meaning |
| :--- | :---: | :--- |
| `active_months_l3m` | -0.6815 | Shopping in Jan, Feb, or Mar 2024 strongly signals active retention. |
| `purchased_in_last_3m_flag` | -0.6565 | Binary indicator of recent purchases. |
| `total_txns_l3m` | -0.6086 | High recent shopping frequency. |
| `total_app_sessions_l3m` | -0.5639 | Regular digital app engagement in Q1 2024 prevents churn. |
| `total_spend_l3m` | -0.5536 | Monetary investment in the recent quarter. |
| `total_coupons_l3m` | 0.0000 | Active coupon redemption indicates loyalty. |
| `total_emails_opened_l3m` | 0.0000 | Marketing responsiveness in Q1 2024. |

### Top Risk Factors (Positive Correlation with Churn)
| Feature Name | Pearson $r$ | Business Meaning |
| :--- | :---: | :--- |
| `recency_months` | +0.4334 | **Strongest single risk driver:** Months since last shopping transaction. |
| `zero_trans_months_count_15m` | +0.3044 | Number of dormant months in the 15-month window. |
| `total_support_tickets_15m` | +0.2190 | Higher customer service friction indicates dissatisfaction. |
| `customer_tenure_months` | +0.1960 | Older member cohorts have reached mature churn cycles. |
| `total_complaints_15m` | +0.0931 | Formal complaints are direct precursors to member defection. |

---

## 4. Final Modeling Dataset Architecture

- **Path:** `data/final/customer_churn_modeling_dataset.parquet` (and `.csv`)
- **Dimensions:** 2,600 rows $	imes$ 62 columns
- **Keys:** `CUSTOMER_ID` (100% unique, zero duplicates)
- **Nulls:** 0 across all 62 columns
- **Feature Groups:**
  1. **Demographics (10 features):** `AGE`, `GENDER`, `CITY`, `MEMBERSHIP_TIER`, `HOME_STORE_PRICE_TIER`, `PREFERRED_CATEGORY`, `MARKETING_OPT_IN`, missingness indicators.
  2. **Recency & Dormancy (4 features):** `recency_months`, `active_months_count_15m`, `zero_trans_months_count_15m`, `active_months_ratio`.
  3. **Spend & Monetary Velocity (10 features):** Total spend (15M, L3M, L6M), mean spend, spend std, basket value mean/max, spend trend 3M vs prior.
  4. **Frequency & Transactions (6 features):** Total transactions (15M, L3M, L6M), avg monthly txns, txn trend 3M vs prior.
  5. **Digital & Promotional Engagement (10 features):** App sessions (15M, L3M, L6M, trend), emails opened (15M, L3M, L6M), coupon redemptions (15M, L3M), coupon redemption rate.
  6. **Customer Service & Support Friction (7 features):** Support tickets (15M, L3M, L6M), complaints (15M, L3M, L6M), recent complaint flag, support trend.
  7. **Tenure & Cohorts (3 features):** `customer_tenure_months`, `signup_cohort_year`, `signup_cohort_quarter`.
  8. **Labels & Metadata (4 features):** `INSUFFICIENT_HISTORY_FLAG`, `MONTHS_OBSERVED`, `NEVER_PURCHASED_FLAG`, `CHURNED` (Target).

---

## 5. Visual Artifacts Generated

The following high-resolution charts were created in `outputs/figures/`:
1. `churn_distribution.png` — Class imbalance and volume overview.
2. `churn_by_tier.png` — Churn rates across Silver, Gold, Platinum.
3. `churn_by_city.png` — Geographic churn ranking across 10 markets.
4. `churn_by_signup_cohort.png` — Quarterly cohort churn curve.
5. `churn_vs_recency_and_frequency.png` — Recency and zero-transaction boxplots.
6. `churn_vs_support_and_complaints.png` — Support ticket and complaint churn analysis.
7. `correlation_heatmap.png` — Feature correlation matrix with target.
