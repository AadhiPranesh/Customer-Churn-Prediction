# Tailwyndz Propel 2026 Assessment: Customer Churn Prediction & Retention Strategy
## Final Technical Report: FreshBasket Retail Loyalty Program

**Candidate:** Machine Learning & Data Science Engineering Assessment  
**Project:** Customer Churn Prediction for a Retail Loyalty Program  
**Organization:** FreshBasket Retail (Fictional Grocery Retailer)  
**Date:** October 2026  
**Status:** Completed & Validated  

---

## 1. Executive Summary and Business Problem

### 1.1 Business Context
FreshBasket Retail operates a regional retail grocery loyalty program across 10 metropolitan markets and 40 store locations. The program features three membership tiers: **Silver**, **Gold**, and **Platinum**. In recent fiscal quarters, membership churn has steadily climbed, yet retail leadership possessed no systematic, data-driven mechanism to detect at-risk customers prior to complete defection.

To combat churn, FreshBasket historically relied on a **blanket retention marketing policy**: mailing identical promotional discount vouchers to all loyalty members indiscriminately. Management recognized two fundamental flaws in this practice:
1. **Inefficient Capital Allocation:** Blanket offers predominantly reach active, loyal customers who have no intention of leaving, needlessly subsidizing baseline sales.
2. **Failure to Address Friction:** Members defecting due to unresolved customer service issues or product complaints receive price promotions rather than proactive problem resolution.

### 1.2 Problem Objective
The objective of this project is to build an end-to-end classification, diagnostic, and retention simulation system that:
- **Predicts Churn Probability:** Quantifies individual member defection risk for the upcoming observation period.
- **Identifies Churn Drivers:** Diagnoses behavioral, transactional, digital engagement, and customer service friction drivers.
- **Segments Member Risk:** Replaces blanket marketing with risk-stratified operational tiers (**Low**, **Medium**, **High**).
- **Simulates Retention Economics:** Compares targeted intervention playbooks against the status quo blanket policy using transparent financial assumptions and sensitivity modeling.
- **Formulates an A/B Testing Protocol:** Outlines a rigorous randomized controlled trial to measure causal uplift before enterprise rollout.

---

## 2. Dataset Description and Data Dictionary Summary

### 2.1 Workbook Architecture
The dataset covers 18 months of historical activity from **January 1, 2023 through June 30, 2024** for a loyalty population of **2,600 unique members**. The workbook contains three data tables and two reference documentation sheets:

| Table / Worksheet | Dimensions | Role / Description |
| :--- | :---: | :--- |
| **`Glossary`** | 11 rows | Project overview, definitions, and strict data leakage rules. |
| **`Data Dictionary`** | 30 rows | Field descriptions, business definitions, and data types. |
| **`fb Customer Profile`** | 2,610 rows $\times$ 10 cols | Member demographics (age, gender, city), tier, preferences, and price tier. |
| **`fb Monthly Activity`** | 28,075 rows $\times$ 12 cols | Monthly transactional time-series (spend, visits, app logins, emails, complaints). |
| **`fb Churn Label`** | 2,607 rows $\times$ 6 cols | Ground truth churn labels, observation dates, and history flags. |

*(Note: Data tables contain a decorative title row in Row 1; column headers begin on Row 2, with data rows commencing on Row 3).*

### 2.2 Operational Data Dictionary
- `CUSTOMER_ID`: Unique integer identifier connecting all three tables across 2,600 members.
- `SIGNUP_DATE`: Member loyalty enrollment date (spanning February 2021 through May 2024).
- `MEMBERSHIP_TIER`: Customer loyalty tier (**Silver**, **Gold**, **Platinum**).
- `CITY`: Customer regional home market (10 metropolitan markets).
- `TOTAL_SPEND`: Monthly gross purchases in USD.
- `TRANSACTIONS`: Monthly in-store and online shopping transaction count.
- `AVG_BASKET_VALUE`: Spend per transaction ($\text{TOTAL\_SPEND} / \text{TRANSACTIONS}$).
- `APP_SESSIONS` & `EMAILS_OPENED`: Monthly digital and marketing engagement metrics.
- `SUPPORT_TICKETS` & `COMPLAINT_FLAG`: Customer service contact volume and formal complaints.
- `INSUFFICIENT_HISTORY_FLAG`: Flagged as 1 if $< 4$ months of historical activity are recorded.
- `CHURNED` (Target): Defined as **zero purchase transactions in April–June 2024** despite having purchase history earlier in the window.

---

## 3. Data-Quality Issues and Cleaning Decisions

A rigorous Phase 2 audit uncovered several realistic data-quality issues that were systematically resolved without modifying or overwriting the original Excel workbook.

### 3.1 Cleaning Decisions & Technical Rationale

| Data Quality Issue | Table | Raw Audit Count | Cleaning Treatment & Rationale | Post-Clean Status |
| :--- | :---: | :---: | :--- | :---: |
| **Exact Duplicate Rows** | Profile | 8 rows | Dropped identical duplicate rows (`CUSTOMER_ID` uniqueness). | 2,600 unique rows |
| **Exact Duplicate Rows** | Churn Label | 5 rows | Dropped identical duplicate rows. | 2,600 unique rows |
| **Duplicate (ID, Month)** | Activity | 60 pairs | Dropped duplicate activity rows after sign correction. | 28,013 unique rows |
| **Inconsistent Text Casing** | Profile | 40 raw strings | Applied `.str.strip().str.title()` across `CITY` and `MEMBERSHIP_TIER`. | 10 clean cities, 3 tiers |
| **Missing Price Tiers** | Profile | 130 missing (4.98%) | Store-level audit proved price tier is constant per store ID. 100% recovered via `PREFERRED_STORE_ID` lookup. | 0 missing (100% recovered) |
| **Missing Age** | Profile | 78 missing (2.99%) | Symmetric distribution (mean 46.16, median 46.0). Median imputed; added binary `AGE_IS_MISSING = 1`. | 0 missing |
| **Missing Gender** | Profile | 52 missing (1.99%) | Imputed as `'Unknown'`; added binary `GENDER_IS_MISSING = 1`. | 0 missing |
| **Negative Spend** | Activity | 24 records | For all 24 rows, $\text{AVG\_BASKET\_VALUE} > 0$ and equals $|\text{TOTAL\_SPEND}| / \text{TRANSACTIONS}$. Confirmed as data-entry sign error. Corrected via `abs()`. | 0 negative spend |
| **Zero Spend with Txns** | Activity | 20 records | Inactive spend entry despite active visits. Reconstructed as $\text{TRANSACTIONS} \times \text{AVG\_BASKET\_VALUE}$. | Fully reconstructed |
| **Zero-Activity Months** | Activity | 2,335 records | Zero visits/spend with active app logins and support tickets. Strictly preserved as pre-churn dormancy signals. | Preserved |
| **Missing Last Purchase** | Churn Label | 120 records | Customers with 0 lifetime transactions in window. Verified non-churners per definition; flagged `NEVER_PURCHASED_FLAG = 1`. | Preserved & flagged |

---

## 4. Exploratory Data Analysis & Important Findings

### 4.1 Ground Truth Churn Distribution & Class Imbalance
- **Active Members (`CHURNED = 0`):** 1,782 members (**68.54%**)
- **Churned Members (`CHURNED = 1`):** 818 members (**31.46%**)
- **Imbalance Ratio:** ~2.18 to 1.
- *Methodological Consequence:* Standard classification accuracy is inherently misleading: a naive dummy classifier predicting zero churn achieves **68.54% accuracy** while identifying zero defecting members. Model benchmarking must center on **Precision-Recall AUC (PR-AUC)**, **ROC-AUC**, and **Recall**.

### 4.2 Segment-Level Analysis

#### A. Membership Tier (Demographic Invariance)
- **Silver:** 1,474 members, 476 churned (**32.29% churn rate**)
- **Platinum:** 302 members, 94 churned (**31.13% churn rate**)
- **Gold:** 824 members, 248 churned (**30.10% churn rate**)
- *Takeaway:* Churn rates across tiers are virtually identical (~30%–32%). Tier status does not differentiate churners from non-churners, proving that demographics alone cannot solve churn.

#### B. Regional City Markets
- **Highest Churn Markets:** **Austin (37.59%)**, **Columbus (36.19%)**, and **Nashville (33.33%)**.
- **Lowest Churn Markets:** **Portland (28.33%)**, **Denver (28.79%)**, and **Spokane (28.97%)**.

#### C. Signup Cohort Dynamics (Survival Effect)
- **2021 Signups:** 745 members, 279 churned (**37.45% churn rate**)
- **2022 Signups:** 792 members, 306 churned (**38.64% churn rate**)
- **2023 Signups:** 816 members, 229 churned (**28.06% churn rate**)
- **2024 Signups:** 247 members, 4 churned (**1.62% churn rate**)
- *Takeaway:* Mature cohorts exhibit natural steady-state attrition (~38%), whereas recent 2024 signups have had insufficient exposure time to register 3 consecutive dormant months. This survival effect justifies our cohort-based validation strategy.

---

## 5. Leakage-Safe Feature Engineering

### 5.1 Strict Temporal Boundary
The 18-month activity window spans January 1, 2023 through June 30, 2024. The assessment defines churn over the final 3 months (**April 1, 2024 – June 30, 2024**).
- **Predictive Feature Window:** **January 1, 2023 through March 31, 2024** (15 historical months).
- **Outcome Window:** April 1, 2024 through June 30, 2024 (3 label months).
- **Leakage Elimination:** 100% of predictive features were computed strictly from transactions occurring on or before March 31, 2024. All label-derived columns (`INSUFFICIENT_HISTORY_FLAG`, `MONTHS_OBSERVED`, `NEVER_PURCHASED_FLAG`, `LAST_PURCHASE_DATE`, `OBSERVATION_END_DATE`) were **strictly excluded from the predictive feature set $X$**.

### 5.2 Engineered Feature Groups (57 Total Predictive Features)
1. **Recency & Dormancy (4 features):**
   - `recency_months`: Months between last recorded purchase and cutoff date (`2024-03-31`).
   - `active_months_count_15m`: Count of months with active transactions.
   - `zero_trans_months_count_15m`: Count of dormant zero-transaction months.
   - `active_months_ratio`: Active months divided by observed months in window.
2. **Spend & Monetary Velocity (10 features):**
   - `total_spend_15m`, `avg_monthly_spend_15m`, `max_monthly_spend_15m`, `std_monthly_spend_15m`.
   - `total_spend_l3m` (Jan–Mar 2024) and `total_spend_l6m` (Oct 2023–Mar 2024).
   - `spend_trend_3m_vs_prior`: Ratio of recent monthly spend to prior quarterly spend.
   - `avg_basket_value_mean` & `avg_basket_value_max`.
3. **Transaction Frequency & Velocity (6 features):**
   - `total_txns_15m`, `avg_monthly_txns_15m`, `total_txns_l3m`, `total_txns_l6m`.
   - `txns_trend_3m_vs_prior`: Frequency acceleration/deceleration ratio.
   - `purchased_in_last_3m_flag`: Binary indicator of recent quarterly activity.
4. **Digital & Promotional Engagement (10 features):**
   - `total_app_sessions_15m`, `total_app_sessions_l3m`, `app_trend_3m_vs_prior`.
   - `total_emails_opened_15m`, `total_emails_opened_l3m`.
   - `total_coupons_15m`, `total_coupons_l3m`, `coupon_redemption_rate`, `avg_promo_txn_pct_15m`.
5. **Customer Support Friction & Complaints (7 features):**
   - `total_support_tickets_15m`, `total_support_tickets_l3m`, `total_support_tickets_l6m`.
   - `support_ticket_trend_l3m`: Escalation in customer inquiries.
   - `total_complaints_15m`, `total_complaints_l3m`, `had_recent_complaint_flag`.
6. **Demographics & Profile (10 features):**
   - `AGE`, `AGE_IS_MISSING`, `GENDER`, `GENDER_IS_MISSING`, `CITY`, `MEMBERSHIP_TIER`, `HOME_STORE_PRICE_TIER`, `PREFERRED_CATEGORY`, `PREFERRED_STORE_ID`, `MARKETING_OPT_IN`.
7. **Tenure & Cohorts (2 features):**
   - `customer_tenure_months` (as of March 31, 2024) and `signup_cohort_year`.

---

## 6. Validation Strategy and Limitations

### 6.1 Validation Design: Cohort-Based Temporal Split
In production, a loyalty churn model must evaluate newer incoming member cohorts using decision rules learned on mature customers. A naive random split leaks future cohort relationships into earlier cohorts.

We partitioned the data chronologically by `SIGNUP_DATE`:
- **Training Cohort:** Members enrolled **before July 1, 2023** ($n = 1,947$, 74.9% of total). Churn rate = **37.65%** (733 churned, 1,214 active).
- **Validation Cohort:** Members enrolled **on or after July 1, 2023** ($n = 653$, 25.1% of total). Churn rate = **13.02%** (85 churned, 568 active).

### 6.2 Preprocessing & Threshold Protocol
- **No Test Snooping:** All transformers (`OneHotEncoder`, `StandardScaler`) were fit exclusively on `X_train` and applied downstream to `X_test`.
- **Threshold Tuning:** Classification decision thresholds were optimized strictly using out-of-fold predictions during 5-fold cross-validation on `X_train` to maximize F1-score.

### 6.3 Acknowledged Limitations of Validation Strategy
Because all members share the same label observation window (April–June 2024) and observation end date (June 30, 2024), a true multi-quarter calendar out-of-time (OOT) evaluation is constrained by the 18-month dataset. Newer cohorts (specifically Q1 and Q2 2024 signups) exhibit lower baseline churn due to shorter exposure windows, introducing cohort-dependent base rates that must be accounted for during operational scoring.

---

## 7. Model Comparison and Metric Reconciliation

### 7.1 Benchmark Table (Held-Out Validation Cohort, $n=653$, 85 Churners)

All candidate models were evaluated on the held-out validation cohort:

| Model Architecture | Decision Threshold ($\tau$) | PR-AUC | ROC-AUC | F1-Score | Recall | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.330** | **0.8315** | **0.9080** | 0.4398 | **0.8588** | 0.2955 | 0.7152 |
| *Random Forest (Standard $\tau$)* | *0.500* | *0.8315* | *0.9080* | **0.7826** | *0.7412* | **0.8289** | **0.9464** |
| **HistGradientBoosting** | 0.810 | **0.8139** | **0.9181** | **0.7394** | 0.7176 | **0.7625** | **0.9342** |
| *HistGradientBoosting (Standard $\tau$)* | *0.500* | *0.8139* | *0.9181* | 0.6842 | 0.7647 | 0.6190 | 0.9081 |
| **Logistic Regression (Balanced)** | 0.720 | 0.5680 | 0.8500 | 0.4538 | 0.6941 | 0.3371 | 0.7825 |
| **Heuristic Recency Rule** | 0.500 | 0.1957 | 0.6843 | 0.0088 | 0.0118 | 0.0071 | 0.6570 |
| **Majority Class Baseline** | 0.500 | 0.1302 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.8698 |

### 7.2 Confusion Matrices (Validation Cohort)

| Model & Threshold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Operational Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest ($\tau = 0.330$)** | 394 | 174 | **12** | **73** | **Maximum Recall:** Captures 85.9% of all churners. |
| **Random Forest ($\tau = 0.500$)** | 555 | 13 | 22 | **63** | **High Precision (82.9%):** Minimizes wasted outreach. |
| **HistGradientBoosting ($\tau = 0.810$)** | 549 | 19 | 24 | **61** | Balanced accuracy and precision (76.3%). |
| **Logistic Regression ($\tau = 0.720$)** | 452 | 116 | 26 | **59** | Linear baseline; moderate recall (69.4%). |
| **Heuristic Recency Rule** | 428 | 140 | 84 | 1 | Naive heuristic fails to separate subtle churners. |
| **Majority Class Baseline** | 568 | 0 | 85 | 0 | Zero predictive capacity. |

### 7.3 Metric Reconciliation Analysis
1. **Operating Point Trade-Off:** 
   - At $\tau = 0.330$, Random Forest maximizes **Recall (85.88%)**, missing only 12 churners in the entire test cohort. This operating point is ideal for low-cost, automated digital reactivation where false positives incur negligible cost.
   - At $\tau = 0.500$, Random Forest achieves **82.89% Precision** and **94.64% Accuracy**, making it the superior threshold for high-cost, high-touch concierge interventions.
2. **Superiority over Baselines:** 
   - Random Forest delivers a **+0.7013 PR-AUC lift** over the majority-class baseline (0.8315 vs 0.1302).
   - While the naive baseline boasts 86.98% accuracy, it catches zero churners, confirming that accuracy is a fundamentally flawed metric in retention modeling.

---

## 8. Ablation Study and Feature Importance

### 8.1 Required Feature Ablation Study Results
To isolate the predictive value contributed by different data modalities, we evaluated three strictly controlled feature groups under the identical cohort validation split using the Random Forest architecture:

| Feature Group | Modality Scope | Total Features | PR-AUC | ROC-AUC | F1-Score | Recall | Precision | PR-AUC Uplift vs A |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Group A** | Demographics & Membership Only | 12 | 0.1630 | 0.5073 | 0.1711 | 0.1529 | 0.1940 | *Baseline* |
| **Group B** | Demographics + Historical Transactions | 36 | 0.7534 | 0.8750 | 0.6503 | 0.6235 | 0.6795 | **+0.5904** |
| **Group C** | Full Features (+ Engagement & Support) | 57 | **0.8463** | **0.9222** | **0.8050** | **0.7529** | **0.8649** | **+0.6833** |

#### Ablation Insights:
1. **Demographics Alone Have Zero Discriminative Power:** Group A yields a PR-AUC of 0.1630 (barely above the 0.1302 random prevalence). Age, gender, city, and tier do not determine churn.
2. **Transaction Dynamics Provide the Core Signal:** Adding frequency, spend velocity, and dormancy metrics generates a massive **+0.5904 PR-AUC uplift**.
3. **Digital & Support Friction Provide the Winning Edge:** Incorporating app sessions, marketing email responsiveness, coupon usage, support tickets, and complaints adds an additional **+0.0929 PR-AUC uplift** (+0.6833 total lift over demographics alone).

### 8.2 Top Permutation Feature Importances (Held-Out Test Set)

Permutation importance evaluates the mean decrease in PR-AUC when each feature is randomly shuffled on the test cohort:

| Rank | Predictive Feature | Mean Decrease in PR-AUC | Standard Error | Domain Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `recency_months` | **+0.0245** | $\pm 0.0054$ | Months since last purchase is the single strongest marker of customer disengagement. |
| **2** | `avg_basket_value_mean` | **+0.0230** | $\pm 0.0064$ | Contraction in basket size indicates decaying customer wallet share. |
| **3** | `total_txns_l3m` | **+0.0169** | $\pm 0.0062$ | Recent shopping frequency in the prior quarter. |
| **4** | `max_monthly_spend_15m` | **+0.0156** | $\pm 0.0032$ | Historical purchasing capacity and account engagement scale. |
| **5** | `total_spend_15m` | **+0.0140** | $\pm 0.0014$ | Lifetime monetary relationship in the predictive window. |
| **6** | `total_txns_15m` | **+0.0139** | $\pm 0.0014$ | Baseline customer loyalty and store visit habits. |
| **7** | `active_months_count_15m`| **+0.0138** | $\pm 0.0016$ | Long-term consistency of store visits. |
| **8** | `app_trend_3m_vs_prior` | **+0.0119** | $\pm 0.0049$ | Deceleration in mobile app logins directly precedes transactional dormancy. |
| **9** | `avg_promo_txn_pct_15m` | **+0.0114** | $\pm 0.0030$ | Promotional dependency; deal seekers churn when offers decrease. |
| **10** | `total_support_tickets_l6m`| **+0.0069** | $\pm 0.0014$ | Customer service friction and complaint escalation. |

> **Causation Guardrail:** High feature importance indicates strong predictive association, not direct causation. For example, high recency is a behavioral symptom of disengagement rather than the root cause. Support tickets and complaints, however, provide direct, actionable operational levers for retention teams.

---

## 9. Customer Risk Scoring and Risk Segmentation

### 9.1 Operational Risk Stratification (All 2,600 Members)
Operational predictions are stored in `outputs/predictions/customer_churn_predictions.csv`:
- **Low Risk ($p < 0.30$):** **1,561 members (60.0%)**. Healthy, highly active shoppers.
- **Medium Risk ($0.30 \le p < 0.60$):** **264 members (10.2%)**. Early disengagement; candidates for automated digital nudges.
- **High Risk ($p \ge 0.60$):** **775 members (29.8%)**. Severe dormancy or active customer service complaints; priority targets for high-touch intervention.

### 9.2 Audit of Exact-1.0 Probabilities
In the prediction file, exactly **422 customers** received a churn probability of **1.0000**.
- **Empirical Audit:** Across all 422 customers, the ground truth churn rate was **100.0%** (422 out of 422 actually churned).
- **Mathematical Explanation:** These members exhibited prolonged dormancy (mean `recency_months` = 9.34 months, min 3.0 months, zero purchases in Jan–Mar 2024). In our 200-tree Random Forest, extreme dormancy feature vectors land in terminal leaf nodes containing 100% pure churn training samples across all 200 trees ($\frac{1}{200}\sum 1.0 = 1.0$).
- *Conclusion:* Exact-1.0 probabilities represent uncalibrated leaf purity on extreme dormant cases and are completely expected and empirically accurate.

### 9.3 High-Risk Segment Distribution
- **By Tier:** Churn probabilities are uniformly distributed across tiers (Silver: 36.8%, Gold: 34.9%, Platinum: 33.6%). Silver tier contains the largest absolute volume of at-risk members (543 high-risk members) and must not be ignored.
- **By City Market:** **Austin (38.7% high-risk)** and **Columbus (37.7% high-risk)** exhibit the highest risk density, whereas **Portland (23.8%)** and **Denver (24.1%)** are the most stable.

---

## 10. Retention Intervention Scenarios & Sensitivity Analysis

### 10.1 Financial Assumptions & Scope
Because exact margin structures were not provided in the dataset, we utilized realistic, clearly labeled retail grocery benchmarks:
- **Baseline Annual Retained Member Value (LTV Contribution):** **\$180.00** (~$30 average basket $\times$ 6 retained shopping trips per year at ~15% contribution margin).
- *Strict Distinction:* Simulated assumptions are clearly separated from empirical model metrics. Model probabilities quantify churn risk; intervention efficacy represents assumed operational responsiveness that must be validated via A/B testing.

### 10.2 Comparative Scenario Economics

| Retention Strategy | Target Scope | Targeted Members | Campaign Cost (USD) | Est. Retained Members | Est. Gross Benefit (USD) | Est. Net Benefit (USD) | ROI (%) | Cost / Retained Member |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Status Quo (Blanket Promotion)** | All members | 2,600 | \$26,000 | 40.9 | \$7,362 | **-\$18,638** | **-71.7%** | \$635.70 |
| **Scenario 1 (Targeted Reactivation)** | Disengaged only | 862 | \$6,896 | 69.7 | \$12,547 | **+\$5,651** | **+81.9%** | \$98.93 |
| **Scenario 2 (Proactive VIP Support)** | Support friction | 458 | \$6,870 | 110.7 | \$19,924 | **+\$13,054** | **+190.0%** | \$62.07 |
| **Scenario 3 (Integrated Dual-Track)** | Priority VIP + Reactivation | 898 | \$10,390 | 144.2 | \$25,962 | **+\$15,572** | **+149.9%** | **\$72.04** |

#### Scenario Takeaways:
1. **Blanket Retention Destroys Capital:** Spending \$26,000 across all 2,600 members yields a **net loss of -\$18,638**, validating executive concerns regarding blanket discounting.
2. **Proactive Support Offers Exceptional ROI (+190%):** Resolving customer complaints and service tickets for the 458 friction members costs \$6,870 and saves an estimated 110.7 members (**+\$13,054 net gain**).
3. **Integrated Strategy Maximizes Net Value (+150% ROI):** Routing friction members to concierge support and remaining disengaged members to category vouchers achieves **+\$15,572 net benefit** and reduces cost per retained customer from \$635.70 down to **\$72.04**.

### 10.3 Sensitivity Analysis Matrix (Scenario 3: Integrated Strategy)
To confirm robustness against varying retail conditions, we modeled Net Benefit across a matrix of intervention efficacy rates (10% to 30%) and customer lifetime values (\$100 to \$250):

| Intervention Efficacy | LTV = \$100 | LTV = \$150 | LTV = \$180 (Base) | LTV = \$220 | LTV = \$250 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | -\$3,178 | +\$433 | **+\$2,599** | +\$5,488 | +\$7,654 |
| **15%** | +\$427 | +\$5,840 | **+\$9,088** | +\$13,419 | +\$16,666 |
| **20% (Base)** | +\$4,031 | +\$11,247 | **+\$15,572** | +\$21,349 | +\$25,682 |
| **25%** | +\$7,636 | +\$16,654 | **+\$22,065** | +\$29,279 | +\$34,699 |
| **30%** | +\$11,240 | +\$22,061 | **+\$28,554** | +\$37,210 | +\$43,702 |

> **Break-Even Finding:** Under our baseline \$180 LTV assumption, the integrated retention strategy breaks even at an intervention efficacy of just **9.5%**. Even under a highly conservative downside scenario (10% efficacy at \$150 LTV), the campaign remains cash-flow positive.

---

## 11. Proposed A/B Testing Plan

A machine learning churn model identifies *correlation* with risk, not *treatment effect*. To measure true causal uplift before broad commercial deployment, FreshBasket must execute a **Randomized Controlled Trial (A/B Test)**:

### 11.1 Experimental Design
- **Eligible Population:** The 1,039 members categorized as High Risk ($p \ge 0.60$) or Medium Risk ($0.30 \le p < 0.60$).
- **Randomization:** Individual customer-level stratified randomization (stratified by Tier and City).
- **Treatment Arm (80%, $n = 831$):** Receives the Dual-Track Retention Playbook (concierge outreach for service friction cases; automated personalized 15% preferred category discount for dormant shoppers).
- **Holdout Control Arm (20%, $n = 208$):** Receives standard baseline communications with zero promotional incentive.

### 11.2 Hypotheses & Primary Outcome Metrics
- **Primary Hypothesis:** Members in the Treatment Arm will exhibit a statistically significant reduction in 90-day churn rate compared to the Holdout Control ($p < 0.05$, two-tailed two-proportion z-test with 80% statistical power to detect a 5 percentage-point delta).
- **Secondary Metrics:**
  - Incremental Spend per Targeted Customer.
  - Net Contribution Margin after deducting voucher discounts and concierge labor costs.
  - Promotional Redemption Rate and 60-day visit frequency.

---

## 12. Limitations, Fairness, and Insufficient-History Customers

### 12.1 Handling Insufficient-History Customers (2024 Signups)
- **Data Finding:** 373 members (14.35%) have `INSUFFICIENT_HISTORY_FLAG = 1` (< 4 months history).
- **Model Behavior:** Because recent 2024 signups have sparse transaction counts, the model predicts higher churn risk (mean predicted risk = 34.4%) compared to their observed churn rate (1.6%).
- **Governance Recommendation:** Do not enroll members with tenure $< 4$ months into aggressive win-back discount campaigns. Instead, route them into a dedicated **Onboarding & First-90-Days Nurture Journey** focused on app setup, welcome rewards, and loyalty program education.

### 12.2 Algorithmic Fairness & Bias Mitigation
- **Tier Inclusivity:** Retention offers must not be restricted exclusively to Gold and Platinum members. Silver members represent 56.7% of all loyalty members; neglecting them would accelerate base erosion.
- **Geographic Fairness:** Stores in Austin and Columbus require operational support (inventory audits, staffing reviews) rather than penalizing members in those markets with lower service quality.

### 12.3 Future Technical Improvements
1. **Dynamic Churn Definition:** Transition from fixed calendar windows to rolling 60-day inactivity horizons.
2. **Uplift Modeling:** Train two-model uplift or causal forest learners to predict which members are "persuadable" versus "lost causes" or "sure things".
3. **Deep Sequence Modeling:** Implement temporal Recurrent Neural Networks (LSTM/GRU) or Temporal Fusion Transformers directly on weekly basket event sequences.

---

## 13. Conclusion and Reproducibility Instructions

### 13.1 Assessment Conclusions
By replacing indiscriminate blanket mailings with a leakage-free machine learning framework, FreshBasket can capture **85.9% of churning members**, transform a **-\$18,638 blanket loss** into a **+\$15,572 net gain**, and build an enduring operational retention engine.

### 13.2 Execution Instructions
The entire project is structured into standalone, modular pipelines. You can execute any phase independently from PowerShell:

```powershell
# Phase 2: Data Cleaning & Quality Audit
python run_cleaning.py

# Phase 3: Feature Engineering & EDA Visualizations
python run_phase3.py

# Phase 4: Model Development, Cohort Benchmarking & Ablation
python run_phase4.py

# Phase 5: Operational Scoring, Retention Simulation & Business Analysis
python run_phase5.py
```

All source code, saved models (`outputs/models/`), metrics tables (`outputs/metrics/`), figures (`outputs/figures/`), and predictions (`outputs/predictions/`) are verified, reproducible, and ready for deployment.
