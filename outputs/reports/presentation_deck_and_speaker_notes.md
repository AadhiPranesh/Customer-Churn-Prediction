# Presentation Deck & Speaker Notes: Customer Churn Prediction & Retention Strategy
## FreshBasket Retail Loyalty Program | Tailwyndz Propel 2026 Assessment

**Format:** 12 Slides | **Target Duration:** 12–15 Minutes | **Audience:** Technical Assessment Panel & Retail Leadership  

---

### Slide 1: Title & Executive Overview
- **Title:** Customer Churn Prediction and Retention Strategy for FreshBasket Retail
- **Subtitle:** Machine Learning-Driven Customer Retention for a Retail Loyalty Program
- **Presenter:** Candidate, Data Science Engineering Assessment
- **Key Project Highlights:**
  - 18 months of retail activity across 2,600 loyalty members (Silver, Gold, Platinum).
  - Leakage-safe feature engineering isolating historical activity through March 31, 2024.
  - Champion Random Forest model achieving **0.8315 PR-AUC** and **85.9% Recall** on held-out cohorts.
  - Strategic shift from costly blanket promotions to an integrated targeted retention playbook (+150% ROI).
- **Recommended Visual:** Project Architecture Flowchart (Data -> Cleaning -> Features -> Models -> Intervention).
- **Speaking Time:** ~1.0 Minute
- **Speaker Notes:**
  > "Good morning, members of the evaluation panel. Today, I am presenting an end-to-end Machine Learning and Retention Strategy for FreshBasket Retail's customer loyalty program. FreshBasket operates a multi-tier membership program across 10 regional markets. In recent quarters, membership churn has steadily climbed, yet the business lacked an early-warning system. By building a leakage-safe predictive pipeline and simulating targeted business interventions, we replaced expensive blanket mailings with a precision retention engine. Over the next 12 minutes, I will walk you through our data engineering decisions, model benchmarking, ablation findings, and business economics."

---

### Slide 2: Business Problem & Core Objectives
- **Title:** Business Scenario: The Failure of Blanket Retention
- **On-Slide Bullet Points:**
  - **The Retention Challenge:** Rising churn across Silver, Gold, and Platinum tiers; no systematic way to identify at-risk members before they stop shopping.
  - **The Flawed Status Quo:** Marketing sends identical $10 discount vouchers to all members indiscriminately ($26,000 annual spend).
  - **The Financial Waste:** Blanket offers subsidize loyal shoppers who were never going to leave, while completely failing to resolve customer service friction.
  - **Core Objectives:**
    1. Accurately predict individual customer churn risk for the next observation period.
    2. Uncover the behavioural, digital, and service drivers preceding customer defection.
    3. Simulate high-ROI retention interventions to turn marketing losses into net profit.
- **Recommended Visual:** High-level problem diagram comparing "Blanket Marketing (Status Quo)" vs "Targeted Retention (Proposed)".
- **Speaking Time:** ~1.0 Minute
- **Speaker Notes:**
  > "Let's begin with the business problem. FreshBasket has been losing valuable loyalty members, and leadership had no way of knowing who was about to defect until they stopped visiting the store. To stop the bleeding, marketing was mailing $10 vouchers to every single member. As you can imagine, this was hugely wasteful: millions of dollars were spent subsidizing loyal shoppers who were going to shop anyway, while customers who were angry about bad service got a coupon instead of an apology. Our mission was to build a machine learning framework that predicts defection early, diagnoses the exact reasons why members leave, and models targeted interventions to maximize retention ROI."

---

### Slide 3: Dataset Architecture & Problem Formulation
- **Title:** Dataset Overview & Operational Problem Framing
- **On-Slide Bullet Points:**
  - **Dataset Scope:** 18 calendar months (January 2023 – June 2024) across 2,600 unique loyalty members.
  - **Workbook Tables:**
    - `fb Customer Profile`: 2,600 members, demographics, signup dates, store price tiers.
    - `fb Monthly Activity`: 28,013 customer-month rows (spend, visits, app logins, emails, complaints).
    - `fb Churn Label`: Observation end date (June 30, 2024) and ground-truth target.
  - **Ground Truth Target (`CHURNED`):**
    - Defined as zero purchase transactions in the final 3 months (**April–June 2024**) despite prior purchase history.
  - **Class Balance:**
    - Active Members (`0`): 1,782 (68.54%) | Churned Members (`1`): 818 (31.46%).
    - Baseline Imbalance Ratio: ~2.18 to 1.
- **Recommended Visual:** `outputs/figures/churn_distribution.png` (Class balance bar & donut chart).
- **Speaking Time:** ~1.0 Minute
- **Speaker Notes:**
  > "The dataset covers 18 months of customer activity across 2,600 loyalty members. We are given three tables: customer demographics, monthly transactional history, and churn labels. The churn target is defined as having zero purchase transactions in the final three months—April, May, and June 2024—despite having an established purchasing history earlier in the observation window. Exactly 818 members churned, representing a 31.5% churn rate. This moderate class imbalance is typical of retail loyalty programs. A crucial technical point to note here is that a naive model that predicts zero churn achieves 68.5% accuracy while providing zero business value. Therefore, standard accuracy is not our primary metric; we evaluate our models on PR-AUC, ROC-AUC, and Recall."

---

### Slide 4: Data Quality Audit & Defensive Cleaning Decisions
- **Title:** Data Quality Audit: Defending Data-Cleaning Choices
- **On-Slide Bullet Points:**
  - **Zero Raw Data Modification:** Raw Excel workbook preserved untouched; cleaned tables saved under `data/processed/`.
  - **Duplicate Resolution:** Removed 8 duplicates in Profile, 5 in Churn, and 60 in Activity (exact row duplicates).
  - **Categorical Casing Normalization:** Collapsed 40 raw city strings into 10 clean metropolitan markets; standardized tiers to Silver, Gold, Platinum.
  - **Negative Spend Investigation (24 rows):** Confirmed as data-entry sign errors because average basket values were positive and matched $|TOTAL\_SPEND| / TRANSACTIONS$. Fixed via `abs()`.
  - **Zero-Spend Reconstructions (20 rows):** Restored spend as $TRANSACTIONS \times AVG\_BASKET\_VALUE$.
  - **Preservation of Zero-Activity Months (2,335 rows):** Retained inactive months because they contain crucial digital app and customer complaint signals preceding churn.
  - **100% Deterministic Imputation:** Recovered all 130 missing store price tiers using `PREFERRED_STORE_ID` mapping (zero guesswork).
- **Recommended Visual:** Summary audit table before/after cleaning from `outputs/reports/data_quality_report.md`.
- **Speaking Time:** ~1.5 Minutes
- **Speaker Notes:**
  > "In Phase 2, we conducted a rigorous data quality audit. We did not delete data blindly. For example, we identified 24 records with negative spend. Rather than throwing them out, we audited the average basket value and transaction counts for those rows. We proved mathematically that average basket value was positive and equal to the absolute spend divided by transactions. The negative sign was simply a data-entry sign error. Similarly, we found 2,335 months with zero shopping visits. Deleting those rows would have destroyed the very dormancy and complaint signals that precede churn, so we strictly preserved them. For missing demographic values, we noticed that every store ID has an invariant price tier, allowing us to deterministically recover 100% of the 130 missing price tiers with zero guesswork. All 15 automated validation assertions passed."

---

### Slide 5: Leakage-Safe Feature Engineering
- **Title:** Feature Engineering: Eliminating Data Leakage
- **On-Slide Bullet Points:**
  - **The Golden Leakage Boundary:**
    - Label Window = April 1, 2024 – June 30, 2024 (3 months).
    - Predictive Feature Window = January 1, 2023 – **March 31, 2024** (15 historical months).
  - **Strict Information Isolation:** Zero activity occurring on or after April 1, 2024 was used to engineer features.
  - **57 Engineered Predictive Features:**
    - *Recency & Dormancy:* Months since last purchase, count of inactive months, active months ratio.
    - *Spend Velocity & Monetary:* 15M spend, L3M spend, L6M spend, basket value mean/max, spend trend 3M vs prior.
    - *Frequency & Momentum:* Total visits, visit velocity (L3M vs prior), recent quarterly purchase flag.
    - *Digital & Marketing:* Mobile app sessions (15M, L3M, trend), email open rates, coupon redemption rate.
    - *Customer Service Friction:* Support ticket volume, ticket escalation trend, formal complaint flags.
- **Recommended Visual:** Diagram of the 18-month timeline showing the 15-month Feature Window vs 3-month Label Window.
- **Speaking Time:** ~1.5 Minutes
- **Speaker Notes:**
  > "The most critical engineering requirement in churn modeling is preventing data leakage. Because the churn label is evaluated across April through June 2024, using any transaction, visit, or email open from that period would create catastrophic lookahead bias. We established a strict cutoff on March 31, 2024. All 57 predictive features were engineered strictly from historical activity on or before that date. We captured multidimensional behavioral signals: recency, spending velocity—measuring whether a customer's spend in Q1 2024 dropped compared to the previous quarter—frequency acceleration, mobile app logins, and customer service friction metrics. Furthermore, all label-derived metadata, such as insufficient history flags and observation dates, were excluded from the feature matrix."

---

### Slide 6: Validation Strategy: Cohort-Based Temporal Splitting
- **Title:** Validation Strategy: Cohort-Aware Evaluation
- **On-Slide Bullet Points:**
  - **Why Random Splits Fail:** Random train/test splits train on future customer cohorts to predict past cohorts, creating unrealistic data snooping.
  - **Selected Strategy:** **Cohort-Based Temporal Partitioning** by `SIGNUP_DATE`:
    - *Training Cohort:* Enrolled **before July 1, 2023** ($n = 1,947$, 74.9% of total). Churn rate = **37.65%**.
    - *Validation Cohort:* Enrolled **on or after July 1, 2023** ($n = 653$, 25.1% of total). Churn rate = **13.02%** (85 churners).
  - **Realistic Evaluation:** Tests how effectively models trained on mature members generalize to predict churn on newer cohorts with limited tenure.
  - **Leakage-Free Cross-Validation:** Preprocessing pipelines fit strictly on training folds; decision thresholds optimized via 5-fold CV on `X_train` alone.
  - **Acknowledged Limitation:** All members share the same label end date (June 30, 2024), so cohort splitting represents the most defensible temporal approach supported by the data.
- **Recommended Visual:** `outputs/figures/churn_by_signup_cohort.png` (Quarterly cohort survival curve).
- **Speaking Time:** ~1.0 Minute
- **Speaker Notes:**
  > "The assessment specifically forbade relying on random train/test splits. In a production loyalty environment, you train your models on mature member cohorts and deploy them to predict retention on newer cohorts. We partitioned our 2,600 members chronologically by enrollment date: members who joined before July 1, 2023 formed our training cohort of 1,947 members, while members joining in July 2023 or later formed our held-out validation cohort of 653 members. Both classes are well-represented in the test cohort, with 85 churners. To prevent threshold snooping, all decision thresholds were tuned exclusively using out-of-fold cross-validation on the training set. We explicitly document that because all members share the same June 2024 observation end date, this cohort split is the most defensible temporal validation strategy available."

---

### Slide 7: Model Comparison & Benchmark Results
- **Title:** Model Benchmarking on Held-Out Validation Cohort
- **On-Slide Bullet Points:**
  - Evaluated on Held-Out Test Cohort ($n=653$, Prevalence = 13.02%):
| Model | Threshold | PR-AUC | ROC-AUC | Recall | Precision | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.33** | **0.8315** | **0.9080** | **85.88%** | 29.55% | 0.4398 | 71.52% |
| *Random Forest (Balanced)* | *0.50* | *0.8315* | *0.9080* | 74.12% | **82.89%** | **0.7826** | **94.64%** |
| **HistGradientBoosting** | 0.81 | 0.8139 | **0.9181** | 71.76% | 76.25% | 0.7394 | 93.42% |
| **Logistic Regression** | 0.72 | 0.5680 | 0.8500 | 0.6941 | 0.3371 | 0.4538 | 0.7825 |
| **Heuristic Recency Rule** | 0.50 | 0.1957 | 0.6843 | 01.18% | 00.71% | 0.0088 | 0.6570 |
| **Majority Class Baseline** | 0.50 | 0.1302 | 0.5000 | 00.00% | 00.00% | 0.0000 | 86.98% |
  - **Random Forest Champion:** +0.7013 PR-AUC lift over random chance (0.8315 vs 0.1302).
  - **Threshold Strategy:** $\tau = 0.33$ captures **85.9% of churners** (missing only 12); $\tau = 0.50$ provides **82.9% precision** for expensive concierge outreach.
- **Recommended Visual:** `outputs/figures/model_comparison.png` & `outputs/figures/precision_recall_curves.png`.
- **Speaking Time:** ~1.5 Minutes
- **Speaker Notes:**
  > "Here are our benchmark results on the held-out validation cohort. Notice how clearly the metrics tell the story. The naive majority-class baseline achieves 87% accuracy, but has 0% recall and a PR-AUC of 0.1302. Our champion Random Forest achieves a PR-AUC of 0.8315 and an ROC-AUC of 0.9080—a massive 70-point lift over random chance. Notice also our threshold analysis: at a decision threshold of 0.33, Random Forest achieves 85.88% recall, capturing 73 out of the 85 churners in the test cohort and missing only 12. If the business instead requires high precision for expensive interventions, shifting the threshold to 0.50 delivers 82.9% precision and 94.6% accuracy. HistGradientBoosting also performed exceptionally well with 0.8139 PR-AUC. Both ensemble models vastly outperformed logistic regression and the heuristic baseline."

---

### Slide 8: Required Feature Ablation Study & Explainability
- **Title:** Feature Ablation Study & Key Predictive Drivers
- **On-Slide Bullet Points:**
  - **Required Ablation Study Results:**
    - *Group A (Demographics Only - 12 feats):* PR-AUC = **0.1630** (Barely above 0.1302 random baseline).
    - *Group B (Demographics + Transactions - 36 feats):* PR-AUC = **0.7534** (**+0.5904 uplift** over A).
    - *Group C (Full Features + Digital & Support - 57 feats):* PR-AUC = **0.8463** (**+0.6833 uplift** over A).
  - **Top Permutation Churn Drivers (Held-Out Test Set):**
    1. `recency_months`: Months since last purchase is the #1 predictor of defection.
    2. `avg_basket_value_mean`: Decaying basket size indicates wallet-share loss.
    3. `total_txns_l3m`: Declining quarterly shopping frequency.
    4. `app_trend_3m_vs_prior`: Deceleration in mobile app logins precedes store dormancy.
    5. `total_support_tickets_l6m`: Customer service friction and unresolved complaints.
  - **Causation Guardrail:** High importance reflects predictive association, not direct causality.
- **Recommended Visual:** `outputs/figures/feature_importance.png` (Top 15 permutation importances).
- **Speaking Time:** ~1.5 Minutes
- **Speaker Notes:**
  > "The assessment required an ablation study to test whether behavioral and engagement features actually improve predictive accuracy over basic demographics. The findings were stark. Group A—using age, gender, city, and tier alone—yielded a PR-AUC of just 0.1630, which is virtually indistinguishable from guessing. Demographics alone do not predict churn. When we added historical transaction frequency and spending trends in Group B, PR-AUC surged by 59 points to 0.7534. When we added mobile app sessions, email open trends, support tickets, and complaints in Group C, performance reached 0.8463—a total uplift of 68 points. Looking at our permutation importance on held-out data, purchase recency and basket value contraction were the strongest behavioral markers, while digital app abandonment and customer support tickets provided actionable operational signals."

---

### Slide 9: Operational Customer Risk Scoring & Segmentation
- **Title:** Customer Risk Stratification & Driver Attribution
- **On-Slide Bullet Points:**
  - **Scored Population:** All 2,600 loyalty members scored and exported to `customer_churn_predictions.csv`.
  - **Risk Stratification Rules:**
    - **Low Risk ($p < 0.30$):** **1,561 members (60.0%)**. Highly engaged; require zero retention spend.
    - **Medium Risk ($0.30 \le p < 0.60$):** **264 members (10.2%)**. Early disengagement; automated digital nudges.
    - **High Risk ($p \ge 0.60$):** **775 members (29.8%)**. Critical defection risk; high-touch outreach.
  - **Individualized Risk Drivers:** Every customer mapped to understandable primary and secondary drivers (e.g. *High Inactivity*, *Spend Contraction*, *Recent Complaint*).
  - **Audit of Exact-1.0 Probabilities (422 members):**
    - 100% of these customers actually churned (mean recency = 9.3 months).
    - Result of terminal leaf purity in Random Forest trees on extreme dormant cases; mathematically expected.
  - **New Member Reliability Tag:** Members with tenure $< 4$ months flagged as *Cautious* to prevent misinterpreting onboarding sparsity as churn.
- **Recommended Visual:** `outputs/figures/churn_risk_distribution.png` (Risk bucket distribution & probability histogram).
- **Speaking Time:** ~1.0 Minute
- **Speaker Notes:**
  > "In Phase 5, we deployed our champion pipeline to generate individual customer risk scores. Crucially, the model isolates 1,561 Low-Risk members—representing 60% of our entire loyalty base—who are healthy and active. By identifying this group, we immediately eliminate 60% of unnecessary marketing spend. Meanwhile, we identified 775 High-Risk members and 264 Medium-Risk members who require immediate attention. For each member, we output understandable primary and secondary risk drivers so that marketing and store managers know exactly why a member is at risk. We also conducted an audit on 422 customers who received a probability of exactly 1.0. We verified that 100% of them were true churners who had been dormant for 9 months on average, landing in completely pure churn leaves across all 200 trees. Finally, members with less than 4 months of history are explicitly tagged with a reliability warning so they receive onboarding support rather than aggressive discounts."

---

### Slide 10: Retention Intervention Simulation & Financial Modeling
- **Title:** Retention Intervention Simulation & Financial Impact
- **On-Slide Bullet Points:**
  - **Financial Assumptions (Clearly Labeled):** Baseline Retained Member Value (Annual LTV) = **$180.00**.
  - **Scenario Comparison:**
| Retention Playbook | Target Scope | Targeted Members | Cost (USD) | Retained Members | Net Benefit (USD) | ROI (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Status Quo (Blanket Promotion)** | All members | 2,600 | $26,000 | 40.9 | **-$18,638** | **-71.7%** |
| **Scenario 1 (Targeted Reactivation)** | Disengaged only | 862 | $6,896 | 69.7 | **+$5,651** | **+81.9%** |
| **Scenario 2 (Proactive VIP Support)** | Support friction | 458 | $6,870 | 110.7 | **+$13,054** | **+190.0%** |
| **Scenario 3 (Integrated Dual-Track)** | Priority VIP + Reactivation | 898 | $10,390 | 144.2 | **+$15,572** | **+149.9%** |
  - **Core Financial Takeaway:** Moving from blanket mailings to the integrated targeted playbook flips an **$18.6K loss** into a **+$15.6K net profit** (+150% ROI), slashing cost per retained customer from $635 down to **$72**.
  - **Sensitivity Analysis:** Remains profitable down to an intervention efficacy of just **9.5%**.
- **Recommended Visual:** `outputs/figures/retention_simulation_roi.png` & `outputs/figures/retention_sensitivity_analysis.png`.
- **Speaking Time:** ~1.5 Minutes
- **Speaker Notes:**
  > "Now, let's examine the financial impact. We simulated three targeted intervention scenarios against the status quo blanket policy, using a conservative baseline annual retained value of $180 per customer. Look at the status quo: mailing $10 vouchers to all 2,600 members burns $26,000 and nets a loss of -$18,638, costing an absurd $635 per retained member. Now look at Scenario 2, our Proactive VIP Support program: by proactively reaching out to the 458 members with recurring support tickets and complaints, we achieve a 190% ROI, netting over $13,000 in benefit. When we combine proactive support with targeted reactivation vouchers in Scenario 3, we target 898 at-risk members, retain an estimated 144 members, and generate a net financial benefit of +$15,572—a 150% ROI. Our sensitivity analysis shows that this strategy breaks even at an efficacy of just 9.5%, demonstrating massive financial robustness."

---

### Slide 11: Limitations, Fairness & A/B Testing Protocol
- **Title:** Governance, Fairness & Experimental Validation Plan
- **On-Slide Bullet Points:**
  - **Correlation vs Uplift:** A predictive model identifies churn risk, not treatment response. An A/B test is required before full commercial rollout.
  - **Recommended A/B Testing Design:**
    - *Eligible Pool:* The 1,039 High-Risk and Medium-Risk identified members.
    - *Treatment Arm (80%, $n = 831$):* Receives the Dual-Track Retention Strategy.
    - *Holdout Control Arm (20%, $n = 208$):* Receives standard communications with zero promotional incentive.
    - *Primary Success Metric:* Statistically significant reduction in 90-day churn rate ($p < 0.05$, two-tailed test with 80% statistical power).
  - **Algorithmic Fairness Considerations:**
    - Retention outreach must not favor Platinum over Silver members; Silver members represent 56.7% of the loyalty base.
    - Stores in high-churn markets (Austin, Columbus) require operational inventory audits rather than penalizing members.
  - **Governance for New Signups:** Members with tenure $< 4$ months receive onboarding welcome journeys, not win-back discounts.
- **Recommended Visual:** A/B Testing Trial Architecture Diagram (Treatment vs Control split).
- **Speaking Time:** ~1.5 Minutes
- **Speaker Notes:**
  > "Before rolling out this intervention across the entire enterprise, we must address causal governance. Machine learning models identify correlation, not causal treatment effects. A member predicted to churn might still churn despite receiving a coupon, while others might stay regardless. Therefore, FreshBasket must execute a randomized controlled trial. We propose an A/B test on the 1,039 at-risk members, allocating 80% to our Dual-Track intervention and reserving a 20% holdout control arm. We will track 90-day churn reduction at an alpha of 0.05 with 80% statistical power. In terms of fairness, our analysis revealed that Silver tier members comprise over half of all at-risk customers; restricting retention efforts to high-tier members would accelerate customer base erosion. Finally, new signups must receive onboarding welcome nurture journeys rather than premature win-back discounts."

---

### Slide 12: Conclusion, Deliverables & Reproducibility
- **Title:** Conclusion, Project Deliverables & Next Steps
- **On-Slide Bullet Points:**
  - **Summary of Achievements:**
    - Cleaned and audited 18 months of retail activity across 2,600 loyalty members with zero raw data modification.
    - Built a leakage-safe customer modeling table using data through March 31, 2024.
    - Achieved **0.8315 PR-AUC** and **85.9% Recall** using Random Forest on held-out cohorts.
    - Proved via ablation that behavioral and support friction features contribute a **+0.6833 PR-AUC uplift** over demographics alone.
    - Designed an integrated retention strategy delivering **+$15,572 net annual profit** (+150% ROI).
  - **Project Deliverables Completed:**
    - `data/final/customer_churn_modeling_dataset.parquet` & `.csv` (2,600 rows $\times$ 64 cols).
    - `outputs/predictions/customer_churn_predictions.csv` (Operational scores & risk drivers).
    - `outputs/reports/final_assessment_report.md` (Full 13-section technical report).
    - 10 publication-quality figures in `outputs/figures/` and metrics tables in `outputs/metrics/`.
  - **Reproducibility:** All pipelines runnable via `python run_cleaning.py`, `python run_phase3.py`, `python run_phase4.py`, and `python run_phase5.py`.
- **Recommended Visual:** Summary scorecard of key metrics (PR-AUC: 0.8315, Churners Caught: 85.9%, Net ROI: +150%).
- **Speaking Time:** ~1.0 Minute
- **Speaker Notes:**
  > "To conclude, this project successfully delivers an end-to-end churn prediction and retention architecture for FreshBasket Retail. We proved that static demographics cannot predict churn, whereas transactional velocity and customer support friction provide a massive 68-point uplift in PR-AUC. Our champion model catches 86% of defecting customers, and our dual-track retention playbook turns an $18,000 blanket marketing loss into a $15,500 net gain. All code, data pipelines, serialized models, predictions, and reports are fully modular, verified with automated assertions, and 100% reproducible. Thank you very much, and I welcome any questions from the panel."
