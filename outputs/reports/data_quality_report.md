# Data Quality Audit & Cleaning Report
**Project:** Customer Churn Prediction for Retail Loyalty Program  
**Dataset:** FreshBasket Retail Loyalty Dataset  
**Phase:** Phase 2 - Data Cleaning & Quality Audit  

---

## 1. Executive Summary
This report documents the end-to-end data audit and cleaning process applied to the FreshBasket Loyalty Churn dataset. All transformations are reproducible, deterministic, and follow senior data science best practices. The raw Excel workbook was preserved untouched, and cleaned tables are stored under `data/processed/`.

---

## 2. Table-by-Table Cleaning Results

### A. Customer Profile (`fb Customer Profile`)
- **Raw Dimensions:** 2608 rows, 10 columns
- **Exact Duplicates Removed:** 8 duplicate rows removed.
- **Unique Customers Retained:** 2600 unique customers (100% of loyalty population).
- **Categorical Normalization:**
  - `CITY`: Standardized from **40 dirty string variations** down to **10 clean cities** via title-casing and whitespace trimming.
  - `MEMBERSHIP_TIER`: Standardized from mixed casing/whitespace variations into exactly 3 tiers: {'Silver': 1474, 'Gold': 824, 'Platinum': 302}.
- **Missing Demographic Value Imputation & Recovery:**
  - `HOME_STORE_PRICE_TIER`: **130 missing values** (4.98%). Store-level audit proved that each store ID has an invariant price tier. All 130 missing values were **100% recovered deterministically** via `PREFERRED_STORE_ID` mapping (0 unresolved).
  - `AGE`: **78 missing values** (2.99%). Imputed using the median age (46.0 years). Added binary indicator `AGE_IS_MISSING = 1` to preserve missingness signal.
  - `GENDER`: **52 missing values** (1.99%). Imputed as `'Unknown'`. Added binary indicator `GENDER_IS_MISSING = 1`.

---

### B. Monthly Activity (`fb Monthly Activity`)
- **Raw Dimensions:** 28073 rows, 12 columns
- **Negative Spend Investigation:**
  - **24 records** had `TOTAL_SPEND < 0`.
  - Audit revealed that for all 24 records, `AVG_BASKET_VALUE` was positive and equal to `|TOTAL_SPEND| / TRANSACTIONS`. This proves the negative sign was a data-entry sign error rather than a refund or return.
  - **Treatment:** Corrected using `TOTAL_SPEND = TOTAL_SPEND.abs()`.
- **Duplicate Records Removed:**
  - **60 duplicate rows** on `(CUSTOMER_ID, MONTH)` were removed.
- **Reconstruction of Missing Spend:**
  - **20 records** had `TOTAL_SPEND == 0` despite having `TRANSACTIONS > 0` and `AVG_BASKET_VALUE > 0`.
  - **Treatment:** Deterministically reconstructed spend as `TRANSACTIONS * AVG_BASKET_VALUE`.
- **Preservation of Inactive Months:**
  - **2330 records** with zero transactions and zero spend were **strictly preserved**. These represent inactive months containing vital app sessions, email opens, and support ticket complaints preceding churn.
- **Leakage Prevention Partitioning:**
  - Total records in predictive window (`<= 2024-03-31`): **23253** (15 months).
  - Total records in outcome window (`>= 2024-04-01`): **4760** (3 months).
  - Explicit boolean flag `IS_PREDICTIVE_WINDOW` added to prevent data leakage during feature engineering.

---

### C. Churn Label (`fb Churn Label`)
- **Raw Dimensions:** 2605 rows, 6 columns
- **Exact Duplicates Removed:** 5 duplicate rows removed.
- **Unique Customers Retained:** 2600 customers (matches Customer Profile perfectly).
- **Target Distribution (`CHURNED`):**
  - Active (`0`): **1782** (68.54%)
  - Churned (`1`): **818** (31.46%)
- **Missing `LAST_PURCHASE_DATE` Investigation:**
  - **120 records** have null `LAST_PURCHASE_DATE`.
  - Verification confirmed that all 120 members had 0 lifetime transactions in the observation window (signed up late or never made a purchase). All 120 are correctly marked `CHURNED = 0` per definition.
  - Added explicit indicator `NEVER_PURCHASED_FLAG = 1`.
- **Insufficient History Flag (`INSUFFICIENT_HISTORY_FLAG`):**
  - Fully preserved: **373 members** have fewer than 4 months of history (14.35%).

---

## 3. Data Integrity & Validation Checklist
All automated validation checks passed:

| Validation Check | Description | Status |
| :--- | :--- | :---: |
| `profile_customer_id_unique` | Every row in Customer Profile has a unique ID | **PASS** |
| `profile_no_null_customer_id` | Zero missing customer IDs | **PASS** |
| `profile_no_null_price_tier` | 100% price tiers recovered from store IDs | **PASS** |
| `profile_cities_standardized_to_10` | Exact 10 clean metropolitan markets | **PASS** |
| `profile_tiers_standardized_to_3` | Silver, Gold, Platinum standardized | **PASS** |
| `activity_no_duplicate_customer_month` | Zero duplicate customer-month pairs | **PASS** |
| `activity_no_negative_spend` | Zero negative spend records remaining | **PASS** |
| `activity_all_months_18` | All 18 calendar months present | **PASS** |
| `activity_predictive_window_flag_present` | Leakage guard flag populated | **PASS** |
| `churn_customer_id_unique` | Zero duplicate labels | **PASS** |
| `churn_has_2600_customers` | Exactly 2,600 unique members matching Profile | **PASS** |
| `churn_target_binary_only` | Target strictly contains binary values (0, 1) | **PASS** |

---
**Report generated by Phase 2 Data Cleaning Pipeline.**
