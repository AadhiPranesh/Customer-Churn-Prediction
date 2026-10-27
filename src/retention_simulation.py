"""
Retention Intervention Simulation and Financial Modeling Module.
Simulates targeted business interventions, calculates incremental ROI,
conducts sensitivity analysis, and generates comparative visualizations.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COL

METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"


def run_retention_simulation(
    df_scored: pd.DataFrame,
    baseline_ltv: float = 180.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate business retention scenarios using clearly labeled operational assumptions.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Target Audience Segmentation
    # -------------------------------------------------------------
    # At-risk candidates identified by champion model
    high_med_risk = df_scored["CHURN_PROBABILITY"] >= 0.33
    
    # Disengagement subgroup: High recency or severe spend contraction
    disengaged_mask = high_med_risk & (
        (df_scored["recency_months"] >= 2) | (df_scored["spend_trend_3m_vs_prior"] < 0.60)
    )
    
    # Service friction subgroup: High support tickets or formal complaints
    service_mask = high_med_risk & (
        (df_scored["total_support_tickets_15m"] >= 2) | (df_scored["had_recent_complaint_flag"] == 1)
    )
    
    # Dual-track mutually exclusive assignment
    # Service friction members get priority VIP support; remaining disengaged get reactivation offers
    vip_support_mask = service_mask
    reactivation_mask = disengaged_mask & (~service_mask)
    
    # -------------------------------------------------------------
    # 2. Scenario Assumptions & Economics
    # -------------------------------------------------------------
    # Status Quo: Blanket voucher to ALL 2,600 members
    cost_blanket_per_cust = 10.0
    total_blanket_target = len(df_scored) # 2600
    cost_blanket = total_blanket_target * cost_blanket_per_cust # $26,000
    # Blanket offers have low response and mostly reach non-churners; efficacy = 5% on true churners
    retained_blanket = 818 * 0.05
    gross_blanket = retained_blanket * baseline_ltv
    net_blanket = gross_blanket - cost_blanket
    roi_blanket = (net_blanket / cost_blanket) * 100
    
    # Scenario 1: Targeted Disengagement Reactivation Offer
    # Personalized 15% preferred category discount + app push
    cost_s1_per_cust = 8.0
    n_s1 = int(disengaged_mask.sum())
    cost_s1 = n_s1 * cost_s1_per_cust
    # Assumptions: 45% redemption response, 20% relative churn reduction among responders
    retained_s1 = float((df_scored.loc[disengaged_mask, "CHURN_PROBABILITY"] * 0.45 * 0.20).sum())
    gross_s1 = retained_s1 * baseline_ltv
    net_s1 = gross_s1 - cost_s1
    roi_s1 = (net_s1 / cost_s1) * 100
    
    # Scenario 2: Proactive VIP Support & Concierge Resolution
    # Dedicated CSR outreach + $15 compensation credit for members with support friction
    cost_s2_per_cust = 15.0
    n_s2 = int(service_mask.sum())
    cost_s2 = n_s2 * cost_s2_per_cust
    # Assumptions: 75% contact rate, 35% relative churn reduction among reached members
    retained_s2 = float((df_scored.loc[service_mask, "CHURN_PROBABILITY"] * 0.75 * 0.35).sum())
    gross_s2 = retained_s2 * baseline_ltv
    net_s2 = gross_s2 - cost_s2
    roi_s2 = (net_s2 / cost_s2) * 100
    
    # Scenario 3: Integrated Dual-Track Retention Strategy (Recommended)
    # VIP outreach for service friction members + reactivation offers for remaining disengaged members
    n_vip = int(vip_support_mask.sum())
    n_react = int(reactivation_mask.sum())
    cost_s3 = (n_vip * cost_s2_per_cust) + (n_react * cost_s1_per_cust)
    retained_s3_vip = float((df_scored.loc[vip_support_mask, "CHURN_PROBABILITY"] * 0.75 * 0.35).sum())
    retained_s3_react = float((df_scored.loc[reactivation_mask, "CHURN_PROBABILITY"] * 0.45 * 0.20).sum())
    retained_s3 = retained_s3_vip + retained_s3_react
    gross_s3 = retained_s3 * baseline_ltv
    net_s3 = gross_s3 - cost_s3
    roi_s3 = (net_s3 / cost_s3) * 100
    
    scenarios = [
        {
            "Scenario_Name": "Status Quo (Blanket Promotion)",
            "Targeted_Members": total_blanket_target,
            "Target_Share_Pct": round((total_blanket_target / len(df_scored)) * 100, 1),
            "Campaign_Cost_USD": round(cost_blanket, 2),
            "Estimated_Retained_Members": round(retained_blanket, 1),
            "Estimated_Gross_Benefit_USD": round(gross_blanket, 2),
            "Estimated_Net_Benefit_USD": round(net_blanket, 2),
            "ROI_Pct": round(roi_blanket, 1),
            "Cost_Per_Retained_Member": round(cost_blanket / retained_blanket, 2)
        },
        {
            "Scenario_Name": "Scenario 1 (Targeted Reactivation)",
            "Targeted_Members": n_s1,
            "Target_Share_Pct": round((n_s1 / len(df_scored)) * 100, 1),
            "Campaign_Cost_USD": round(cost_s1, 2),
            "Estimated_Retained_Members": round(retained_s1, 1),
            "Estimated_Gross_Benefit_USD": round(gross_s1, 2),
            "Estimated_Net_Benefit_USD": round(net_s1, 2),
            "ROI_Pct": round(roi_s1, 1),
            "Cost_Per_Retained_Member": round(cost_s1 / retained_s1, 2)
        },
        {
            "Scenario_Name": "Scenario 2 (Proactive Support Concierge)",
            "Targeted_Members": n_s2,
            "Target_Share_Pct": round((n_s2 / len(df_scored)) * 100, 1),
            "Campaign_Cost_USD": round(cost_s2, 2),
            "Estimated_Retained_Members": round(retained_s2, 1),
            "Estimated_Gross_Benefit_USD": round(gross_s2, 2),
            "Estimated_Net_Benefit_USD": round(net_s2, 2),
            "ROI_Pct": round(roi_s2, 1),
            "Cost_Per_Retained_Member": round(cost_s2 / retained_s2, 2)
        },
        {
            "Scenario_Name": "Scenario 3 (Integrated Dual-Track Strategy)",
            "Targeted_Members": n_vip + n_react,
            "Target_Share_Pct": round(((n_vip + n_react) / len(df_scored)) * 100, 1),
            "Campaign_Cost_USD": round(cost_s3, 2),
            "Estimated_Retained_Members": round(retained_s3, 1),
            "Estimated_Gross_Benefit_USD": round(gross_s3, 2),
            "Estimated_Net_Benefit_USD": round(net_s3, 2),
            "ROI_Pct": round(roi_s3, 1),
            "Cost_Per_Retained_Member": round(cost_s3 / retained_s3, 2)
        }
    ]
    
    df_scenarios = pd.DataFrame(scenarios)
    df_scenarios.to_csv(METRICS_DIR / "retention_scenario_comparison.csv", index=False)
    
    # -------------------------------------------------------------
    # 3. Sensitivity Analysis Matrix
    # -------------------------------------------------------------
    # Test varying intervention efficacy (10% to 30%) and LTV ($100 to $250) for Scenario 3
    efficacies = [0.10, 0.15, 0.20, 0.25, 0.30]
    ltv_values = [100.0, 150.0, 180.0, 220.0, 250.0]
    
    sens_rows = []
    for eff in efficacies:
        for val in ltv_values:
            # Scale retention by efficacy ratio relative to base (0.20 base efficacy for react, 0.35 base for vip)
            scale = eff / 0.20
            sens_retained = retained_s3 * scale
            sens_gross = sens_retained * val
            sens_net = sens_gross - cost_s3
            sens_roi = (sens_net / cost_s3) * 100
            sens_rows.append({
                "Intervention_Efficacy": f"{int(eff*100)}%",
                "Assumed_LTV_USD": val,
                "Retained_Members": round(sens_retained, 1),
                "Gross_Benefit_USD": round(sens_gross, 2),
                "Net_Benefit_USD": round(sens_net, 2),
                "ROI_Pct": round(sens_roi, 1)
            })
            
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(METRICS_DIR / "retention_sensitivity_matrix.csv", index=False)
    
    return df_scenarios, df_sens


def plot_retention_visualizations(
    df_scored: pd.DataFrame,
    df_scenarios: pd.DataFrame,
    df_sens: pd.DataFrame
) -> Dict[str, Path]:
    """Generate high-resolution charts for predicted risk distribution and simulation ROI."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated = {}
    
    # 1. Churn Risk Distribution & Buckets
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Histogram of probabilities
    sns.histplot(df_scored["CHURN_PROBABILITY"], bins=25, kde=True, color="#2980b9", ax=axes[0], edgecolor="black")
    axes[0].axvline(0.30, color="#f39c12", linestyle="--", linewidth=1.5, label="Medium Risk Threshold (0.30)")
    axes[0].axvline(0.60, color="#e74c3c", linestyle="--", linewidth=1.5, label="High Risk Threshold (0.60)")
    axes[0].set_title("Customer Churn Probability Distribution", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Predicted Churn Probability")
    axes[0].set_ylabel("Customer Count")
    axes[0].legend(loc="upper right")
    
    # Risk Bucket Count Bar
    bucket_counts = df_scored["RISK_BUCKET"].value_counts()[["Low Risk", "Medium Risk", "High Risk"]]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = axes[1].bar(bucket_counts.index, bucket_counts.values, color=colors, edgecolor="black", width=0.5, alpha=0.85)
    axes[1].set_title("Operational Risk Bucket Allocation (2,600 Members)", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Customer Count")
    axes[1].set_ylim(0, 1800)
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(f"{h:,}\n({h/len(df_scored)*100:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")
                         
    plt.suptitle("Model Inference: Loyalty Member Churn Risk Stratification", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p1 = FIGURES_DIR / "churn_risk_distribution.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    generated["risk_distribution"] = p1
    
    # 2. Retention Simulation Net Financial Benefit
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    names = df_scenarios["Scenario_Name"].tolist()
    net_vals = df_scenarios["Estimated_Net_Benefit_USD"].tolist()
    bar_colors = ["#e74c3c" if v < 0 else "#27ae60" for v in net_vals]
    
    bars = ax.barh(names, net_vals, color=bar_colors, edgecolor="black", height=0.5, alpha=0.85)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_title("Estimated Net Financial Benefit by Retention Strategy\n(Hypothetical LTV = $180)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Estimated Net Benefit (USD)", fontsize=11)
    
    for bar in bars:
        w = bar.get_width()
        offset = 500 if w >= 0 else -3000
        sign = "+" if w > 0 else ""
        ax.annotate(f"{sign}${w:,.0f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(offset, 0), textcoords="offset points",
                    ha="left", va="center", fontweight="bold", fontsize=10)
                    
    plt.subplots_adjust(left=0.35, right=0.92, top=0.88, bottom=0.12)
    p2 = FIGURES_DIR / "retention_simulation_roi.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    generated["retention_roi"] = p2

    # 3. Sensitivity Analysis Plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    for ltv in [100.0, 150.0, 180.0, 220.0, 250.0]:
        sub = df_sens[df_sens["Assumed_LTV_USD"] == ltv]
        ax.plot(sub["Intervention_Efficacy"], sub["Net_Benefit_USD"], marker="o", linewidth=2.0, label=f"LTV = ${int(ltv)}")
        
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2, label="Break-Even ($0 Net)")
    ax.set_title("Sensitivity Analysis: Net Financial Benefit vs Intervention Efficacy (Scenario 3)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Intervention Efficacy (% Relative Churn Reduction)", fontsize=11)
    ax.set_ylabel("Estimated Net Benefit (USD)", fontsize=11)
    ax.legend(title="Customer LTV", loc="upper left")
    
    plt.tight_layout()
    p3 = FIGURES_DIR / "retention_sensitivity_analysis.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    generated["sensitivity_analysis"] = p3

    return generated
