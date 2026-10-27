"""
Script to generate high-resolution publication-grade EDA visualizations
for the Customer Churn Prediction project.
"""

from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import FIGURES_DIR


# Set global aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

COLORS = {"Active": "#2ecc71", "Churned": "#e74c3c", "Primary": "#3498db", "Dark": "#2c3e50"}


def generate_all_eda_charts(df: pd.DataFrame) -> Dict[str, Path]:
    """Generate and save all 7 EDA visualizations."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated = {}
    
    # -------------------------------------------------------------
    # 1. Churn Distribution & Class Balance
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    counts = df["CHURNED"].value_counts().sort_index()
    labels = ["Active (0)", "Churned (1)"]
    palette = [COLORS["Active"], COLORS["Churned"]]
    
    # Bar Chart
    bars = axes[0].bar(labels, counts, color=palette, width=0.5, edgecolor="black", alpha=0.85)
    axes[0].set_title("Customer Churn Volume", fontweight="bold")
    axes[0].set_ylabel("Number of Customers")
    axes[0].set_ylim(0, 2100)
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f"{h:,}\n({h/len(df)*100:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")
    
    # Donut Chart
    axes[1].pie(counts, labels=labels, autopct="%1.1f%%", startangle=90,
                colors=palette, explode=(0, 0.05),
                wedgeprops=dict(width=0.4, edgecolor="black"))
    axes[1].set_title("Class Proportion (31.5% Churn Rate)", fontweight="bold")
    
    plt.suptitle("FreshBasket Loyalty Program: Overall Churn Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p1 = FIGURES_DIR / "churn_distribution.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    generated["churn_distribution"] = p1
    
    # -------------------------------------------------------------
    # 2. Churn Rate by Membership Tier
    # -------------------------------------------------------------
    tier_stats = df.groupby("MEMBERSHIP_TIER")["CHURNED"].agg(["count", "mean"]).reset_index()
    tier_stats["churn_pct"] = tier_stats["mean"] * 100
    tier_stats = tier_stats.sort_values("churn_pct", ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    bars = ax1.bar(tier_stats["MEMBERSHIP_TIER"], tier_stats["churn_pct"], color="#3498db", width=0.45, edgecolor="black", alpha=0.85)
    ax1.set_title("Churn Rate by Loyalty Membership Tier", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Churn Rate (%)", fontsize=11)
    ax1.set_ylim(0, 45)
    
    # Overlay benchmark line
    overall_rate = df["CHURNED"].mean() * 100
    ax1.axhline(overall_rate, color="red", linestyle="--", linewidth=1.5, label=f"Overall Average ({overall_rate:.1f}%)")
    
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold")
                     
    ax1.legend(loc="upper right")
    plt.tight_layout()
    p2 = FIGURES_DIR / "churn_by_tier.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    generated["churn_by_tier"] = p2

    # -------------------------------------------------------------
    # 3. Churn Rate by City Market
    # -------------------------------------------------------------
    city_stats = df.groupby("CITY")["CHURNED"].agg(["count", "mean"]).reset_index()
    city_stats["churn_pct"] = city_stats["mean"] * 100
    city_stats = city_stats.sort_values("churn_pct", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(city_stats["CITY"], city_stats["churn_pct"], color="#16a085", height=0.55, edgecolor="black", alpha=0.85)
    ax.set_title("Churn Rate Across 10 Regional City Markets", fontsize=13, fontweight="bold")
    ax.set_xlabel("Churn Rate (%)", fontsize=11)
    ax.set_xlim(0, 45)
    ax.axvline(overall_rate, color="red", linestyle="--", linewidth=1.5, label=f"Overall Average ({overall_rate:.1f}%)")
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontweight="bold")
                    
    ax.legend(loc="lower right")
    plt.tight_layout()
    p3 = FIGURES_DIR / "churn_by_city.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    generated["churn_by_city"] = p3

    # -------------------------------------------------------------
    # 4. Churn Rate by Signup Cohort
    # -------------------------------------------------------------
    cohort_stats = df.groupby("signup_cohort_quarter")["CHURNED"].agg(["count", "mean"]).reset_index()
    cohort_stats["churn_pct"] = cohort_stats["mean"] * 100
    cohort_stats = cohort_stats.sort_values("signup_cohort_quarter")
    
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cohort_stats["signup_cohort_quarter"], cohort_stats["churn_pct"], marker="o", color="#8e44ad", linewidth=2.5, markersize=7)
    ax.set_title("Churn Rate by Quarterly Signup Cohort (Tenure & Cohort Survival)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Member Signup Quarter")
    ax.set_ylim(-2, 50)
    ax.axhline(overall_rate, color="red", linestyle="--", linewidth=1.2, label=f"Overall Average ({overall_rate:.1f}%)")
    plt.xticks(rotation=45)
    
    for _, row in cohort_stats.iterrows():
        ax.annotate(f"{row['churn_pct']:.1f}%",
                    xy=(row["signup_cohort_quarter"], row["churn_pct"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold")
                    
    ax.legend(loc="upper right")
    plt.tight_layout()
    p4 = FIGURES_DIR / "churn_by_signup_cohort.png"
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close()
    generated["churn_by_signup_cohort"] = p4

    # -------------------------------------------------------------
    # 5. Purchase Recency & Inactive Months
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.boxplot(data=df, x="CHURNED", y="recency_months", hue="CHURNED", ax=axes[0], palette=[COLORS["Active"], COLORS["Churned"]], legend=False)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Active (0)", "Churned (1)"])
    axes[0].set_title("Purchase Recency (Months Since Last Purchase)", fontweight="bold")
    axes[0].set_ylabel("Months (as of Cutoff 2024-03-31)")
    
    sns.boxplot(data=df, x="CHURNED", y="zero_trans_months_count_15m", hue="CHURNED", ax=axes[1], palette=[COLORS["Active"], COLORS["Churned"]], legend=False)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Active (0)", "Churned (1)"])
    axes[1].set_title("Dormancy: Zero-Transaction Months Count (15M)", fontweight="bold")
    axes[1].set_ylabel("Count of Inactive Months")
    
    plt.suptitle("Behavioral Signal: Dormancy and Purchase Recency Preceding Churn", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p5 = FIGURES_DIR / "churn_vs_recency_and_frequency.png"
    plt.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close()
    generated["churn_vs_recency"] = p5

    # -------------------------------------------------------------
    # 6. Customer Support Friction & Complaints vs Churn
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Support tickets binned
    df["ticket_bucket"] = pd.cut(df["total_support_tickets_15m"], bins=[-1, 0, 2, 5, 20], labels=["0 Tickets", "1-2 Tickets", "3-5 Tickets", "6+ Tickets"])
    ticket_churn = df.groupby("ticket_bucket", observed=False)["CHURNED"].agg(["count", "mean"]).reset_index()
    ticket_churn["churn_pct"] = ticket_churn["mean"] * 100
    
    axes[0].bar(ticket_churn["ticket_bucket"], ticket_churn["churn_pct"], color="#e67e22", width=0.5, edgecolor="black", alpha=0.85)
    axes[0].set_title("Churn Rate by Lifetime Support Tickets (15M)", fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_ylim(0, 60)
    for idx, row in ticket_churn.iterrows():
        axes[0].annotate(f"{row['churn_pct']:.1f}%\n(n={row['count']})",
                         xy=(idx, row["churn_pct"]),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold", fontsize=9)
                         
    # Complaint flag
    complaint_churn = df.groupby("had_recent_complaint_flag")["CHURNED"].agg(["count", "mean"]).reset_index()
    complaint_churn["churn_pct"] = complaint_churn["mean"] * 100
    
    axes[1].bar(["No Recent Complaint", "Had Complaint (Last 3M)"], complaint_churn["churn_pct"], color=["#3498db", "#c0392b"], width=0.45, edgecolor="black", alpha=0.85)
    axes[1].set_title("Churn Rate by Recent Complaint Status (Jan-Mar 2024)", fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_ylim(0, 60)
    for idx, row in complaint_churn.iterrows():
        axes[1].annotate(f"{row['churn_pct']:.1f}%\n(n={row['count']})",
                         xy=(idx, row["churn_pct"]),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold", fontsize=9)
                         
    plt.suptitle("Support Friction as a Churn Driver", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p6 = FIGURES_DIR / "churn_vs_support_and_complaints.png"
    plt.savefig(p6, dpi=300, bbox_inches="tight")
    plt.close()
    generated["churn_vs_support"] = p6

    # -------------------------------------------------------------
    # 7. Correlation Heatmap
    # -------------------------------------------------------------
    key_cols = [
        "CHURNED",
        "recency_months",
        "zero_trans_months_count_15m",
        "total_support_tickets_15m",
        "had_recent_complaint_flag",
        "total_txns_l3m",
        "total_spend_l3m",
        "total_app_sessions_l3m",
        "total_emails_opened_l3m",
        "purchased_in_last_3m_flag",
        "active_months_ratio",
        "customer_tenure_months"
    ]
    
    corr_matrix = df[key_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.8, vmax=0.8, cbar=True, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap: Key Predictive Drivers vs Churn Outcome", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p7 = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(p7, dpi=300, bbox_inches="tight")
    plt.close()
    generated["correlation_heatmap"] = p7

    return generated
