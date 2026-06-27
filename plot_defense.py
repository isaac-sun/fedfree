#!/usr/bin/env python3
"""
IEEE-Style Defense Visualization for FL Free-Rider Detection.

Reads results/defense_history.csv and generates three publication-quality charts:

  Chart 1 — Defense Separation Scatter (positive_sum × variance)
  Chart 2 — Fingerprint Bar Chart (variance comparison)
  Chart 3 — Convergence Curves (Macro-F1 over rounds)

All figures are saved as 300 DPI PDFs in results/plots/.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════════════
# IEEE Style Setup
# ═══════════════════════════════════════════════════════════════════════════════

IEEE_COLORS = {
    "honest": "#2166AC",        # deep blue
    "attacker": "#B2182B",      # deep red
    "baseline": "#4D4D4D",      # dark gray
    "defense": "#1B7837",       # dark green
    "threshold": "#B2182B",     # red dashed
    "honest_edge": "#053061",
    "attacker_edge": "#67001F",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
    "text.usetex": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    path_png = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Defense Separation Scatter Plot
# ═══════════════════════════════════════════════════════════════════════════════

def chart_scatter(
    df: pd.DataFrame,
    pos_threshold: float = 0.01,
    var_threshold: float = 0.001,
):
    """Scatter: positive_sum (X) vs variance (Y), colored by attacker status.

    Two dashed threshold lines mark the defense decision boundaries.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    honest = df[~df["is_attacker"]]
    attackers = df[df["is_attacker"]]

    # Scatter
    ax.scatter(
        honest["positive_sum"], honest["variance"],
        c=IEEE_COLORS["honest"], s=30, marker="o", alpha=0.7,
        edgecolors=IEEE_COLORS["honest_edge"], linewidth=0.3,
        label="Honest Clients", zorder=3,
    )
    ax.scatter(
        attackers["positive_sum"], attackers["variance"],
        c=IEEE_COLORS["attacker"], s=40, marker="^", alpha=0.8,
        edgecolors=IEEE_COLORS["attacker_edge"], linewidth=0.3,
        label="Free-Rider Attackers", zorder=3,
    )

    # Threshold lines
    ax.axvline(x=pos_threshold, color=IEEE_COLORS["threshold"],
               linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Pos-Sum Threshold = {pos_threshold}")
    ax.axhline(y=var_threshold, color=IEEE_COLORS["threshold"],
               linestyle=":", linewidth=1.2, alpha=0.7,
               label=f"Variance Threshold = {var_threshold}")

    # Shaded defense zones
    ax.axvspan(0, pos_threshold, alpha=0.06, color="red", zorder=0)
    ax.axhspan(0, var_threshold, xmin=0, xmax=pos_threshold / (ax.get_xlim()[1] or 1),
               alpha=0.08, color="orange", zorder=0)

    ax.set_xlabel("Positive Per-Class SV Sum  $\\Sigma_c \\max(SV_{i,c}, 0)$")
    ax.set_ylabel("Per-Class SV Variance  $\\mathrm{Var}_c(SV_{i,c})$")
    ax.set_title("Defense Separation: Honest vs Free-Rider Clients")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray", fontsize=7.5)
    ax.grid(True, alpha=0.2)

    # Annotation
    ax.text(0.98, 0.06,
            "Shaded region = Defense detection zone\n"
            "(low positive sum & low variance → suspected free-rider)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, fontstyle="italic", color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    _save(fig, "fig1_defense_scatter")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Variance Fingerprint Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def chart_fingerprint(df: pd.DataFrame, var_threshold: float = 0.001):
    """Grouped bar chart: variance distribution of typical honest vs attacker clients."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate: mean variance per client across all rounds
    client_stats = df.groupby(["client_id", "is_attacker"])["variance"].mean().reset_index()
    client_stats = client_stats.sort_values("variance", ascending=False)

    honest_stats = client_stats[~client_stats["is_attacker"]]
    attacker_stats = client_stats[client_stats["is_attacker"]]

    # Plot bars
    x_h = np.arange(len(honest_stats))
    x_a = np.arange(len(attacker_stats)) + len(honest_stats) + 1  # gap between groups

    bars_h = ax.bar(
        x_h, honest_stats["variance"].values,
        color=IEEE_COLORS["honest"], alpha=0.85, edgecolor=IEEE_COLORS["honest_edge"],
        linewidth=0.5, label="Honest Clients",
    )
    bars_a = ax.bar(
        x_a, attacker_stats["variance"].values,
        color=IEEE_COLORS["attacker"], alpha=0.85, edgecolor=IEEE_COLORS["attacker_edge"],
        linewidth=0.5, label="Free-Rider Attackers",
    )

    # Threshold line
    ax.axhline(y=var_threshold, color=IEEE_COLORS["threshold"],
               linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Detection Threshold = {var_threshold}")

    # Labels
    all_labels = [f"C{c}" for c in honest_stats["client_id"].values] + \
                 [f"A{c}" for c in attacker_stats["client_id"].values]
    all_x = np.concatenate([x_h, x_a])
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=45, fontsize=8)

    ax.set_ylabel("Mean Per-Class SV Variance  $\\mathrm{Var}_c(SV_{i,c})$")
    ax.set_xlabel("Client ID  (C = Honest,  A = Attacker)")
    ax.set_title("Variance Fingerprint: Honest vs Free-Rider Clients")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)

    # Annotation
    ax.text(0.98, 0.95,
            "Honest clients exhibit high variance\n(uneven class contributions).\n"
            "Free-riders show near-zero variance\n(flat, uninformative SV profile).",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, fontstyle="italic", color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    _save(fig, "fig2_variance_fingerprint")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Convergence: Baseline vs Defense
# ═══════════════════════════════════════════════════════════════════════════════

def chart_convergence(baseline_f1s: list[float], defense_f1s: list[float]):
    """Convergence curve: Macro-F1 over rounds, baseline vs with defense."""
    fig, ax = plt.subplots(figsize=(7, 5))

    rounds = np.arange(1, len(baseline_f1s) + 1)

    ax.plot(rounds, baseline_f1s,
            color=IEEE_COLORS["baseline"], linewidth=1.6,
            linestyle="--", dashes=(4, 2),
            label="FedAvg Baseline (No Defense)")

    ax.plot(rounds, defense_f1s,
            color=IEEE_COLORS["defense"], linewidth=2.0,
            linestyle="-",
            label="FedAvg + DefenseController (Ours)")

    # Highlight the gap at the end
    final_b = baseline_f1s[-1]
    final_d = defense_f1s[-1]
    ax.annotate(
        f"$\\Delta$ = {final_d - final_b:+.3f}",
        xy=(len(rounds), final_d),
        xytext=(len(rounds) - 5, final_d + 0.02),
        fontsize=9, fontweight="bold",
        color=IEEE_COLORS["defense"],
        arrowprops=dict(arrowstyle="->", color=IEEE_COLORS["defense"], lw=1.2),
    )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Model Convergence: Baseline vs Defense-Enhanced FedAvg")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
    ax.grid(True, alpha=0.2)

    # Inset: zoom on the last 10 rounds
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax, width="40%", height="35%", loc="center right",
                          bbox_to_anchor=(-0.02, -0.02, 1, 1),
                          bbox_transform=ax.transAxes)
    last_n = min(10, len(rounds))
    ax_inset.plot(rounds[-last_n:], baseline_f1s[-last_n:],
                  color=IEEE_COLORS["baseline"], linewidth=1.4, linestyle="--")
    ax_inset.plot(rounds[-last_n:], defense_f1s[-last_n:],
                  color=IEEE_COLORS["defense"], linewidth=1.6)
    ax_inset.set_title(f"Last {last_n} Rounds (Zoom)", fontsize=7)
    ax_inset.tick_params(labelsize=6)
    ax_inset.grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, "fig3_convergence")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — Fancy: Defense Phase Classification (Sankey-style stacked bar)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_defense_phases(df: pd.DataFrame):
    """Show how many clients pass each defense phase per round."""
    fig, ax = plt.subplots(figsize=(8, 5))

    rounds = sorted(df["round"].unique())
    phase_counts = []
    for r in rounds:
        rdf = df[df["round"] == r]
        phase_counts.append({
            "round": r,
            "clean": (rdf["phase"] == "none").sum(),
            "phase1": (rdf["phase"] == "phase1_pos_sum").sum(),
            "phase2": (rdf["phase"] == "phase2_variance").sum(),
        })
    pc = pd.DataFrame(phase_counts)

    x = np.arange(len(rounds))
    w = 0.6
    ax.bar(x, pc["clean"], w, color="#4393C3", alpha=0.85, label="Passed Both Phases (Clean)")
    ax.bar(x, pc["phase2"], w, bottom=pc["clean"], color="#F4A582", alpha=0.85,
           label="Flagged: Phase 2 (Low Variance)")
    ax.bar(x, pc["phase1"], w, bottom=pc["clean"] + pc["phase2"], color="#CA0020", alpha=0.85,
           label="Flagged: Phase 1 (Low Pos-Sum)")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Number of Clients")
    ax.set_title("Defense Phase Classification per Round")
    ax.set_xticks(x[::5])
    ax.set_xticklabels(rounds[::5])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    _save(fig, "fig4_defense_phases")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5 — Fancy: 3D-style Defense Decision Boundary
# ═══════════════════════════════════════════════════════════════════════════════

def chart_decision_contour(df: pd.DataFrame,
                           pos_threshold: float = 0.01,
                           var_threshold: float = 0.001):
    """Density contour showing the defense decision boundary."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    honest = df[~df["is_attacker"]]
    attackers = df[df["is_attacker"]]

    # KDE contours for honest clients
    if len(honest) > 5:
        sns.kdeplot(
            data=honest, x="positive_sum", y="variance",
            ax=ax, levels=5, color=IEEE_COLORS["honest"], alpha=0.4,
            linewidths=0.8, label="Honest Density",
        )

    # KDE contours for attackers
    if len(attackers) > 5:
        sns.kdeplot(
            data=attackers, x="positive_sum", y="variance",
            ax=ax, levels=3, color=IEEE_COLORS["attacker"], alpha=0.5,
            linewidths=0.8, linestyles="--", label="Attacker Density",
        )

    # Scatter overlay
    ax.scatter(honest["positive_sum"], honest["variance"],
               c=IEEE_COLORS["honest"], s=15, alpha=0.5, zorder=3)
    ax.scatter(attackers["positive_sum"], attackers["variance"],
               c=IEEE_COLORS["attacker"], s=20, marker="^", alpha=0.6, zorder=3)

    # Threshold lines
    ax.axvline(pos_threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(var_threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    # Quadrant labels
    ax.text(pos_threshold * 0.5, var_threshold * 0.5,
            "REMOVED\n(Both low)", fontsize=6, ha="center", va="center",
            color="darkred", fontstyle="italic")
    ax.text(pos_threshold * 0.5, var_threshold * 5,
            "Phase 1\n(No positive\ncontribution)", fontsize=6, ha="center", va="center",
            color="darkred", fontstyle="italic")

    ax.set_xlabel("Positive Per-Class SV Sum")
    ax.set_ylabel("Per-Class SV Variance")
    ax.set_title("Defense Decision Boundary: Density Contours")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    _save(fig, "fig5_decision_contour")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    csv_path = "results/defense_history.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run main.py first to generate data.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    print(f"  Rounds: {df['round'].nunique()}")
    print(f"  Clients: {df['client_id'].nunique()}")
    print(f"  Attackers detected: {df[df['is_attacker']]['suspected'].sum()} / {df[df['is_attacker']].shape[0]}")

    thresholds = {
        "pos_threshold": df["positive_sum"].quantile(0.05) if len(df) > 0 else 0.01,
        "var_threshold": df["variance"].quantile(0.05) if len(df) > 0 else 0.001,
    }
    print(f"  Auto-detected thresholds: {thresholds}")

    print("\nGenerating charts...")
    chart_scatter(df, **thresholds)
    chart_fingerprint(df, thresholds["var_threshold"])
    chart_defense_phases(df)
    chart_decision_contour(df, **thresholds)

    # Convergence chart needs separate data; placeholder message
    print("\n  Note: fig3_convergence requires F1 arrays from main.py.")
    print("  To generate, save baseline_f1s and defense_f1s as JSON and pass here.")
    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
