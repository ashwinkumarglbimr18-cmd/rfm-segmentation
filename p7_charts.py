"""
P7 - RFM Segmentation Charts
Author: Ashwin Kumar

Produces three hiring-manager-friendly charts:
  1. segment_treemap.png   - revenue treemap by RFM segment
  2. pareto_segments.png   - customer share vs revenue share Pareto bars
  3. rfm_scatter.png       - recency vs frequency scatter (log/log, dot size = monetary)

Inputs (workspace root):
  customer_rfm_scores.csv
  segment_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import squarify

# ---------------------------------------------------------------------------
# Paths and tokens
# ---------------------------------------------------------------------------
WS = Path("/home/user/workspace")
RFM_CSV = WS / "customer_rfm_scores.csv"
SUM_CSV = WS / "segment_summary.csv"

OUT_TREEMAP = WS / "segment_treemap.png"
OUT_PARETO  = WS / "pareto_segments.png"
OUT_SCATTER = WS / "rfm_scatter.png"

INK    = "#28251D"
MUTED  = "#7A7974"
FAINT  = "#BAB9B4"
RULE   = "#D4D1CA"
BG     = "#FFFFFF"
FOOTER = "Ashwin Kumar \u2014 Performance Marketing Portfolio"

# Distinct palette - Champions highlighted in the strongest teal.
SEGMENT_COLORS = {
    "Champions":            "#01696F",  # signature teal - strongest
    "Loyal Customers":      "#1B474D",  # dark teal
    "At Risk":              "#A84B2F",  # terra
    "Cannot Lose Them":     "#964219",  # dark warning
    "Potential Loyalists":  "#20808D",  # mid teal
    "New Customers":        "#FFC553",  # gold
    "Promising":            "#E8AF34",  # darker gold
    "Need Attention":       "#944454",  # mauve
    "About To Sleep":       "#848456",  # olive
    "Hibernating":          "#7A7974",  # muted gray
    "Lost":                 "#BAB9B4",  # faint gray
    "Other":                "#D4D1CA",  # rule gray
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.edgecolor":    RULE,
    "axes.labelcolor":   INK,
    "axes.titlecolor":   INK,
    "xtick.color":       INK,
    "ytick.color":       INK,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "savefig.facecolor": BG,
    "savefig.dpi":       300,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
rfm = pd.read_csv(RFM_CSV)
summary = pd.read_csv(SUM_CSV)

# Sort summary by revenue descending (already sorted on save, but be defensive)
summary = summary.sort_values("total_revenue", ascending=False).reset_index(drop=True)

total_customers = int(rfm.shape[0])
total_revenue   = float(summary["total_revenue"].sum())
pct_customers_col = "pct_of_total_customers"
pct_revenue_col   = "pct_of_total_revenue"


# ===========================================================================
# CHART 1 - Treemap
# ===========================================================================
def chart_treemap():
    fig, ax = plt.subplots(figsize=(15, 8.5))

    sizes  = summary["total_revenue"].to_numpy()
    labels = []
    for _, r in summary.iterrows():
        labels.append(
            f"{r['segment']}\n"
            f"{int(r['customer_count']):,} customers\n"
            f"{r[pct_revenue_col]:.1f}% of revenue"
        )
    colors = [SEGMENT_COLORS.get(s, "#CCCCCC") for s in summary["segment"]]

    # Lay out the treemap ourselves so we can control label colours per cell.
    norm_sizes = squarify.normalize_sizes(sizes, 100, 60)
    rects = squarify.squarify(norm_sizes, 0, 0, 100, 60)

    # Pass 1: draw all rectangles
    rect_info = []
    for rect, color, seg, cust, rev_share in zip(
        rects, colors, summary["segment"],
        summary["customer_count"], summary[pct_revenue_col],
    ):
        x, y, dx, dy = rect["x"], rect["y"], rect["dx"], rect["dy"]
        ax.add_patch(plt.Rectangle(
            (x, y), dx, dy, facecolor=color, edgecolor="white", linewidth=2,
        ))
        rect_info.append((x, y, dx, dy, color, seg, int(cust), float(rev_share)))

    # Pass 2: adaptive labels by cell area, anchored top-left
    unlabeled = []  # for legend strip
    for x, y, dx, dy, color, seg, cust, rev_share in rect_info:
        area = dx * dy
        is_pale = color in ("#FFC553", "#E8AF34", "#BAB9B4", "#D4D1CA")
        txt_color = INK if is_pale else "white"

        if area > 600:
            fs_main, fs_sub = 14, 10.5
            ax.text(x + 1.6, y + 1.8, seg,
                    fontsize=fs_main, color=txt_color, fontweight="bold",
                    va="top", ha="left")
            ax.text(x + 1.6, y + 5.5, f"{cust:,} customers",
                    fontsize=fs_sub, color=txt_color, va="top", ha="left")
            ax.text(x + 1.6, y + 7.8, f"{rev_share:.1f}% of revenue",
                    fontsize=fs_sub, color=txt_color, va="top", ha="left")
        elif area > 180:
            fs_main, fs_sub = 10.5, 8.5
            ax.text(x + 1.2, y + 1.4, seg,
                    fontsize=fs_main, color=txt_color, fontweight="bold",
                    va="top", ha="left")
            ax.text(x + 1.2, y + 4.0, f"{cust:,} customers",
                    fontsize=fs_sub, color=txt_color, va="top", ha="left")
            ax.text(x + 1.2, y + 6.0, f"{rev_share:.1f}% of revenue",
                    fontsize=fs_sub, color=txt_color, va="top", ha="left")
        elif area > 60 and dx > 4 and dy > 4:
            fs_main, fs_sub = 8.5, 7.2
            ax.text(x + 0.8, y + 1.0, seg,
                    fontsize=fs_main, color=txt_color, fontweight="bold",
                    va="top", ha="left")
            ax.text(x + 0.8, y + 3.2, f"{rev_share:.1f}%",
                    fontsize=fs_sub, color=txt_color, va="top", ha="left")
        elif dx > 3.5 and dy > 2.2:
            # Only label if the segment name actually fits the cell width.
            # Approx 0.55 data units per char at 6.8pt; require some padding.
            if dx >= 0.55 * len(seg) + 1.0:
                ax.text(x + 0.5, y + dy / 2, seg,
                        fontsize=6.8, color=txt_color, fontweight="bold",
                        va="center", ha="left")
            else:
                unlabeled.append((color, seg, cust, rev_share))
        else:
            unlabeled.append((color, seg, cust, rev_share))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_aspect("auto")
    ax.invert_yaxis()  # display origin top-left; labels anchored at top of cell
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "Where the Revenue Lives: RFM Segment Treemap",
        fontsize=16, fontweight="bold", color=INK, x=0.04, ha="left", y=0.965,
    )
    fig.text(
        0.04, 0.918,
        "Cell area = total lifetime revenue   \u00b7   "
        f"5,878 customers   \u00b7   \u00a3{total_revenue/1e6:.1f}M total revenue   \u00b7   "
        "Champions alone account for ~51%",
        fontsize=10.5, color=MUTED, ha="left",
    )
    fig.text(0.99, 0.012, FOOTER, fontsize=8, color=FAINT, ha="right")

    # Small legend strip below for any cells too tiny to label in-place
    if unlabeled:
        parts = [f"{seg} ({cust:,} \u00b7 {rev:.1f}%)"
                 for _, seg, cust, rev in unlabeled]
        legend_txt = "Tiny cells (left to right):  " + "   \u00b7   ".join(parts)
        fig.text(0.04, 0.038, legend_txt, fontsize=8.5, color=MUTED, ha="left")

    fig.tight_layout(rect=[0.02, 0.06, 0.99, 0.90])
    fig.savefig(OUT_TREEMAP, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ===========================================================================
# CHART 2 - Pareto bars
# ===========================================================================
def chart_pareto():
    # Order segments by revenue desc; keep the same order on the y axis
    seg_order = summary["segment"].tolist()
    cust_pct  = summary[pct_customers_col].to_numpy()
    rev_pct   = summary[pct_revenue_col].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 8))

    y = np.arange(len(seg_order))
    bar_h = 0.38

    # Light bars (customer share) on top, dark bars (revenue share) on bottom
    bars_cust = ax.barh(
        y + bar_h / 2, cust_pct, height=bar_h, color="#BCE2E7",
        edgecolor="none", label="% of customers",
    )
    bars_rev = ax.barh(
        y - bar_h / 2, rev_pct, height=bar_h, color="#01696F",
        edgecolor="none", label="% of revenue",
    )

    # Value labels (1 decimal)
    for bars, vals in ((bars_cust, cust_pct), (bars_rev, rev_pct)):
        for b, v in zip(bars, vals):
            ax.text(
                b.get_width() + 0.6, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", ha="left", fontsize=8.6, color=INK,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(seg_order, fontsize=10)
    ax.invert_yaxis()  # highest-revenue segment at top
    ax.set_xlabel("Share of total (%)", fontsize=10, color=INK)
    ax.set_xlim(0, max(rev_pct.max(), cust_pct.max()) * 1.18)
    ax.grid(axis="x", color=RULE, linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="both", which="major", length=0)

    # Champions + Loyal cumulative annotation
    champ_loyal_cust = summary.loc[
        summary["segment"].isin(["Champions", "Loyal Customers"]),
        pct_customers_col,
    ].sum()
    champ_loyal_rev = summary.loc[
        summary["segment"].isin(["Champions", "Loyal Customers"]),
        pct_revenue_col,
    ].sum()

    annot = (
        f"Champions + Loyal Customers:\n"
        f"{champ_loyal_cust:.1f}% of customers, {champ_loyal_rev:.1f}% of revenue"
    )
    ax.annotate(
        annot,
        xy=(rev_pct[1] + 1.2, 1 - bar_h / 2),  # near the Loyal Customers revenue bar
        xytext=(rev_pct.max() * 0.55, 4.5),
        fontsize=10.5, color=INK, fontweight="semibold",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FBF6F0", ec="#A84B2F", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#A84B2F", lw=1.2,
                        connectionstyle="arc3,rad=-0.15"),
        zorder=5,
    )

    ax.legend(loc="lower right", frameon=False, fontsize=10)

    fig.suptitle(
        "Pareto Check: 33% of Customers Drive 79% of Revenue",
        fontsize=16, fontweight="bold", color=INK, x=0.04, ha="left", y=0.965,
    )
    fig.text(
        0.04, 0.918,
        "Light bar = share of customers   \u00b7   Dark bar = share of revenue   "
        "\u00b7   Segments sorted by total revenue (descending)",
        fontsize=10.5, color=MUTED, ha="left",
    )
    fig.text(0.99, 0.012, FOOTER, fontsize=8, color=FAINT, ha="right")

    fig.tight_layout(rect=[0.02, 0.03, 0.99, 0.90])
    fig.savefig(OUT_PARETO, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ===========================================================================
# CHART 3 - Recency vs Frequency scatter, log/log, size by monetary
# ===========================================================================
def chart_scatter():
    fig, ax = plt.subplots(figsize=(15, 8.6))

    # Slight jitter so coincident integer points don't fully overlap
    rng = np.random.default_rng(42)
    jitter_x = np.exp(rng.normal(0, 0.04, size=len(rfm)))
    jitter_y = np.exp(rng.normal(0, 0.04, size=len(rfm)))
    x_vals = rfm["recency_days"].clip(lower=1).to_numpy() * jitter_x
    y_vals = rfm["frequency"].clip(lower=1).to_numpy() * jitter_y

    # Dot size scaled by monetary (sqrt for visual fairness, capped)
    monetary = rfm["monetary"].clip(lower=1).to_numpy()
    sizes = 6 + 90 * np.sqrt(monetary / monetary.max())

    # Plot one segment at a time so the legend works and z-order is controlled
    seg_order_for_legend = list(SEGMENT_COLORS.keys())
    for seg in seg_order_for_legend:
        if seg not in rfm["segment"].unique():
            continue
        mask = (rfm["segment"] == seg).to_numpy()
        ax.scatter(
            x_vals[mask], y_vals[mask],
            s=sizes[mask],
            c=SEGMENT_COLORS[seg],
            alpha=0.65,
            edgecolor="white",
            linewidth=0.4,
            label=seg,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()  # most recent customers on the right
    ax.set_xlabel("Recency (days since last purchase, log scale, recent \u2192)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("Frequency (distinct invoices, log scale)",
                  fontsize=10.5, color=INK)

    # Friendly tick labels (avoid "10^1" formatting)
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else f"{v:.1f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    ax.grid(True, which="major", color=RULE, linewidth=0.5, alpha=0.6, zorder=0)
    ax.grid(True, which="minor", color=RULE, linewidth=0.3, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)

    # Legend (segment colour). Force consistent dot size in legend.
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=SEGMENT_COLORS[s], markersize=9,
                   label=s, markeredgecolor="white", markeredgewidth=0.4)
        for s in seg_order_for_legend if s in rfm["segment"].unique()
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=9.5, title="Segment", title_fontsize=10,
              ncol=2, columnspacing=1.2, handletextpad=0.6, labelspacing=0.4)

    # Size legend - separate small annotation explaining dot size
    fig.text(
        0.99, 0.04,
        "Dot size \u221d \u221amonetary (lifetime spend, \u00a3)",
        fontsize=9, color=MUTED, ha="right",
    )

    fig.suptitle(
        "The RFM Map: Recency, Frequency, and Segment Membership",
        fontsize=16, fontweight="bold", color=INK, x=0.04, ha="left", y=0.965,
    )
    fig.text(
        0.04, 0.918,
        "Each dot is one customer   \u00b7   Top-right corner = recent + frequent (Champions)   "
        "\u00b7   Bottom-left = lapsed + single purchase (Lost / Hibernating)",
        fontsize=10.5, color=MUTED, ha="left",
    )
    fig.text(0.99, 0.012, FOOTER, fontsize=8, color=FAINT, ha="right")

    fig.tight_layout(rect=[0.02, 0.05, 0.99, 0.90])
    fig.savefig(OUT_SCATTER, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run all three
# ---------------------------------------------------------------------------
chart_treemap()
chart_pareto()
chart_scatter()

# Confirmation
import os
files = [OUT_TREEMAP, OUT_PARETO, OUT_SCATTER, WS / "p7_charts.py"]
print("=" * 60)
print("CHARTS COMPLETE")
print("=" * 60)
for p in files:
    size_kb = os.path.getsize(p) / 1024
    print(f"  saved : {p}  ({size_kb:,.1f} KB)")
print(f"\nTotal chart count: 3 PNGs + 1 script")
