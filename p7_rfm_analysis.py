"""
P7 - RFM Segmentation
Author: Ashwin Kumar

Builds an RFM (Recency, Frequency, Monetary) segmentation from the cleaned
Online Retail II dataset (UCI, 2009-2011).

Inputs
------
online_retail_II_cleaned.csv
    Columns: Invoice, StockCode, Description, Quantity, InvoiceDate,
             Price, Customer ID, Country, Revenue
    Already filtered: returns excluded, null Customer IDs excluded,
                      zero-price rows excluded, exact duplicates removed.

Outputs
-------
data/customer_rfm_scores.csv
data/segment_summary.csv
report/p7_methodology.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WS = Path("/home/user/workspace")
SRC_CANDIDATES = [
    WS / "online_retail_II_cleaned.csv",
    WS / "01_Datasets_Raw" / "online_retail_II_cleaned.csv",
    WS / "02_Projects" / "P6_Cohort_Retention" / "data" / "online_retail_II_cleaned.csv",
]

P7 = WS / "02_Projects" / "P7_RFM_Segmentation"
DATA_DIR   = P7 / "data"
REPORT_DIR = P7 / "report"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RFM     = DATA_DIR / "customer_rfm_scores.csv"
OUT_SUMMARY = DATA_DIR / "segment_summary.csv"
OUT_METHOD  = REPORT_DIR / "p7_methodology.txt"


# ---------------------------------------------------------------------------
# 1. Locate or rebuild the cleaned source
# ---------------------------------------------------------------------------
def _load_cleaned():
    for p in SRC_CANDIDATES:
        if p.exists():
            print(f"Using cleaned source: {p}")
            df = pd.read_csv(p, encoding="utf-8", parse_dates=["InvoiceDate"])
            return df
    # Fallback: rebuild from raw Online Retail II if a raw file is present.
    raw = WS / "online_retail_II.csv"
    if not raw.exists():
        raise FileNotFoundError(
            "Neither a cleaned file nor a raw 'online_retail_II.csv' was "
            "found in the workspace."
        )
    print(f"Rebuilding from raw: {raw}")
    df = pd.read_csv(raw, encoding="utf-8", parse_dates=["InvoiceDate"])
    df = df[[
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
    ]].copy()
    df = df[df["Customer ID"].notna()]
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df = df.drop_duplicates().reset_index(drop=True)
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df


df = _load_cleaned()

# Defensive: ensure Revenue exists, integer customer IDs, datetime InvoiceDate.
if "Revenue" not in df.columns:
    df["Revenue"] = df["Quantity"] * df["Price"]
df["Customer ID"] = df["Customer ID"].astype("int64")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])


# ---------------------------------------------------------------------------
# 2. Snapshot date and per-customer RFM aggregates
# ---------------------------------------------------------------------------
# Snapshot = one day AFTER the maximum transaction date so that recency for
# customers who transacted on the very last day is 1 (not 0). This keeps the
# distribution strictly positive and avoids divide-by-zero edge cases in
# downstream LTV math.
max_dt = df["InvoiceDate"].max()
SNAPSHOT = (max_dt.normalize() + pd.Timedelta(days=1))
print(f"Max transaction date : {max_dt}")
print(f"Snapshot date        : {SNAPSHOT.date()}")

rfm = (
    df.groupby("Customer ID")
      .agg(
          last_purchase = ("InvoiceDate", "max"),
          frequency     = ("Invoice", "nunique"),
          monetary      = ("Revenue", "sum"),
      )
      .reset_index()
)
rfm["recency_days"] = (SNAPSHOT - rfm["last_purchase"]).dt.days
rfm = rfm[["Customer ID", "recency_days", "frequency", "monetary"]]
rfm = rfm.rename(columns={"Customer ID": "customer_id"})


# ---------------------------------------------------------------------------
# 3. Quintile scoring (1-5)
#    R: lower days  -> higher score (invert)
#    F: higher freq -> higher score
#    M: higher rev  -> higher score
# ---------------------------------------------------------------------------
def _qscore(series, ascending=True, labels=None):
    """Quintile-bin a series into 1-5 with rank-based pre-bucketing.

    `pd.qcut` on raw values fails when many tied values land at a bin edge
    (very common for `frequency`, where a large mass of customers has
    frequency = 1). Pre-ranking with method='first' breaks ties deterministically
    by row order, which is the standard RFM workaround.
    """
    if labels is None:
        labels = [1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=5, labels=labels).astype(int)

# Recency: invert so 5 = most recent (smallest recency_days)
rfm["r_score"] = _qscore(rfm["recency_days"], ascending=False)  # labels [5,4,3,2,1]
rfm["f_score"] = _qscore(rfm["frequency"],    ascending=True)
rfm["m_score"] = _qscore(rfm["monetary"],     ascending=True)

rfm["rfm_score"] = (
    rfm["r_score"].astype(str)
    + rfm["f_score"].astype(str)
    + rfm["m_score"].astype(str)
)
rfm["rfm_sum"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]


# ---------------------------------------------------------------------------
# 4. Segment assignment
#
# Rules are applied in PRIORITY ORDER. The first matching rule wins, so more
# specific labels (Champions, Cannot Lose Them) are checked before broader
# labels (Loyal Customers, At Risk).
# ---------------------------------------------------------------------------
def assign_segment(r, f, m):
    # 1. Cannot Lose Them : top spenders who have lapsed badly
    if r == 1 and f >= 4 and m >= 4:
        return "Cannot Lose Them"
    # 2. Champions : recent + frequent + high spend
    if r == 5 and f >= 4 and m >= 4:
        return "Champions"
    # 3. Loyal Customers : still active, frequent, decent spend (not already Champions)
    if r >= 3 and f >= 4 and m >= 3:
        return "Loyal Customers"
    # 4. At Risk : not recent but used to be valuable
    if r <= 2 and f >= 2 and m >= 3:
        return "At Risk"
    # 5. Potential Loyalists : recent, mid-frequency
    if r >= 4 and 2 <= f <= 3:
        return "Potential Loyalists"
    # 6. New Customers : recent, only one purchase
    if r >= 4 and f == 1:
        return "New Customers"
    # 7. Promising : recent-ish, single purchase
    if 3 <= r <= 4 and f == 1:
        return "Promising"
    # 8. Need Attention : middling on both R and F
    if 2 <= r <= 3 and 2 <= f <= 3:
        return "Need Attention"
    # 9. About To Sleep : middling recency, single purchase
    if 2 <= r <= 3 and f == 1:
        return "About To Sleep"
    # 10. Lost : never came back, low value
    if r == 1 and f == 1 and m <= 2:
        return "Lost"
    # 11. Hibernating : lapsed, low frequency, low value
    if r <= 2 and f <= 2 and m <= 2:
        return "Hibernating"
    # Catch-all (covers rare cells that none of the rules above hit)
    return "Other"

rfm["segment"] = [
    assign_segment(r, f, m)
    for r, f, m in zip(rfm["r_score"], rfm["f_score"], rfm["m_score"])
]


# ---------------------------------------------------------------------------
# 5. Round monetary, finalise column order, save
# ---------------------------------------------------------------------------
rfm["monetary"] = rfm["monetary"].round(2)
rfm = rfm[[
    "customer_id", "recency_days", "frequency", "monetary",
    "r_score", "f_score", "m_score",
    "rfm_score", "rfm_sum", "segment",
]]
rfm.to_csv(OUT_RFM, index=False)


# ---------------------------------------------------------------------------
# 6. Segment summary
# ---------------------------------------------------------------------------
total_customers = len(rfm)
total_revenue   = float(rfm["monetary"].sum())

summary = (
    rfm.groupby("segment")
       .agg(
           customer_count   = ("customer_id", "size"),
           total_revenue    = ("monetary",    "sum"),
           avg_recency_days = ("recency_days","mean"),
           avg_frequency    = ("frequency",   "mean"),
           avg_monetary     = ("monetary",    "mean"),
       )
       .reset_index()
)
summary["pct_of_total_customers"] = summary["customer_count"] / total_customers * 100
summary["pct_of_total_revenue"]   = summary["total_revenue"]   / total_revenue   * 100

# Tidy rounding
summary["total_revenue"]          = summary["total_revenue"].round(2)
summary["pct_of_total_customers"] = summary["pct_of_total_customers"].round(2)
summary["pct_of_total_revenue"]   = summary["pct_of_total_revenue"].round(2)
summary["avg_recency_days"]       = summary["avg_recency_days"].round(1)
summary["avg_frequency"]          = summary["avg_frequency"].round(2)
summary["avg_monetary"]           = summary["avg_monetary"].round(2)

# Final column order
summary = summary[[
    "segment", "customer_count", "pct_of_total_customers",
    "total_revenue", "pct_of_total_revenue",
    "avg_recency_days", "avg_frequency", "avg_monetary",
]]
summary = summary.sort_values("total_revenue", ascending=False).reset_index(drop=True)
summary.to_csv(OUT_SUMMARY, index=False)


# ---------------------------------------------------------------------------
# 7. Methodology document
# ---------------------------------------------------------------------------
methodology = f"""\
P7 - RFM SEGMENTATION: METHODOLOGY
============================================================
Author : Ashwin Kumar
Dataset: Online Retail II (UCI, 2009-2011), cleaned per P6 rules.

INPUT
------------------------------------------------------------
online_retail_II_cleaned.csv
  - 779,425 transaction rows (post-cleaning)
  - 5,878 unique customers
  - 2009-12-01 to 2011-12-09

Cleaning carried over from P6:
  - Dropped rows with null Customer ID.
  - Dropped rows with Quantity <= 0 (returns / cancellations).
  - Dropped rows with Price <= 0 (manual adjustments).
  - Dropped exact duplicate rows.
  - Added Revenue = Quantity * Price.

SNAPSHOT DATE
------------------------------------------------------------
Snapshot date = max(InvoiceDate) + 1 day = {SNAPSHOT.date()}.

Rationale: using max(InvoiceDate) directly would assign a recency of 0 days
to any customer whose last purchase fell on the dataset's final day, which
breaks downstream calculations that expect recency >= 1 (e.g. ratios such
as 1/recency, log(recency)). Adding one day shifts the entire distribution
to be strictly positive without distorting relative ranking - every
customer's recency moves by exactly one day.

PER-CUSTOMER AGGREGATION
------------------------------------------------------------
For each Customer ID:
  recency_days = (snapshot_date - max(InvoiceDate)).days
  frequency    = number of DISTINCT Invoice IDs
  monetary     = sum of Revenue (= Quantity * Price)

QUINTILE SCORING (1-5)
------------------------------------------------------------
Each of R, F, M is split into five quintiles using pd.qcut on the
rank-transformed series (method='first'). Pre-ranking is the standard
RFM workaround for ties: many customers share frequency = 1, so qcut
on raw values would fail to allocate exactly 20% per bin.

  r_score: lower recency_days -> HIGHER score (5 = most recent)
  f_score: higher frequency   -> HIGHER score (5 = most frequent)
  m_score: higher monetary    -> HIGHER score (5 = highest spend)

Composite fields:
  rfm_score = concatenation of r,f,m as a 3-character string, e.g. "555".
  rfm_sum   = r_score + f_score + m_score, integer in [3, 15].

SEGMENT RULES (priority order; first match wins)
------------------------------------------------------------
  1. Cannot Lose Them  : R=1, F=4-5, M=4-5
  2. Champions         : R=5, F=4-5, M=4-5
  3. Loyal Customers   : R=3-5, F=4-5, M=3-5  (not already Champions)
  4. At Risk           : R=1-2, F=2-5, M=3-5
  5. Potential Loyalists: R=4-5, F=2-3
  6. New Customers     : R=4-5, F=1
  7. Promising         : R=3-4, F=1
  8. Need Attention    : R=2-3, F=2-3
  9. About To Sleep    : R=2-3, F=1
 10. Lost              : R=1, F=1, M=1-2
 11. Hibernating       : R=1-2, F=1-2, M=1-2

Rule precedence is intentional. "Cannot Lose Them" is checked before
"At Risk" so the highest-value lapsed customers are surfaced as a
distinct, urgent action group rather than blended into the broader
At Risk pool. Similarly, Champions is checked before Loyal Customers
because every Champion is technically also Loyal, but the labels
serve different campaigns.

LIMITATIONS
------------------------------------------------------------
  * Overlap and edge cells: a small number of (r, f, m) combinations
    do not match any rule above (e.g. R=1, F=3, M=2). These customers
    are labelled "Other" so the row count is preserved and the gap is
    transparent.
  * Tie-breaking in qcut: rank-based pre-bucketing breaks ties by row
    order. Two customers with identical raw R/F/M values can therefore
    end up in different score bins. The effect is negligible at scale
    but exists.
  * Quintiles are dataset-relative. A customer's "5" recency is "most
    recent in this dataset", not an absolute industry benchmark.
  * Returns are excluded upstream; monetary is GROSS revenue, not net.
  * Wholesale skew: the source retailer leans toward B2B wholesale
    buyers, so frequency and monetary distributions are heavier-tailed
    than a pure B2C dataset would be.

DATASET REFERENCE
------------------------------------------------------------
Online Retail II, UCI Machine Learning Repository (2009-2011).
https://archive.ics.uci.edu/dataset/502/online+retail+ii
"""
OUT_METHOD.write_text(methodology, encoding="utf-8")


# ---------------------------------------------------------------------------
# 8. Console summary
# ---------------------------------------------------------------------------
sep = "=" * 64
print(sep)
print("RFM SEGMENTATION COMPLETE")
print(sep)
print(f"Total customers scored      : {total_customers:,}")
print(f"Total monetary revenue      : £{total_revenue:,.2f}")
print(f"customer_rfm_scores shape   : {rfm.shape}")
print()
print("Segments by revenue (descending):")
print(f"  {'segment':<22s} {'customers':>10s} {'rev_share':>10s} {'avg_R':>7s} {'avg_F':>7s} {'avg_M':>10s}")
for _, row in summary.iterrows():
    print(
        f"  {row['segment']:<22s} "
        f"{int(row['customer_count']):>10,} "
        f"{row['pct_of_total_revenue']:>9.2f}% "
        f"{row['avg_recency_days']:>7.1f} "
        f"{row['avg_frequency']:>7.2f} "
        f"£{row['avg_monetary']:>9,.2f}"
    )
print()
print("First 5 rows of segment_summary.csv:")
print(summary.head(5).to_string(index=False))
print()
print(sep)
print(f"Saved -> {OUT_RFM.relative_to(WS)}")
print(f"Saved -> {OUT_SUMMARY.relative_to(WS)}")
print(f"Saved -> {OUT_METHOD.relative_to(WS)}")
