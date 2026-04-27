# RFM Customer Segmentation

Turning 5,878 raw customers into 11 named, action-ready segments using Recency, Frequency, and Monetary scoring.

## 1. Overview

This project scores every customer in a UK online retailer's two-year transaction log on three axes — Recency, Frequency, and Monetary value — and then maps them into 11 named lifecycle segments (plus an "Other" edge bucket). The goal is to turn raw invoice data into a concrete spending and messaging plan: who to protect, who to upsell, who to win back, and who to let go.

I built this as the companion piece to my P6 Cohort Retention analysis. P6 answered the question *"which acquisition months produced good customers?"* — a planning lens on the past. P7 answers *"right now, which individual customers should I spend on, win back, or let go?"* — an operational lens on the present. Together they cover both sides of the lifecycle marketing brief: cohort quality and customer quality.

## 2. Dataset

- **Source**: Online Retail II, UCI Machine Learning Repository
- **Link**: [https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Scope**: 779,425 cleaned transactions across 5,878 unique customers, December 2009 to December 2011
- **Cleaning rules** (carried over from P6 so both projects sit on the same base):
  - Dropped rows with null Customer ID
  - Dropped rows with Quantity ≤ 0 (returns and cancellations)
  - Dropped rows with Price ≤ 0 (manual adjustments)
  - Dropped exact duplicate rows
  - Added `Revenue = Quantity × Price`

## 3. Methodology

- **Snapshot date**: `2011-12-10` (= max InvoiceDate + 1 day). I add the extra day so every customer's recency is strictly positive — otherwise customers who bought on the dataset's final day get `recency = 0`, which breaks any downstream `1 / recency` or `log(recency)` calculation.
- **Per-customer aggregation**:
  - `recency_days` = days since last purchase
  - `frequency` = number of distinct invoices
  - `monetary` = sum of `Quantity × Price` across all invoices
- **Quintile scoring** (1–5): I score each of R, F, M using `pd.qcut` on the *rank-transformed* values (`method='first'`). The pre-rank step is the standard RFM workaround for ties — many customers share `frequency = 1`, so `qcut` on raw values fails to allocate exactly 20% per bin.
  - `r_score`: lower recency → higher score (5 = most recent)
  - `f_score`: higher frequency → higher score
  - `m_score`: higher monetary → higher score
- **Segment assignment** (priority order, first match wins):
  1. Cannot Lose Them — `R=1, F=4–5, M=4–5`
  2. Champions — `R=5, F=4–5, M=4–5`
  3. Loyal Customers — `R=3–5, F=4–5, M=3–5` (those not already Champions)
  4. At Risk — `R=1–2, F=2–5, M=3–5`
  5. Potential Loyalists — `R=4–5, F=2–3`
  6. New Customers — `R=4–5, F=1`
  7. Promising — `R=3–4, F=1`
  8. Need Attention — `R=2–3, F=2–3`
  9. About To Sleep — `R=2–3, F=1`
  10. Lost — `R=1, F=1, M=1–2`
  11. Hibernating — `R=1–2, F=1–2, M=1–2`

  Rule precedence is intentional. Cannot Lose Them is checked before At Risk so that the highest-value lapsed customers surface as a distinct, urgent action group rather than dissolving into the broader At Risk pool.
- **Edge cases**: a small set of `(r, f, m)` combinations (e.g. `R=1, F=3, M=2`) do not match any rule above. I label these customers `Other` rather than force-fitting them, so the row count is preserved and the gap is transparent.

## 4. Key Findings

- **33.1%** of customers (Champions + Loyal Customers, 1,947 people) drive **78.8%** of all lifetime revenue — a textbook Pareto outcome.
- **Champions** alone: **741** customers (12.6%), **£8.83M** of revenue (50.8%), with average lifetime spend of **£11,919** and a typical recency of ~8 days.
- **Cannot Lose Them**: only **60** customers, but they have averaged **486 days** of silence and were worth **£4,115** each — a small, concentrated, urgent win-back target.
- **At Risk + Need Attention**: a 1,531-customer reactivation pool worth **£2.05M** in lifetime revenue — the main pipeline for automated win-back sequences.
- The bottom 28% of customers (Lost, Hibernating, About To Sleep, Promising, New, Other) produce only **3.3%** of revenue. These segments justify acquisition-funnel spend at most — no lifecycle investment.

## 5. Strategic Recommendations

- **Champions → VIP protection.** Tiered loyalty programme, dedicated account manager for the top quintile, surprise-and-delight gifts at anniversary milestones. Goal: keep churn near zero, because losing a Champion erases roughly £12K of lifetime value.
- **Loyal Customers → Upsell and cross-sell.** Bundled offers and complementary-category recommendations. Convert the top decile into Champions via personalised AOV-lift campaigns. Target a 15–20% lift in average monetary over the next quarter.
- **Cannot Lose Them → Immediate one-to-one win-back.** Hand-pick the 60-customer list, send personal email from a senior account manager, offer 25–30% off plus free shipping. Even a 25% recovery rate adds ~£60K back to the book.
- **At Risk → Personalised three-touch reactivation.** Automated sequence: "we miss you" email with category-relevant picks, then 15% off after 7 days, then 20% off plus free shipping after 14 days. Target a 12% reactivation rate, ~£175K back in pipeline.

## 6. Files

| File | What it is |
|------|------------|
| `README.md` | This file |
| `rfm_segmentation_report.pdf` | 2-page hiring-manager-friendly summary report |
| `customer_rfm_scores.csv` | One row per customer with R/F/M values, scores, rfm_score, rfm_sum, and assigned segment |
| `segment_summary.csv` | Per-segment customer counts, revenue totals, and average R/F/M |
| `p7_methodology.txt` | Plain-text methodology and limitations |
| `segment_treemap.png` | Revenue treemap by segment |
| `pareto_segments.png` | Customer-share vs revenue-share Pareto bars |
| `rfm_scatter.png` | Recency vs frequency log/log scatter, dot size = monetary |
| `p7_rfm_analysis.py` | Reproducible script: cleaning → RFM scoring → segment assignment → CSV outputs |
| `p7_charts.py` | Reproducible script: reads the two CSVs and generates all three visualisations |

## 7. Tools

Python, pandas, numpy, matplotlib, squarify, reportlab.

## 8. How to Reproduce

1. Download the Online Retail II dataset from the UCI link above and place the raw CSV in the project root.
2. Run `python p7_rfm_analysis.py` to clean the data and produce `customer_rfm_scores.csv`, `segment_summary.csv`, and `p7_methodology.txt`.
3. Run `python p7_charts.py` to generate `segment_treemap.png`, `pareto_segments.png`, and `rfm_scatter.png` from the two CSVs.
4. Open `rfm_segmentation_report.pdf` for the 2-page summary; this README ties everything together.

## 9. Contact

- Email: [ashwinkumarglbimr18@gmail.com](mailto:ashwinkumarglbimr18@gmail.com)
- LinkedIn: [https://www.linkedin.com/in/ashwin-kumar-180816174/](https://www.linkedin.com/in/ashwin-kumar-180816174/)
