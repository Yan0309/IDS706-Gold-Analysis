# IDS706-Gold-Analysis

# Gold Price Analysis (2015–2025)

## Overview
This project explores the `gold_data_2015_25.csv` dataset, which contains daily
values for SPX (S&P 500), GLD (gold ETF price), USO (oil), SLV (silver), and
the EUR/USD exchange rate from 2015 to 2025. The goal was to explore whether
GLD price shows a stable relationship with EUR/USD and SPX, and to practice
basic data analysis and a first machine learning experiment with linear
regression.

## Setup & How to Run

### Requirements
- Python 3.11+
- Rust (`rustc`/`cargo`) and the `evcxr_jupyter` kernel, for the Rust notebook (Question 2)
- VS Code with the Python and Jupyter extensions (recommended)

### Steps
1. Clone this repository:
```bash
   git clone git@github.com:Yan0309/IDS706-Gold-Analysis.git
   cd IDS706-Gold-Analysis
```
2. Create and activate a virtual environment:
```bash
   python3 -m venv .venv
   source .venv/bin/activate
```
3. Install dependencies:
```bash
   pip install pandas matplotlib seaborn scikit-learn polars jupyter
```
4. Run the main analysis script:
```bash
   python analysis.py
```
   This prints data inspection output, grouping/correlation results, and
   regression results to the terminal, and saves plots to `figures/`.

5. (Optional) Run the Pandas vs Polars comparison:
```bash
   python polars_comparison.py
```
6. For the Rust notebook (Question 2): install the Rust Jupyter kernel with
```bash
   cargo install evcxr_jupyter
   evcxr_jupyter --install
```
   then open `notebooks/rust_vs_python_intro.ipynb` in VS Code and select the
   **Rust** kernel.

## Data Import & Inspection
- Loaded the CSV with pandas and inspected it using `.head()`, `.info()`,
  and `.describe()`.
- No missing values and no duplicate rows were found.
- The `Date` column was originally stored as a string, which would have
  limited time-based analysis (e.g., extracting year, sorting chronologically).
  It was converted to `datetime` using `pd.to_datetime()`, and a `Year` column
  was extracted for grouping.

## Part 1: GLD vs EUR/USD
Using `groupby("Year")` and `.mean()`, I computed the yearly average GLD price
and EUR/USD rate. GLD shows a strong, almost continuous upward trend over the
decade (from ~$111 in 2015 to ~$289 in 2025), while EUR/USD mostly moved
within a narrower range (roughly 0.96–1.25) without a clear long-term trend.

I then computed the correlation between GLD and EUR/USD two ways:
- **Overall (all 10 years combined):** correlation ≈ -0.146 (weak negative)
- **Year by year:** correlation varied widely and was often *positive*
  (e.g., 0.85 in 2018, 0.94 in 2025), with only a couple of years showing
  negative correlation (e.g., -0.73 in 2019)

This mismatch between the overall and yearly correlations is an example of
**Simpson's Paradox**: the weak negative correlation at the 10-year level is
likely driven by the fact that GLD trends upward over time while EUR/USD does
not, rather than the two variables consistently moving in opposite directions
within any given year. The exact cause of the year-to-year variation is not
clear from this dataset alone and would need further research (e.g., macro
events per year, or simply small yearly sample sizes of ~250 trading days).

## Part 2: GLD vs SPX — Linear Regression
I trained a simple linear regression model (`scikit-learn`) using SPX to
predict GLD.

- **Overall model (10 years of data):** R² ≈ 0.859 — SPX appears to explain
  about 86% of the variance in GLD.
- **Year-by-year models:** R² values were mostly much lower and inconsistent
  (e.g., 0.016 in 2015, 0.003 in 2021, 0.03 in 2025), with 2024 as a notable
  exception (R² ≈ 0.826).

This gap suggests the high overall R² is likely inflated by the fact that
**both SPX and GLD trend upward over the decade**, rather than reflecting a
consistently strong relationship between them within any single year. This is
a useful reminder that a high R² over a long time span doesn't necessarily
mean two variables are meaningfully predictive of each other in the short
term — it may partly reflect a shared long-term trend instead. It's an
interesting parallel to the idea that long-term and short-term views of the
same assets can tell different stories, though confirming that would require
analysis beyond the scope of this project.

## Visualizations
- **`figures/gld_vs_time.png`** — Line chart of GLD price over time. Chosen
  to clearly show the trend and timing of major shifts (e.g., a spike around
  2020, and a sharp rise from 2024 onward).
- **`figures/spx_vs_gld_scatter.png`** — Scatter plot of SPX vs GLD with the
  fitted regression line overlaid. Chosen because this is a relationship
  between two continuous variables, and the plot makes visible how the fitted
  line fails to closely track the data in the middle and later ranges of SPX.

## Limitations & Next Steps
- This analysis only tests two candidate variables (EUR/USD, SPX) against
  GLD; many other factors likely influence gold prices (inflation, interest
  rates, central bank demand, geopolitical events) that are outside the scope
  of this dataset.
- Yearly correlation/R² values are based on relatively small samples (~250
  trading days per year), which may make them less stable.
- Findings here are correlational, not causal — no claims are made about
  *why* GLD and SPX or EUR/USD move as they do.

# Polars vs Pandas Performance Comparison

I also compared Pandas and Polars performance on two operations: reading the
CSV file and grouping by year to compute yearly averages (polars_comparison.py).

Results were inconsistent between two separate runs — in one run Polars was
faster at reading the CSV, but in the groupby comparison Polars was slower
than Pandas. This inconsistency suggests the results are not reliable from a
single run and would need to be averaged over multiple runs to draw a solid
conclusion.

More importantly, this dataset is quite small (2,666 rows), which likely
limits any real performance difference between the two libraries. Polars is
built on a compiled Rust engine and is generally expected to show its
advantage on larger datasets and more complex operations, where the
per-operation overhead becomes negligible relative to the actual work done.
On a small dataset like this, that overhead itself (e.g. starting up
Polars's engine) may outweigh any gains, which could explain why Polars was
not consistently faster here.

## Files
- `analysis.py` — main data analysis and modeling script
- `data/gold_data_2015_25.csv` — dataset
- `figures/` — generated plots
- `notebooks/rust_vs_python_intro.ipynb` — Rust ownership notebook (Question 2)
- `polars_comparison.py` — Pandas vs Polars performance comparison