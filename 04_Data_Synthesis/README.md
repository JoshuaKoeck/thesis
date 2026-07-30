# Data Synthesis

This folder combines the article sentiment data (from `02_Categorization`) with the market return data (from `03_Market_Data`) into one table, then tests the thesis hypotheses (H1–H3) via pooled panel regressions.

- **main.py**: builds the combined table. Reads `results_big.csv` (the concatenated categorization output — see the root README for how this file is assembled) and `category_analytics.xlsx`, aggregates article sentiment by date/region/asset class, forward-fills both series over the full analysis window, and writes the result to `combined_trend.xlsx` / `combined_trend.csv`, plus an interactive `combined_trend.html` chart for a quick visual check.
- **panel_regression.py**: tests H1–H3 via pooled panel regressions with entity fixed effects, one regression per (asset, horizon) combination. Reads `combined_trend.xlsx`, writes `panel_regression_results.xlsx`.

Run `main.py` first, then `panel_regression.py`.

## Glossary

Terms used throughout this folder's code and output tables:

- **H1 (regional effect)** — does the market's reaction to ESG sentiment differ by region (Europe vs. North America vs. Asia)?
- **H2 (asymmetry)** — does negative sentiment move markets more than equally-sized positive sentiment?
- **H3 (size effect)** — do small-cap stocks react more strongly to sentiment than large-cap stocks?
- **CAR** — cumulative (abnormal) return: the compounded market return over a forward window `[t, t+horizon]`, used as the dependent variable in the panel regressions instead of a single day's return.
- **BH-FDR (Benjamini-Hochberg)** — a multiple-testing correction that controls the expected false discovery rate across a family of tests, applied separately to the main-effect terms, the H1 regional comparisons, and the H2 asymmetry tests. Reported in the output tables as `sig_bh` / `sig_bh_diff`.
- **Wald test** — used for H2, to test whether the positive- and negative-sentiment coefficients differ, and for the H1 regional total effects; the test statistic is the coefficient difference divided by its standard error, built from the joint covariance of the two coefficients (delta method).
- **Entity fixed effects** — in the panel regressions, one fixed effect per (region, cap) entity, which absorbs each entity's average return so the estimated sentiment coefficient reflects within-entity variation over time rather than cross-entity level differences.
- **Cluster-robust standard errors** — standard errors clustered by date rather than by entity, since there are only a handful of entities per asset (too few for entity-clustering) but hundreds of trading days; this also accounts for common market-wide shocks hitting all entities on the same day.
