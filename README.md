# Thesis

This repository contains the code and documentation for my thesis project. The project focuses on analyzing the performance and characteristics of ESG (Environmental, Social, and Governance) and energy sector indices across different regions and market capitalizations. For more Information read Thesis.pdf in the root directory of this repository.

## Pipeline

The code runs as a four-stage pipeline. Each stage is a folder, run in order; each stage's output file(s) become the next stage's input. See each folder's own README for details on the scripts inside it.

| Stage | Folder                                             | What it does                                                                                                                                                  | Reads                                                                 | Writes                                                                                     |
| ----- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 1     | [`01_Extraction`](01_Extraction/README.md)         | Pulls raw news events from GDELT day by day and scrapes the article text for each URL                                                                         | — (queries the GDELT API directly)                                    | `output.parquet`                                                                           |
| 2     | [`02_Categorization`](02_Categorization/README.md) | Cleans and keyword-prefilters the scraped articles, then classifies the survivors with Claude (ESG relevance, sentiment, region, etc.)                        | `output.parquet`                                                      | `filtered_articles.parquet`, `batch_id.txt` → `results_final.csv`                          |
| 3     | [`03_Market_Data`](03_Market_Data/README.md)       | Downloads daily OHLCV price data for the ESG/energy index constituents (Refinitiv Eikon) and builds category-level return/volatility analytics                | `indices.json`                                                        | `prices_daily.xlsx`, `category_analytics.xlsx`                                             |
| 4     | [`04_Data_Synthesis`](04_Data_Synthesis/README.md) | Combines article sentiment and market return trends into one table, then tests the thesis hypotheses (H1–H3) via pooled panel regressions | `results_final.csv` (as `results_big.csv`), `category_analytics.xlsx` | `combined_trend.xlsx` → `panel_regression_results.xlsx` |

A couple of filenames change between stages — this is a manual/naming step, not a bug in the scripts:

- Stage 2 can be run in multiple batches, producing several `results_final*.csv` files; these are concatenated by hand into `results_big.csv` before stage 4 (`ARTICLES_PATH` in `04_Data_Synthesis/main.py`).
