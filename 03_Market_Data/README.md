# Market Data

In this section it's about extracting the market data for the thesis. The data is extracted from Refintiv Eikon. Code for running tests against the extracted data are also to be found in this folder

- main.py: Code for extracting market data from refinitv eikon for a set timeframe using OHLC. Reads `indices.json`, writes `prices_daily.xlsx` (long format, one row per ticker/day) and `category_analytics.xlsx` (daily category-level index levels, returns, and rolling volatility).
- timeframe_check.py: Check how many rows of data were extracted for given output data. Reads `prices_daily.xlsx` and prints a coverage summary per category (no file output).
- market_data_tests.py: Code for running mathematical tests against the market data to check for consistency and correctness (stationarity, normality, autocorrelation, outliers). Reads `prices_daily.xlsx`, writes `data_quality_tests.xlsx`.
- descriptive_table.py: This file contains the logic for the descriptive table in the thesis. It takes the extracted market data and outputs a table with descriptive statistics for each index. Reads `prices_daily.xlsx`, writes `descriptive_table.xlsx`.

---

The following table lists the tickers of the indices used in this thesis. The "Green" column contains the tickers for the ESG indices, while the "Brown" column contains the tickers for the corresponding energy indices.

| Region        | Size  | Green/Brown | Name                                                      | Ticker   | ISIN         |
| ------------- | ----- | ----------- | --------------------------------------------------------- | -------- | ------------ |
| Europe        | Large | Green       | Xtrackers MSCI Europe ESG UCITS ETF 1C                    | XZEU.F   | IE00BFMNHK08 |
| Europe        | Large | Brown       | iShares MSCI Europe Energy Sector UCITS ETF               | ESIE.F   | IE00BMW42637 |
| Europe        | Small | Green       | BNP Paribas Easy MSCI Europe Small Caps SRI PAB UCITS ETF | EESM.F   | LU1291101555 |
| Europe        | Small | Brown       | Amundi STOXX Europe 600 Energy Screened UCITS ETF Acc     | ENRG.PA  | LU1834988278 |
| North America | Large | Green       | iShares ESG Optimized MSCI USA ETF                        | SUSA     | —            |
| North America | Large | Brown       | iShares S&P 500 Energy Sector UCITS ETF                   | IESU.L   | IE00B42NKQ00 |
| North America | Small | Green       | UBS MSCI USA Small Cap Selection UCITS ETF                | USSMC.MI | IE000XFXBGR0 |
| North America | Small | Brown       | Invesco S&P SmallCap Energy ETF                           | PSCE     | US46138G4745 |
| Asia          | Large | Green       | Global X China Clean Energy ETF                           | 9809.HK  | —            |
| Asia          | Large | Brown       | Global X MSCI China Energy ETF                            | CHIE     | —            |
