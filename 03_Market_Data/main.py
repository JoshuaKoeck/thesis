import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import refinitiv.data as rd
from refinitiv.data.content import historical_pricing
from tqdm import tqdm

INPUT_FILE = "indices.json"
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
OUTPUT_DAILY = "prices_daily.xlsx"
OUTPUT_CATEGORY = "category_analytics.xlsx"
BATCH_SIZE = 50
VOL_WINDOW = 20

_FIELDS = ["OPEN", "HIGH", "LOW", "TRDPRC_1", "ACVOL_UNS"]


def load_tickers(json_path: str) -> pd.DataFrame:
    """Load ticker metadata from the ESG index JSON file.

    The JSON must follow the structure:
        { "region__type__cap": [ {ticker, company, name, description}, ... ], ... }

    Each category key is split on "__" to extract region, type (green/brown),
    and market cap tier. Every asset in the list becomes one row in the output.

    Args:
        json_path: Path to the JSON file containing the index definitions.

    Returns:
        DataFrame with columns: ticker, company, name, description,
        region, type, cap, category. One row per ticker.
    """
    path = Path(json_path)
    if not path.exists():
        print(f"[ERROR] File not found: {json_path}")
        sys.exit(1)

    with open(path) as f:
        indices = json.load(f)

    rows = []
    for key, assets in indices.items():
        parts = key.split("__")
        try:
            region = parts[0]
            typ = parts[1]
            cap = parts[2]
        except IndexError:
            print(f"[WARN] Skipping malformed category key: {key}")
            continue

        for asset in assets:
            rows.append(
                {
                    "ticker": asset.get("ticker", ""),
                    "company": asset.get("company", ""),
                    "name": asset.get("name", ""),
                    "description": asset.get("description", ""),
                    "region": region,
                    "type": typ,
                    "cap": cap,
                    "category": key,
                }
            )

    df = pd.DataFrame(rows)
    print(f"[INFO] Loaded {len(df)} tickers from {json_path}")
    return df


def fetch_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data for a list of RICs from Refinitiv.

    Refinitiv returns a (ticker, field) MultiIndex on the columns with a
    DatetimeIndex as the row index. Because duplicate field labels can appear
    for a single ticker, we iterate per-ticker rather than using stack(), drop
    any duplicate field columns, then concatenate into long format.

    The Refinitiv end date is inclusive, so ``end`` is shifted back by one day
    internally to preserve the exclusive-end semantics used throughout the pipeline.

    The avg_price column is computed as the OHLC4 average:
        (open + high + low + close) / 4
    Rows where close is NaN (no trade recorded that day) are dropped.

    Args:
        tickers: Refinitiv RIC symbols to fetch. If your JSON uses a different
            convention (e.g. Bloomberg tickers) a mapping layer is needed before
            calling this function.
        start: Start date in YYYY-MM-DD format (inclusive).
        end: End date in YYYY-MM-DD format (exclusive; shifted to inclusive internally).

    Returns:
        Long-format DataFrame with columns: date, ticker, open, high, low,
        close, volume, avg_price. Returns an empty DataFrame if Refinitiv
        returns no data or raises an exception.
    """
    end_inclusive = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        response = historical_pricing.summaries.Definition(
            universe=tickers,
            start=start,
            end=end_inclusive,
            fields=_FIELDS,
            interval=historical_pricing.Intervals.DAILY,
        ).get_data()
    except Exception as exc:
        print(f"[WARN] Refinitiv fetch failed for batch starting {tickers[0]}: {exc}")
        return pd.DataFrame()

    if response is None or response.data.df is None or response.data.df.empty:
        return pd.DataFrame()

    raw: pd.DataFrame = response.data.df.copy()

    if isinstance(raw.columns, pd.MultiIndex):
        ticker_names = raw.columns.get_level_values(0).unique()
        frames = []
        for tic in ticker_names:
            sub = raw[tic]
            sub = sub.loc[:, ~sub.columns.duplicated()]
            sub = sub.copy().reset_index()
            sub.columns = [str(c) for c in sub.columns]
            sub = sub.rename(columns={sub.columns[0]: "date"})
            sub["ticker"] = tic
            frames.append(sub)
        raw = pd.concat(frames, ignore_index=True)
    elif isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index()
        raw.columns = [str(c) for c in raw.columns]
        raw = raw.rename(columns={raw.columns[0]: "date", raw.columns[1]: "ticker"})
    else:
        raw = raw.reset_index()
        raw.columns = [str(c) for c in raw.columns]
        raw = raw.rename(columns={raw.columns[0]: "date"})
        if "Instrument" in raw.columns:
            raw = raw.rename(columns={"Instrument": "ticker"})
        elif len(tickers) == 1:
            raw["ticker"] = tickers[0]

    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={"trdprc_1": "close", "acvol_uns": "volume"})

    keep = [
        c
        for c in ["date", "ticker", "open", "high", "low", "close", "volume"]
        if c in raw.columns
    ]
    raw = raw[keep].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.dropna(subset=["close"])

    price_cols = [c for c in ["open", "high", "low", "close"] if c in raw.columns]
    raw["avg_price"] = raw[price_cols].mean(axis=1)

    return raw


def fetch_all(
    tickers: list[str], start: str, end: str, batch_size: int
) -> pd.DataFrame:
    """Fetch price data for all tickers by splitting them into batches.

    Calls fetch_batch for each chunk and concatenates the results into a single
    long-format DataFrame. Batching is necessary because Refinitiv imposes a
    practical limit on the number of RICs per request. Progress is shown via a
    tqdm progress bar.

    Args:
        tickers: Full list of Refinitiv RIC symbols to fetch.
        start: Start date in YYYY-MM-DD format (inclusive).
        end: End date in YYYY-MM-DD format (exclusive).
        batch_size: Maximum number of tickers per Refinitiv request. Reduce if
            you hit API or memory limits.

    Returns:
        Long-format DataFrame with columns: date, ticker, open, high, low,
        close, volume, avg_price. Returns an empty DataFrame if no batch
        returned any data.
    """
    frames = []
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch in tqdm(batches, desc="Fetching batches"):
        df = fetch_batch(batch, start, end)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("[WARN] No price data retrieved.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_category_analytics(prices_df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """Compute daily cross-sectional analytics per category from raw price data.

    For each (date, category) pair the function produces:
    - Cross-sectional price statistics across all tickers (mean, median, min, max).
    - An equal-weighted index level rebased to 100 on the first date, built from
      the daily mean of per-ticker percentage returns.
    - Annualised rolling volatility of the index return series.

    The category key (e.g. "europe__green__large") is split into separate region,
    type, and cap columns for easier filtering downstream.

    Args:
        prices_df: Long-format DataFrame with columns: date, ticker, avg_price,
            category. Additional columns are ignored.
        vol_window: Rolling window length in trading days used for the annualised
            volatility calculation. Requires at least 5 observations (min_periods=5).

    Returns:
        DataFrame sorted by (category, date) with columns: date, category, region,
        type, cap, n_tickers, mean_price, median_price, min_price, max_price,
        mean_daily_return, index_return, index_level, rolling_vol_{vol_window}d.
    """
    df = prices_df[["date", "ticker", "avg_price", "category"]].copy()
    df = df.dropna(subset=["avg_price"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["category", "ticker", "date"])

    df["ticker_return"] = df.groupby(["category", "ticker"])["avg_price"].pct_change()

    agg = (
        df.groupby(["date", "category"])
        .agg(
            n_tickers=("ticker", "nunique"),
            mean_price=("avg_price", "mean"),
            median_price=("avg_price", "median"),
            min_price=("avg_price", "min"),
            max_price=("avg_price", "max"),
            mean_daily_return=("ticker_return", "mean"),
        )
        .reset_index()
    )

    vol_col = f"rolling_vol_{vol_window}d"
    frames = []
    for cat, grp in agg.groupby("category"):
        grp = grp.sort_values("date").copy()
        grp["index_return"] = grp["mean_daily_return"].fillna(0)
        grp["index_level"] = 100 * (1 + grp["index_return"]).cumprod()
        grp[vol_col] = grp["index_return"].rolling(
            vol_window, min_periods=5
        ).std() * np.sqrt(252)
        frames.append(grp)

    result = pd.concat(frames, ignore_index=True)

    split = result["category"].str.split("__", expand=True)
    result.insert(2, "region", split[0])
    result.insert(3, "type", split[1])
    result.insert(4, "cap", split[2])

    green = result[result["type"] == "green"][
        ["date", "region", "cap", "index_level", "index_return"]
    ].rename(columns={"index_level": "_g_level", "index_return": "_g_return"})
    brown = result[result["type"] == "brown"][
        ["date", "region", "cap", "index_level", "index_return"]
    ].rename(columns={"index_level": "_b_level", "index_return": "_b_return"})
    paired = green.merge(brown, on=["date", "region", "cap"], how="inner")

    paired = paired[["date", "region", "cap"]]

    result = result.merge(paired, on=["date", "region", "cap"], how="left")

    float_cols = [
        "mean_price",
        "median_price",
        "min_price",
        "max_price",
        "mean_daily_return",
        "index_return",
        "index_level",
        vol_col,
    ]
    result[float_cols] = result[float_cols].round(6)

    col_order = [
        "date",
        "category",
        "region",
        "type",
        "cap",
        "n_tickers",
        "mean_price",
        "median_price",
        "min_price",
        "max_price",
        "mean_daily_return",
        "index_return",
        "index_level",
        vol_col,
    ]
    return result[col_order].sort_values(["category", "date"]).reset_index(drop=True)


def main():
    """Entry point: open a Refinitiv session, fetch all prices, and write outputs.

    Reads credentials from ~/.refinitiv/config.json by default. Set the
    environment variable RD_LIB_CFG_PATH to override the config location.

    Produces two Excel files:
        - OUTPUT_DAILY: long-format price data, one row per (date, ticker).
        - OUTPUT_CATEGORY: daily category-level analytics including index levels,
          rolling volatility, and green-vs-brown comparison metrics.

    Prints a coverage summary and snapshot tables for the last available date
    before closing the Refinitiv session.
    """
    print("[INFO] Opening Refinitiv session …")
    rd.open_session()

    try:
        meta_df = load_tickers(INPUT_FILE)
        tickers = meta_df["ticker"].tolist()

        print(f"[INFO] Fetching prices from {START_DATE} to {END_DATE} …")
        prices_df = fetch_all(tickers, START_DATE, END_DATE, BATCH_SIZE)

        if prices_df.empty:
            print("[ERROR] No data fetched. Check RICs and date range.")
            sys.exit(1)

        prices_df = prices_df.merge(
            meta_df[["ticker", "company", "name", "region", "type", "cap", "category"]],
            on="ticker",
            how="left",
        )
        prices_df = prices_df.sort_values(["ticker", "date"]).reset_index(drop=True)

        prices_df.to_excel(OUTPUT_DAILY, index=False)
        print(
            f"[OK] Long format saved       → {OUTPUT_DAILY}  ({len(prices_df):,} rows)"
        )

        cat_df = build_category_analytics(prices_df, VOL_WINDOW)
        cat_df.to_excel(OUTPUT_CATEGORY, index=False)
        n_cats = cat_df["category"].nunique()
        n_days = cat_df["date"].nunique()
        print(
            f"[OK] Category analytics saved → {OUTPUT_CATEGORY}  ({n_cats} categories × {n_days} days)"
        )

        print("\n── Coverage ──────────────────────────────────────────────────────")
        missing = set(tickers) - set(prices_df["ticker"].unique())
        if missing:
            print(f"[WARN] {len(missing)} tickers returned no data: {sorted(missing)}")
        else:
            print("[OK]  All tickers returned data.")
        print(
            f"      {prices_df['ticker'].nunique()} tickers  ×  {prices_df['date'].nunique()} trading days"
        )

        vol_col = f"rolling_vol_{VOL_WINDOW}d"
        last = (
            cat_df.sort_values("date")
            .groupby("category")
            .tail(1)[["category", "index_level", vol_col, "n_tickers"]]
            .sort_values("category")
        )
        print("\n── Index levels on last date ─────────────────────────────────────")
        print(last.to_string(index=False))

    finally:
        rd.close_session()
        print("[INFO] Refinitiv session closed.")


if __name__ == "__main__":
    main()
