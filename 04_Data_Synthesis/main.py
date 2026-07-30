from __future__ import annotations

from typing import Final

import pandas as pd
import plotly.graph_objects as go

REGIONS: Final[list[str]] = ["asia", "north_america", "europe", "row", "other"]
CAPS: Final[list[str]] = ["large", "small"]
START: Final[str] = "2021-01-01"
END: Final[str] = "2025-12-31"

VALID_ASSETS: Final[list[str]] = ["brown", "green"]

ARTICLES_PATH: Final[str] = "results_big.csv"
MARKET_PATH: Final[str] = "category_analytics.xlsx"
OUTPUT_XLSX: Final[str] = "combined_trend.xlsx"
OUTPUT_CSV: Final[str] = "combined_trend.csv"
OUTPUT_HTML: Final[str] = "combined_trend.html"


def load_articles(path: str) -> pd.DataFrame:
    """Load the raw article data (results_big.csv) from disk."""
    return pd.read_csv(path)


def load_market(path: str) -> pd.DataFrame:
    """Load the market data (category_analytics.xlsx).

    The source file uses German number formatting (comma as decimal
    separator, dot as thousands separator), which is passed explicitly to pandas.
    """
    return pd.read_excel(path, decimal=",", thousands=".")


def get_full_day_index(start: str, end: str) -> pd.DatetimeIndex:
    """Create a gap-free daily date index covering the full analysis window."""
    return pd.date_range(start, end, freq="D")


def prepare_article_rows(articles: pd.DataFrame) -> pd.DataFrame:
    """Prepare the raw article data for aggregation.

    Normalizes region names, converts the date column, filters to articles
    with a valid asset_category (brown/green), and computes the
    reach-weighted sentiment value.
    """
    articles = articles.copy()
    articles["region_norm"] = articles["region"].str.replace("-", "_", regex=False)
    articles["date"] = pd.to_datetime(articles["SQLDATE"], format="%Y%m%d")

    rows = articles[articles["asset_category"].isin(VALID_ASSETS)].copy()
    rows["asset"] = rows["asset_category"]
    rows["signed_reach"] = rows["sentiment"] * rows["reach"]
    return rows


def aggregate_article_trend(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reach-weighted sentiment values by date, region, and asset class."""
    grp = rows.groupby(["date", "region_norm", "asset"])
    return (grp["signed_reach"].sum() / grp["reach"].sum()).rename("est_return").reset_index()


def build_asset_grid(trend: pd.DataFrame, asset: str, full_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Pivot the aggregated trend of a single asset class into a date x region matrix.

    The index is expanded to the full analysis window and the columns to the
    fixed region order. Days without articles are forward-filled with the
    value from the most recent day that had coverage, so every day in the
    analysis window carries a value. Days before the first covered day
    remain NaN, since there is no prior value to fill from.
    """
    pivot = trend[trend["asset"] == asset].pivot_table(
        index="date", columns="region_norm", values="est_return"
    )
    return pivot.reindex(index=full_days, columns=REGIONS).ffill()


def build_articles_trend(articles: pd.DataFrame, full_days: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[str]]:
    """Build the full article sentiment table (date x region x asset class).

    Returns the combined wide table together with the fixed column order.
    """
    rows = prepare_article_rows(articles)
    trend = aggregate_article_trend(rows)

    art_brown = build_asset_grid(trend, "brown", full_days).add_prefix("art_brown_")
    art_green = build_asset_grid(trend, "green", full_days).add_prefix("art_green_")

    articles_trend = pd.concat([art_brown, art_green], axis=1)
    articles_trend.index.name = "date"

    art_cols = [f"art_brown_{r}" for r in REGIONS] + [f"art_green_{r}" for r in REGIONS]
    return articles_trend[art_cols], art_cols


def build_market_wide(market: pd.DataFrame, full_days: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[str]]:
    """Build the market data table (date x asset class x region x market cap).

    Days without market data (e.g. weekends and holidays) are forward-filled
    with the value from the most recent trading day, so every day in the
    analysis window carries a value. Days before the first available trading
    day remain NaN, since there is no prior value to fill from. Returns the
    wide table with flattened column names together with the fixed column
    order.
    """
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])
    market["region_norm"] = market["region"].str.replace("-", "_", regex=False)

    market_wide = market.pivot_table(
        index="date",
        columns=["type", "region_norm", "cap"],
        values="index_return",
    )
    market_wide.columns = [f"mkt_{asset}_{region}_{cap}" for (asset, region, cap) in market_wide.columns]
    market_wide = market_wide.reindex(full_days).ffill()
    market_wide.index.name = "date"

    mkt_cols = [
        f"mkt_{asset}_{region}_{cap}"
        for asset in ("brown", "green")
        for region in REGIONS
        for cap in CAPS
        if f"mkt_{asset}_{region}_{cap}" in market_wide.columns
    ]
    return market_wide[mkt_cols], mkt_cols


def combine_trends(articles_trend: pd.DataFrame, market_wide: pd.DataFrame) -> pd.DataFrame:
    """Combine article and market data side by side into a single table."""
    combined = pd.concat([articles_trend, market_wide], axis=1)
    combined.index.name = "date"
    return combined


def save_tabular_outputs(combined: pd.DataFrame, xlsx_path: str, csv_path: str) -> None:
    """Write the combined table to an Excel file and a CSV file."""
    combined.to_excel(xlsx_path, sheet_name="data")
    combined.to_csv(csv_path)


def add_article_traces(fig: go.Figure, combined: pd.DataFrame, art_cols: list[str]) -> None:
    """Add a line-with-markers trace for each article sentiment column on the left y-axis.

    Markers ensure that isolated single-day points (surrounded by NaN) remain visible.
    """
    for col in art_cols:
        fig.add_trace(
            go.Scatter(
                x=combined.index,
                y=combined[col],
                name=col,
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=7),
                connectgaps=True,
                yaxis="y1",
                visible=True,
            )
        )


def add_market_traces(fig: go.Figure, combined: pd.DataFrame, mkt_cols: list[str]) -> None:
    """Add a dotted-line trace for each market return column on the right y-axis.

    Gaps (days without market data) are bridged so the line stays continuous
    across non-trading days instead of breaking.
    """
    for col in mkt_cols:
        fig.add_trace(
            go.Scatter(
                x=combined.index,
                y=combined[col],
                name=col,
                mode="lines",
                line=dict(width=1, dash="dot"),
                connectgaps=True,
                yaxis="y2",
                visible="legendonly",
            )
        )


def build_figure(combined: pd.DataFrame, art_cols: list[str], mkt_cols: list[str]) -> go.Figure:
    """Build the interactive Plotly figure combining article and market data.

    Legend entries are clickable (show/hide), dragging zooms into a range,
    and the range slider allows scrubbing through the date axis.
    """
    fig = go.Figure()
    add_article_traces(fig, combined, art_cols)
    add_market_traces(fig, combined, mkt_cols)

    fig.update_layout(
        title="Article sentiment vs. market index_return by region/asset/cap",
        hovermode="x unified",
        xaxis=dict(title="date", rangeslider=dict(visible=True)),
        yaxis=dict(title="article sentiment [-1, 1]", range=[-1, 1], side="left"),
        yaxis2=dict(
            title="market index_return", overlaying="y", side="right", showgrid=False
        ),
        legend=dict(font=dict(size=9)),
        height=700,
    )
    return fig


def print_summary(combined: pd.DataFrame, art_cols: list[str], mkt_cols: list[str]) -> None:
    """Print a preview of the combined table together with the column names used."""
    pd.set_option("display.max_columns", None, "display.width", 200)
    print("=== combined (articles + market, side by side) ===")
    print(combined.head(15))
    print("\narticle cols:", art_cols)
    print("market  cols:", mkt_cols)


def main() -> None:
    """Run the full pipeline: load, aggregate, combine, save, visualize."""
    full_days = get_full_day_index(START, END)

    articles = load_articles(ARTICLES_PATH)
    market = load_market(MARKET_PATH)

    articles_trend, art_cols = build_articles_trend(articles, full_days)
    market_wide, mkt_cols = build_market_wide(market, full_days)

    combined = combine_trends(articles_trend, market_wide)
    print_summary(combined, art_cols, mkt_cols)

    save_tabular_outputs(combined, OUTPUT_XLSX, OUTPUT_CSV)

    fig = build_figure(combined, art_cols, mkt_cols)
    fig.write_html(OUTPUT_HTML)
    print(f"\nwrote {OUTPUT_HTML} (open in browser, toggle via legend)")


if __name__ == "__main__":
    main()