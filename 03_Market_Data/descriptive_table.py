import numpy as np
import pandas as pd

INPUT_FILE = "prices_daily.xlsx"
OUTPUT_FILE = "descriptive_table.xlsx"
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"


def build_descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive return statistics per category for the thesis table.

    Expects a long-format DataFrame with columns: date, category, and either
    close or avg_price. For each category, computes daily returns and derives
    trading day count, average daily return, annualised volatility (daily std
    scaled by sqrt(252) trading days), and the cumulative return over the
    full window.

    Returns:
        DataFrame sorted by Category with one row per category.
    """
    rows = []
    price_col = "close" if "close" in df.columns else "avg_price"

    for category, grp in df.groupby("category"):
        grp = grp.sort_values("date")
        returns = grp[price_col].pct_change().dropna()
        cumulative_return = (grp[price_col].iloc[-1] / grp[price_col].iloc[0]) - 1

        rows.append(
            {
                "Category": category,
                "Trading Days": len(returns),
                "Avg Daily Returns (%)": round(returns.mean() * 100, 4),
                "Annual volatility (%)": round(returns.std() * np.sqrt(252) * 100, 2),
                "Cumulative Returns (%)": round(cumulative_return * 100, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("Category")


def main():
    """Load daily prices, restrict to the analysis window, build the
    descriptive statistics table, and write it to OUTPUT_FILE."""
    df = pd.read_excel(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

    table = build_descriptive_table(df)
    table.to_excel(OUTPUT_FILE, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
