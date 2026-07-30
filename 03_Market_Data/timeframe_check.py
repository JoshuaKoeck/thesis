import pandas as pd

INPUT_FILE = "prices_daily.xlsx"


def build_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise data coverage per category: row/day/ticker counts and date range.

    Useful for spotting categories with missing days, too few tickers, or a
    shorter history than the rest of the dataset.
    """
    return (
        df.groupby("category")
        .agg(
            n_rows=("date", "size"),
            n_unique_days=("date", "nunique"),
            n_tickers=("ticker", "nunique"),
            first_day=("date", "min"),
            last_day=("date", "max"),
        )
        .sort_values("n_rows")
    )


def main():
    """Load the daily prices file and print the per-category coverage summary."""
    df = pd.read_excel(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    check = build_coverage_table(df)
    print(check.to_string())


if __name__ == "__main__":
    main()
