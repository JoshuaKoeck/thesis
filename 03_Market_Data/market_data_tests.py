import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

INPUT_FILE = "prices_daily.xlsx"
OUTPUT_FILE = "data_quality_tests.xlsx"


def run_tests_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Run the four standard data-quality tests per category.

    Expects a long-format DataFrame with columns: date, category, close (or
    avg_price). Computes daily returns per category, then runs, for each
    return series:
      - ADF test: checks whether the return series is stationary (returns
        fluctuate around a constant mean and variance). A non-stationary
        series would imply exploitable trends, which is inconsistent with a
        stable, efficient market.
      - Jarque-Bera test: checks whether the returns are normally
        distributed. This commonly fails for financial data, but the result
        still informs the choice of models and methods in later analyses.
      - Ljung-Box test (lag 10): checks for autocorrelation in the return
        series, i.e. whether today's return is predictable from the last 10
        days without outside information. Significant autocorrelation would
        be inconsistent with a stable, efficient market.
      - Outlier check: flags days where the return is more than 5 standard
        deviations from the mean, as a way to catch potential data errors.

    Also reports descriptive statistics for each series, including excess
    kurtosis (kurtosis relative to the normal distribution's value of 3).
    """
    results = []

    price_col = "close" if "close" in df.columns else "avg_price"

    for category, grp in df.groupby("category"):
        grp = grp.sort_values("date").copy()
        returns = grp[price_col].pct_change().dropna()

        if len(returns) < 30:
            print(f"[WARN] {category}: too few observations ({len(returns)})")
            continue

        adf_stat, adf_p, *_ = adfuller(returns, autolag="AIC")
        is_stationary = adf_p < 0.05

        jb_stat, jb_p = stats.jarque_bera(returns)
        is_normal = jb_p > 0.05

        lb = acorr_ljungbox(returns, lags=[10], return_df=True)
        lb_p = lb["lb_pvalue"].iloc[0]
        has_autocorr = lb_p < 0.05

        z_scores = np.abs((returns - returns.mean()) / returns.std())
        n_outliers = (z_scores > 5).sum()

        results.append(
            {
                "category": category,
                "n_observations": len(returns),
                "mean_return": round(returns.mean(), 6),
                "std_return": round(returns.std(), 6),
                "skewness": round(stats.skew(returns), 4),
                "kurtosis": round(stats.kurtosis(returns), 4),
                "adf_statistic": round(adf_stat, 4),
                "adf_pvalue": round(adf_p, 4),
                "stationary": "Yes" if is_stationary else "No",
                "jb_statistic": round(jb_stat, 2),
                "jb_pvalue": round(jb_p, 4),
                "normal_dist": "Yes" if is_normal else "No",
                "ljungbox_pvalue": round(lb_p, 4),
                "autocorrelation": "Yes" if has_autocorr else "No",
                "n_outliers_5sd": int(n_outliers),
            }
        )

    return pd.DataFrame(results)


def main():
    """Load the daily price file, run the per-category data-quality tests,
    and write the results to an Excel file."""
    df = pd.read_excel(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    print(f"[INFO] Loaded {len(df)} rows, {df['category'].nunique()} categories")

    results = run_tests_per_category(df)
    results = results.sort_values("category").reset_index(drop=True)

    results.to_excel(OUTPUT_FILE, index=False)

    print(f"\n[OK] Saved → {OUTPUT_FILE}\n")
    print(
        results[
            [
                "category",
                "stationary",
                "normal_dist",
                "autocorrelation",
                "n_outliers_5sd",
            ]
        ].to_string(index=False)
    )
    print(
        "\n[INFO] Stored the results of the data quality tests for each category in the output excel file."
    )


if __name__ == "__main__":
    main()
