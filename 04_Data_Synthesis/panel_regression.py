"""
Test H1-H3 with pooled panel regressions instead of per-cell correlations.

This replaces the 110-cell correlation grid (compare_trends_inference.py) with
one panel regression per asset x horizon combination. The dependent variable is
the cumulative return over [t, t+horizon] rather than the single-day return at
lag k, since a one-day-isolated effect k days later is a much narrower target
than a drift that builds up over the window.

H1 (regional effects) is tested via the sentiment:C(region) interaction terms,
with Asia as the baseline category. H3 (small-cap sensitivity) is tested via
the sentiment:small_cap interaction term. Both are estimated in the same
"main effect" model, pooling all regions and caps for a given asset and
horizon, which gives far more statistical power than testing each region/cap
cell in isolation.

H2 (asymmetry) needs its own model, since a single sentiment coefficient
cannot show whether negative and positive sentiment have different-sized
effects; averaging over both signs would cancel the asymmetry. The asymmetry
model instead regresses on sentiment_pos and sentiment_neg separately, and a
Wald test compares the two coefficients.

Entity fixed effects (one per region x cap combination) absorb any
region/cap-level average return, so the coefficients reflect within-entity
variation over time. Standard errors are clustered by date rather than by
entity, since there are only up to 6 entities per asset (too few for
cluster-robust asymptotics) but hundreds of trading days, which better
satisfies the requirements for cluster-robust inference and also accounts for
common daily market-wide shocks hitting all entities at once.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import PanelEffectsResults
from scipy import stats

REGIONS: Final[list[str]] = ["asia", "north_america", "europe"]
CAPS: Final[list[str]] = ["large", "small"]
ASSETS: Final[list[str]] = ["brown", "green"]
HORIZONS: Final[list[int]] = [0, 1, 2, 3, 5, 10]

ALPHA: Final[float] = 0.05
MIN_OBSERVATIONS: Final[int] = 30

COMBINED_TREND_PATH: Final[str] = "combined_trend.xlsx"
OUTPUT_XLSX: Final[str] = "panel_regression_results.xlsx"


def load_combined_trend(path: str) -> pd.DataFrame:
    """Load the combined article/market trend table produced by the trend pipeline."""
    combined = pd.read_excel(path, sheet_name="data", index_col=0)
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "date"
    return combined


def compute_cumulative_return(returns: pd.Series, horizon: int) -> pd.Series:
    """Compute the forward-looking cumulative return over [t, t+horizon].

    Daily returns are converted to log returns, summed over the forward
    window, and converted back to a simple return. This compounds correctly
    regardless of how large the individual daily returns are.
    """
    log_returns = np.log1p(returns)
    reversed_log_returns = log_returns.iloc[::-1]
    cumulative_log = reversed_log_returns.rolling(window=horizon + 1, min_periods=horizon + 1).sum()
    cumulative_log = cumulative_log.iloc[::-1]
    return np.expm1(cumulative_log)


def build_long_panel(
    combined: pd.DataFrame,
    asset: str,
    regions: list[str],
    caps: list[str],
    horizon: int,
) -> pd.DataFrame:
    """Build a long-format panel for one asset class and horizon.

    Each row is a (region, cap, date) observation with the article sentiment
    on that date and the cumulative market return over [date, date+horizon].
    The same sentiment series is reused for the large- and small-cap rows of
    a region, since sentiment is not measured separately by market cap.
    """
    frames = []
    for region in regions:
        art_col = f"art_{asset}_{region}"
        if art_col not in combined.columns:
            continue
        sentiment = combined[art_col]
        for cap in caps:
            mkt_col = f"mkt_{asset}_{region}_{cap}"
            if mkt_col not in combined.columns:
                continue
            car = compute_cumulative_return(combined[mkt_col], horizon)
            frames.append(pd.DataFrame({
                "date": combined.index,
                "region": region,
                "cap": cap,
                "entity": f"{region}_{cap}",
                "sentiment": sentiment.to_numpy(),
                "car": car.to_numpy(),
            }))

    if not frames:
        return pd.DataFrame(columns=["date", "region", "cap", "entity", "sentiment", "car"])

    panel = pd.concat(frames, ignore_index=True)
    return panel.dropna(subset=["sentiment", "car"])


def add_regression_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns needed by the regression formulas.

    small_cap is a 0/1 dummy used for the H3 interaction. sentiment_pos and
    sentiment_neg split the sentiment score by sign for the H2 asymmetry
    model; each is zero on days where it does not apply, so their sum equals
    the original sentiment score.
    """
    panel = panel.copy()
    panel["small_cap"] = (panel["cap"] == "small").astype(int)
    panel["sentiment_pos"] = panel["sentiment"].clip(lower=0)
    panel["sentiment_neg"] = panel["sentiment"].clip(upper=0)
    return panel


def to_panel_index(panel: pd.DataFrame) -> pd.DataFrame:
    """Set the (entity, date) MultiIndex required by linearmodels' PanelOLS."""
    return panel.set_index(["entity", "date"])


REGION_BASELINE: Final[str] = "north_america"


def region_interaction_term(region: str) -> str:
    """Build the formula's coefficient name for a non-baseline region's interaction term."""
    return f"sentiment:C(region, Treatment(reference='{REGION_BASELINE}'))[T.{region}]"


def run_main_effect_model(panel_indexed: pd.DataFrame) -> PanelEffectsResults:
    """Fit the H1/H3 model: sentiment plus region and small-cap interactions.

    North America is set as the region baseline (via Treatment()) instead of
    the default alphabetical baseline, so the Europe interaction term is
    directly the Europe-vs-North-America comparison that H1 asks for, with
    its own standard error and p-value read straight off the fit, no
    post-estimation Wald test required. Entity fixed effects absorb
    region/cap-level average returns. Standard errors are clustered by date.
    """
    model = PanelOLS.from_formula(
        f"car ~ 1 + sentiment + sentiment:small_cap + sentiment:C(region, Treatment(reference='{REGION_BASELINE}')) + EntityEffects",
        data=panel_indexed,
    )
    return model.fit(cov_type="clustered", cluster_entity=False, cluster_time=True)


def run_asymmetry_model(panel_indexed: pd.DataFrame) -> PanelEffectsResults:
    """Fit the H2 model: separate coefficients for positive and negative sentiment."""
    model = PanelOLS.from_formula(
        "car ~ 1 + sentiment_pos + sentiment_neg + EntityEffects",
        data=panel_indexed,
    )
    return model.fit(cov_type="clustered", cluster_entity=False, cluster_time=True)


def wald_diff_test(results: PanelEffectsResults, param_a: str, param_b: str) -> tuple[float, float, float]:
    """Test whether two regression coefficients differ, using their joint covariance.

    Returns (difference, t-statistic, two-sided p-value). This is the test
    behind H2: whether the negative-sentiment coefficient differs from the
    positive-sentiment coefficient.
    """
    params = results.params
    cov = results.cov
    diff = params[param_a] - params[param_b]
    var_diff = cov.loc[param_a, param_a] + cov.loc[param_b, param_b] - 2 * cov.loc[param_a, param_b]
    se_diff = np.sqrt(var_diff)
    t_stat = diff / se_diff
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=results.df_resid))
    return diff, t_stat, p_value


def extract_main_effect_summary(results: PanelEffectsResults, asset: str, horizon: int) -> list[dict]:
    """Extract one row per coefficient from a main-effect (H1/H3) model."""
    rows = []
    for term in results.params.index:
        rows.append(dict(
            asset=asset,
            horizon=horizon,
            term=term,
            coef=results.params[term],
            se=results.std_errors[term],
            t_stat=results.tstats[term],
            p_value=results.pvalues[term],
        ))
    return rows


def region_total_effect(results: PanelEffectsResults, region: str) -> tuple[float, float, float, float]:
    """Compute one region's total sentiment effect and its inference statistics.

    For the baseline region (REGION_BASELINE) this is just the sentiment
    coefficient, read directly off the fit. For any other region it is
    sentiment + that region's interaction term, with the standard error
    built from the joint covariance of both coefficients (delta method for
    a linear combination). Returns (coef, se, t_stat, p_value).
    """
    if region == REGION_BASELINE:
        coef = results.params["sentiment"]
        var = results.cov.loc["sentiment", "sentiment"]
    else:
        term = region_interaction_term(region)
        coef = results.params["sentiment"] + results.params[term]
        var = (
            results.cov.loc["sentiment", "sentiment"]
            + results.cov.loc[term, term]
            + 2 * results.cov.loc["sentiment", term]
        )
    se = np.sqrt(var)
    t_stat = coef / se
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=results.df_resid))
    return coef, se, t_stat, p_value


def extract_h1_summary(results: PanelEffectsResults, asset: str, horizon: int) -> dict:
    """Extract the H1 regional comparison: Europe vs. North America directly.

    Because North America is the model's baseline region, the Europe
    interaction term already equals the Europe-minus-North-America
    difference, with its own standard error and p-value read directly from
    the fit rather than computed via a post-estimation Wald test. Each
    region's own total effect is included alongside it so the direction
    (not just the difference) can be read off.
    """
    europe_term = region_interaction_term("europe")
    diff = results.params[europe_term]
    t_stat = results.tstats[europe_term]
    p_value = results.pvalues[europe_term]

    asia_coef, asia_se, asia_t, asia_p = region_total_effect(results, "asia")
    europe_coef, europe_se, europe_t, europe_p = region_total_effect(results, "europe")
    na_coef, na_se, na_t, na_p = region_total_effect(results, "north_america")

    return dict(
        asset=asset,
        horizon=horizon,
        coef_asia=asia_coef, p_asia=asia_p,
        coef_europe=europe_coef, p_europe=europe_p,
        coef_north_america=na_coef, p_north_america=na_p,
        diff_europe_minus_namerica=diff,
        t_diff=t_stat,
        p_diff=p_value,
    )


def extract_asymmetry_summary(results: PanelEffectsResults, asset: str, horizon: int) -> dict:
    """Extract the H2 asymmetry summary: both coefficients plus their difference test."""
    diff, t_stat, p_value = wald_diff_test(results, "sentiment_neg", "sentiment_pos")
    return dict(
        asset=asset,
        horizon=horizon,
        coef_pos=results.params["sentiment_pos"],
        p_pos=results.pvalues["sentiment_pos"],
        coef_neg=results.params["sentiment_neg"],
        p_neg=results.pvalues["sentiment_neg"],
        diff_neg_minus_pos=diff,
        t_diff=t_stat,
        p_diff=p_value,
    )


def benjamini_hochberg(pvals: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Return a boolean mask of hypotheses rejected under BH-FDR control."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.zeros_like(p, dtype=bool)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    if passed.any():
        cutoff = np.max(np.where(passed)[0])
        out[order[: cutoff + 1]] = True
    return out


def run_all_horizons(
    combined: pd.DataFrame,
    assets: list[str],
    regions: list[str],
    caps: list[str],
    horizons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the main-effect, H1 regional, and H2 asymmetry models for every asset x horizon.

    Returns three long tables: one row per coefficient for the main-effect
    models, one row per asset x horizon for the H1 regional comparison, and
    one row per asset x horizon for the H2 asymmetry test.
    """
    main_rows: list[dict] = []
    h1_rows: list[dict] = []
    asym_rows: list[dict] = []

    for asset in assets:
        for horizon in horizons:
            panel = build_long_panel(combined, asset, regions, caps, horizon)
            panel = add_regression_features(panel)
            if panel["entity"].nunique() < 2 or len(panel) < MIN_OBSERVATIONS:
                continue

            indexed = to_panel_index(panel)

            main_results = run_main_effect_model(indexed)
            main_rows.extend(extract_main_effect_summary(main_results, asset, horizon))
            h1_rows.append(extract_h1_summary(main_results, asset, horizon))

            asym_results = run_asymmetry_model(indexed)
            asym_rows.append(extract_asymmetry_summary(asym_results, asset, horizon))

    main_df = pd.DataFrame(main_rows)
    h1_df = pd.DataFrame(h1_rows)
    asym_df = pd.DataFrame(asym_rows)

    if len(main_df):
        main_df["sig_bh"] = False
        is_sentiment_term = main_df["term"].isin(["sentiment", "sentiment:small_cap"]) | main_df["term"].str.startswith("sentiment:C(region")
        main_df.loc[is_sentiment_term, "sig_bh"] = benjamini_hochberg(main_df.loc[is_sentiment_term, "p_value"].to_numpy())

    if len(h1_df):
        h1_df["sig_bh_diff"] = benjamini_hochberg(h1_df["p_diff"].to_numpy())

    if len(asym_df):
        asym_df["sig_bh_diff"] = benjamini_hochberg(asym_df["p_diff"].to_numpy())

    return main_df, h1_df, asym_df


def save_outputs(main_df: pd.DataFrame, h1_df: pd.DataFrame, asym_df: pd.DataFrame, path: str) -> None:
    """Write the main-effect, H1 regional, and H2 asymmetry result tables to one Excel file."""
    with pd.ExcelWriter(path) as xl:
        main_df.to_excel(xl, sheet_name="h1_h3_main_effects", index=False)
        h1_df.to_excel(xl, sheet_name="h1_region_pairwise", index=False)
        asym_df.to_excel(xl, sheet_name="h2_asymmetry", index=False)
    print(f"\nwrote {path}")


def print_summary(main_df: pd.DataFrame, h1_df: pd.DataFrame, asym_df: pd.DataFrame) -> None:
    """Print a compact overview of the significant terms across all three model families."""
    pd.set_option("display.max_columns", None, "display.width", 200,
                  "display.float_format", lambda v: f"{v:.4f}")

    print("=== H1/H3 main-effect terms surviving BH-FDR ===")
    sig_main = main_df[main_df.get("sig_bh", False)]
    if len(sig_main):
        print(sig_main[["asset", "horizon", "term", "coef", "p_value"]].to_string(index=False))
    else:
        print("None.")

    print("\n=== H1: Europe vs. North America, direct comparison ===")
    print(h1_df[["asset", "horizon", "coef_europe", "coef_north_america", "diff_europe_minus_namerica", "p_diff", "sig_bh_diff"]].to_string(index=False))

    print("\n=== H2 asymmetry tests surviving BH-FDR ===")
    sig_asym = asym_df[asym_df.get("sig_bh_diff", False)]
    if len(sig_asym):
        print(sig_asym[["asset", "horizon", "coef_pos", "coef_neg", "diff_neg_minus_pos", "p_diff"]].to_string(index=False))
    else:
        print("None.")


def main() -> None:
    """Run the full panel regression pipeline for H1-H3 across all horizons."""
    combined = load_combined_trend(COMBINED_TREND_PATH)
    main_df, h1_df, asym_df = run_all_horizons(combined, ASSETS, REGIONS, CAPS, HORIZONS)
    print_summary(main_df, h1_df, asym_df)
    save_outputs(main_df, h1_df, asym_df, OUTPUT_XLSX)


if __name__ == "__main__":
    main()