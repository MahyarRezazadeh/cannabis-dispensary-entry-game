"""
Continuous-distance version of the paper's most robust specification
(Grouping D: local barrier CO+MT vs free NM+OR-2015-2021, Washington
dropped). Replaces the binary border_70mi indicator with continuous
log-distance to the nearest nonlegal-state county, interacted with the
regime classification, controlling for log(population) and
log(1+captive population) -- the nonlegal-side "potential demand"
population within 70 miles already used throughout this project's later
specifications.

y = alpha + beta1 log(dist) + beta2 regime + beta3 log(dist) x regime
    + gamma1 log(population) + gamma2 log(1+captive_population)
    + year FE (+ state FE if within-group variation) + eps
county-clustered SEs. beta3 is the coefficient of interest: does the
distance gradient itself differ by entry regime.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def add_regime_D(df):
    d = df.copy()
    is_local = d["state_abbr"].isin(["CO", "MT"])
    is_free = (d["state_abbr"] == "NM") | ((d["state_abbr"] == "OR") & (d["year"] <= 2021))
    d["regime_D"] = np.select([is_local, is_free], [1, 0], default=np.nan)
    return d[d["regime_D"].notna()].copy()


def load_with_captive(panel_file):
    df = pd.read_csv(os.path.join(TBL, panel_file), dtype={"fips5": str})
    cap = pd.read_csv(os.path.join(TBL, "captive_population_all5states.csv"), dtype={"fips5": str})
    df = df.merge(cap[["fips5", "state_abbr", "captive_population"]], on=["fips5", "state_abbr"], how="left")
    df["captive_population"] = df["captive_population"].fillna(0)
    df["log1p_captive_pop"] = np.log1p(df["captive_population"])
    df["log_dist"] = np.log(df["dist_nonlegal_mi"])
    return add_regime_D(df)


def run_spec(df, yvar, label):
    d = df.dropna(subset=[yvar, "log_dist", "log_population", "log1p_captive_pop"]).copy()
    d = d[np.isfinite(d["log_dist"])]
    d["year"] = d["year"].astype(str)
    within_var = d.groupby("state_abbr")["regime_D"].nunique()
    state_fe = not (within_var <= 1).all()
    fe = " + C(year) + C(state_abbr)" if state_fe else " + C(year)"
    formula = f"{yvar} ~ log_dist * regime_D + log_population + log1p_captive_pop" + fe
    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\n=== {label} (n={int(m_hc1.nobs)}, counties={d.fips5.nunique()}, state_FE={state_fe}) ===")
    rows = []
    for term in ["log_dist", "regime_D", "log_dist:regime_D", "log1p_captive_pop"]:
        b = m_hc1.params.get(term, np.nan)
        p_hc1 = m_hc1.pvalues.get(term, np.nan)
        p_cl = m_cl.pvalues.get(term, np.nan)
        se_cl = m_cl.bse.get(term, np.nan)
        print(f"  {term:22s} b={b:+.4f}  se(cl)={se_cl:.4f}  p_HC1={p_hc1:.4f}  p_countyCl={p_cl:.4f}")
        rows.append({"outcome": label, "term": term, "coef": b, "se_cl": se_cl, "p_hc1": p_hc1, "p_cl": p_cl,
                      "n": int(m_hc1.nobs)})
    return rows


def main():
    percapita = load_with_captive("master_percapita_panel.csv")
    perstore = load_with_captive("master_perstore_panel.csv")
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    print("=" * 90)
    print("CONTINUOUS DISTANCE x REGIME-D (local CO+MT vs free NM+OR-2015-2021, WA dropped)")
    print("=" * 90)
    all_rows = []
    all_rows += run_spec(percapita, "log_revenue_per_capita", "Revenue per capita")
    all_rows += run_spec(perstore, "log_revenue_per_store", "Revenue per store")
    all_rows += run_spec(perstore, "log_dispensary_per_1000", "Dispensary density")
    pd.DataFrame(all_rows).to_csv(os.path.join(TBL, "continuous_distance_groupD_results.csv"), index=False)
    print(f"\nSaved continuous_distance_groupD_results.csv")


if __name__ == "__main__":
    main()
