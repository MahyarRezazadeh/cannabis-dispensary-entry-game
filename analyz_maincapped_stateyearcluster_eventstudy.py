"""
Event study for the paper's main entry-regulation classification (Grouping A,
already used throughout as `entry_capped`): Capped = Washington, Colorado,
Montana, and Oregon's own capped period (2022+); Free = New Mexico and
Oregon's own free period (2015-2021). This is the FIRST time this
classification is run as a year-by-year event study (it has previously only
been reported as the single pooled coefficient in table_main_pooled.tex) and
the first time clustering is done at the state x year level rather than by
county, per direct request.

Spec: y_ct = alpha + beta*Border_c + sum_k theta_k(Border_c x 1{year=k})
      + gamma1 ln(population) + gamma2 ln(1+captive population)
      + year FE (+ state FE where >1 state contributes) + eps_ct
clustered by state x year (not county). All three outcomes: revenue per
store, revenue per capita, dispensary density (density event study carries
the same frozen-CO/WA-count caveat documented earlier in this project).
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import os

TBL = "tables"


def load_with_captive(panel_file):
    df = pd.read_csv(os.path.join(TBL, panel_file), dtype={"fips5": str})
    df = df[df["year"] != 2026].copy()  # drop incomplete current year
    cap = pd.read_csv(os.path.join(TBL, "captive_population_all5states.csv"), dtype={"fips5": str})
    df = df.merge(cap[["fips5", "state_abbr", "captive_population"]], on=["fips5", "state_abbr"], how="left")
    df["captive_population"] = df["captive_population"].fillna(0)
    df["log1p_captive_pop"] = np.log1p(df["captive_population"])
    return df


def pooled_eventstudy(df, yvar, regime_val, label):
    d = df[df.entry_capped == regime_val].dropna(
        subset=[yvar, "border_70mi", "log_population", "log1p_captive_pop"]).copy()
    d["year"] = d["year"].astype(int)
    d["state_year"] = d["state_abbr"].astype(str) + "_" + d["year"].astype(str)
    years = sorted(d["year"].unique())
    n_states = d["state_abbr"].nunique()
    for y in years:
        d[f"D_{y}"] = ((d["year"] == y) & (d["border_70mi"] == 1)).astype(int)
    ref = years[len(years) // 2]
    dvars = [f"D_{y}" for y in years if y != ref]
    fe = " + C(year)" + (" + C(state_abbr)" if n_states > 1 else "")
    formula = f"{yvar} ~ border_70mi + " + " + ".join(dvars) + " + log_population + log1p_captive_pop" + fe
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["state_year"]})
    n_clusters = d["state_year"].nunique()
    print(f"\n--- {label}: {yvar}, ref={ref}, n={int(m.nobs)}, counties={d.fips5.nunique()}, "
          f"states={sorted(d.state_abbr.unique())}, state*year clusters={n_clusters} ---")
    b_ref, se_ref, p_ref = m.params["border_70mi"], m.bse["border_70mi"], m.pvalues["border_70mi"]
    rows = [{"year": ref, "coef": b_ref, "se": se_ref, "p": p_ref, "ref": True}]
    for y in years:
        if y == ref:
            continue
        term = f"D_{y}"
        if term not in m.params.index:
            continue
        d_b = m.params[term]
        b_y = b_ref + d_b
        se_y = float(np.sqrt(m.cov_params().loc["border_70mi", "border_70mi"]
                              + m.cov_params().loc[term, term]
                              + 2 * m.cov_params().loc["border_70mi", term]))
        t_y = b_y / se_y
        p_y = 2 * (1 - stats.t.cdf(abs(t_y), m.df_resid))
        rows.append({"year": y, "coef": b_y, "se": se_y, "p": p_y, "ref": False})
    out = pd.DataFrame(rows).sort_values("year")
    out["pct"] = (np.exp(out["coef"]) - 1) * 100
    print(out.to_string(index=False))
    return out


def main():
    percapita = load_with_captive("master_percapita_panel.csv")
    perstore = load_with_captive("master_perstore_panel.csv")
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    for outcome, df, yvar in [("percapita", percapita, "log_revenue_per_capita"),
                              ("perstore", perstore, "log_revenue_per_store"),
                              ("density", perstore, "log_dispensary_per_1000")]:
        print(f"\n{'='*90}\n{outcome.upper()}\n{'='*90}")
        out_capped = pooled_eventstudy(df, yvar, 1, "Capped (WA+CO+MT+OR-capped)")
        out_capped["regime"] = "capped"
        out_free = pooled_eventstudy(df, yvar, 0, "Free (NM+OR-free)")
        out_free["regime"] = "free"
        combined = pd.concat([out_capped, out_free], ignore_index=True)
        combined.to_csv(os.path.join(TBL, f"maincapped_stateyearcluster_eventstudy_{outcome}.csv"), index=False)
        print(f"Saved maincapped_stateyearcluster_eventstudy_{outcome}.csv")


if __name__ == "__main__":
    main()
