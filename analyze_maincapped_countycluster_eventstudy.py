"""
Part (b) of the requested comparison: event study using the ORIGINAL
Table 2 classification (entry_capped: WA always capped, CO/MT capped,
OR switches 2022, NM free), county-clustered, with the same captive-
population control and year(+state) FE as the WA-2016-split event study,
so the two are directly comparable side by side.
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
    years = sorted(d["year"].unique())
    n_states = d["state_abbr"].nunique()
    for y in years:
        d[f"D_{y}"] = ((d["year"] == y) & (d["border_70mi"] == 1)).astype(int)
    ref = years[len(years) // 2]
    dvars = [f"D_{y}" for y in years if y != ref]
    fe = " + C(year)" + (" + C(state_abbr)" if n_states > 1 else "")
    formula = f"{yvar} ~ border_70mi + " + " + ".join(dvars) + " + log_population + log1p_captive_pop" + fe
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    n_clusters = d["fips5"].nunique()
    print(f"\n--- {label}: {yvar}, ref={ref}, n={int(m.nobs)}, states={sorted(d.state_abbr.unique())}, "
          f"county clusters={n_clusters} ---")
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
    return out, n_clusters


def main():
    percapita = load_with_captive("master_percapita_panel.csv")
    perstore = load_with_captive("master_perstore_panel.csv")
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    cluster_counts = {}
    for outcome, df, yvar in [("percapita", percapita, "log_revenue_per_capita"),
                              ("perstore", perstore, "log_revenue_per_store"),
                              ("density", perstore, "log_dispensary_per_1000")]:
        print(f"\n{'='*90}\n{outcome.upper()}\n{'='*90}")
        out_capped, n_c = pooled_eventstudy(df, yvar, 1, "Capped (WA+CO+MT+OR-capped, original Table 2 def)")
        out_capped["regime"] = "capped"
        out_free, n_f = pooled_eventstudy(df, yvar, 0, "Free (NM+OR-free, original Table 2 def)")
        out_free["regime"] = "free"
        combined = pd.concat([out_capped, out_free], ignore_index=True)
        combined.to_csv(os.path.join(TBL, f"maincapped_countycluster_eventstudy_{outcome}.csv"), index=False)
        cluster_counts[outcome] = {"capped": n_c, "free": n_f}
        print(f"Saved maincapped_countycluster_eventstudy_{outcome}.csv")

    print("\nCounty cluster counts:", cluster_counts)


if __name__ == "__main__":
    main()
