"""
Same grouping and spec as analyze_wa2016split_stateyearcluster.py (Capped:
MT+CO+OR-capped+WA-pre2016; Free: WA-2016on+OR-free+NM), but clustered by
county (fips5) instead of state x year, to compare against the small-cluster-
count state x year version (14-20 clusters, below the usual reliability
threshold). County clustering gives many more clusters and is the
convention used everywhere else in this project.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import os

TBL = "tables"


def add_regime(df):
    d = df.copy()
    is_capped = (d["state_abbr"].isin(["MT", "CO"])
                 | ((d["state_abbr"] == "OR") & (d["year"] >= 2022))
                 | ((d["state_abbr"] == "WA") & (d["year"] < 2016)))
    is_free = (((d["state_abbr"] == "WA") & (d["year"] >= 2016))
               | ((d["state_abbr"] == "OR") & (d["year"] <= 2021))
               | (d["state_abbr"] == "NM"))
    d["regime2"] = np.select([is_capped, is_free], [1, 0], default=np.nan)
    return d


def load_with_captive(panel_file):
    df = pd.read_csv(os.path.join(TBL, panel_file), dtype={"fips5": str})
    df = df[df["year"] != 2026].copy()  # drop incomplete current year
    cap = pd.read_csv(os.path.join(TBL, "captive_population_all5states.csv"), dtype={"fips5": str})
    df = df.merge(cap[["fips5", "state_abbr", "captive_population"]], on=["fips5", "state_abbr"], how="left")
    df["captive_population"] = df["captive_population"].fillna(0)
    df["log1p_captive_pop"] = np.log1p(df["captive_population"])
    return add_regime(df)


def pooled_eventstudy(df, yvar, regime_val, label):
    d = df[df.regime2 == regime_val].dropna(
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
        out_capped, n_c = pooled_eventstudy(df, yvar, 1, "Capped (MT+CO+OR-capped+WA-pre2016)")
        out_capped["regime"] = "capped"
        out_free, n_f = pooled_eventstudy(df, yvar, 0, "Free (WA-2016on+OR-free+NM)")
        out_free["regime"] = "free"
        combined = pd.concat([out_capped, out_free], ignore_index=True)
        combined.to_csv(os.path.join(TBL, f"wa2016split_countycluster_eventstudy_{outcome}.csv"), index=False)
        cluster_counts[outcome] = {"capped": n_c, "free": n_f}
        print(f"Saved wa2016split_countycluster_eventstudy_{outcome}.csv")

    print("\nCounty cluster counts:", cluster_counts)


if __name__ == "__main__":
    main()
