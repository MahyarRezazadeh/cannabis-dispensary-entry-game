"""
Investigate whether 100 miles is a defensible threshold for the nearest-
nonlegal-city border definition, per explicit user question: is there
supportive evidence in OUR OWN DATA (not just literature precedent)?

Two checks, both using the continuous dist_nearest_nonlegal_city_mi measure
already computed in tables/border_100mi_by_year.csv (not re-derived):

1. Threshold sensitivity scan: refit the main pooled border x capped
   interaction specification at a grid of thresholds (25 to 250 miles) and
   report how the border and interaction coefficients move, to see whether
   100mi sits in a stable plateau or is a knife-edge/cherry-picked choice.

2. A simple split-sample "distance bin" version: instead of an arbitrary
   single cutoff, bin distance into quartiles/deciles and see where revenue
   and dispensary density actually start flattening out with distance --
   an atheoretical, nonparametric look at the same question the dose-response
   cubic (analyze_city_distance_doseresponse.py / _revpc.py) already
   addressed parametrically (turning points at 127.6mi and 205.9mi there).
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def load():
    dist = pd.read_csv(os.path.join(TBL, "border_100mi_by_year.csv"), dtype={"fips5": str})[
        ["fips5", "year", "dist_nearest_nonlegal_city_mi"]]
    pc = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    ps = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    ps["dispensary_per_1000"] = 1000 * pd.to_numeric(ps.dispensary_count, errors="coerce") / ps.population
    ps["log_dispensary_per_1000"] = np.log(ps.dispensary_per_1000)
    pc = pc.merge(dist, on=["fips5", "year"], how="left")
    ps = ps.merge(dist, on=["fips5", "year"], how="left")
    pc = pc[~((pc.state_abbr == "NM") & (pc.year == 2026))].copy()
    ps = ps[~((ps.state_abbr == "NM") & (ps.year == 2026))].copy()
    return pc, ps


def threshold_scan(pc, ps):
    thresholds = [25, 50, 75, 100, 125, 150, 175, 200, 250]
    specs = [
        ("Revenue per store", ps, "log_revenue_per_store",
         "{y} ~ border_t * entry_capped + log_population + C(year)"),
        ("Revenue per capita", pc, "log_revenue_per_capita",
         "{y} ~ border_t * entry_capped + log_population + C(year) + C(state_abbr)"),
        ("Dispensary density", ps, "log_dispensary_per_1000",
         "{y} ~ border_t * entry_capped + log_population + C(year)"),
    ]
    rows = []
    for label, df, yvar, formula_tmpl in specs:
        for t in thresholds:
            d = df.dropna(subset=[yvar, "dist_nearest_nonlegal_city_mi", "log_population"]).copy()
            d["border_t"] = (d["dist_nearest_nonlegal_city_mi"] <= t).astype(int)
            d["year"] = d["year"].astype(str)
            n_border = (d["border_t"] == 1).sum()
            if d["border_t"].nunique() < 2 or n_border < 10:
                continue
            formula = formula_tmpl.format(y=yvar)
            m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
            b = m.params.get("border_t", np.nan)
            p = m.pvalues.get("border_t", np.nan)
            bi = m.params.get("border_t:entry_capped", np.nan)
            pi = m.pvalues.get("border_t:entry_capped", np.nan)
            rows.append({"outcome": label, "threshold_mi": t, "n_border_countyyears": n_border,
                         "border_b": b, "border_pct": (np.exp(b) - 1) * 100, "border_p_cl": p,
                         "inter_b": bi, "inter_pct": (np.exp(bi) - 1) * 100, "inter_p_cl": pi})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TBL, "threshold_scan_100mi_family.csv"), index=False)
    print("Threshold sensitivity scan (county-clustered p-values):\n")
    for label in res["outcome"].unique():
        sub = res[res.outcome == label]
        print(f"--- {label} ---")
        print(sub[["threshold_mi", "n_border_countyyears", "border_pct", "border_p_cl",
                    "inter_pct", "inter_p_cl"]].round(3).to_string(index=False))
        print()
    return res


def distance_bins(pc, ps):
    print("=" * 70)
    print("Nonparametric check: mean log outcome by distance decile")
    print("=" * 70)
    specs = [("Revenue per capita", pc, "log_revenue_per_capita"),
             ("Dispensary density", ps, "log_dispensary_per_1000")]
    for label, df, yvar in specs:
        d = df.dropna(subset=[yvar, "dist_nearest_nonlegal_city_mi"]).copy()
        d["decile"] = pd.qcut(d["dist_nearest_nonlegal_city_mi"], 10, duplicates="drop")
        summary = d.groupby("decile", observed=True).agg(
            mean_dist=("dist_nearest_nonlegal_city_mi", "mean"),
            mean_y=(yvar, "mean"),
            n=(yvar, "size")).reset_index(drop=True)
        print(f"\n{label}:")
        print(summary.round(3).to_string(index=False))


def distance_distribution(pc):
    print("\n" + "=" * 70)
    print("Distribution of dist_nearest_nonlegal_city_mi (2025 cross-section, one row per county)")
    print("=" * 70)
    d25 = pc[pc.year == 2025].drop_duplicates("fips5")
    print(d25["dist_nearest_nonlegal_city_mi"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1))
    # histogram bin counts to look for a natural gap near 100mi
    bins = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400]
    counts = pd.cut(d25["dist_nearest_nonlegal_city_mi"], bins).value_counts().sort_index()
    print("\nCounty counts by distance band:")
    print(counts)


def main():
    pc, ps = load()
    distance_distribution(pc)
    threshold_scan(pc, ps)
    distance_bins(pc, ps)


if __name__ == "__main__":
    main()
