"""
Final primary-specification event studies: Grouping D (local barrier
CO+MT vs free NM+OR-2015-2021, Washington dropped), balanced panel
(counties restricted to those reporting every year within their own
state's window -- validated as the more robust variant in the FE
comparison step). Revenue per capita and revenue per store use the full
border x year design (border main effect + D_year interactions,
population + captive-population controls, state/year FE, county-clustered
SEs). Dispensary density is run too, but flagged explicitly: Colorado's
dispensary_count is frozen per county across years (confirmed data
artifact) and Montana contributes only a single 2025 cross-section, so
the "local" line is not a genuine time series; the "free" line is
NM-only and only covers 2022-2026 (Oregon contributes zero density
observations to either side of Grouping D). Reported for completeness,
captioned accordingly, not read as evidence of a real trend on the local
side.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import os

TBL = "tables"

LOCAL_STATES = ["CO", "MT"]


def add_regime(df):
    d = df.copy()
    is_local = d["state_abbr"].isin(LOCAL_STATES)
    is_free = (d["state_abbr"] == "NM") | ((d["state_abbr"] == "OR") & (d["year"] <= 2021))
    d["regime"] = np.select([is_local, is_free], ["local", "free"], default=None)
    return d[d["regime"].notna()].copy()


def load_with_captive(panel_file):
    df = pd.read_csv(os.path.join(TBL, panel_file), dtype={"fips5": str})
    cap = pd.read_csv(os.path.join(TBL, "captive_population_all5states.csv"), dtype={"fips5": str})
    df = df.merge(cap[["fips5", "state_abbr", "captive_population"]], on=["fips5", "state_abbr"], how="left")
    df["captive_population"] = df["captive_population"].fillna(0)
    df["log1p_captive_pop"] = np.log1p(df["captive_population"])
    return add_regime(df)


def balanced_panel(d, regime):
    sub = d[d.regime == regime]
    keep = []
    for state, g in sub.groupby("state_abbr"):
        state_years = set(g["year"].unique())
        years_present = g.groupby("fips5")["year"].apply(set)
        keep.extend(years_present[years_present == state_years].index.tolist())
    return d[(d.regime == regime) & (d.fips5.isin(keep))]


def pooled_eventstudy(d, yvar, label):
    d = d.dropna(subset=[yvar, "border_70mi", "log_population", "log1p_captive_pop"]).copy()
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
    print(f"  [{label}] n={int(m.nobs)}, counties={d.fips5.nunique()}, states={sorted(d.state_abbr.unique())}")
    b_ref, se_ref = m.params["border_70mi"], m.bse["border_70mi"]
    rows = [{"year": ref, "coef": b_ref, "se": se_ref, "ref": True}]
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
        rows.append({"year": y, "coef": b_y, "se": se_y, "ref": False})
    out = pd.DataFrame(rows).sort_values("year")
    out["pct"] = (np.exp(out["coef"]) - 1) * 100
    return out


def main():
    percapita = load_with_captive("master_percapita_panel.csv")
    perstore = load_with_captive("master_perstore_panel.csv")
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    for outcome, df, yvar in [("percapita", percapita, "log_revenue_per_capita"),
                              ("perstore", perstore, "log_revenue_per_store"),
                              ("density", perstore, "log_dispensary_per_1000")]:
        print(f"\n=== {outcome} ===")
        d_local = balanced_panel(df, "local")
        d_free = balanced_panel(df, "free")
        out_local = pooled_eventstudy(d_local, yvar, "local (balanced)")
        out_local["regime"] = "local"
        out_free = pooled_eventstudy(d_free, yvar, "free (balanced)")
        out_free["regime"] = "free"
        combined = pd.concat([out_local, out_free], ignore_index=True)
        combined.to_csv(os.path.join(TBL, f"groupD_balanced_eventstudy_{outcome}.csv"), index=False)
        print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
