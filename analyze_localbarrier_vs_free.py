"""
New grouping requested directly: drop Washington entirely. Compare
Local-barrier states (Colorado, Montana -- all years) against genuinely
Free-entry states/periods (New Mexico, all years; Oregon, but ONLY its
free period 2015-2021, excluding its post-2022 capped years). This is
narrower than the earlier local_barrier-vs-nonlocal-barrier cut (which
kept Washington as the "non-local" comparison group) -- here the
comparison is specifically local-government restriction vs. no
restriction at all, dropping the state-level-cap case (Washington)
entirely rather than pooling it with the free states.

Within this filtered subsample, regime membership has NO within-state
variation (CO and MT are always "local"; NM and OR-free-years are always
"free" -- Oregon's capped years are simply excluded), so state fixed
effects are included per-group only where more than one state contributes
(matching the established convention in analyze_localbarrier_eventstudy_v2.py).

Outputs:
  1. A descriptive summary table (states, years, county-year counts) for
     the two groups.
  2. A pooled single-coefficient border x regime interaction table for
     revenue per capita, revenue per store, and dispensary density
     (cross-sectional identification, safe under the known CO/WA frozen-
     dispensary-count issue -- see build_border_vs_interior_trends.py notes).
  3. Pooled year-by-year event studies (border x year interaction, with
     county population, log(1+captive population), state/year FE where
     applicable, county-clustered SEs) for revenue per capita and revenue
     per store. Density is NOT run as a time series: Colorado's
     dispensary_count is frozen per county across years (a confirmed data
     artifact) and Montana contributes only a single 2025 cross-section,
     so there is no genuine within-group time variation to plot for density.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import os

TBL = "tables"
OUT = "figures_reframe"
os.makedirs(OUT, exist_ok=True)

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


# ---------------- 1. Descriptive summary table ----------------
percapita = load_with_captive("master_percapita_panel.csv")
perstore = load_with_captive("master_perstore_panel.csv")
perstore["log_dispensary_per_1000"] = np.log(
    1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

print("=" * 80)
print("DESCRIPTIVE: Local-barrier (CO+MT) vs Free (NM + Oregon 2015-2021), WA dropped")
print("=" * 80)
for label, df in [("percapita panel", percapita), ("perstore panel", perstore)]:
    print(f"\n--- {label} ---")
    summ = df.groupby(["regime", "state_abbr"]).agg(
        n_county_years=("fips5", "size"), n_counties=("fips5", "nunique"),
        year_min=("year", "min"), year_max=("year", "max")).reset_index()
    print(summ.to_string(index=False))

summary_rows = []
for regime in ["local", "free"]:
    d = percapita[percapita.regime == regime]
    summary_rows.append({
        "regime": "Local barrier (CO+MT)" if regime == "local" else "Free (NM, OR 2015-2021)",
        "states": ", ".join(sorted(d.state_abbr.unique())),
        "years": f"{d.year.min()}-{d.year.max()}",
        "n_counties": d.fips5.nunique(),
        "n_county_years": len(d),
        "n_border_counties": d[d.border_70mi == 1].fips5.nunique(),
    })
desc_table = pd.DataFrame(summary_rows)
desc_table.to_csv(os.path.join(TBL, "localbarrier_vs_free_summary.csv"), index=False)
print("\n--- Group summary table ---")
print(desc_table.to_string(index=False))


# ---------------- 2. Pooled single-coefficient interaction table ----------------
def run_pooled_spec(df, yvar, label):
    d = df.dropna(subset=[yvar, "border_70mi", "log_population"]).copy()
    d["regime_local"] = (d["regime"] == "local").astype(int)
    d["year"] = d["year"].astype(str)
    within_var = d.groupby("state_abbr")["regime_local"].nunique()
    state_fe = not (within_var <= 1).all()
    fe = " + C(year) + C(state_abbr)" if state_fe else " + C(year)"
    formula = f"{yvar} ~ border_70mi * regime_local + log_population" + fe
    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\n=== {label} (n={int(m_hc1.nobs)}, counties={d.fips5.nunique()}, state_FE={state_fe}) ===")
    rows = []
    for term in ["border_70mi", "regime_local", "border_70mi:regime_local"]:
        b = m_hc1.params.get(term, np.nan)
        p_hc1 = m_hc1.pvalues.get(term, np.nan)
        p_cl = m_cl.pvalues.get(term, np.nan)
        pct = (np.exp(b) - 1) * 100 if pd.notna(b) else np.nan
        print(f"  {term:28s} b={b:+.4f} ({pct:+.1f}%)  p_HC1={p_hc1:.4f}  p_countyCl={p_cl:.4f}")
        rows.append({"outcome": label, "term": term, "coef": b, "pct": pct, "p_hc1": p_hc1, "p_cl": p_cl,
                      "n": int(m_hc1.nobs)})
    return rows


print("\n" + "=" * 80)
print("POOLED SINGLE-COEFFICIENT SPEC")
print("=" * 80)
all_rows = []
all_rows += run_pooled_spec(percapita, "log_revenue_per_capita", "Revenue per capita")
all_rows += run_pooled_spec(perstore, "log_revenue_per_store", "Revenue per store")
all_rows += run_pooled_spec(perstore, "log_dispensary_per_1000", "Dispensary density")
pd.DataFrame(all_rows).to_csv(os.path.join(TBL, "localbarrier_vs_free_pooled_results.csv"), index=False)


# ---------------- 3. Event studies (revenue outcomes only) ----------------
def pooled_eventstudy(df, yvar, regime, label):
    d = df[df.regime == regime].dropna(
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
    print(f"\n--- {label}: {yvar}, ref year={ref}, n={int(m.nobs)}, "
          f"counties={d.fips5.nunique()}, states={sorted(d.state_abbr.unique())} ---")
    b_ref, se_ref, p_ref = m.params["border_70mi"], m.bse["border_70mi"], m.pvalues["border_70mi"]
    rows = [{"year": ref, "coef": b_ref, "se": se_ref, "p": p_ref, "ref": True}]
    for y in years:
        if y == ref:
            continue
        term = f"D_{y}"
        d_b = m.params.get(term, np.nan)
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


print("\n" + "=" * 80)
print("EVENT STUDIES (revenue outcomes only -- density excluded, see module docstring)")
print("=" * 80)
for outcome, df, yvar in [("percapita", percapita, "log_revenue_per_capita"),
                          ("perstore", perstore, "log_revenue_per_store")]:
    out_local = pooled_eventstudy(df, yvar, "local", "Local barrier (CO+MT)")
    out_local["regime"] = "local"
    out_free = pooled_eventstudy(df, yvar, "free", "Free (NM + OR 2015-2021)")
    out_free["regime"] = "free"
    combined = pd.concat([out_local, out_free], ignore_index=True)
    combined.to_csv(os.path.join(TBL, f"localbarrier_vs_free_eventstudy_{outcome}.csv"), index=False)
