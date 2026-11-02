"""
Reclassification experiment requested directly: what if Washington is
coded as NON-capped (free) instead of capped, given that its statutory
cap is only ~79% utilized (480 of up to 608 licenses, per the corrected
WSLCB pull in wa_dispensary_counts_corrected.csv) rather than saturated?

new_capped(state, year): CO=1, MT=1, WA=0 (reclassified from 1), NM=0,
OR = 0 if year<=2021 else 1 (unchanged -- Oregon's own switch is
independent of this WA question).

Capped group: CO, MT, Oregon 2022-onward.
Free group:   WA, NM, Oregon 2015-2021.

CAVEAT carried through: WA's revenue_per_store and dispensary density
numbers still rest on the known-undercounted dispensary_count (98 vs the
true 480 active licenses); revenue_per_capita is unaffected since it never
touches dispensary_count.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import os

TBL = "tables"
OUT = "figures_reframe"
os.makedirs(OUT, exist_ok=True)

CAPPED_STATES = ["CO", "MT"]


def add_regime(df):
    d = df.copy()
    is_capped = d["state_abbr"].isin(CAPPED_STATES) | ((d["state_abbr"] == "OR") & (d["year"] >= 2022))
    d["wa_free_capped"] = is_capped.astype(int)
    return d


def load_with_captive(panel_file):
    df = pd.read_csv(os.path.join(TBL, panel_file), dtype={"fips5": str})
    cap = pd.read_csv(os.path.join(TBL, "captive_population_all5states.csv"), dtype={"fips5": str})
    df = df.merge(cap[["fips5", "state_abbr", "captive_population"]], on=["fips5", "state_abbr"], how="left")
    df["captive_population"] = df["captive_population"].fillna(0)
    df["log1p_captive_pop"] = np.log1p(df["captive_population"])
    return add_regime(df)


percapita = load_with_captive("master_percapita_panel.csv")
perstore = load_with_captive("master_perstore_panel.csv")
perstore["log_dispensary_per_1000"] = np.log(
    1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

print("=" * 80)
print("Group composition under WA-as-free reclassification")
print("=" * 80)
for label, df in [("percapita", percapita), ("perstore", perstore)]:
    print(f"\n--- {label} ---")
    summ = df.groupby(["wa_free_capped", "state_abbr"]).agg(
        n=("fips5", "size"), counties=("fips5", "nunique"),
        year_min=("year", "min"), year_max=("year", "max")).reset_index()
    print(summ.to_string(index=False))


def run_pooled_spec(df, yvar, label):
    d = df.dropna(subset=[yvar, "border_70mi", "log_population"]).copy()
    d["year"] = d["year"].astype(str)
    within_var = d.groupby("state_abbr")["wa_free_capped"].nunique()
    state_fe = not (within_var <= 1).all()
    fe = " + C(year) + C(state_abbr)" if state_fe else " + C(year)"
    print(f"  [state_FE={state_fe}]")
    formula = f"{yvar} ~ border_70mi * wa_free_capped + log_population" + fe
    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\n=== {label} (n={int(m_hc1.nobs)}, counties={d.fips5.nunique()}) ===")
    rows = []
    for term in ["border_70mi", "wa_free_capped", "border_70mi:wa_free_capped"]:
        b = m_hc1.params.get(term, np.nan)
        p_hc1 = m_hc1.pvalues.get(term, np.nan)
        p_cl = m_cl.pvalues.get(term, np.nan)
        pct = (np.exp(b) - 1) * 100 if pd.notna(b) else np.nan
        print(f"  {term:30s} b={b:+.4f} ({pct:+.1f}%)  p_HC1={p_hc1:.4f}  p_countyCl={p_cl:.4f}")
        rows.append({"outcome": label, "term": term, "coef": b, "pct": pct, "p_hc1": p_hc1, "p_cl": p_cl,
                      "n": int(m_hc1.nobs)})
    return rows


print("\n" + "=" * 80)
print("POOLED SINGLE-COEFFICIENT SPEC (WA reclassified as free)")
print("=" * 80)
all_rows = []
all_rows += run_pooled_spec(percapita, "log_revenue_per_capita", "Revenue per capita")
all_rows += run_pooled_spec(perstore, "log_revenue_per_store", "Revenue per store")
all_rows += run_pooled_spec(perstore, "log_dispensary_per_1000", "Dispensary density")
pd.DataFrame(all_rows).to_csv(os.path.join(TBL, "wa_as_free_pooled_results.csv"), index=False)


def pooled_eventstudy(df, yvar, regime_val, label):
    d = df[df.wa_free_capped == regime_val].dropna(
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
print("EVENT STUDIES (revenue outcomes; density skipped -- CO/WA count is frozen, see prior audit)")
print("=" * 80)
for outcome, df, yvar in [("percapita", percapita, "log_revenue_per_capita"),
                          ("perstore", perstore, "log_revenue_per_store")]:
    out_capped = pooled_eventstudy(df, yvar, 1, "Capped (CO+MT+OR 2022-2025)")
    out_capped["regime"] = "capped"
    out_free = pooled_eventstudy(df, yvar, 0, "Free (WA+NM+OR 2015-2021)")
    out_free["regime"] = "free"
    combined = pd.concat([out_capped, out_free], ignore_index=True)
    combined.to_csv(os.path.join(TBL, f"wa_as_free_eventstudy_{outcome}.csv"), index=False)
