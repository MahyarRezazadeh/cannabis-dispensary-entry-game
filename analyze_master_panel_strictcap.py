"""
Robustness check: rerun the main pooled border x entry-regime interaction
using entry_capped_strict (state-level numerical cap only: WA + Oregon's
restricted period) instead of the paper's primary entry_capped (which also
treats Colorado's local bans and Montana's opt-in requirement as capped).
Free group under this stricter definition: CO, MT, NM, and Oregon's free
period. Capped group: WA (entire panel) and Oregon's restricted period only.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def run_spec(df, yvar, label, capvar):
    d = df.dropna(subset=[yvar, "border_70mi", capvar, "log_population"]).copy()
    d["year"] = d["year"].astype(str)
    within_var = d.groupby("state_abbr")[capvar].nunique()
    state_fe = not (within_var <= 1).all()
    fe = " + C(year) + C(state_abbr)" if state_fe else " + C(year)"
    formula = f"{yvar} ~ border_70mi * {capvar} + log_population" + fe

    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})

    print(f"\n=== {label} (n={int(m_hc1.nobs)}, counties={d['fips5'].nunique()}, state_FE={state_fe}) ===")
    for term in ["border_70mi", capvar, f"border_70mi:{capvar}"]:
        b = m_hc1.params.get(term, np.nan)
        p_hc1 = m_hc1.pvalues.get(term, np.nan)
        p_cl = m_cl.pvalues.get(term, np.nan)
        pct = (np.exp(b) - 1) * 100 if pd.notna(b) else np.nan
        print(f"  {term:32s} b={b:+.4f} ({pct:+.1f}%)  p_HC1={p_hc1:.4f}  p_countyCl={p_cl:.4f}")


def main():
    percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    print("=" * 90)
    print("ROBUSTNESS: STRICT state-cap-only classification (CO, MT reclassified as FREE)")
    print("Free: CO, MT, NM, Oregon (2015-2021)   |   Capped: WA, Oregon (2022-2025)")
    print("=" * 90)
    run_spec(perstore, "log_revenue_per_store", "Revenue PER STORE", "entry_capped_strict")
    run_spec(percapita, "log_revenue_per_capita", "Revenue PER CAPITA", "entry_capped_strict")
    run_spec(perstore, "log_dispensary_per_1000", "Dispensary DENSITY", "entry_capped_strict")


if __name__ == "__main__":
    main()
