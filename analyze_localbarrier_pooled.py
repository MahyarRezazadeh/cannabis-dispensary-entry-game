"""
Pooled single-interaction spec for the local_barrier classification (CO+MT,
where the binding restriction is imposed by local/county government, vs
WA+OR+NM, where it is not), parallel to the paper's main entry_capped and
entry_capped_strict pooled specs (analyze_master_panel.py,
analyze_master_panel_strictcap.py). Reuses the same run_spec pattern.
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
    rows = []
    for term in ["border_70mi", capvar, f"border_70mi:{capvar}"]:
        b = m_hc1.params.get(term, np.nan)
        se_hc1 = m_hc1.bse.get(term, np.nan)
        p_hc1 = m_hc1.pvalues.get(term, np.nan)
        p_cl = m_cl.pvalues.get(term, np.nan)
        pct = (np.exp(b) - 1) * 100 if pd.notna(b) else np.nan
        print(f"  {term:32s} b={b:+.4f} ({pct:+.1f}%)  p_HC1={p_hc1:.4f}  p_countyCl={p_cl:.4f}")
        rows.append({"outcome": label, "term": term, "coef": b, "pct": pct, "p_hc1": p_hc1, "p_cl": p_cl,
                      "n": int(m_hc1.nobs)})
    return rows


def main():
    percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    print("=" * 90)
    print("ROBUSTNESS: local_barrier classification (CO+MT local-government restriction")
    print("vs WA+OR+NM, no local-government-imposed restriction)")
    print("=" * 90)
    all_rows = []
    all_rows += run_spec(perstore, "log_revenue_per_store", "Revenue per store", "local_barrier")
    all_rows += run_spec(percapita, "log_revenue_per_capita", "Revenue per capita", "local_barrier")
    all_rows += run_spec(perstore, "log_dispensary_per_1000", "Dispensary density", "local_barrier")

    pd.DataFrame(all_rows).to_csv(os.path.join(TBL, "localbarrier_pooled_results.csv"), index=False)
    print(f"\nWrote {os.path.join(TBL, 'localbarrier_pooled_results.csv')}")


if __name__ == "__main__":
    main()
