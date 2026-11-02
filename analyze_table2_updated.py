"""
Re-estimate Table 2 (table_main_pooled.tex) with the ORIGINAL entry_capped
classification (Washington capped for its entire panel, consistent with
the WSLCB research confirming the general application window closed
permanently in March 2016 and has never reopened -- so the WA-2016-split
reclassification is abandoned), but with the incomplete 2026 partial year
dropped, matching the convention adopted for all other analyses in this
project since that filter was introduced. Same spec as the original table:
border x capped interaction, log(population) control, year + state FE
(percapita) / year FE only (perstore, density -- collinearity guard),
county-clustered SEs, NO captive-population control (matching the
original Table 2 exactly, for a clean before/after comparison).
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def run_spec(df, yvar, label):
    d = df.dropna(subset=[yvar, "border_70mi", "entry_capped", "log_population"]).copy()
    d["year"] = d["year"].astype(str)
    within_var = d.groupby("state_abbr")["entry_capped"].nunique()
    state_fe = not (within_var <= 1).all()
    fe = " + C(year) + C(state_abbr)" if state_fe else " + C(year)"
    formula = f"{yvar} ~ border_70mi * entry_capped + log_population" + fe
    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\n=== {label} (n={int(m_hc1.nobs)}, counties={d.fips5.nunique()}, state_FE={state_fe}) ===")
    for term in ["border_70mi", "entry_capped", "border_70mi:entry_capped"]:
        b = m_hc1.params.get(term, np.nan)
        p_hc1 = m_hc1.pvalues.get(term, np.nan)
        p_cl = m_cl.pvalues.get(term, np.nan)
        pct = (np.exp(b) - 1) * 100 if pd.notna(b) else np.nan
        print(f"  {term:26s} b={b:+.4f} ({pct:+.1f}%)  p_HC1={p_hc1:.4f}  p_countyCl={p_cl:.4f}")
    b_border = m_hc1.params.get("border_70mi", np.nan)
    b_int = m_hc1.params.get("border_70mi:entry_capped", np.nan)
    print(f"  Implied FREE-state border premium:   {(np.exp(b_border)-1)*100:+.1f}%")
    print(f"  Implied CAPPED-state border premium: {(np.exp(b_border+b_int)-1)*100:+.1f}%")


def main():
    percapita_full = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore_full = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    perstore_full["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore_full["dispensary_count"], errors="coerce") / perstore_full["population"])

    for tag, percapita, perstore in [
        ("ORIGINAL (2026 included, if present)", percapita_full, perstore_full),
        ("UPDATED (2026 dropped)", percapita_full[percapita_full.year != 2026],
         perstore_full[perstore_full.year != 2026]),
    ]:
        print("\n" + "=" * 90)
        print(f"TABLE 2, {tag}")
        print("=" * 90)
        run_spec(percapita, "log_revenue_per_capita", "Revenue per capita")
        run_spec(perstore, "log_revenue_per_store", "Revenue per store")
        run_spec(perstore, "log_dispensary_per_1000", "Dispensary density")


if __name__ == "__main__":
    main()
