"""
Main pooled analysis for the reframed paper: does entry regulation (capped vs.
free) determine whether the cross-border demand premium shows up in revenue
per store or in revenue per capita?

Central spec: y ~ border_70mi * entry_capped + log_population + year FE + state FE
Run separately for log(revenue per store) and log(revenue per capita).
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def run_spec(df, yvar, label, state_fe=True):
    d = df.dropna(subset=[yvar, "border_70mi", "entry_capped", "log_population"]).copy()
    d["year"] = d["year"].astype(str)
    # Guard against the collinearity trap: if entry_capped has essentially no
    # within-state variation in this sample (e.g. the per-store panel, where
    # OR contributes only a single all-capped 2025 cross-section), state FE
    # will absorb entry_capped's identifying variation and produce an
    # unstable, meaningless main-effect coefficient -- drop state FE in that
    # case and say so, rather than report the resulting garbage estimate.
    within_var = d.groupby("state_abbr")["entry_capped"].nunique()
    if state_fe and (within_var <= 1).all():
        print(f"\n[{label}] entry_capped has zero within-state variation for every state in this sample "
              f"(collinear with state FE) -- dropping state FE for this spec.")
        state_fe = False
    fe = " + C(year) + C(state_abbr)" if state_fe else " + C(year)"
    formula = f"{yvar} ~ border_70mi * entry_capped + log_population" + fe

    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})

    print(f"\n=== {label} (n={int(m_hc1.nobs)}, counties={d['fips5'].nunique()}, states={sorted(d.state_abbr.unique())}, "
          f"state_FE={state_fe}) ===")
    for term in ["border_70mi", "entry_capped", "border_70mi:entry_capped"]:
        b = m_hc1.params.get(term, np.nan)
        p_hc1 = m_hc1.pvalues.get(term, np.nan)
        p_cl = m_cl.pvalues.get(term, np.nan)
        pct = (np.exp(b) - 1) * 100 if pd.notna(b) else np.nan
        print(f"  {term:32s} b={b:+.4f} ({pct:+.1f}%)  p_HC1={p_hc1:.4f}  p_countyCl={p_cl:.4f}")
    return m_hc1, m_cl, d


def state_specific(df, yvar, label):
    print(f"\n--- {label}: single-state border coefficient (free vs capped, all with log(pop)+year FE) ---")
    for st in sorted(df.state_abbr.unique()):
        d = df[df.state_abbr == st].dropna(subset=[yvar, "border_70mi", "log_population"]).copy()
        if d["border_70mi"].nunique() < 2:
            print(f"  {st}: no border/interior variation within-state (all counties same status) -- skipped")
            continue
        d["year"] = d["year"].astype(str)
        yrs = d["year"].nunique()
        formula = f"{yvar} ~ border_70mi + log_population" + (" + C(year)" if yrs > 1 else "")
        try:
            m = smf.ols(formula, data=d).fit(cov_type="HC1")
            b = m.params.get("border_70mi", np.nan)
            p = m.pvalues.get("border_70mi", np.nan)
            regime = df[df.state_abbr == st]["entry_capped"].mode()[0]
            print(f"  {st} (n={int(m.nobs)}, entry_capped mode={regime}): b={b:+.4f} ({(np.exp(b)-1)*100:+.1f}%), p={p:.4f}")
        except Exception as e:
            print(f"  {st}: failed -- {e}")


def main():
    percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})

    print("=" * 90)
    print("MAIN POOLED SPECS: border x entry-regime interaction")
    print("=" * 90)
    run_spec(perstore, "log_revenue_per_store", "Revenue PER STORE")
    run_spec(percapita, "log_revenue_per_capita", "Revenue PER CAPITA")

    print("\n" + "=" * 90)
    print("STATE-SPECIFIC ESTIMATES (transparency check)")
    print("=" * 90)
    state_specific(perstore, "log_revenue_per_store", "Revenue PER STORE")
    state_specific(percapita, "log_revenue_per_capita", "Revenue PER CAPITA")


if __name__ == "__main__":
    main()
