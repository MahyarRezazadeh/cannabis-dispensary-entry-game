"""
Extensive margin: does the NUMBER of dispensaries (not revenue) respond to
cross-border demand, and does entry regulation moderate that response?
Uses the same master per-store panel (CO/WA/OR/MT/NM, verified entry-regime
coding) so results are directly comparable to the revenue-per-store and
revenue-per-capita specs already run.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def main():
    df = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    df["dispensary_per_1000"] = 1000 * pd.to_numeric(df["dispensary_count"], errors="coerce") / df["population"]
    df["log_dispensary_per_1000"] = np.log(df["dispensary_per_1000"])
    df = df.dropna(subset=["log_dispensary_per_1000", "border_70mi", "entry_capped", "log_population"]).copy()
    df["year"] = df["year"].astype(str)

    within_var = df.groupby("state_abbr")["entry_capped"].nunique()
    state_fe = not (within_var <= 1).all()
    fe = " + C(year) + C(state_abbr)" if state_fe else " + C(year)"
    formula = f"log_dispensary_per_1000 ~ border_70mi * entry_capped + log_population" + fe
    print(f"Extensive margin: state_FE={state_fe} (dispensary_count has the same within-state-variation "
          f"constraint as revenue_per_store, since it's built from the same per-store panel)")

    m_hc1 = smf.ols(formula, data=df).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["fips5"]})
    print(f"\n=== Dispensary density (n={int(m_hc1.nobs)}, counties={df['fips5'].nunique()}) ===")
    for term in ["border_70mi", "entry_capped", "border_70mi:entry_capped"]:
        b = m_hc1.params.get(term, np.nan)
        print(f"  {term:32s} b={b:+.4f} ({(np.exp(b)-1)*100:+.1f}%)  "
              f"p_HC1={m_hc1.pvalues.get(term, np.nan):.4f}  p_countyCl={m_cl.pvalues.get(term, np.nan):.4f}")

    print("\n--- State-specific dispensary-density border coefficient ---")
    for st in sorted(df.state_abbr.unique()):
        d = df[df.state_abbr == st].copy()
        if d["border_70mi"].nunique() < 2:
            print(f"  {st}: no border/interior variation -- skipped")
            continue
        yrs = d["year"].nunique()
        f = "log_dispensary_per_1000 ~ border_70mi + log_population" + (" + C(year)" if yrs > 1 else "")
        m = smf.ols(f, data=d).fit(cov_type="HC1")
        b = m.params.get("border_70mi", np.nan)
        p = m.pvalues.get("border_70mi", np.nan)
        regime = d["entry_capped"].mode()[0]
        print(f"  {st} (n={int(m.nobs)}, entry_capped mode={regime}): b={b:+.4f} ({(np.exp(b)-1)*100:+.1f}%), p={p:.4f}")


if __name__ == "__main__":
    main()
