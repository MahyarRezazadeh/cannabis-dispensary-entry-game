"""
Oregon before/after robustness check: Oregon is the one state in the sample
with a genuine within-state entry-regime switch inside its own panel --
free/open retail licensing 2015-10-01 through 2022-01-01, then a licensing
freeze (Jan 2022) followed by a permanent per-capita cap (Mar 2024). This
script splits Oregon's own panel at that switch and asks, independently of
the pooled cross-state comparison: in the FREE period, is the border effect
bigger on revenue per capita or revenue per store (whichever is estimable)?
Does that flip once Oregon becomes an entry barrier?

Revenue per capita has real panel coverage on both sides of the switch
(2017-2021 free, 2022-2025 capped). Revenue per store does NOT: Oregon's
per-store data only exists as a single 2025 cross-section (see project data
notes on why -- the geocoded dispensary panel is unreliable for Oregon and
OLCC's license registry only supports a current snapshot), so there is no
pre-2022 per-store observation to compare against. This is reported as a
real limitation, not glossed over.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def run(d, yvar, label):
    d = d.dropna(subset=[yvar, "border_70mi", "log_population"]).copy()
    if d["border_70mi"].nunique() < 2:
        print(f"  {label}: no border/interior variation -- skipped")
        return None
    d["year"] = d["year"].astype(str)
    yrs = d["year"].nunique()
    formula = f"{yvar} ~ border_70mi + log_population" + (" + C(year)" if yrs > 1 else "")
    m = smf.ols(formula, data=d).fit(cov_type="HC1")
    b = m.params.get("border_70mi", np.nan)
    p = m.pvalues.get("border_70mi", np.nan)
    print(f"  {label} (n={int(m.nobs)}, years={sorted(d.year.unique())}): "
          f"b={b:+.4f} ({(np.exp(b)-1)*100:+.1f}%), p={p:.4f}")
    return m


def main():
    pc = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    ps = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})

    or_pc = pc[pc.state_abbr == "OR"].copy()
    or_ps = ps[ps.state_abbr == "OR"].copy()

    print("=" * 80)
    print("OREGON WITHIN-STATE BEFORE/AFTER: 2022-01-01 entry freeze")
    print("=" * 80)

    print("\n--- Revenue PER CAPITA ---")
    free = run(or_pc[or_pc.year <= 2021], "log_revenue_per_capita", "FREE period (2017-2021)")
    capped = run(or_pc[or_pc.year >= 2022], "log_revenue_per_capita", "CAPPED period (2022-2025)")

    print("\n--- Revenue PER STORE ---")
    print("  FREE period (pre-2022): NO DATA -- Oregon's per-store panel is a 2025-only cross-section")
    run(or_ps, "log_revenue_per_store", "CAPPED period (2025 cross-section only)")

    print("\n--- Formal test: does the border coefficient differ across the two periods? ---")
    d = or_pc.dropna(subset=["log_revenue_per_capita", "border_70mi", "log_population"]).copy()
    d["year"] = d["year"].astype(str)
    d["post_freeze"] = (d["year"].astype(int) >= 2022).astype(int)
    m = smf.ols("log_revenue_per_capita ~ border_70mi * post_freeze + log_population + C(year)",
                data=d).fit(cov_type="HC1")
    m_cl = smf.ols("log_revenue_per_capita ~ border_70mi * post_freeze + log_population + C(year)",
                   data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    b_int = m.params.get("border_70mi:post_freeze", np.nan)
    print(f"  border_70mi x post_freeze: b={b_int:+.4f} ({(np.exp(b_int)-1)*100:+.1f}%), "
          f"p_HC1={m.pvalues.get('border_70mi:post_freeze', np.nan):.4f}, "
          f"p_countyCl={m_cl.pvalues.get('border_70mi:post_freeze', np.nan):.4f}, n={int(m.nobs)}")

    if free is not None and capped is not None:
        b_free = free.params["border_70mi"]
        b_capped = capped.params["border_70mi"]
        print(f"\n  Summary: free-period border coefficient {b_free:+.4f} vs. capped-period {b_capped:+.4f} "
              f"({'shrinks' if abs(b_capped) < abs(b_free) else 'grows'} after the freeze, "
              f"{'same sign' if np.sign(b_free)==np.sign(b_capped) else 'FLIPS SIGN'})")


if __name__ == "__main__":
    main()
