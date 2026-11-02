"""
Part (a) of the requested comparison: re-run Table 2's EXACT pooled
single-coefficient interaction spec (border x regime + log_population +
year FE + state FE where applicable, county-clustered, NO captive
population control -- matching table_main_pooled.tex exactly) but with
the WA-2016-split classification substituted for the original entry_capped:
  Capped: MT, CO, Oregon (2022+), Washington before 2016
  Free:   Washington from 2016 on, Oregon (2015-2021), New Mexico
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def add_regime(df):
    d = df.copy()
    is_capped = (d["state_abbr"].isin(["MT", "CO"])
                 | ((d["state_abbr"] == "OR") & (d["year"] >= 2022))
                 | ((d["state_abbr"] == "WA") & (d["year"] < 2016)))
    is_free = (((d["state_abbr"] == "WA") & (d["year"] >= 2016))
               | ((d["state_abbr"] == "OR") & (d["year"] <= 2021))
               | (d["state_abbr"] == "NM"))
    d["regime2"] = np.select([is_capped, is_free], [1, 0], default=np.nan)
    return d


def run_spec(df, yvar, label):
    d = df.dropna(subset=[yvar, "border_70mi", "regime2", "log_population"]).copy()
    d["year"] = d["year"].astype(str)
    within_var = d.groupby("state_abbr")["regime2"].nunique()
    state_fe = not (within_var <= 1).all()
    fe = " + C(year) + C(state_abbr)" if state_fe else " + C(year)"
    formula = f"{yvar} ~ border_70mi * regime2 + log_population" + fe
    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\n=== {label} (n={int(m_hc1.nobs)}, counties={d.fips5.nunique()}, state_FE={state_fe}) ===")
    for term in ["border_70mi", "regime2", "border_70mi:regime2"]:
        b = m_hc1.params.get(term, np.nan)
        p_hc1 = m_hc1.pvalues.get(term, np.nan)
        p_cl = m_cl.pvalues.get(term, np.nan)
        pct = (np.exp(b) - 1) * 100 if pd.notna(b) else np.nan
        print(f"  {term:24s} b={b:+.4f} ({pct:+.1f}%)  p_HC1={p_hc1:.4f}  p_countyCl={p_cl:.4f}")
    # implied levels
    b_border = m_hc1.params.get("border_70mi", np.nan)
    b_int = m_hc1.params.get("border_70mi:regime2", np.nan)
    print(f"  Implied FREE-state border premium:   {(np.exp(b_border)-1)*100:+.1f}%")
    print(f"  Implied CAPPED-state border premium: {(np.exp(b_border+b_int)-1)*100:+.1f}%")


def main():
    percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    percapita = percapita[percapita["year"] != 2026].copy()  # drop incomplete current year
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    perstore = perstore[perstore["year"] != 2026].copy()
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])
    percapita = add_regime(percapita)
    perstore = add_regime(perstore)

    print("=" * 90)
    print("TABLE-2-STYLE POOLED SPEC, WA-2016-split classification (no captive pop, matching Table 2 exactly)")
    print("=" * 90)
    run_spec(percapita, "log_revenue_per_capita", "Revenue per capita")
    run_spec(perstore, "log_revenue_per_store", "Revenue per store")
    run_spec(perstore, "log_dispensary_per_1000", "Dispensary density")


if __name__ == "__main__":
    main()
