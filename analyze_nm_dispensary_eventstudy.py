"""
Dispensary-level border-premium regression for New Mexico, using
tables/nm_dispensary_panel_annual.csv (individual-store revenue, not
county aggregates). NM's border classification (proximity to TX/AZ, both
non-legal throughout the sample) is time-invariant at the county level, so
there is no single policy-switch date to peg a pre/post design to -- this
is the same year-by-year border x year-dummy design already used for the
county-level regime event studies (analyze_localbarrier_eventstudy_v2.py),
run here on individual dispensary-year revenue instead of county-year
aggregates, as a firm-level robustness check on the county-level NM result.

Spec: log(total_sales)_dt = alpha + sum_k theta_k (Border_d x 1(year=k))
      + gamma1 ln(county population) + gamma2 ln(1+captive_population) + year FE + eps_dt
clustered by county (fips5), consistent with the county-clustering convention
already adopted for the local-barrier event study.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def main():
    df = pd.read_csv(os.path.join(TBL, "nm_dispensary_panel_annual.csv"), dtype={"county_fips": str})
    cap = pd.read_csv(os.path.join(TBL, "captive_population_all5states.csv"), dtype={"fips5": str})
    df = df.merge(cap[["fips5", "captive_population"]], left_on="county_fips", right_on="fips5", how="left")
    df["captive_population"] = df["captive_population"].fillna(0)
    df["log1p_captive_pop"] = np.log1p(df["captive_population"])
    df["log_population"] = np.log(df["population"])
    df = df[df["total_sales"] > 0].copy()
    df["log_total_sales"] = np.log(df["total_sales"])

    # keep full calendar years only for the event-study coefficients to be comparable
    # (2022 is April-Dec only, 2026 is partial-year) -- flag but do not drop, note in output
    years = sorted(df["year"].unique())
    print(f"Years present: {years} (2022 partial: Apr-Dec; latest year may be partial)")
    print(f"N dispensary-year obs: {len(df)}, unique dispensaries: {df.dispensary_id.nunique()}, "
          f"counties: {df.county_fips.nunique()}")

    for y in years:
        df[f"D_{y}"] = ((df["year"] == y) & (df["border_70mi"] == 1)).astype(int)
    ref = years[len(years) // 2]
    dvars = [f"D_{y}" for y in years if y != ref]

    # border_70mi main effect = border premium in the reference year itself;
    # each D_y (y != ref) is the INCREMENTAL deviation from that reference-year
    # premium, so the year-y premium = border_70mi + D_y. Omitting the main
    # effect (as an earlier draft of this script did) forces the reference-year
    # premium to zero instead of estimating it -- confirmed as a real bug here
    # since the pooled single-coefficient spec below shows a large, significant
    # premium that the no-main-effect event-study spec completely missed.
    formula = ("log_total_sales ~ border_70mi + " + " + ".join(dvars) +
               " + log_population + log1p_captive_pop + C(year)")
    m = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["county_fips"]})

    print(f"\n--- NM dispensary-level border premium, ref year={ref} ---")
    b_ref = m.params["border_70mi"]
    se_ref = m.bse["border_70mi"]
    p_ref = m.pvalues["border_70mi"]
    rows = [{"year": ref, "coef": b_ref, "se": se_ref, "p": p_ref, "ref": True}]
    for y in years:
        if y == ref:
            continue
        term = f"D_{y}"
        d_b, d_se, d_p = m.params.get(term, np.nan), m.bse.get(term, np.nan), m.pvalues.get(term, np.nan)
        # level coefficient for year y = border_70mi + D_y; combine via linear combination test
        b_y = b_ref + d_b
        se_y = float(np.sqrt(m.cov_params().loc["border_70mi", "border_70mi"]
                              + m.cov_params().loc[term, term]
                              + 2 * m.cov_params().loc["border_70mi", term]))
        from scipy import stats
        t_y = b_y / se_y
        p_y = 2 * (1 - stats.t.cdf(abs(t_y), m.df_resid))
        rows.append({"year": y, "coef": b_y, "se": se_y, "p": p_y, "ref": False})
    out = pd.DataFrame(rows).sort_values("year")
    out["pct"] = (np.exp(out["coef"]) - 1) * 100
    print(out.to_string(index=False))
    print(f"\nlog_population coef: {m.params.get('log_population', np.nan):+.4f} "
          f"(p={m.pvalues.get('log_population', np.nan):.4f})")
    print(f"log1p_captive_pop coef: {m.params.get('log1p_captive_pop', np.nan):+.4f} "
          f"(p={m.pvalues.get('log1p_captive_pop', np.nan):.4f})")

    out.to_csv(os.path.join(TBL, "nm_dispensary_eventstudy_percapita.csv"), index=False)

    # pooled (non-event-study) single-coefficient version for a headline number
    formula_pooled = "log_total_sales ~ border_70mi + log_population + log1p_captive_pop + C(year)"
    mp = smf.ols(formula_pooled, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["county_fips"]})
    b, se, p = mp.params["border_70mi"], mp.bse["border_70mi"], mp.pvalues["border_70mi"]
    pct = (np.exp(b) - 1) * 100
    print(f"\n--- Pooled single-coefficient spec ---")
    print(f"Border premium (dispensary-level, all years pooled): {pct:+.1f}% (b={b:.4f}, se={se:.4f}, p={p:.4f})")

    # compare to the county-level NM-only result for context
    print("\n--- For comparison: entry margin, border vs interior ---")
    ever = df.drop_duplicates("dispensary_id")
    share = ever.groupby("border_70mi")["true_new_entrant"].mean()
    print(share)


if __name__ == "__main__":
    main()
