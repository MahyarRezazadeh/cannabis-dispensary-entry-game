"""
Corrected event-study design for local-barrier vs non-local-barrier states,
addressing three issues with the first version (analyze_localbarrier_eventstudy.py):
  1. No captive-market-population control -- added here as log(1+captive_population),
     since interior counties have captive_population=0 by construction (they are,
     by definition, >70mi from any nonlegal-state county) and a plain log() is
     undefined at zero.
  2. Separate per-year cross-sectional regressions -- replaced with ONE pooled
     regression per regime group, with border x year interaction terms, so year
     and state fixed effects are estimated jointly rather than year-by-year.
  3. HC1 standard errors -- replaced with county-clustered only, per instruction.

Spec (estimated separately for the local-barrier group and the non-local-barrier
group): y_ct = alpha + sum_k theta_k (Border_c x 1(year=k)) + gamma1 ln(pop_ct)
              + gamma2 ln(1+captive_population_c) + state FE + year FE + eps_ct
clustered by county (fips5). theta_k is the border premium in year k.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"

LOCAL_STATES = ["CO", "MT"]
NONLOCAL_STATES = ["WA", "NM"]
NONLOCAL_STATES_WITH_OR = ["WA", "NM", "OR"]


def load_with_captive(panel_file):
    df = pd.read_csv(os.path.join(TBL, panel_file), dtype={"fips5": str})
    cap = pd.read_csv(os.path.join(TBL, "captive_population_all5states.csv"), dtype={"fips5": str})
    df = df.merge(cap[["fips5", "state_abbr", "captive_population"]], on=["fips5", "state_abbr"], how="left")
    df["captive_population"] = df["captive_population"].fillna(0)
    df["log1p_captive_pop"] = np.log1p(df["captive_population"])
    return df


def pooled_eventstudy(df, yvar, states, label):
    d = df[df.state_abbr.isin(states)].dropna(
        subset=[yvar, "border_70mi", "log_population", "log1p_captive_pop"]).copy()
    d["year"] = d["year"].astype(int)
    years = sorted(d["year"].unique())
    n_states = d["state_abbr"].nunique()

    for y in years:
        d[f"D_{y}"] = ((d["year"] == y) & (d["border_70mi"] == 1)).astype(int)
    ref = years[len(years) // 2]  # a middle year as reference (arbitrary but stated)
    dvars = [f"D_{y}" for y in years if y != ref]

    # border_70mi main effect = border premium in the reference year; each D_y
    # (y != ref) is the incremental deviation from that reference-year premium,
    # so year-y premium = border_70mi + D_y. Omitting the main effect (as an
    # earlier version of this function did) forces the reference-year premium
    # to zero instead of estimating it, since border_70mi varies within state
    # (not absorbed by state FE) -- confirmed as a real bug via a side-by-side
    # comparison against a pooled single-coefficient spec on the NM dispensary
    # panel, where the no-main-effect design produced near-zero, insignificant
    # coefficients while the correctly-specified design and the pooled spec
    # agreed on a large, significant premium.
    fe = " + C(year)" + (" + C(state_abbr)" if n_states > 1 else "")
    formula = f"{yvar} ~ border_70mi + " + " + ".join(dvars) + " + log_population + log1p_captive_pop" + fe
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})

    print(f"\n--- {label}: {yvar}, ref year={ref}, n={int(m.nobs)}, "
          f"counties={d.fips5.nunique()}, states={sorted(d.state_abbr.unique())} ---")
    from scipy import stats
    b_ref = m.params["border_70mi"]
    se_ref = m.bse["border_70mi"]
    p_ref = m.pvalues["border_70mi"]
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
    print(f"log1p_captive_pop coef: {m.params.get('log1p_captive_pop', np.nan):+.4f} "
          f"(p={m.pvalues.get('log1p_captive_pop', np.nan):.4f})")
    return out


def main():
    percapita = load_with_captive("master_percapita_panel.csv")
    perstore = load_with_captive("master_perstore_panel.csv")
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    for outcome, df, yvar in [("percapita", percapita, "log_revenue_per_capita"),
                              ("perstore", perstore, "log_revenue_per_store")]:
        out_local = pooled_eventstudy(df, yvar, LOCAL_STATES, "Local barrier (CO+MT)")
        out_local["regime"] = "local"
        out_nonlocal = pooled_eventstudy(df, yvar, NONLOCAL_STATES, "Non-local barrier (WA+NM, Oregon excluded)")
        out_nonlocal["regime"] = "nonlocal"
        combined = pd.concat([out_local, out_nonlocal], ignore_index=True)
        combined.to_csv(os.path.join(TBL, f"localbarrier_eventstudy_v2_{outcome}.csv"), index=False)


if __name__ == "__main__":
    main()
