"""
Test two candidate fixes for the noisy "free" group event-study estimates
(WA-as-free grouping: Capped=CO+MT+OR2022+, Free=WA+NM+OR2015-2021),
against the baseline spec already built in analyze_wa_as_free.py:

  BASELINE: border + D_year + log_population + log1p_captive_pop
            + C(year) + C(state_abbr) [additive], county-clustered SE
  VARIANT A: same, but state x year fully interacted (C(state_year))
             instead of additive C(year)+C(state_abbr)
  VARIANT B: baseline spec, restricted to a BALANCED panel -- only
             counties present in every year their regime group covers
             (the same "always-reporting" fix already used for the
             Oregon Baker County problem in analyze_oregon_eventstudy.py)

Reported side by side per year, per outcome, per regime, so the actual
effect on standard errors can be read off directly rather than assumed.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import os

TBL = "tables"
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


def eventstudy(d, yvar, fe_type, label):
    """fe_type: 'additive' (year+state FE), 'stateyear' (state x year FE)"""
    d = d.dropna(subset=[yvar, "border_70mi", "log_population", "log1p_captive_pop"]).copy()
    if d.empty or d["border_70mi"].nunique() < 2:
        print(f"  [{label}] insufficient data (n={len(d)}, border variation={d['border_70mi'].nunique() if len(d) else 0})")
        return None
    d["year"] = d["year"].astype(int)
    years = sorted(d["year"].unique())
    n_states = d["state_abbr"].nunique()
    for y in years:
        d[f"D_{y}"] = ((d["year"] == y) & (d["border_70mi"] == 1)).astype(int)
    ref = years[len(years) // 2]
    dvars = [f"D_{y}" for y in years if y != ref]

    if fe_type == "additive":
        fe = " + C(year)" + (" + C(state_abbr)" if n_states > 1 else "")
    elif fe_type == "stateyear":
        d["state_year"] = d["state_abbr"].astype(str) + "_" + d["year"].astype(str)
        fe = " + C(state_year)"
    else:
        raise ValueError(fe_type)

    formula = f"{yvar} ~ border_70mi + " + " + ".join(dvars) + " + log_population + log1p_captive_pop" + fe
    try:
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        return None
    if "border_70mi" not in m.params.index or pd.isna(m.bse.get("border_70mi", np.nan)):
        print(f"  [{label}] border_70mi not identified (collinear/singular)")
        return None

    b_ref, se_ref = m.params["border_70mi"], m.bse["border_70mi"]
    rows = [{"year": ref, "coef": b_ref, "se": se_ref, "ref": True}]
    for y in years:
        if y == ref:
            continue
        term = f"D_{y}"
        if term not in m.params.index:
            continue
        d_b = m.params[term]
        b_y = b_ref + d_b
        try:
            se_y = float(np.sqrt(m.cov_params().loc["border_70mi", "border_70mi"]
                                  + m.cov_params().loc[term, term]
                                  + 2 * m.cov_params().loc["border_70mi", term]))
        except KeyError:
            se_y = np.nan
        rows.append({"year": y, "coef": b_y, "se": se_y, "ref": False})
    out = pd.DataFrame(rows).sort_values("year")
    out["pct"] = (np.exp(out["coef"]) - 1) * 100
    avg_se = out["se"].mean()
    print(f"  [{label}] n={int(m.nobs)}, counties={d.fips5.nunique()}, states={sorted(d.state_abbr.unique())}, "
          f"avg SE (log pts)={avg_se:.3f}")
    return out


def balanced_panel(d, regime_val):
    """Keep only counties present in every year WITHIN THEIR OWN STATE's coverage window
    (not the combined group's full span, which no single county can satisfy when the group
    splices together states/periods that never overlap in time, e.g. Oregon 2015-2021 +
    New Mexico 2022-2026 -- there is no calendar year both contribute to)."""
    sub = d[d.wa_free_capped == regime_val]
    keep_fips = []
    for state, g in sub.groupby("state_abbr"):
        state_years = set(g["year"].unique())
        years_present = g.groupby("fips5")["year"].apply(set)
        keep_fips.extend(years_present[years_present == state_years].index.tolist())
    return d[(d.wa_free_capped == regime_val) & (d.fips5.isin(keep_fips))]


def main():
    percapita = load_with_captive("master_percapita_panel.csv")
    perstore = load_with_captive("master_perstore_panel.csv")

    results = {}
    for outcome, df, yvar in [("percapita", percapita, "log_revenue_per_capita"),
                              ("perstore", perstore, "log_revenue_per_store")]:
        print(f"\n{'='*80}\n{outcome.upper()}\n{'='*80}")
        for regime_val, regime_name in [(1, "capped"), (0, "free")]:
            print(f"\n--- {regime_name} ---")
            d_regime = df[df.wa_free_capped == regime_val]

            print(f"BASELINE (additive year+state FE):")
            out_base = eventstudy(d_regime, yvar, "additive", "baseline")

            print(f"VARIANT A (state x year FE):")
            out_a = eventstudy(d_regime, yvar, "stateyear", "state x year FE")

            print(f"VARIANT B (balanced panel, additive FE):")
            d_bal = balanced_panel(df, regime_val)
            n_dropped_counties = d_regime.fips5.nunique() - d_bal.fips5.nunique()
            print(f"  Balanced panel: {d_bal.fips5.nunique()} counties kept "
                  f"(dropped {n_dropped_counties} that don't report every year)")
            out_b = eventstudy(d_bal, yvar, "additive", "balanced panel")

            for tag, out in [("baseline", out_base), ("stateyear_fe", out_a), ("balanced_panel", out_b)]:
                if out is not None:
                    out["regime"] = regime_name
                    out["variant"] = tag
                    out["outcome"] = outcome
                    results[f"{outcome}_{regime_name}_{tag}"] = out

    all_out = pd.concat(results.values(), ignore_index=True)
    all_out.to_csv(os.path.join(TBL, "wa_as_free_fe_comparison.csv"), index=False)
    print(f"\nSaved wa_as_free_fe_comparison.csv")

    print(f"\n{'='*80}\nSUMMARY: average SE (log points) by variant\n{'='*80}")
    summ = all_out.groupby(["outcome", "regime", "variant"])["se"].mean().unstack()
    print(summ.to_string())


if __name__ == "__main__":
    main()
