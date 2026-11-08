"""
Year-by-year border premium under local-barrier (CO+MT) vs. non-local-barrier
(WA+OR+NM) states, for revenue outcomes. Unlike the earlier capped/free
event study, local_barrier does not vary within any state's panel, so this
is a fixed two-group comparison, not a switch-based design.

Produces two versions per outcome: Oregon excluded (clean) and Oregon
included (shows the same Baker-County/missing-pre-2020-data problem
documented in analyze_regime_eventstudy.py and analyze_oregon_eventstudy.py).
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"

LOCAL_STATES = ["CO", "MT"]
NONLOCAL_STATES = ["WA", "NM"]
NONLOCAL_STATES_WITH_OR = ["WA", "NM", "OR"]


def year_coef(df, yvar, states, year, min_border=2):
    d = df[(df.state_abbr.isin(states)) & (df.year == year)].dropna(
        subset=[yvar, "border_70mi", "log_population"]).copy()
    if d["border_70mi"].sum() < min_border or d[d.border_70mi == 0].shape[0] < 2:
        return None
    n_states = d["state_abbr"].nunique()
    formula = f"{yvar} ~ border_70mi + log_population" + (" + C(state_abbr)" if n_states > 1 else "")
    try:
        m = smf.ols(formula, data=d).fit(cov_type="HC1")
    except Exception:
        return None
    b = m.params.get("border_70mi", np.nan)
    se = m.bse.get("border_70mi", np.nan)
    p = m.pvalues.get("border_70mi", np.nan)
    return {"year": year, "coef": b, "se": se, "p": p, "n": int(m.nobs),
            "n_border": int(d.border_70mi.sum())}


def run(df, yvar, include_oregon, fname):
    nonlocal_states = NONLOCAL_STATES_WITH_OR if include_oregon else NONLOCAL_STATES
    tag = "Oregon included" if include_oregon else "Oregon excluded"
    print(f"\n=== {fname} ({tag}) ===")
    rows = []
    years = sorted(df.year.unique())
    for regime_label, states in [("local", LOCAL_STATES), ("nonlocal", nonlocal_states)]:
        for y in years:
            r = year_coef(df, yvar, states, y)
            if r is not None:
                r["regime"] = regime_label
                rows.append(r)
    out = pd.DataFrame(rows).sort_values(["regime", "year"])
    out["pct"] = (np.exp(out["coef"]) - 1) * 100
    out.to_csv(os.path.join(TBL, fname), index=False)
    print(out.to_string(index=False))


def main():
    percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    run(percapita, "log_revenue_per_capita", False, "localbarrier_eventstudy_percapita.csv")
    run(percapita, "log_revenue_per_capita", True, "localbarrier_eventstudy_percapita_withOR.csv")
    run(perstore, "log_revenue_per_store", False, "localbarrier_eventstudy_perstore.csv")
    run(perstore, "log_revenue_per_store", True, "localbarrier_eventstudy_perstore_withOR.csv")


if __name__ == "__main__":
    main()
