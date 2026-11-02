"""
Year-by-year border premium under capped vs. free entry, for revenue outcomes.
Oregon is EXCLUDED (its only always-border counties, Baker/Malheur/Wallowa, have
a severe data problem: Malheur and Wallowa have no data before 2020, and Baker's
own 2017-2019 revenue-per-capita figures are implausibly high -- $955-$1,784 per
resident in a 16,912-person county -- likely non-retail transactions folded into
this county-level series; the free-period result reported earlier for Oregon does
not survive excluding Baker (drops from +173% p=0.007 to +90% p=0.21) or
restricting to the reliable 2020-2021 window (+138% p=0.12), so it is dropped
here rather than presented as if it were reliable).

This is not an event study around a single switch date (no other state has one
with adequate data): Colorado, Washington, and Montana are capped for their
entire panels, and New Mexico is free for its entire panel, so instead this
shows the border premium's own year-by-year path under each regime -- a
persistent-divergence check rather than a before/after-a-policy-change check.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"

CAPPED_STATES = ["CO", "WA", "MT"]
FREE_STATES = ["NM"]


def regime_groups_for_year(year, include_oregon):
    """Oregon is free through 2021 and capped from 2022 (its own within-state
    switch, per the entry_capped coding used throughout this project) -- so
    when included, it moves between the two groups depending on the year."""
    capped = list(CAPPED_STATES)
    free = list(FREE_STATES)
    if include_oregon:
        if year <= 2021:
            free = free + ["OR"]
        else:
            capped = capped + ["OR"]
    return capped, free


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


def run(percapita_full, perstore_full, include_oregon):
    suffix = "_withOR" if include_oregon else ""
    percapita = percapita_full if include_oregon else percapita_full[percapita_full.state_abbr != "OR"]
    perstore = perstore_full if include_oregon else perstore_full[perstore_full.state_abbr != "OR"]

    for label, df, yvar, fname in [
        ("Revenue per capita", percapita, "log_revenue_per_capita", f"regime_eventstudy_percapita{suffix}.csv"),
        ("Revenue per store", perstore, "log_revenue_per_store", f"regime_eventstudy_perstore{suffix}.csv"),
    ]:
        tag = "Oregon included" if include_oregon else "Oregon excluded"
        print(f"\n=== {label} ({tag}) ===")
        rows = []
        years = sorted(df.year.unique())
        for y in years:
            capped_states, free_states = regime_groups_for_year(y, include_oregon)
            for regime_label, states in [("capped", capped_states), ("free", free_states)]:
                r = year_coef(df, yvar, states, y)
                if r is not None:
                    r["regime"] = regime_label
                    rows.append(r)
        out = pd.DataFrame(rows).sort_values(["regime", "year"])
        out["pct"] = (np.exp(out["coef"]) - 1) * 100
        out.to_csv(os.path.join(TBL, fname), index=False)
        print(out.to_string(index=False))


def main():
    percapita_full = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore_full = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    perstore_full["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore_full["dispensary_count"], errors="coerce") / perstore_full["population"])

    run(percapita_full, perstore_full, include_oregon=False)
    run(percapita_full, perstore_full, include_oregon=True)


if __name__ == "__main__":
    main()
