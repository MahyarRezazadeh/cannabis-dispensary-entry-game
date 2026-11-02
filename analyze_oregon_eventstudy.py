"""
Event study: does Oregon's border-county revenue premium (per capita) move
smoothly around its own Jan-2022 entry freeze, or is the before/after split
already reported (analyze_oregon_regime_switch.py) hiding a jump/pre-trend
that a single two-period split can't reveal?

Spec: y_ct = alpha + sum_{k != -1} theta_k * (Border_c x 1(tau_ct=k))
             + county FE + year FE + eps_ct
tau = year - 2022 (freeze year = 0), reference period tau=-1 (2021) omitted.
Same convention as the project's existing CO-NM switch-off event study.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def main():
    df = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    orr = df[df.state_abbr == "OR"].dropna(subset=["log_revenue_per_capita", "border_70mi", "log_population"]).copy()

    orr["tau"] = orr["year"].astype(int) - 2022

    # border_70mi is now TIME-VARYING (Section on time-varying border classification):
    # Josephine/Curry/Jackson are only "border" in 2017 (via still-nonlegal California),
    # unrelated to Oregon's own entry-freeze policy. Mixing them in would make the
    # "border group" composition jump between tau=-5 and tau=-4 for a reason that has
    # nothing to do with the entry-freeze event. Restrict this event study to Oregon's
    # ALWAYS-border counties (border_70mi==1 in every year of the panel, i.e. the
    # Idaho-adjacent core), which is the clean treatment group for a cap-policy event study.
    # "always border" = border_70mi==1 in EVERY year this county actually has data for
    # (some OR counties are missing from the panel in 2017 -- 22 of 30 counties have a
    # 2017 row -- for data-availability reasons unrelated to border status; requiring
    # presence in literally all 9 panel years would wrongly drop those as "transient").
    border_share = orr.groupby("fips5")["border_70mi"].mean()
    always_border = border_share[border_share == 1.0].index.tolist()
    county_names = orr[orr.fips5.isin(always_border)][["fips5", "county_name"]].drop_duplicates()
    print(f"Always-border Oregon counties (n={len(always_border)}):")
    print(county_names.to_string(index=False))

    transient = sorted(set(orr[orr.border_70mi == 1].fips5.unique()) - set(always_border))
    transient_names = orr[orr.fips5.isin(transient)][["fips5", "county_name"]].drop_duplicates()
    print(f"\nExcluded transient-border counties (CA/NV-adjacent, 2017 only, n={len(transient)}):")
    print(transient_names.to_string(index=False))

    orr["always_border"] = orr["fips5"].isin(always_border).astype(int)

    print("\nOregon per-capita panel event-time coverage:")
    print(orr.groupby("tau")["fips5"].nunique())

    # dummy for each tau except reference (-1), interacted with always_border
    taus = sorted(orr["tau"].unique())
    ref = -1
    tau_label = {k: (f"m{abs(k)}" if k < 0 else f"p{k}") for k in taus}
    for k in taus:
        if k == ref:
            continue
        orr[f"D_{tau_label[k]}"] = ((orr["tau"] == k) & (orr["always_border"] == 1)).astype(int)

    dvars = [f"D_{tau_label[k]}" for k in taus if k != ref]
    formula = "log_revenue_per_capita ~ " + " + ".join(dvars) + " + log_population + C(year) + C(fips5)"
    m = smf.ols(formula, data=orr).fit(cov_type="cluster", cov_kwds={"groups": orr["fips5"]})

    rows = []
    for k in taus:
        if k == ref:
            rows.append({"tau": k, "year": 2022 + k, "coef": 0.0, "se": 0.0, "p": np.nan, "ref": True})
            continue
        term = f"D_{tau_label[k]}"
        b = m.params.get(term, np.nan)
        se = m.bse.get(term, np.nan)
        p = m.pvalues.get(term, np.nan)
        rows.append({"tau": k, "year": 2022 + k, "coef": b, "se": se, "p": p, "ref": False})

    out = pd.DataFrame(rows).sort_values("tau")
    out["pct"] = (np.exp(out["coef"]) - 1) * 100
    out.to_csv(os.path.join(TBL, "oregon_eventstudy_percapita.csv"), index=False)
    print("\n--- Event-time coefficients (border x year, county-clustered SE, ref=2021) ---")
    print(out.to_string(index=False))
    print(f"\nn={int(m.nobs)}, counties={orr.fips5.nunique()}")


if __name__ == "__main__":
    main()
