"""
Systematic comparison: run the pooled border x regime interaction test
across every entry-regulation grouping built so far in this project, each
under both the standard (unbalanced) panel and the balanced-panel
restriction validated last step (counties kept only if they report every
year within their own state's window), to see which grouping x outcome
combination actually reaches significance under county-clustered SEs --
and whether that significance is robust to the balanced-panel fix or
was an artifact of panel churn.

Groupings:
  A. Main Capped        CO/WA/MT capped, OR switches 2022, NM free
                         (the paper's primary classification)
  B. Strict state-cap    Only WA + OR-2022+ capped; CO/MT/NM/OR-free free
  C. Local barrier       CO+MT=1 (local) vs WA+OR+NM=0 (non-local)
  D. Local vs free       CO+MT (local) vs NM+OR-2015-2021 (free); WA dropped
  E. WA-as-free          CO+MT+OR2022+ capped vs WA+NM+OR2015-2021 free
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"


def add_groupings(df):
    d = df.copy()
    # A, B already exist as entry_capped, entry_capped_strict; C exists as local_barrier
    # D: local vs free, WA dropped entirely
    is_local = d["state_abbr"].isin(["CO", "MT"])
    is_free_d = (d["state_abbr"] == "NM") | ((d["state_abbr"] == "OR") & (d["year"] <= 2021))
    d["grp_D"] = np.select([is_local, is_free_d], [1, 0], default=np.nan)
    # E: WA-as-free
    is_capped_e = d["state_abbr"].isin(["CO", "MT"]) | ((d["state_abbr"] == "OR") & (d["year"] >= 2022))
    d["grp_E"] = is_capped_e.astype(int)
    return d


def balanced_panel(d, groupcol):
    """Restrict to counties present in every year within their own state's window,
    computed separately for each side of the grouping."""
    keep = []
    sub = d.dropna(subset=[groupcol])
    for (state, grp), g in sub.groupby(["state_abbr", groupcol]):
        state_years = set(g["year"].unique())
        years_present = g.groupby("fips5")["year"].apply(set)
        keep.extend(years_present[years_present == state_years].index.tolist())
    return d[d.fips5.isin(keep)]


def run_spec(df, yvar, groupcol, label, balanced=False):
    d = df.dropna(subset=[yvar, "border_70mi", groupcol, "log_population"]).copy()
    if balanced:
        d = balanced_panel(d, groupcol)
    if len(d) == 0 or d[groupcol].nunique() < 2:
        return {"label": label, "balanced": balanced, "yvar": yvar, "n": len(d),
                "pct_interaction": np.nan, "p_hc1": np.nan, "p_cl": np.nan, "note": "insufficient data"}
    d["year"] = d["year"].astype(str)
    within_var = d.groupby("state_abbr")[groupcol].nunique()
    state_fe = not (within_var <= 1).all()
    fe = " + C(year) + C(state_abbr)" if state_fe else " + C(year)"
    formula = f"{yvar} ~ border_70mi * {groupcol} + log_population" + fe
    try:
        m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
        m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    except Exception as e:
        return {"label": label, "balanced": balanced, "yvar": yvar, "n": len(d),
                "pct_interaction": np.nan, "p_hc1": np.nan, "p_cl": np.nan, "note": f"failed: {e}"}
    term = f"border_70mi:{groupcol}"
    if term not in m_hc1.params.index:
        return {"label": label, "balanced": balanced, "yvar": yvar, "n": len(d),
                "pct_interaction": np.nan, "p_hc1": np.nan, "p_cl": np.nan, "note": "not identified"}
    b = m_hc1.params[term]
    pct = (np.exp(b) - 1) * 100
    return {"label": label, "balanced": balanced, "yvar": yvar, "n": int(m_hc1.nobs),
            "counties": d.fips5.nunique(), "pct_interaction": pct,
            "p_hc1": m_hc1.pvalues[term], "p_cl": m_cl.pvalues[term], "note": "ok"}


def main():
    percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])
    percapita = add_groupings(percapita)
    perstore = add_groupings(perstore)

    groupings = [
        ("A. Main Capped (CO/WA/MT capped, OR switch, NM free)", "entry_capped"),
        ("B. Strict state-cap (WA+OR2022 capped only)", "entry_capped_strict"),
        ("C. Local barrier (CO+MT) vs non-local (WA+OR+NM)", "local_barrier"),
        ("D. Local (CO+MT) vs free (NM+OR-free), WA dropped", "grp_D"),
        ("E. WA-as-free (CO+MT+OR2022 capped vs WA+NM+OR-free)", "grp_E"),
    ]
    outcomes = [("Revenue per capita", percapita, "log_revenue_per_capita"),
                ("Revenue per store", perstore, "log_revenue_per_store"),
                ("Dispensary density", perstore, "log_dispensary_per_1000")]

    rows = []
    for label, groupcol in groupings:
        for oname, df, yvar in outcomes:
            for balanced in [False, True]:
                r = run_spec(df, yvar, groupcol, label, balanced=balanced)
                r["outcome"] = oname
                rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TBL, "grouping_comparison_full.csv"), index=False)

    print("=" * 110)
    print("Border x regime INTERACTION p-values (county-clustered), by grouping / outcome / balance")
    print("=" * 110)
    for label, _ in groupings:
        print(f"\n--- {label} ---")
        sub = res[res.label == label]
        piv = sub.pivot_table(index="outcome", columns="balanced", values=["pct_interaction", "p_cl"], aggfunc="first")
        print(piv.to_string())

    print("\n" + "=" * 110)
    print("SIGNIFICANT (p_cl < 0.05) combinations:")
    print("=" * 110)
    sig = res[res.p_cl < 0.05].sort_values(["outcome", "p_cl"])
    print(sig[["label", "outcome", "balanced", "pct_interaction", "p_cl", "n", "counties"]].to_string(index=False))


if __name__ == "__main__":
    main()
