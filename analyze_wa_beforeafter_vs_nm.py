"""
Task 1: DD analysis of Washington's OWN border premium before vs. after its
2016 cap saturation/application-window closure, plus a direct comparison
against New Mexico (always free) as a benchmark for what a genuinely
free-entry border premium looks like. Three pieces:

  (a) Within-WA DD: does WA's own border premium change after 2016?
      y_ct = a + b1*border + b2*post2016 + b3*border*post2016 + controls + year FE
  (b) Level comparison, WA-post-2016 vs NM: is WA's post-2016 border
      premium statistically distinguishable from NM's (the paper's
      cleanest "genuinely free" benchmark)? If not distinguishable, that
      supports treating WA post-2016 as behaving like a free-entry state;
      if distinguishable, it undermines that reclassification.
      y_ct = a + b1*border + b2*is_WA + b3*border*is_WA + controls + year FE
  (c) Event-study version of (a): border x year interactions within WA
      only, for a visual before/after picture.

Captive-population control, year (+state) FE, county-clustered SEs,
2026 dropped throughout, consistent with the rest of this project's
current specifications.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import os

TBL = "tables"
OUT = "figures_reframe"


def load_with_captive(panel_file):
    df = pd.read_csv(os.path.join(TBL, panel_file), dtype={"fips5": str})
    df = df[df["year"] != 2026].copy()
    cap = pd.read_csv(os.path.join(TBL, "captive_population_all5states.csv"), dtype={"fips5": str})
    df = df.merge(cap[["fips5", "state_abbr", "captive_population"]], on=["fips5", "state_abbr"], how="left")
    df["captive_population"] = df["captive_population"].fillna(0)
    df["log1p_captive_pop"] = np.log1p(df["captive_population"])
    return df


def within_wa_dd(df, yvar, label):
    d = df[df.state_abbr == "WA"].dropna(subset=[yvar, "border_70mi", "log_population", "log1p_captive_pop"]).copy()
    d["post2016"] = (d["year"] >= 2016).astype(int)
    d["year"] = d["year"].astype(str)
    formula = f"{yvar} ~ border_70mi * post2016 + log_population + log1p_captive_pop + C(year)"
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\n=== WITHIN-WA DD: {label} (n={int(m.nobs)}, counties={d.fips5.nunique()}) ===")
    for term in ["border_70mi", "post2016", "border_70mi:post2016"]:
        b = m.params.get(term, np.nan)
        p = m.pvalues.get(term, np.nan)
        pct = (np.exp(b) - 1) * 100 if pd.notna(b) else np.nan
        print(f"  {term:24s} b={b:+.4f} ({pct:+.1f}%)  p(cl)={p:.4f}")
    b_border = m.params.get("border_70mi", np.nan)
    b_dd = m.params.get("border_70mi:post2016", np.nan)
    print(f"  Pre-2016 WA border premium:  {(np.exp(b_border)-1)*100:+.1f}%")
    print(f"  Post-2016 WA border premium: {(np.exp(b_border+b_dd)-1)*100:+.1f}%")


def wa_vs_nm_level(df, yvar, label):
    d = df[((df.state_abbr == "WA") & (df.year >= 2016)) | (df.state_abbr == "NM")].dropna(
        subset=[yvar, "border_70mi", "log_population", "log1p_captive_pop"]).copy()
    d["is_WA"] = (d.state_abbr == "WA").astype(int)
    d["year"] = d["year"].astype(str)
    formula = f"{yvar} ~ border_70mi * is_WA + log_population + log1p_captive_pop + C(year)"
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\n=== WA(2016+) vs NM LEVEL COMPARISON: {label} (n={int(m.nobs)}, counties={d.fips5.nunique()}) ===")
    for term in ["border_70mi", "is_WA", "border_70mi:is_WA"]:
        b = m.params.get(term, np.nan)
        p = m.pvalues.get(term, np.nan)
        pct = (np.exp(b) - 1) * 100 if pd.notna(b) else np.nan
        print(f"  {term:24s} b={b:+.4f} ({pct:+.1f}%)  p(cl)={p:.4f}")
    b_border = m.params.get("border_70mi", np.nan)
    b_diff = m.params.get("border_70mi:is_WA", np.nan)
    p_diff = m.pvalues.get("border_70mi:is_WA", np.nan)
    print(f"  NM border premium (benchmark):     {(np.exp(b_border)-1)*100:+.1f}%")
    print(f"  WA(2016+) border premium:          {(np.exp(b_border+b_diff)-1)*100:+.1f}%")
    verdict = "NOT distinguishable from NM (supports free reclassification)" if p_diff > 0.10 else \
        "DISTINGUISHABLE from NM (undermines free reclassification)"
    print(f"  Difference p-value: {p_diff:.4f} -- {verdict}")


def eventstudy_wa_beforeafter(df, yvar, label):
    d = df[df.state_abbr == "WA"].dropna(subset=[yvar, "border_70mi", "log_population", "log1p_captive_pop"]).copy()
    d["year"] = d["year"].astype(int)
    years = sorted(d["year"].unique())
    for y in years:
        d[f"D_{y}"] = ((d["year"] == y) & (d["border_70mi"] == 1)).astype(int)
    ref = 2015
    dvars = [f"D_{y}" for y in years if y != ref]
    formula = f"{yvar} ~ border_70mi + " + " + ".join(dvars) + " + log_population + log1p_captive_pop + C(year)"
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\n=== WA EVENT STUDY: {label} (n={int(m.nobs)}, counties={d.fips5.nunique()}) ===")
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
        se_y = float(np.sqrt(m.cov_params().loc["border_70mi", "border_70mi"]
                              + m.cov_params().loc[term, term]
                              + 2 * m.cov_params().loc["border_70mi", term]))
        rows.append({"year": y, "coef": b_y, "se": se_y, "ref": False})
    out = pd.DataFrame(rows).sort_values("year")
    out["pct"] = (np.exp(out["coef"]) - 1) * 100
    print(out.to_string(index=False))
    return out


def main():
    percapita = load_with_captive("master_percapita_panel.csv")
    perstore = load_with_captive("master_perstore_panel.csv")
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    print("#" * 90)
    print("PART (a): WITHIN-WASHINGTON DD, before vs after 2016")
    print("#" * 90)
    within_wa_dd(percapita, "log_revenue_per_capita", "Revenue per capita")
    within_wa_dd(perstore, "log_revenue_per_store", "Revenue per store")
    within_wa_dd(perstore, "log_dispensary_per_1000", "Dispensary density")

    print("\n" + "#" * 90)
    print("PART (b): WA (2016+) vs NM level comparison")
    print("#" * 90)
    wa_vs_nm_level(percapita, "log_revenue_per_capita", "Revenue per capita")
    wa_vs_nm_level(perstore, "log_revenue_per_store", "Revenue per store")
    wa_vs_nm_level(perstore, "log_dispensary_per_1000", "Dispensary density")

    print("\n" + "#" * 90)
    print("PART (c): WA event study (border x year), 2015 = reference")
    print("#" * 90)
    out_pc = eventstudy_wa_beforeafter(percapita, "log_revenue_per_capita", "Revenue per capita")
    out_pc.to_csv(os.path.join(TBL, "wa_beforeafter_eventstudy_percapita.csv"), index=False)
    out_ps = eventstudy_wa_beforeafter(perstore, "log_revenue_per_store", "Revenue per store")
    out_ps.to_csv(os.path.join(TBL, "wa_beforeafter_eventstudy_perstore.csv"), index=False)


if __name__ == "__main__":
    main()
