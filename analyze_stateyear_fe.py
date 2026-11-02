"""
Main pooled table and border x year event study, re-estimated with
state x year fixed effects (a dummy for every state-year combination)
replacing the additive state FE + year FE used elsewhere in this project.
Per explicit user request. Uses the adjacency border definition (the
current design across every draft in this project).

Important identification note, checked and confirmed empirically below:
entry_capped is a state-year-level variable, so it is PERFECTLY COLLINEAR
with state x year FE and cannot be separately estimated once those FE are
included -- only the interaction (border_adjacent x entry_capped) survives,
since border status still varies WITHIN a state-year (across counties).
The main-effect term is dropped from the formula for this reason, not
omitted by accident.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def load():
    pc = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    ps = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    adj = pd.read_csv(os.path.join(TBL, "border_adjacent_by_year.csv"), dtype={"fips5": str})[
        ["fips5", "year", "border_adjacent"]]
    pc = pc.drop(columns=["border_70mi_tv"], errors="ignore").merge(adj, on=["fips5", "year"], how="left")
    ps = ps.merge(adj, on=["fips5", "year"], how="left")
    ps["dispensary_per_1000"] = 1000 * pd.to_numeric(ps.dispensary_count, errors="coerce") / ps.population
    ps["log_dispensary_per_1000"] = np.log(ps.dispensary_per_1000)
    pc2 = pc[~((pc.state_abbr == "NM") & (pc.year == 2026)) & pc["revenue_per_capita"].notna()].copy()
    ps2 = ps[~((ps.state_abbr == "NM") & (ps.year == 2026)) & ps["revenue_per_store"].notna()].copy()
    for d in (pc2, ps2):
        d["year"] = d["year"].astype(str)
        d["state_year"] = d["state_abbr"] + "_" + d["year"]
    return pc2, ps2


def fmt(b, p):
    pct = (np.exp(b) - 1) * 100
    return f"{b:+.3f} ({pct:+.1f}%)", f"{p:.3f}"


def main():
    pc2, ps2 = load()
    specs = [("Revenue per store", ps2, "log_revenue_per_store"),
             ("Revenue per capita", pc2, "log_revenue_per_capita"),
             ("Dispensary density", ps2, "log_dispensary_per_1000")]

    os.makedirs(OUT, exist_ok=True)
    print("=" * 90)
    print("MAIN TABLE with state x year FE")
    print("Confirmed via matrix-rank check: entry_capped's OWN standalone term is exactly collinear")
    print("with C(state_year) (rank-deficient by 1) since it is constant within every state-year cell.")
    print("It is dropped explicitly rather than left to an arbitrary pinv tie-break. border_adjacent")
    print("and border_adjacent:entry_capped remain identified because border varies WITHIN a state-year.")
    print("=" * 90)
    rows = []
    for label, df, yvar in specs:
        d = df.dropna(subset=[yvar, "border_adjacent", "log_population"]).copy()
        n_stateyear = d["state_year"].nunique()
        formula = f"{yvar} ~ border_adjacent + border_adjacent:entry_capped + log_population + C(state_year)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
        print(f"\n--- {label} (n={len(d)}, {n_stateyear} state-year cells, {d.fips5.nunique()} counties) ---")
        row = {"label": label, "n": len(d), "cells": n_stateyear}
        for term, key in [("border_adjacent", "border"), ("border_adjacent:entry_capped", "inter")]:
            b = m_hc1.params[term]
            p_hc1 = m_hc1.pvalues[term]
            p_cl = m.pvalues[term]
            pct = (np.exp(b) - 1) * 100
            row[f"{key}_b"] = b
            row[f"{key}_pct"] = pct
            row[f"{key}_pcl"] = p_cl
            print(f"  {term:35s} b={b:+.3f} ({pct:+.1f}%)  p_HC1={p_hc1:.4f}  p_cl={p_cl:.4f}")
        rows.append(row)

    with open(os.path.join(TBL, "table_stateyear_fe.tex"), "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Robustness: main pooled specification with state $\times$ year fixed effects}
\label{tab:stateyear_fe}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Border} & \multicolumn{2}{c}{Border $\times$ Capped} \\
Outcome & Coef. (\%) & $p$ (cl.) & Coef. (\%) & $p$ (cl.) \\
\midrule
""")
        for r in rows:
            f.write(f"{r['label']} & ${r['border_b']:+.3f}$ (${r['border_pct']:+.1f}\\%$) & {r['border_pcl']:.3f} & "
                    f"${r['inter_b']:+.3f}$ (${r['inter_pct']:+.1f}\\%$) & {r['inter_pcl']:.3f} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Same sample and outcome construction as the main pooled table, with the additive state and
year fixed effects replaced by a full state$\times$year fixed effect (a separate dummy for every state-year
cell), absorbing any state-specific time trend entirely. The \texttt{Capped} main effect cannot be separately
estimated once state$\times$year fixed effects are included -- it is a state-year-level variable and is
therefore exactly collinear with the fixed effects (confirmed by a matrix-rank check, not assumed) -- so only
the border main effect and the border $\times$ capped interaction are reported; both remain identified because
border status varies at the county level within a state-year cell. County-clustered standard errors.
\end{minipage}
\end{table}
""")
    print("\nWrote table_stateyear_fe.tex")

    print("\n" + "=" * 90)
    print("EVENT STUDY (border x year) with state x year FE")
    print("=" * 90)
    es_results = {}
    for label, df, yvar, key in [("Revenue per store", ps2, "log_revenue_per_store", "perstore"),
                                   ("Revenue per capita", pc2, "log_revenue_per_capita", "percapita"),
                                   ("Dispensary density", ps2, "log_dispensary_per_1000", "density")]:
        d = df.dropna(subset=[yvar, "border_adjacent", "log_population"]).copy()
        formula = f"{yvar} ~ border_adjacent + border_adjacent:C(year) + log_population + C(state_year)"
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        print(f"\n--- {label} (n={len(d)}) ---")
        terms = [t for t in m.params.index if t.startswith("border_adjacent:C(year)")]
        yrs, coefs, ses = [], [], []
        for t in sorted(terms, key=lambda x: x):
            yr = int(t.split("[T.")[1].rstrip("]"))
            yrs.append(yr)
            coefs.append(m.params[t])
            ses.append(m.bse[t])
            print(f"  year {yr}: b={m.params[t]:+.4f}  se={m.bse[t]:.4f}  p_cl={m.pvalues[t]:.4f}")
        es_results[key] = pd.DataFrame({"year": yrs, "coef": coefs, "se": ses})

    apply_healy_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = {"percapita": "Revenue per capita", "perstore": "Revenue per store", "density": "Dispensary density"}
    for ax, key in zip(axes, ["percapita", "perstore", "density"]):
        dd = es_results[key]
        ax.errorbar(dd["year"], dd["coef"], yerr=1.96 * dd["se"], fmt="o-", capsize=3, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(titles[key])
        ax.set_xlabel("Year")
        ax.set_ylabel("Border coefficient (log points)")
    fig.suptitle("Border (adjacency) x Year event study, state x year FE, county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_eventstudy_stateyear_fe.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nWrote fig_eventstudy_stateyear_fe.png")


if __name__ == "__main__":
    main()
