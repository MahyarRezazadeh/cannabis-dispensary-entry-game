"""
Re-estimate the main pooled table and the border x year event study using the
NEW border definition requested 2026-08-18: literal county-to-state-line
geometric adjacency (border_adjacent, from build_border_adjacency_5states.py),
replacing the 70-mile-threshold design (border_70mi) used everywhere else in
this project up to this point. Per explicit instruction, this is a SEPARATE
exploratory pass -- outputs go to tables_reframe/*_adjacency.tex and
figures_reframe/*_adjacency.png, and nothing here is wired into any draft
manuscript until the user reviews it.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"
OUT = "tables_reframe"
FIGOUT = "figures_reframe"


def fit(df, yvar, formula):
    d = df.dropna(subset=[yvar]).copy()
    d["year"] = d["year"].astype(str)
    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    return m_hc1, m_cl, int(m_hc1.nobs), d["fips5"].nunique()


def fmt(b, p_hc1, p_cl):
    pct = (np.exp(b) - 1) * 100
    return f"${b:+.3f}$ (${pct:+.1f}\\%$)", f"{p_hc1:.3f}", f"{p_cl:.3f}"


def load_panels_with_adjacency():
    adj = pd.read_csv(os.path.join(TBL, "border_adjacent_by_year.csv"), dtype={"fips5": str})
    adj = adj[["fips5", "year", "border_adjacent"]]

    percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    perstore["dispensary_per_1000"] = 1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"]
    perstore["log_dispensary_per_1000"] = np.log(perstore["dispensary_per_1000"])

    for name, df in [("percapita", percapita), ("perstore", perstore)]:
        before = len(df)
        df2 = df.merge(adj, on=["fips5", "year"], how="left")
        n_missing = df2["border_adjacent"].isna().sum()
        print(f"{name}: {before} rows, {n_missing} unmatched to adjacency table (should be 0)")
        if name == "percapita":
            percapita = df2
        else:
            perstore = df2

    # exclude NM's incomplete 2026 partial year, matching every other table in this project
    percapita = percapita[~((percapita.state_abbr == "NM") & (percapita.year == 2026))].copy()
    perstore = perstore[~((perstore.state_abbr == "NM") & (perstore.year == 2026))].copy()
    return percapita, perstore


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGOUT, exist_ok=True)
    percapita, perstore = load_panels_with_adjacency()

    print("\n--- Adjacency vs 70mi agreement check ---")
    for name, df in [("percapita", percapita), ("perstore", perstore)]:
        tab = pd.crosstab(df["border_70mi"], df["border_adjacent"])
        print(f"\n{name} (rows=border_70mi, cols=border_adjacent):\n{tab}")

    specs = [
        ("Revenue per store", perstore, "log_revenue_per_store",
         "log_revenue_per_store ~ border_adjacent * entry_capped + log_population + C(year)"),
        ("Revenue per capita", percapita, "log_revenue_per_capita",
         "log_revenue_per_capita ~ border_adjacent * entry_capped + log_population + C(year) + C(state_abbr)"),
        ("Dispensary density", perstore, "log_dispensary_per_1000",
         "log_dispensary_per_1000 ~ border_adjacent * entry_capped + log_population + C(year)"),
    ]

    rows = []
    for label, df, yvar, formula in specs:
        m_hc1, m_cl, n, ncl = fit(df, yvar, formula)
        row = {"Outcome": label, "n": n, "Counties": ncl}
        for term, short in [("border_adjacent", "Border"), ("entry_capped", "Capped"),
                             ("border_adjacent:entry_capped", "Border x Capped")]:
            b = m_hc1.params.get(term, np.nan)
            p1 = m_hc1.pvalues.get(term, np.nan)
            p2 = m_cl.pvalues.get(term, np.nan)
            row[f"{short}_b"], row[f"{short}_pHC1"], row[f"{short}_pCl"] = fmt(b, p1, p2)
        rows.append(row)
        print(f"\n{label}: n={n}, counties={ncl}")
        print(m_hc1.summary().tables[1])

    with open(os.path.join(OUT, "table_main_pooled_adjacency.tex"), "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{EXPLORATORY -- Main pooled specification under adjacency-based border definition (not yet in any draft)}
\label{tab:main_pooled_adjacency}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
& \multicolumn{2}{c}{Border} & \multicolumn{2}{c}{Capped} & \multicolumn{2}{c}{Border $\times$ Capped} & \\
Outcome & Coef. (\%) & $p$ (cl.) & Coef. (\%) & $p$ (cl.) & Coef. (\%) & $p$ (cl.) & $n$ \\
\midrule
""")
        for r in rows:
            f.write(f"{r['Outcome']} & {r['Border_b']} & {r['Border_pCl']} & {r['Capped_b']} & {r['Capped_pCl']} & "
                    f"{r['Border x Capped_b']} & {r['Border x Capped_pCl']} & {r['n']} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Border is now literal county-to-state-line geometric adjacency (touches a nonlegal-state
county as of that observation's own year), not the 70-mile threshold used in the main draft. Otherwise
identical specification to the 70-mile version.
\end{minipage}
\end{table}
""")
    print("\nWrote table_main_pooled_adjacency.tex")

    # ---------- event study: border x year, same design as fig44-46 ----------
    event_specs = [
        ("percapita", percapita, "log_revenue_per_capita", "figs_percapita"),
        ("perstore", perstore, "log_revenue_per_store", "figs_perstore"),
        ("density", perstore, "log_dispensary_per_1000", "figs_density"),
    ]
    es_results = {}
    for name, df, yvar, _ in event_specs:
        d = df.dropna(subset=[yvar]).copy()
        d["year_s"] = d["year"].astype(str)
        d["byear"] = d["border_adjacent"] * d["entry_capped"]
        formula = (f"{yvar} ~ border_adjacent + border_adjacent:C(year_s) + log_population + C(year_s)"
                   + (" + C(state_abbr)" if name == "percapita" else ""))
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        terms = [t for t in m.params.index if t.startswith("border_adjacent:C(year_s)")]
        yrs = [int(t.split("[T.")[1].rstrip("]")) for t in terms]
        coefs = [m.params[t] for t in terms]
        ses = [m.bse[t] for t in terms]
        es_results[name] = pd.DataFrame({"year": yrs, "coef": coefs, "se": ses}).sort_values("year")
        es_results[name].to_csv(os.path.join(TBL, f"eventstudy_adjacency_{name}.csv"), index=False)
        print(f"\nEvent study ({name}), border x year coefficients:")
        print(es_results[name].to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
    try:
        from healy_style import apply_healy_style, PALETTE
        apply_healy_style()
    except Exception:
        PALETTE = {"blue": "#2A6F97"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = {"percapita": "Revenue per capita", "perstore": "Revenue per store", "density": "Dispensary density"}
    for ax, name in zip(axes, ["percapita", "perstore", "density"]):
        d = es_results[name]
        ax.errorbar(d["year"], d["coef"], yerr=1.96 * d["se"], fmt="o-", capsize=3, color=PALETTE.get("blue", "#2A6F97"))
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(titles[name])
        ax.set_xlabel("Year")
        ax.set_ylabel("Border coefficient (log points)")
    fig.suptitle("Border (adjacency) x Year event study, county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGOUT, "fig_eventstudy_adjacency.png"), dpi=200, bbox_inches="tight")
    print("\nWrote fig_eventstudy_adjacency.png")


if __name__ == "__main__":
    main()
