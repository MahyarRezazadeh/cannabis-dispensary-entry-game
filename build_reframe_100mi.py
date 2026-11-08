"""
Re-estimate the main pooled table, the border x year event study, and the
premium-by-regime bar figure (fig1) using the 100-mile-to-nearest-nonlegal-
city border definition (border_100mi, from build_border_100mi_city_panel.py),
in place of border_adjacent. Same specifications as
build_reframe_tables_adjacency.py / build_reframe_figures_adjacency.py --
only the border variable and its source table change. Per explicit user
instruction, this is an APPENDIX-ONLY robustness pass: it does not replace
or edit anything used in the main body of any draft.
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
OUT = "tables_reframe"
FIGOUT = "figures_reframe"


def load_panels_with_100mi():
    b100 = pd.read_csv(os.path.join(TBL, "border_100mi_by_year.csv"), dtype={"fips5": str})[
        ["fips5", "year", "border_100mi"]]

    percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    perstore["dispensary_per_1000"] = 1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"]
    perstore["log_dispensary_per_1000"] = np.log(perstore["dispensary_per_1000"])

    for name, df in [("percapita", percapita), ("perstore", perstore)]:
        before = len(df)
        df2 = df.merge(b100, on=["fips5", "year"], how="left")
        n_missing = df2["border_100mi"].isna().sum()
        print(f"{name}: {before} rows, {n_missing} unmatched to border_100mi table (should be 0)")
        if name == "percapita":
            percapita = df2
        else:
            perstore = df2

    percapita = percapita[~((percapita.state_abbr == "NM") & (percapita.year == 2026))].copy()
    perstore = perstore[~((perstore.state_abbr == "NM") & (perstore.year == 2026))].copy()
    return percapita, perstore


def fit(df, yvar, formula):
    d = df.dropna(subset=[yvar]).copy()
    d["year"] = d["year"].astype(str)
    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    return m_hc1, m_cl, int(m_hc1.nobs), d["fips5"].nunique()


def fmt(b, p_hc1, p_cl):
    pct = (np.exp(b) - 1) * 100
    return f"${b:+.3f}$ (${pct:+.1f}\\%$)", f"{p_hc1:.3f}", f"{p_cl:.3f}"


def build_main_table(percapita, perstore):
    specs = [
        ("Revenue per store", perstore, "log_revenue_per_store",
         "log_revenue_per_store ~ border_100mi * entry_capped + log_population + C(year)"),
        ("Revenue per capita", percapita, "log_revenue_per_capita",
         "log_revenue_per_capita ~ border_100mi * entry_capped + log_population + C(year) + C(state_abbr)"),
        ("Dispensary density", perstore, "log_dispensary_per_1000",
         "log_dispensary_per_1000 ~ border_100mi * entry_capped + log_population + C(year)"),
    ]
    rows = []
    for label, df, yvar, formula in specs:
        m_hc1, m_cl, n, ncl = fit(df, yvar, formula)
        row = {"Outcome": label, "n": n, "Counties": ncl}
        for term, short in [("border_100mi", "Border"), ("entry_capped", "Capped"),
                             ("border_100mi:entry_capped", "Border x Capped")]:
            b = m_hc1.params.get(term, np.nan)
            p1 = m_hc1.pvalues.get(term, np.nan)
            p2 = m_cl.pvalues.get(term, np.nan)
            row[f"{short}_b"], row[f"{short}_pHC1"], row[f"{short}_pCl"] = fmt(b, p1, p2)
        rows.append(row)
        print(f"\n{label}: n={n}, counties={ncl}")
        print(m_hc1.summary().tables[1])

    with open(os.path.join(OUT, "table_main_pooled_100mi.tex"), "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Appendix: Main pooled specification under the 100-mile nearest-nonlegal-city border definition}
\label{tab:main_pooled_100mi}
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
\textit{Notes:} Border is redefined as an indicator for lying within 100 miles of the nearest incorporated
place (2023 population $\geq$5,000) in a state with no operating recreational retail market as of that
observation's own year, replacing the county-to-state-line geometric adjacency definition used in the main
specification (Table~\ref{tab:result1}). Time-varying legal status uses the same state-year cutoffs as the
main design (Section~\ref{sec:timevarying}). Otherwise identical specification to the main table.
\end{minipage}
\end{table}
""")
    print("\nWrote table_main_pooled_100mi.tex")


def build_event_study(percapita, perstore):
    event_specs = [
        ("percapita", percapita, "log_revenue_per_capita"),
        ("perstore", perstore, "log_revenue_per_store"),
        ("density", perstore, "log_dispensary_per_1000"),
    ]
    es_results = {}
    for name, df, yvar in event_specs:
        d = df.dropna(subset=[yvar]).copy()
        d["year_s"] = d["year"].astype(str)
        formula = (f"{yvar} ~ border_100mi + border_100mi:C(year_s) + log_population + C(year_s)"
                   + (" + C(state_abbr)" if name == "percapita" else ""))
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
        terms = [t for t in m.params.index if t.startswith("border_100mi:C(year_s)")]
        yrs = [int(t.split("[T.")[1].rstrip("]")) for t in terms]
        coefs = [m.params[t] for t in terms]
        ses = [m.bse[t] for t in terms]
        es_results[name] = pd.DataFrame({"year": yrs, "coef": coefs, "se": ses}).sort_values("year")
        es_results[name].to_csv(os.path.join(TBL, f"eventstudy_100mi_{name}.csv"), index=False)
        print(f"\nEvent study ({name}), border_100mi x year coefficients:")
        print(es_results[name].to_string(index=False))

    apply_healy_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = {"percapita": "Revenue per capita", "perstore": "Revenue per store", "density": "Dispensary density"}
    for ax, name in zip(axes, ["percapita", "perstore", "density"]):
        d = es_results[name]
        ax.errorbar(d["year"], d["coef"], yerr=1.96 * d["se"], fmt="o-", capsize=3, color=PALETTE["blue"])
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(titles[name])
        ax.set_xlabel("Year")
        ax.set_ylabel("Border coefficient (log points)")
    fig.suptitle("Border (100mi nearest nonlegal city) x Year event study, county-clustered 95% CI", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGOUT, "fig_eventstudy_100mi.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nWrote fig_eventstudy_100mi.png")


def build_fig1_premium_by_regime(percapita, perstore):
    pc2 = percapita[percapita["revenue_per_capita"].notna()].copy()
    ps2 = perstore[perstore["revenue_per_store"].notna()].copy()
    specs = [("Revenue per store", ps2, "log_revenue_per_store"),
             ("Revenue per capita", pc2, "log_revenue_per_capita"),
             ("Dispensary density", ps2, "log_dispensary_per_1000")]
    rows = []
    for label, df, yvar in specs:
        for cap, regime in [(0, "Free"), (1, "Capped")]:
            d = df[df.entry_capped == cap].dropna(subset=[yvar, "border_100mi", "log_population"]).copy()
            d["year"] = d.year.astype(str)
            formula = f"{yvar} ~ border_100mi + log_population + C(year)" + (
                " + C(state_abbr)" if d.state_abbr.nunique() > 1 else "")
            m = smf.ols(formula, data=d).fit(cov_type="HC1")
            b, se = m.params.get("border_100mi"), m.bse.get("border_100mi")
            pct = (np.exp(b) - 1) * 100
            lo = (np.exp(b - 1.96 * se) - 1) * 100
            hi = (np.exp(b + 1.96 * se) - 1) * 100
            rows.append({"outcome": label, "regime": regime, "pct": pct, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    print("\nFig1 (100mi) premium-by-regime values:")
    print(df.to_string(index=False))

    apply_healy_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    outcomes = [s[0] for s in specs]
    x = np.arange(len(outcomes))
    width = 0.35
    for i, (regime, color) in enumerate([("Free", PALETTE["blue"]), ("Capped", PALETTE["red"])]):
        sub = df[df.regime == regime].set_index("outcome").loc[outcomes]
        err = np.array([sub["pct"] - sub["lo"], sub["hi"] - sub["pct"]])
        ax.bar(x + (i - 0.5) * width, sub["pct"], width, yerr=err, capsize=4, label=regime, color=color, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes)
    ax.set_ylabel("Border premium (%)")
    ax.set_title("Border premium by outcome and entry regime (100mi nearest-city definition)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGOUT, "fig1_premium_by_regime_100mi.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig1_premium_by_regime_100mi.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGOUT, exist_ok=True)
    percapita, perstore = load_panels_with_100mi()

    print("\n--- 100mi vs adjacency agreement check (all years) ---")
    adj = pd.read_csv(os.path.join(TBL, "border_adjacent_by_year.csv"), dtype={"fips5": str})[
        ["fips5", "year", "border_adjacent"]]
    for name, df in [("percapita", percapita), ("perstore", perstore)]:
        chk = df.merge(adj, on=["fips5", "year"], how="left")
        tab = pd.crosstab(chk["border_adjacent"], chk["border_100mi"])
        print(f"\n{name} (rows=border_adjacent, cols=border_100mi):\n{tab}")

    build_main_table(percapita, perstore)
    build_event_study(percapita, perstore)
    build_fig1_premium_by_regime(percapita, perstore)


if __name__ == "__main__":
    main()
