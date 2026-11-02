"""
Build the main LaTeX regression tables and the state-summary table for the
reframed manuscript.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

TBL = "tables"
OUT = "tables_reframe"


def fit(df, yvar, formula):
    d = df.dropna(subset=[yvar]).copy()
    d["year"] = d["year"].astype(str)
    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    m_cl = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    return m_hc1, m_cl, int(m_hc1.nobs), d["fips5"].nunique()


def fmt(b, p_hc1, p_cl):
    pct = (np.exp(b) - 1) * 100
    return f"${b:+.3f}$ (${pct:+.1f}\\%$)", f"{p_hc1:.3f}", f"{p_cl:.3f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    perstore["dispensary_per_1000"] = 1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"]
    perstore["log_dispensary_per_1000"] = np.log(perstore["dispensary_per_1000"])

    specs = [
        ("Revenue per store", perstore, "log_revenue_per_store",
         "log_revenue_per_store ~ border_70mi * entry_capped + log_population + C(year)"),
        ("Revenue per capita", percapita, "log_revenue_per_capita",
         "log_revenue_per_capita ~ border_70mi * entry_capped + log_population + C(year) + C(state_abbr)"),
        ("Dispensary density", perstore, "log_dispensary_per_1000",
         "log_dispensary_per_1000 ~ border_70mi * entry_capped + log_population + C(year)"),
    ]

    rows = []
    for label, df, yvar, formula in specs:
        m_hc1, m_cl, n, ncl = fit(df, yvar, formula)
        row = {"Outcome": label, "n": n, "Counties": ncl}
        for term, short in [("border_70mi", "Border"), ("entry_capped", "Capped"),
                             ("border_70mi:entry_capped", "Border x Capped")]:
            b = m_hc1.params.get(term, np.nan)
            p1 = m_hc1.pvalues.get(term, np.nan)
            p2 = m_cl.pvalues.get(term, np.nan)
            row[f"{short}_b"], row[f"{short}_pHC1"], row[f"{short}_pCl"] = fmt(b, p1, p2)
        rows.append(row)

    with open(os.path.join(OUT, "table_main_pooled.tex"), "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Main pooled specification: border $\times$ entry-regime interaction}
\label{tab:main_pooled}
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
\textit{Notes:} Pooled across Colorado, Washington, Oregon, Montana, and New Mexico. Coefficients are OLS,
log-linear outcome; \% is $(e^{\hat\beta}-1)\times 100$. $p$ (cl.) is county-clustered. Revenue-per-store and
dispensary-density specs omit state fixed effects because \texttt{entry\_capped} has no within-state variation
in the per-store sample (Oregon's per-store data is a single 2025 cross-section), which would otherwise make
it collinear with the state dummies. Revenue-per-capita includes state FE since Oregon's per-capita panel
spans both its free (2015--2021) and capped (2022--2025) periods, giving genuine within-state identification.
Entry-regime coding: Colorado, Washington, and Montana are capped for their entire panel (local-government
control over licensing -- county opt-out bans in Colorado, opt-in requirements in Montana -- functions as a
binding restriction even where no state-level numerical cap exists); Oregon is free 2015--2021 and capped
2022--2025; New Mexico is free throughout (statutory anti-cap provision, NMSA 1978 \S 26-2C-6(I)).
\end{minipage}
\end{table}
""")
    print("Wrote table_main_pooled.tex")

    # ---------- state-specific table ----------
    with open(os.path.join(OUT, "table_state_specific.tex"), "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{State-specific border coefficients, by outcome}
\label{tab:state_specific}
\begin{tabular}{lccc}
\toprule
State (entry regime) & Revenue/store & Revenue/capita & Dispensary density \\
\midrule
""")
        state_regime = {"CO": "capped", "WA": "capped", "OR": "mixed", "MT": "capped", "NM": "free"}
        for st in ["CO", "WA", "OR", "MT", "NM"]:
            cells = []
            for df, yvar in [(perstore, "log_revenue_per_store"),
                              (percapita, "log_revenue_per_capita"),
                              (perstore, "log_dispensary_per_1000")]:
                d = df[df.state_abbr == st].dropna(subset=[yvar, "border_70mi", "log_population"]).copy()
                if d["border_70mi"].nunique() < 2:
                    cells.append("--")
                    continue
                d["year"] = d["year"].astype(str)
                yrs = d["year"].nunique()
                formula = f"{yvar} ~ border_70mi + log_population" + (" + C(year)" if yrs > 1 else "")
                m = smf.ols(formula, data=d).fit(cov_type="HC1")
                b = m.params.get("border_70mi", np.nan)
                p = m.pvalues.get("border_70mi", np.nan)
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                cells.append(f"${b:+.3f}${stars}")
            f.write(f"{st} ({state_regime[st]}) & {cells[0]} & {cells[1]} & {cells[2]} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.85\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Each cell is the border-county coefficient from a separate within-state OLS regression
(log outcome on border indicator, log population, and year fixed effects where more than one year is
available), HC1 standard errors. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$. ``--'' indicates no
border/interior variation within that state for that outcome.
\end{minipage}
\end{table}
""")
    print("Wrote table_state_specific.tex")


if __name__ == "__main__":
    main()
