"""
Publication-quality summary-statistics table for the Data section, computed
directly on the two estimation samples that feed table_main_pooled.tex:
  - master_percapita_panel.csv (n=1,283 county-years; revenue-per-capita spec)
  - master_perstore_panel.csv  (n=964 county-years; revenue-per-store and
    dispensary-density specs)
Both exclude New Mexico's incomplete 2026 partial year, matching the main
results table exactly (verified by re-running the same regressions on this
filtered sample and confirming an exact match to table_main_pooled.tex).
"""
import pandas as pd
import numpy as np
import os

TBL = "tables"
OUT = "tables_reframe"


def row(label, s, mult=1.0, dec=1):
    s = pd.to_numeric(s, errors="coerce").dropna() * mult
    return (f"{label} & {s.mean():,.{dec}f} & {s.std():,.{dec}f} & "
            f"{s.min():,.{dec}f} & {s.max():,.{dec}f} & {len(s):,} \\\\\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    pc = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    ps = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    ps["dispensary_per_1000"] = 1000 * pd.to_numeric(ps["dispensary_count"], errors="coerce") / ps["population"]

    pc2 = pc[~((pc.state_abbr == "NM") & (pc.year == 2026)) & pc["revenue_per_capita"].notna()].copy()
    ps2 = ps[~((ps.state_abbr == "NM") & (ps.year == 2026)) & ps["revenue_per_store"].notna()].copy()

    lines = []
    lines.append(row("Revenue per capita (\\$)", pc2.revenue_per_capita, dec=2))
    lines.append(row("Revenue per dispensary (\\$000s)", ps2.revenue_per_store, mult=1/1000, dec=1))
    lines.append(row("Dispensaries per 1,000 residents", ps2.dispensary_per_1000, dec=3))
    lines.append(row("County population", ps2.population, dec=0))
    lines.append("\\addlinespace\n")

    def indicator_row(label, s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return f"{label} & {s.mean():.3f} & {s.std():.3f} & {s.min():.0f} & {s.max():.0f} & {len(s):,} \\\\\n"

    lines.append(indicator_row("Border county (70-mile threshold)", ps2.border_70mi))
    lines.append(indicator_row("Capped entry-regime state-year", ps2.entry_capped))

    with open(os.path.join(OUT, "table_summary_stats.tex"), "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Summary statistics, estimation sample}
\label{tab:summary_stats}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Variable & Mean & SD & Min & Max & $n$ \\
\midrule
\multicolumn{6}{l}{\textit{Panel A: Full sample}} \\
""")
        for l in lines:
            f.write(l)

        f.write(r"""\addlinespace
\multicolumn{6}{l}{\textit{Panel B: By entry regime (revenue-per-store sample)}} \\
""")
        for k, lab in [(0, "Free-entry state-years"), (1, "Capped-entry state-years")]:
            d = ps2[ps2.entry_capped == k]
            f.write(row(f"\\quad Revenue per dispensary (\\$000s), {lab}", d.revenue_per_store, mult=1/1000, dec=1))
        for k, lab in [(0, "Free-entry state-years"), (1, "Capped-entry state-years")]:
            d = ps2[ps2.entry_capped == k]
            f.write(row(f"\\quad Dispensaries per 1,000, {lab}", d.dispensary_per_1000, dec=3))

        f.write(r"""\bottomrule
\end{tabular}
}
\begin{minipage}{0.95\textwidth}
\vspace{0.4em}
\footnotesize
\textit{Notes:} Panel A reports the pooled estimation sample: $n=1{,}283$ county-years for revenue per
capita (Colorado, Washington, Oregon, Montana, New Mexico, 2015--2025) and $n=964$ county-years for revenue
per dispensary and dispensary density (same states; Oregon and Montana contribute only their single 2025
cross-section to these two outcomes, Section~\ref{sec:data}). New Mexico's incomplete 2026 partial year is
excluded throughout, matching Table~\ref{tab:main_pooled}. Revenue per capita and revenue per dispensary are
annual dollars; dispensaries per 1,000 residents uses each county's contemporaneous population. Border county
is an indicator for being within 70 miles of a nonlegal-state county as of that observation's own year
(Section~\ref{sec:timevarying}); capped entry-regime state-year is the state-year indicator defined in
Section~\ref{sec:institutional}. Panel B splits the revenue-per-store sample's two dollar/count outcomes by
entry regime; the free-entry group ($n=124$) is New Mexico throughout and Oregon 2015--2021 only (Oregon's
per-store data begins in 2025, its capped period, so Oregon contributes no free-entry per-store observations
in this particular split -- see Section~\ref{sec:data}).
\end{minipage}
\end{table}
""")
    print("Wrote table_summary_stats.tex")
    print(f"pc2 n={len(pc2)}, ps2 n={len(ps2)}")
    print("free-entry perstore n=", len(ps2[ps2.entry_capped == 0]), "states:", sorted(ps2[ps2.entry_capped==0].state_abbr.unique()), "years:", sorted(ps2[ps2.entry_capped==0].year.unique()))


if __name__ == "__main__":
    main()
