"""
Sample-composition table for the Data section: border vs. interior counties
and capped vs. free-entry county-years, computed directly on the two
estimation samples behind table_main_pooled.tex.
"""
import pandas as pd
import os

TBL = "tables"
OUT = "tables_reframe"


def summarize(df, label):
    n = len(df)
    counties = df.fips5.nunique()
    border_cty = df[df.border_70mi == 1].fips5.nunique()
    interior_cty = df[df.border_70mi == 0].fips5.nunique()
    border_cy = (df.border_70mi == 1).sum()
    interior_cy = (df.border_70mi == 0).sum()
    capped_cy = (df.entry_capped == 1).sum()
    free_cy = (df.entry_capped == 0).sum()
    switchers = (df.groupby("fips5")["border_70mi"].nunique() == 2).sum()
    return dict(label=label, n=n, counties=counties, border_cty=border_cty, interior_cty=interior_cty,
                border_cy=border_cy, interior_cy=interior_cy, capped_cy=capped_cy, free_cy=free_cy,
                switchers=switchers)


def main():
    os.makedirs(OUT, exist_ok=True)
    pc = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    ps = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})

    pc2 = pc[~((pc.state_abbr == "NM") & (pc.year == 2026)) & pc["revenue_per_capita"].notna()].copy()
    ps2 = ps[~((ps.state_abbr == "NM") & (ps.year == 2026)) & ps["revenue_per_store"].notna()].copy()

    a = summarize(pc2, "Revenue per capita")
    b = summarize(ps2, "Revenue per store / dispensary density")

    with open(os.path.join(OUT, "table_sample_composition.tex"), "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Sample composition, by estimation sample}
\label{tab:sample_composition}
\begin{tabular}{lcc}
\toprule
 & Revenue per capita & Revenue per store / density \\
\midrule
""")
        f.write(f"County-year observations & {a['n']:,} & {b['n']:,} \\\\\n")
        f.write(f"Unique counties & {a['counties']} & {b['counties']} \\\\\n")
        f.write(r"\addlinespace" + "\n")
        f.write(f"Border counties (ever, in sample) & {a['border_cty']} & {b['border_cty']} \\\\\n")
        f.write(f"Interior counties (ever, in sample) & {a['interior_cty']} & {b['interior_cty']} \\\\\n")
        f.write(f"Counties that switch classification & {a['switchers']} & {b['switchers']} \\\\\n")
        f.write(r"\addlinespace" + "\n")
        f.write(f"Border county-years & {a['border_cy']:,} & {b['border_cy']:,} \\\\\n")
        f.write(f"Interior county-years & {a['interior_cy']:,} & {b['interior_cy']:,} \\\\\n")
        f.write(r"\addlinespace" + "\n")
        f.write(f"Capped-entry county-years & {a['capped_cy']:,} & {b['capped_cy']:,} \\\\\n")
        f.write(f"Free-entry county-years & {a['free_cy']:,} & {b['free_cy']:,} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.92\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Both columns exclude New Mexico's incomplete 2026 partial year and match
Table~\ref{tab:main_pooled} exactly. ``Border (interior) counties, ever'' counts a county once if it is
ever classified as border (interior) in at least one sample year; because the border classification is
time-varying (Section~\ref{sec:timevarying}), a county can appear in both categories across different
years -- ``counties that switch classification'' reports how many do. Capped/free-entry county-years use
the state-year \texttt{Capped} indicator (Section~\ref{sec:institutional}).
\end{minipage}
\end{table}
""")
    print("Wrote table_sample_composition.tex")
    print(a)
    print(b)


if __name__ == "__main__":
    main()
