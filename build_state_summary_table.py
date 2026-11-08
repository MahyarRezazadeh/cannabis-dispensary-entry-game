"""
Build the summary table of state legalization dates, neighbor status, data
coverage, and entry-regulation classification, and the Oregon before/after
table -- both required deliverables from the reframe spec.
"""
import os

OUT = "tables_reframe"

STATES = [
    dict(state="Colorado", retail="2014-01-01", neighbors="KS, NE, OK, UT, WY (always); NM (nonlegal until 2022-04-01)",
         data="County-month sales, CO Dept. of Revenue (Socrata j7a3-jgd3), 2014--2025",
         regime="Capped (local-government opt-out bans; 48.4\\% of counties ban all facilities, Pay\\'an et al. 2021 -- see \\S\\ref{sec:institutional})"),
    dict(state="Washington", retail="2014-07-08", neighbors="ID (always); OR (nonlegal until 2015-10-01)",
         data="County-fiscal-year sales, WA Liquor and Cannabis Board, FY2015--FY2025",
         regime="Capped (I-502 statewide license cap; 334$\\to$556$\\to$556+52 equity slots)"),
    dict(state="Oregon", retail="2015-10-01", neighbors="ID (always); CA (nonlegal until 2018-01-01), NV (nonlegal until 2017-07-01)",
         data="County-month sales, OLCC Socrata series, 2017--2025 (per-capita panel); 2025 cross-section (per-store, license registry)",
         regime="Free 2015--2021, then application freeze (Jan 2022) and permanent per-capita cap (Mar 2024)"),
    dict(state="Montana", retail="2022-01-01", neighbors="ID, ND, SD, WY (always -- the only sample state bordering four nonlegal states)",
         data="County-month sales scraped from static DOR report pages, 2022--2025 (per-capita panel); 2025 cross-section (per-store, license list)",
         regime="Capped (HB 701 local opt-in requirement for counties that voted against I-190 in Nov. 2020, binding from the 2022-01-01 launch; May-2025 moratorium separately restricts license issuance)"),
    dict(state="New Mexico", retail="2022-04-01", neighbors="OK, TX (always)",
         data="County-month sales AND licensee-level sales, NMRLD CROP portal, 2022--2026 (full panel, both outcomes)",
         regime="Free (NMSA 1978 \\S 26-2C-6(I) explicit statutory anti-cap provision; local bans preempted)"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "table_state_summary.tex"), "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Sample states: legalization timing, border exposure, data, and entry regulation}
\label{tab:state_summary}
\footnotesize
\begin{tabular}{p{1.5cm}p{1.1cm}p{3.6cm}p{3.9cm}p{3.9cm}}
\toprule
State & Retail date & Nonlegal-neighbor exposure & Data source \& coverage & Entry-regulation regime \\
\midrule
""")
        for s in STATES:
            f.write(f"{s['state']} & {s['retail']} & {s['neighbors']} & {s['data']} & {s['regime']} \\\\[0.4em]\n")
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.98\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} ``Nonlegal-neighbor exposure'' lists every state sharing a land border, with the date any
neighbor's own retail-legalization date falls inside the sample state's panel window (i.e. requires a
time-varying, not static, border classification -- see \S\ref{sec:timevarying}). Entry-regulation
classifications are from statute-level research (Colorado Retail Marijuana Code / Amendment 64; Washington
RCW 69.50 / I-502; Oregon ORS 475C / HB 4016 / HB 4121; Montana HB 701 (MRTMA); New Mexico Cannabis
Regulation Act, NMSA 1978 Ch.\ 26 Art.\ 2C), not from labels.
\end{minipage}
\end{table}
""")
    print("Wrote table_state_summary.tex")

    with open(os.path.join(OUT, "table_oregon_beforeafter.tex"), "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Oregon within-state before/after: the 2022-01-01 entry freeze}
\label{tab:oregon_beforeafter}
\small
\begin{tabular}{lccc}
\toprule
Period & Border coefficient (\%) & $p$ (HC1) & $n$ \\
\midrule
Free (2017--2021) & $+1.004$ ($+172.9\%$) & 0.007 & 135 \\
Capped (2022--2025) & $+0.723$ ($+106.1\%$) & 0.101 & 120 \\
\midrule
\multicolumn{4}{l}{Formal interaction (border $\times$ post-freeze): $-0.381$ ($-31.7\%$), $p_{HC1}=0.503$,} \\
\multicolumn{4}{l}{$p_{cl}=0.611$, $n=255$} \\
\bottomrule
\end{tabular}
\begin{minipage}{0.85\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Revenue per capita, log-linear, county and log-population controls, year fixed effects.
Oregon's own before/after comparison points the same direction as the pooled cross-state interaction
(Table~\ref{tab:main_pooled}) -- the border effect shrinks and loses significance after the 2022 freeze --
but the formal within-Oregon interaction is not itself significant, consistent with the design being
underpowered on its own (only 3 Oregon border counties). Revenue per store cannot be compared across the
freeze within Oregon: Oregon's per-store data exists only as a single 2025 (post-freeze) cross-section, so
there is no pre-2022 per-store observation -- reported as a limitation, not estimated.
\end{minipage}
\end{table}
""")
    print("Wrote table_oregon_beforeafter.tex")


if __name__ == "__main__":
    main()
