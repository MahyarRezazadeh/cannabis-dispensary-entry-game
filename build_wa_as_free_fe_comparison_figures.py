import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def plot_variant(df, variant, title, outfile):
    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for regime, color, label in [("capped", PALETTE["gold"], "Capped (CO+MT+OR 2022-2025)"),
                                  ("free", PALETTE["blue"], "Free (WA+NM+OR 2015-2021)")]:
        d = df[(df.regime == regime) & (df.variant == variant)].sort_values("year")
        if d.empty:
            continue
        years = d["year"].values
        pct = d["pct"].values
        se_pct = (np.exp(d["coef"] + d["se"]) - 1) * 100 - pct
        ax.errorbar(years, pct, yerr=se_pct, fmt="o-", color=color, label=label,
                    capsize=3, markersize=6, linewidth=1.6)
    ax.axhline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Border premium (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, outfile))
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    all_out = pd.read_csv(os.path.join(TBL, "wa_as_free_fe_comparison.csv"))

    pc = all_out[all_out.outcome == "percapita"]
    ps = all_out[all_out.outcome == "perstore"]

    plot_variant(pc, "balanced_panel",
                 "Border premium on revenue per capita, by year -- BALANCED PANEL\n"
                 "Washington reclassified as free entry; counties restricted to those\n"
                 "reporting every year within their own state's window",
                 "fig25_waasfree_percapita_balancedpanel.png")
    plot_variant(ps, "balanced_panel",
                 "Border premium on revenue per store, by year -- BALANCED PANEL\n"
                 "Washington reclassified as free entry; counties restricted to those\n"
                 "reporting every year within their own state's window",
                 "fig26_waasfree_perstore_balancedpanel.png")
    plot_variant(pc, "stateyear_fe",
                 "Border premium on revenue per capita, by year -- STATE x YEAR FE\n"
                 "Washington reclassified as free entry; fully interacted state-year fixed effects\n"
                 "(no meaningful precision gain over additive FE -- shown for completeness)",
                 "fig27_waasfree_percapita_stateyearFE.png")
    plot_variant(ps, "stateyear_fe",
                 "Border premium on revenue per store, by year -- STATE x YEAR FE\n"
                 "Washington reclassified as free entry; fully interacted state-year fixed effects\n"
                 "(no meaningful precision gain over additive FE -- shown for completeness)",
                 "fig28_waasfree_perstore_stateyearFE.png")


if __name__ == "__main__":
    main()
