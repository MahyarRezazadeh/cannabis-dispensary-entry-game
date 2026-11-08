import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def plot_one(df, title, outfile):
    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for regime, color, label in [("capped", PALETTE["gold"], "Capped (CO+MT+OR 2022-2025)"),
                                  ("free", PALETTE["blue"], "Free (WA+NM+OR 2015-2021)")]:
        d = df[df.regime == regime].sort_values("year")
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
    pc = pd.read_csv(os.path.join(TBL, "wa_as_free_eventstudy_percapita.csv"))
    ps = pd.read_csv(os.path.join(TBL, "wa_as_free_eventstudy_perstore.csv"))
    plot_one(pc, "Border premium on revenue per capita, by year\n"
                 "Washington reclassified as free entry (not capped)\n"
                 "captive-pop + state/year FE, county-clustered SE",
             "fig23_waasfree_percapita.png")
    plot_one(ps, "Border premium on revenue per store, by year\n"
                 "Washington reclassified as free entry (not capped)\n"
                 "captive-pop + state/year FE, county-clustered SE",
             "fig24_waasfree_perstore.png")


if __name__ == "__main__":
    main()
