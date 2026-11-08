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

    for regime, color, label in [("capped", PALETTE["red"], "Capped (CO+WA+MT)"),
                                  ("free", PALETTE["blue"], "Free (NM)")]:
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
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, outfile))
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    os.makedirs(OUT, exist_ok=True)
    pc = pd.read_csv(os.path.join(TBL, "regime_eventstudy_percapita.csv"))
    ps = pd.read_csv(os.path.join(TBL, "regime_eventstudy_perstore.csv"))
    plot_one(pc, "Border premium on revenue per capita, by year and regime\n(Oregon excluded -- see notes)",
             "fig7_regime_eventstudy_percapita.png")
    plot_one(ps, "Border premium on revenue per store, by year and regime\n(Oregon excluded -- see notes)",
             "fig8_regime_eventstudy_perstore.png")

    pc_or = pd.read_csv(os.path.join(TBL, "regime_eventstudy_percapita_withOR.csv"))
    ps_or = pd.read_csv(os.path.join(TBL, "regime_eventstudy_perstore_withOR.csv"))
    plot_one(pc_or, "Border premium on revenue per capita, by year and regime\n"
                     "(Oregon included -- moves capped/free groups in 2022; see notes)",
             "fig7b_regime_eventstudy_percapita_withOR.png")
    plot_one(ps_or, "Border premium on revenue per store, by year and regime\n"
                     "(Oregon included -- moves capped/free groups in 2022; see notes)",
             "fig8b_regime_eventstudy_perstore_withOR.png")


if __name__ == "__main__":
    main()
