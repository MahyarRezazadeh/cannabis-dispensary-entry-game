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

    for regime, color, label in [("local", PALETTE["gold"], "Local barrier (CO+MT)"),
                                  ("nonlocal", PALETTE["blue"], "Non-local barrier (WA+NM)")]:
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
    pc = pd.read_csv(os.path.join(TBL, "localbarrier_eventstudy_v2_percapita.csv"))
    ps = pd.read_csv(os.path.join(TBL, "localbarrier_eventstudy_v2_perstore.csv"))
    plot_one(pc, "Border premium on revenue per capita, by year\n"
                 "Local vs. non-local barrier -- pooled model, captive-pop + state/year FE,\ncounty-clustered SE (Oregon excluded)",
             "fig11_localbarrier_v2_percapita.png")
    plot_one(ps, "Border premium on revenue per store, by year\n"
                 "Local vs. non-local barrier -- pooled model, captive-pop + state/year FE,\ncounty-clustered SE (Oregon excluded)",
             "fig12_localbarrier_v2_perstore.png")


if __name__ == "__main__":
    main()
