import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"

N_CLUSTERS = {
    "percapita": {"capped": 20, "free": 20},
    "perstore": {"capped": 14, "free": 15},
    "density": {"capped": 14, "free": 15},
}


def plot_one(df, outcome, title, outfile):
    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for regime, color, label in [("capped", PALETTE["gold"], "Capped (MT+CO+OR-capped+WA pre-2016)"),
                                  ("free", PALETTE["blue"], "Free (WA 2016+, OR-free, NM)")]:
        d = df[df.regime == regime].sort_values("year")
        years = d["year"].values
        pct = d["pct"].values
        se_pct = (np.exp(d["coef"] + d["se"]) - 1) * 100 - pct
        n_cl = N_CLUSTERS[outcome][regime]
        ax.errorbar(years, pct, yerr=se_pct, fmt="o-", color=color,
                    label=f"{label} ({n_cl} clusters)",
                    capsize=3, markersize=6, linewidth=1.6)
    ax.axhline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Border premium (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.text(0.5, -0.17, "Caution: state×year clusters (14-20) are below the ~20-30 generally\n"
                        "recommended for reliable cluster-robust inference.",
            transform=ax.transAxes, ha="center", fontsize=7.5, style="italic", color="#8B4513")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, outfile))
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    pc = pd.read_csv(os.path.join(TBL, "wa2016split_stateyearcluster_eventstudy_percapita.csv"))
    ps = pd.read_csv(os.path.join(TBL, "wa2016split_stateyearcluster_eventstudy_perstore.csv"))
    dens = pd.read_csv(os.path.join(TBL, "wa2016split_stateyearcluster_eventstudy_density.csv"))

    plot_one(pc, "percapita",
             "Border premium on revenue per capita, by year\n"
             "Capped: MT+CO+OR-capped+WA pre-2016  |  Free: WA 2016+, OR-free, NM\n"
             "Captive-population control, year (+state) FE, state×year-clustered SE",
             "fig38_wa2016split_percapita.png")
    plot_one(ps, "perstore",
             "Border premium on revenue per store, by year\n"
             "Capped: MT+CO+OR-capped+WA pre-2016  |  Free: WA 2016+, OR-free, NM\n"
             "Captive-population control, year (+state) FE, state×year-clustered SE",
             "fig39_wa2016split_perstore.png")
    plot_one(dens, "density",
             "Border premium on dispensary density, by year\n"
             "Capped: MT+CO+OR-capped+WA pre-2016  |  Free: WA 2016+, OR-free, NM\n"
             "Captive-population control, year (+state) FE, state×year-clustered SE",
             "fig40_wa2016split_density.png")


if __name__ == "__main__":
    main()
