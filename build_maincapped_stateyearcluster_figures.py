import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"

N_CLUSTERS = {
    "percapita": {"capped": 30, "free": 10},
    "perstore": {"capped": 24, "free": 5},
    "density": {"capped": 24, "free": 5},
}


def plot_one(df, outcome, title, outfile, ylabel="Border premium (%)"):
    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for regime, color, label in [("capped", PALETTE["gold"], "Capped (WA+CO+MT+OR-capped)"),
                                  ("free", PALETTE["blue"], "Free (NM+OR-free)")]:
        d = df[df.regime == regime].sort_values("year")
        years = d["year"].values
        pct = d["pct"].values
        se_pct = (np.exp(d["coef"] + d["se"]) - 1) * 100 - pct
        n_cl = N_CLUSTERS[outcome][regime]
        ax.errorbar(years, pct, yerr=se_pct, fmt="o-", color=color,
                    label=f"{label} ({n_cl} state×year clusters)",
                    capsize=3, markersize=6, linewidth=1.6)
    ax.axhline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7.5)
    if min(N_CLUSTERS[outcome].values()) < 20:
        ax.text(0.5, -0.16, "Caution: the Free group has as few as 5 state×year clusters -- "
                            "cluster-robust SEs are unreliable below ~20 clusters.",
                transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="#8B4513")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, outfile))
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    pc = pd.read_csv(os.path.join(TBL, "maincapped_stateyearcluster_eventstudy_percapita.csv"))
    ps = pd.read_csv(os.path.join(TBL, "maincapped_stateyearcluster_eventstudy_perstore.csv"))
    dens = pd.read_csv(os.path.join(TBL, "maincapped_stateyearcluster_eventstudy_density.csv"))

    plot_one(pc, "percapita",
             "Border premium on revenue per capita, by year\n"
             "Capped (WA+CO+MT+OR-capped) vs. Free (NM+OR-free)\n"
             "Population + captive-population controls, year (+state) FE, state×year-clustered SE",
             "fig35_maincapped_stateyearcl_percapita.png")
    plot_one(ps, "perstore",
             "Border premium on revenue per store, by year\n"
             "Capped (WA+CO+MT+OR-capped) vs. Free (NM+OR-free)\n"
             "Population + captive-population controls, year (+state) FE, state×year-clustered SE",
             "fig36_maincapped_stateyearcl_perstore.png")
    plot_one(dens, "density",
             "Border premium on dispensary density, by year\n"
             "Capped (WA+CO+MT+OR-capped) vs. Free (NM+OR-free)\n"
             "Population + captive-population controls, year (+state) FE, state×year-clustered SE\n"
             "NOTE: CO/WA dispensary counts are confirmed frozen per county across years (see Data section)",
             "fig37_maincapped_stateyearcl_density.png")


if __name__ == "__main__":
    main()
