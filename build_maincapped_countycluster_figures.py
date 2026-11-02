import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"

N_CLUSTERS = {
    "percapita": {"capped": 146, "free": 62},
    "perstore": {"capped": 133, "free": 32},
    "density": {"capped": 133, "free": 32},
}


def plot_one(df, outcome, title, outfile):
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
                    label=f"{label} ({n_cl} county clusters)",
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
    pc = pd.read_csv(os.path.join(TBL, "maincapped_countycluster_eventstudy_percapita.csv"))
    ps = pd.read_csv(os.path.join(TBL, "maincapped_countycluster_eventstudy_perstore.csv"))
    dens = pd.read_csv(os.path.join(TBL, "maincapped_countycluster_eventstudy_density.csv"))

    plot_one(pc, "percapita",
             "Border premium on revenue per capita, by year -- COUNTY-CLUSTERED\n"
             "ORIGINAL Table 2 classification: Capped=WA+CO+MT+OR-capped, Free=NM+OR-free\n"
             "Captive-population control, year (+state) FE",
             "fig44_maincapped_countycl_percapita.png")
    plot_one(ps, "perstore",
             "Border premium on revenue per store, by year -- COUNTY-CLUSTERED\n"
             "ORIGINAL Table 2 classification: Capped=WA+CO+MT+OR-capped, Free=NM+OR-free\n"
             "Captive-population control, year (+state) FE",
             "fig45_maincapped_countycl_perstore.png")
    plot_one(dens, "density",
             "Border premium on dispensary density, by year -- COUNTY-CLUSTERED\n"
             "ORIGINAL Table 2 classification: Capped=WA+CO+MT+OR-capped, Free=NM+OR-free\n"
             "Captive-population control, year (+state) FE",
             "fig46_maincapped_countycl_density.png")


if __name__ == "__main__":
    main()
