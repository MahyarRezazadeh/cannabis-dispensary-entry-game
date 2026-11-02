import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def plot_one(df, title, outfile, ylabel="Border premium (%)"):
    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    d = df.sort_values("event_year")
    years = d["event_year"].values
    pct = d["pct"].values
    se_pct = (np.exp(d["coef"] + d["se"]) - 1) * 100 - pct
    ax.errorbar(years, pct, yerr=se_pct, fmt="o-", color=PALETTE["red"], capsize=3, markersize=6, linewidth=1.6)
    ax.axhline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_xlabel("Years since own state's retail legalization")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, outfile))
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    pc = pd.read_csv(os.path.join(TBL, "pooled_sincelaunch_eventstudy_percapita.csv"))
    ps = pd.read_csv(os.path.join(TBL, "pooled_sincelaunch_eventstudy_perstore.csv"))
    dens = pd.read_csv(os.path.join(TBL, "pooled_sincelaunch_eventstudy_density.csv"))

    plot_one(pc, "Border premium on revenue per capita, pooled 5-state sample\n"
                 "Event time = years since own state's retail legalization\n"
                 "Captive-pop control, event-year + state FE, county-clustered SE",
             "fig50_pooled_sincelaunch_percapita.png")
    plot_one(ps, "Border premium on revenue per store, pooled 5-state sample\n"
                 "Event time = years since own state's retail legalization\n"
                 "Captive-pop control, event-year + state FE, county-clustered SE",
             "fig51_pooled_sincelaunch_perstore.png")
    plot_one(dens, "Border premium on dispensary density, pooled 5-state sample\n"
                   "Event time = years since own state's retail legalization\n"
                   "Captive-pop control, event-year + state FE, county-clustered SE",
             "fig52_pooled_sincelaunch_density.png")


if __name__ == "__main__":
    main()
