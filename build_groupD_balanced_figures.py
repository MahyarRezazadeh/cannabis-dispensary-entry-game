import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def plot_two_lines(df, title, outfile, free_label="Free (NM + OR 2015-2021)"):
    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for regime, color, label in [("local", PALETTE["gold"], "Local barrier (CO+MT)"),
                                  ("free", PALETTE["blue"], free_label)]:
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


def plot_density_with_caveat(df, outfile):
    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    d_local = df[df.regime == "local"].sort_values("year")
    d_free = df[df.regime == "free"].sort_values("year")

    # shade the years where the local line is a confirmed frozen-count artifact (2015-2024)
    ax.axvspan(2014.5, 2024.5, color=PALETTE["gold"], alpha=0.10, zorder=0)

    for d, color, label in [(d_local, PALETTE["gold"], "Local barrier (CO+MT)"),
                             (d_free, PALETTE["blue"], "Free (NM only)")]:
        years = d["year"].values
        pct = d["pct"].values
        se_pct = (np.exp(d["coef"] + d["se"]) - 1) * 100 - pct
        ax.errorbar(years, pct, yerr=se_pct, fmt="o-", color=color, label=label,
                    capsize=3, markersize=6, linewidth=1.6)
    ax.axhline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Border premium on dispensary density (%)")
    ax.set_title("Border premium on dispensary density, by year -- Grouping D, balanced panel\n"
                 "SHADED REGION: Colorado's dispensary_count is confirmed FROZEN per county across\n"
                 "these years (not a real time series); the gold line's flatness through 2024 is a\n"
                 "data artifact, not evidence of stability. Free line is NM only (genuinely dynamic).")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.5, -0.30, "Caution: local-barrier line is not a genuine year-by-year measure before 2025 "
                        "(see caption and Table notes).",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="#8B4513")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, outfile))
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    pc = pd.read_csv(os.path.join(TBL, "groupD_balanced_eventstudy_percapita.csv"))
    ps = pd.read_csv(os.path.join(TBL, "groupD_balanced_eventstudy_perstore.csv"))
    dens = pd.read_csv(os.path.join(TBL, "groupD_balanced_eventstudy_density.csv"))

    plot_two_lines(pc, "Border premium on revenue per capita, by year -- BALANCED PANEL\n"
                       "Local barrier (CO+MT) vs. free entry (NM + Oregon 2015-2021), Washington dropped\n"
                       "Most robust specification (see grouping comparison); county-clustered SE",
                   "fig32_groupD_balanced_percapita.png")
    plot_two_lines(ps, "Border premium on revenue per store, by year -- BALANCED PANEL\n"
                       "Local barrier (CO+MT) vs. free entry (NM only), Washington dropped\n"
                       "Most robust specification; county-clustered SE",
                   "fig33_groupD_balanced_perstore.png", free_label="Free (NM only)")
    plot_density_with_caveat(dens, "fig34_groupD_balanced_density.png")


if __name__ == "__main__":
    main()
