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
                                  ("nonlocal", PALETTE["blue"], "Non-local barrier (WA+NM"
                                                                 + ("+OR)" if "withOR" in outfile else ")"))]:
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
    specs = [
        ("localbarrier_eventstudy_percapita.csv",
         "Border premium on revenue per capita, by year\nLocal vs. non-local barrier (Oregon excluded)",
         "fig9_localbarrier_percapita.png"),
        ("localbarrier_eventstudy_percapita_withOR.csv",
         "Border premium on revenue per capita, by year\nLocal vs. non-local barrier (Oregon included)",
         "fig9b_localbarrier_percapita_withOR.png"),
        ("localbarrier_eventstudy_perstore.csv",
         "Border premium on revenue per store, by year\nLocal vs. non-local barrier (Oregon excluded)",
         "fig10_localbarrier_perstore.png"),
        ("localbarrier_eventstudy_perstore_withOR.csv",
         "Border premium on revenue per store, by year\nLocal vs. non-local barrier (Oregon included)",
         "fig10b_localbarrier_perstore_withOR.png"),
    ]
    for fname, title, outfile in specs:
        df = pd.read_csv(os.path.join(TBL, fname))
        plot_one(df, title, outfile)


if __name__ == "__main__":
    main()
