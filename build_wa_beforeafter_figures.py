import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def plot_eventstudy(df, title, outfile):
    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    years = df["year"].values
    pct = df["pct"].values
    se_pct = (np.exp(df["coef"] + df["se"]) - 1) * 100 - pct
    ax.errorbar(years, pct, yerr=se_pct, fmt="o-", color=PALETTE["red"], capsize=3, markersize=6, linewidth=1.6)
    ax.axvline(2016, color="#999999", linestyle="--", linewidth=1, label="2016 cap increase / window closure")
    ax.axhline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Border premium (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, outfile))
    plt.close(fig)
    print(f"Wrote {outfile}")


def plot_robustness_bars():
    apply_healy_style()
    labels = ["Revenue\nper capita", "Revenue\nper store", "Dispensary\ndensity"]
    wa_pre = [14.6, 69.4, -32.3]
    wa_post = [63.1, 112.4, -23.2]
    nm = [251.0, 158.7, 35.7]
    diff_p = [0.077, 0.833, 0.0002]

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar(x - width, wa_pre, width, label="WA pre-2016", color=PALETTE["gray"])
    ax.bar(x, wa_post, width, label="WA 2016+", color=PALETTE["gold"])
    ax.bar(x + width, nm, width, label="NM (always free)", color=PALETTE["blue"])
    ax.axhline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Border premium (%)")
    ax.set_title("Robustness check: does WA's post-2016 border premium look like NM's?\n"
                 "WA(2016+) vs. NM difference: per capita p=0.077, per store p=0.833, density p<0.001")
    ax.legend(loc="upper left", fontsize=9)
    for i, p in enumerate(diff_p):
        sig = "*" if p < 0.10 else ""
        ax.text(x[i], max(wa_post[i], nm[i]) + 10, f"WA-NM diff p={p:.3f}{sig}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig49_wa_vs_nm_robustness_bars.png"))
    plt.close(fig)
    print("Wrote fig49_wa_vs_nm_robustness_bars.png")


def main():
    pc = pd.read_csv(os.path.join(TBL, "wa_beforeafter_eventstudy_percapita.csv"))
    ps = pd.read_csv(os.path.join(TBL, "wa_beforeafter_eventstudy_perstore.csv"))
    plot_eventstudy(pc, "Washington's own border premium on revenue per capita, by year\n"
                        "Within-state event study, ref=2015, captive-pop control, county-clustered SE",
                    "fig47_wa_beforeafter_percapita.png")
    plot_eventstudy(ps, "Washington's own border premium on revenue per store, by year\n"
                        "Within-state event study, ref=2015, captive-pop control, county-clustered SE",
                    "fig48_wa_beforeafter_perstore.png")
    plot_robustness_bars()


if __name__ == "__main__":
    main()
