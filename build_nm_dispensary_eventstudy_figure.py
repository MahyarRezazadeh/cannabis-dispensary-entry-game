import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(TBL, "nm_dispensary_eventstudy_percapita.csv"))
    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    df = df.sort_values("year")
    se_pct = (np.exp(df["coef"] + df["se"]) - 1) * 100 - df["pct"]
    ax.errorbar(df["year"], df["pct"], yerr=se_pct, fmt="o-", color=PALETTE["red"],
                capsize=3, markersize=6, linewidth=1.6, label="NM dispensary-level border premium")
    ax.axhline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Border premium (%)")
    ax.set_title("Border premium on individual-dispensary revenue, by year\n"
                  "New Mexico, firm-level panel (n=1,249 dispensaries), county-clustered SE,\n"
                  "controls: log(population), log(1+captive population), year FE")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig13_nm_dispensary_eventstudy.png"))
    plt.close(fig)
    print("Wrote fig13_nm_dispensary_eventstudy.png")


if __name__ == "__main__":
    main()
