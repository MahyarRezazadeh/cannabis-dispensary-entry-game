"""
Event-study comparison requested directly: WITHIN border counties only
(border_70mi==1), do local-cap states (CO+MT) differ from non-local-cap
states (WA+NM) on revenue per dispensary, revenue per capita, and number
of dispensaries? X-axis is event time relative to each state's OWN retail
legalization date (years since launch), not raw calendar year, so CO
(launched 2014) and MT (launched 2022) -- and WA (2014) and NM (2022) --
line up on a comparable clock despite legalizing eight years apart.
Oregon excluded (as in the other local-barrier robustness checks), since
its per-store data is a single 2025 cross-section and its early per-capita
years have the documented Baker County data-quality problem.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"
os.makedirs(OUT, exist_ok=True)

LOCAL_STATES = ["CO", "MT"]
NONLOCAL_STATES = ["WA", "NM"]

legal = pd.read_csv("data/legal_dates_new.csv", parse_dates=["retail_date"])
legal["launch_year"] = legal["retail_date"].dt.year
launch_map = legal.set_index("state_abbr")["launch_year"].to_dict()


def geo_mean_by_eventyear(df, yvar, use_log=True):
    d = df[df[yvar] > 0].copy() if use_log else df.copy()
    d["event_year"] = d["year"] - d["state_abbr"].map(launch_map)
    if use_log:
        d["logy"] = np.log(d[yvar])
        g = d.groupby(["local_barrier", "event_year"])["logy"].agg(["mean", "count"]).reset_index()
        g["value"] = np.exp(g["mean"])
    else:
        g = d.groupby(["local_barrier", "event_year"])[yvar].agg(["mean", "count"]).reset_index()
        g["value"] = g["mean"]
    return g


def plot_two_lines(g, title, ylabel, outfile, min_n=2):
    apply_healy_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for grp, label, color in [(1, "Local-cap (CO+MT)", PALETTE["gold"]), (0, "Non-local-cap (WA+NM)", PALETTE["blue"])]:
        d = g[(g.local_barrier == grp) & (g["count"] >= min_n)].sort_values("event_year")
        ax.plot(d["event_year"], d["value"], marker="o", color=color, label=label, linewidth=1.8, markersize=5)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1, label="Own state's legalization year")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Years since own state's retail legalization")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, outfile))
    plt.close(fig)
    print(f"Wrote {outfile}")


percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})

border_pc = percapita[(percapita.border_70mi == 1) & (percapita.state_abbr.isin(LOCAL_STATES + NONLOCAL_STATES))]
border_ps = perstore[(perstore.border_70mi == 1) & (perstore.state_abbr.isin(LOCAL_STATES + NONLOCAL_STATES))]

print("=== Revenue per capita, border counties only ===")
g1 = geo_mean_by_eventyear(border_pc, "revenue_per_capita")
print(g1.pivot(index="event_year", columns="local_barrier", values=["value", "count"]).to_string())
plot_two_lines(g1, "Revenue per capita, border counties only\nLocal-cap vs. non-local-cap states (raw geometric mean)",
               "Revenue per capita ($)", "fig18_borderonly_percapita_bylocalcap.png")

print("\n=== Revenue per dispensary (per store), border counties only ===")
g2 = geo_mean_by_eventyear(border_ps, "revenue_per_store")
print(g2.pivot(index="event_year", columns="local_barrier", values=["value", "count"]).to_string())
plot_two_lines(g2, "Revenue per dispensary, border counties only\nLocal-cap vs. non-local-cap states (raw geometric mean)",
               "Revenue per dispensary ($)", "fig19_borderonly_perstore_bylocalcap.png")

print("\n=== Number of dispensaries, border counties only ===")
g3 = geo_mean_by_eventyear(border_ps, "dispensary_count", use_log=False)
print(g3.pivot(index="event_year", columns="local_barrier", values=["value", "count"]).to_string())
plot_two_lines(g3, "Number of dispensaries per border county\nLocal-cap vs. non-local-cap states (mean count)",
               "Dispensaries per county (mean)", "fig20_borderonly_count_bylocalcap.png")
