"""
Border-vs-interior "raw trends" event-study figures: two lines per chart
(border group's own geometric-mean outcome by year, interior/control
group's own geometric-mean outcome by year), as opposed to the premium/gap
lines already built elsewhere (fig7-fig13). Geometric mean (exp(mean(log(y))))
is used rather than the arithmetic mean, consistent with the log-linear
framework used throughout this project's regressions, and to avoid a few
large counties/dispensaries dominating the arithmetic average.

Four figures:
  A. NM dispensary-level revenue per dispensary, border vs interior dispensaries
  B. NM county-level revenue per capita, border vs interior counties
  C. Main 5-state revenue per capita, border vs interior, split by entry_capped
  D. Main 5-state revenue per store, border vs interior, split by entry_capped
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"
os.makedirs(OUT, exist_ok=True)


def geo_mean_by_year(df, yvar, group_col, years=None):
    d = df[df[yvar] > 0].copy()
    d["logy"] = np.log(d[yvar])
    g = d.groupby([group_col, "year"])["logy"].agg(["mean", "count"]).reset_index()
    g["geo_mean"] = np.exp(g["mean"])
    return g


def plot_two_lines(ax, g, group_col, group_labels, colors, title, ylabel):
    for grp, label, color in group_labels:
        d = g[g[group_col] == grp].sort_values("year")
        ax.plot(d["year"], d["geo_mean"], marker="o", color=color, label=label, linewidth=1.8, markersize=5)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)


# ---------------- A. NM dispensary-level revenue per dispensary ----------------
apply_healy_style()
disp = pd.read_csv(os.path.join(TBL, "nm_dispensary_panel_annual.csv"))
gA = geo_mean_by_year(disp, "total_sales", "border_70mi")
fig, ax = plt.subplots(figsize=(7.5, 4.8))
plot_two_lines(ax, gA, "border_70mi",
               [(1, "Border dispensaries", PALETTE["red"]), (0, "Interior dispensaries (control)", PALETTE["blue"])],
               None, "New Mexico: revenue per dispensary, border vs. interior\n(raw geometric mean, unadjusted)",
               "Revenue per dispensary ($)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig14_nm_dispensary_trends_perstore.png"))
plt.close(fig)
print("Wrote fig14_nm_dispensary_trends_perstore.png")
print(gA.pivot(index="year", columns="border_70mi", values=["geo_mean", "count"]).to_string())

# ---------------- B. NM county-level revenue per capita ----------------
nmc = pd.read_csv(os.path.join(TBL, "nm_revenue_panel.csv"))
gB = geo_mean_by_year(nmc, "revenue_per_capita", "border_70mi")
fig, ax = plt.subplots(figsize=(7.5, 4.8))
plot_two_lines(ax, gB, "border_70mi",
               [(1, "Border counties", PALETTE["red"]), (0, "Interior counties (control)", PALETTE["blue"])],
               None, "New Mexico: revenue per capita, border vs. interior counties\n(raw geometric mean, unadjusted)",
               "Revenue per capita ($)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig15_nm_county_trends_percapita.png"))
plt.close(fig)
print("\nWrote fig15_nm_county_trends_percapita.png")
print(gB.pivot(index="year", columns="border_70mi", values=["geo_mean", "count"]).to_string())

# ---------------- C. Main 5-state revenue per capita, split by entry_capped ----------------
mpc = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
gC = geo_mean_by_year(mpc, "revenue_per_capita", "border_70mi")
mpc["logrpc"] = np.where(mpc["revenue_per_capita"] > 0, np.log(mpc["revenue_per_capita"]), np.nan)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, cap_val, cap_label in zip(axes, [1, 0], ["Capped-entry states", "Free-entry states"]):
    sub = mpc[mpc["entry_capped"] == cap_val]
    g = geo_mean_by_year(sub, "revenue_per_capita", "border_70mi")
    plot_two_lines(ax, g, "border_70mi",
                   [(1, "Border counties", PALETTE["red"]), (0, "Interior counties (control)", PALETTE["blue"])],
                   None, cap_label, "Revenue per capita ($)")
fig.suptitle("Revenue per capita, border vs. interior counties, by entry regime\n"
             "(raw geometric mean, unadjusted; pooled across Colorado, Washington, Oregon, Montana, New Mexico)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig16_main_trends_percapita_bycapped.png"))
plt.close(fig)
print("\nWrote fig16_main_trends_percapita_bycapped.png")

# ---------------- D. Main 5-state revenue per store, split by entry_capped ----------------
mps = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, cap_val, cap_label in zip(axes, [1, 0], ["Capped-entry states", "Free-entry states"]):
    sub = mps[mps["entry_capped"] == cap_val]
    g = geo_mean_by_year(sub, "revenue_per_store", "border_70mi")
    plot_two_lines(ax, g, "border_70mi",
                   [(1, "Border counties", PALETTE["red"]), (0, "Interior counties (control)", PALETTE["blue"])],
                   None, cap_label, "Revenue per store ($)")
fig.suptitle("Revenue per store, border vs. interior counties, by entry regime\n"
             "(raw geometric mean, unadjusted; pooled across Colorado, Washington, Oregon, Montana, New Mexico)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig17_main_trends_perstore_bycapped.png"))
plt.close(fig)
print("\nWrote fig17_main_trends_perstore_bycapped.png")
