"""
Small-multiples U.S. map: which states have an operating recreational retail
market at each of four snapshot years, with the five sample states outlined.
Directly visualizes why the border classification has to be time-varying --
a state's neighbor set flips from nonlegal to legal partway through several
sample states' own panels (CO-NM 2022, WA-OR 2015, OR-CA/NV 2017-2018).
"""
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from healy_style import apply_healy_style, PALETTE

SHP = os.path.join("data", "us_counties_shp", "cb_2023_us_county_500k.shp")
LEGAL_DATES = os.path.join("data", "legal_dates_new.csv")
OUT = "figures_reframe"
SAMPLE_STATES = ["CO", "WA", "OR", "MT", "NM"]
SNAPSHOT_YEARS = [2014, 2018, 2022, 2026]
EXCLUDE = {"AK", "HI", "PR", "VI", "GU", "AS", "MP"}


def main():
    os.makedirs(OUT, exist_ok=True)
    apply_healy_style()

    counties = gpd.read_file(SHP)
    counties = counties[~counties["STUSPS"].isin(EXCLUDE)]
    states = counties.dissolve(by="STUSPS").reset_index()[["STUSPS", "geometry"]]

    legal = pd.read_csv(LEGAL_DATES)
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    retail_by_abbr = dict(zip(legal["state_abbr"], legal["retail_date"]))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, year in zip(axes.flat, SNAPSHOT_YEARS):
        cutoff = pd.Timestamp(f"{year}-12-31")
        states["legal"] = states["STUSPS"].map(
            lambda s: (s in retail_by_abbr) and pd.notna(retail_by_abbr[s]) and retail_by_abbr[s] <= cutoff)
        states.plot(ax=ax, color=states["legal"].map({True: PALETTE["blue"], False: "#E3E3E3"}),
                    edgecolor="white", linewidth=0.3)
        sample = states[states["STUSPS"].isin(SAMPLE_STATES)]
        sample.boundary.plot(ax=ax, edgecolor=PALETTE["red"], linewidth=1.6)
        ax.set_title(f"{year}", fontsize=12, fontweight="bold")
        ax.set_axis_off()
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)

    legal_patch = mpatches.Patch(color=PALETTE["blue"], label="Operating retail market")
    nonlegal_patch = mpatches.Patch(color="#E3E3E3", label="No retail market")
    sample_patch = mpatches.Patch(facecolor="none", edgecolor=PALETTE["red"], linewidth=1.6, label="Sample state (CO/WA/OR/MT/NM)")
    fig.legend(handles=[legal_patch, nonlegal_patch, sample_patch], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Recreational retail legalization, contiguous U.S., by year", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(os.path.join(OUT, "fig4_legalization_map.png"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig4_legalization_map.png")


if __name__ == "__main__":
    main()
