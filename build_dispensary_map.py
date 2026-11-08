"""
Publication-quality map of the spatial distribution of dispensaries across the
five sample states (CO, WA, OR, MT, NM), using each county's dispensary count
from the exact 2025 cross-section of the estimation dataset behind
table_main_pooled.tex (master_perstore_panel.csv) -- not a separately built or
re-geocoded source, so the map is guaranteed consistent with the regression
sample.

Data-availability note (disclosed in the figure caption, not hidden): a true
point-level geocoded dispensary location exists in this project only for
Colorado and Oregon (data/dispensaries_master_geocoded.csv); Washington's
entry in that same file is a stale partial pull (98 records vs. 480 actual
2025 licenses, see rebuild_wa_dispensary_counts.py) and Montana/New Mexico
were never geocoded to points at all (Section 4.1/4.6 of the manuscript). A
county-choropleth of dispensary density -- the same variable actually used in
the regression -- is therefore the only representation that is both accurate
and comparable across all five states; individual CO/OR points are overlaid
as a supplementary layer where they exist, clearly marked as partial.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
from healy_style import apply_healy_style, PALETTE

SHP = os.path.join("data", "us_counties_shp", "cb_2023_us_county_500k.shp")
TBL = "tables"
OUT = "figures_reframe"
SAMPLE_STATES = ["CO", "WA", "OR", "MT", "NM"]
NEIGHBOR_STATES = ["UT", "WY", "NE", "KS", "OK", "TX", "AZ", "ID", "NV", "CA", "SD", "ND"]
EXCLUDE = {"AK", "HI", "PR", "VI", "GU", "AS", "MP"}

# three small-multiples panels: the five sample states cluster into three
# geographically disjoint groups, so one continental map would waste most of
# its area on empty space between them
PANELS = [
    ("Pacific Northwest", ["WA", "OR"], ["ID"]),
    ("Montana", ["MT"], ["ID", "WY", "ND", "SD"]),
    ("Southwest", ["CO", "NM"], ["UT", "WY", "NE", "KS", "OK", "AZ", "TX"]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    apply_healy_style()

    counties = gpd.read_file(SHP)
    counties = counties[~counties["STUSPS"].isin(EXCLUDE)].copy()

    ps = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    d25 = ps[ps.year == 2025].copy()
    d25["dispensary_per_1000"] = 1000 * pd.to_numeric(d25["dispensary_count"], errors="coerce") / d25["population"]

    m = counties.merge(d25[["fips5", "dispensary_count", "dispensary_per_1000", "border_70mi", "state_abbr"]],
                        left_on="GEOID", right_on="fips5", how="left")
    sample_m = m[m["state_abbr"].isin(SAMPLE_STATES)].copy()

    geo_path = os.path.join("data", "dispensaries_master_geocoded.csv")
    geo = pd.read_csv(geo_path)
    geo = geo[geo["state"].isin(["CO", "OR"])].dropna(subset=["lat", "lon"])

    vmax = sample_m["dispensary_per_1000"].max()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))

    for ax, (panel_title, states, neighbors) in zip(axes, PANELS):
        focus = counties[counties["STUSPS"].isin(states)]
        minx, miny, maxx, maxy = focus.total_bounds
        pad_x, pad_y = 0.6, 0.4

        nb = counties[counties["STUSPS"].isin(neighbors)]
        nb.plot(ax=ax, color="#E3E3E3", edgecolor="white", linewidth=0.3)
        counties[~counties["STUSPS"].isin(states + neighbors)].plot(
            ax=ax, color="#F7F7F7", edgecolor="white", linewidth=0.2)

        panel_m = sample_m[sample_m["state_abbr"].isin(states)]
        panel_m.plot(ax=ax, column="dispensary_per_1000", cmap="Blues", edgecolor="white", linewidth=0.3,
                     vmin=0, vmax=vmax, missing_kwds={"color": "#F7F7F7"})

        border_m = panel_m[panel_m["border_70mi"] == 1]
        border_m.boundary.plot(ax=ax, edgecolor=PALETTE["red"], linewidth=1.4)

        states_diss = counties[counties["STUSPS"].isin(states)].dissolve(by="STUSPS")
        states_diss.boundary.plot(ax=ax, edgecolor="#2B2B2B", linewidth=1.2)

        pts = geo[geo["state"].isin(states)]
        if len(pts):
            ax.scatter(pts["lon"], pts["lat"], s=3, c=PALETTE["dark"], alpha=0.4, linewidths=0, zorder=5)

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_axis_off()
        ax.set_title(panel_title, fontsize=12, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", shrink=0.35, pad=0.02, aspect=30,
                         location="bottom")
    cbar.set_label("Dispensaries per 1,000 residents (2025)", fontsize=10)

    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="#2B2B2B", linewidth=1.2, label="Sample-state boundary"),
        mpatches.Patch(facecolor="none", edgecolor=PALETTE["red"], linewidth=1.4,
                        label="Border county (≤70 mi. of a nonlegal-state county, 2025)"),
        mpatches.Patch(color="#E3E3E3", label="Neighboring state"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["dark"], alpha=0.5,
               markersize=5, label="Individual dispensary (CO, OR only – see note)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=2,
               fontsize=9, frameon=False)

    fig.suptitle("Dispensary density and the nonlegal-state border, 2025", fontsize=15, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.22, wspace=0.05)
    fig.savefig(os.path.join(OUT, "fig7_dispensary_map.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Wrote fig7_dispensary_map.png")

    print("\nCounties merged with no 2025 dispensary data (sample states only):")
    print(sample_m[sample_m["dispensary_per_1000"].isna()][["STUSPS", "NAME"]])


if __name__ == "__main__":
    main()
