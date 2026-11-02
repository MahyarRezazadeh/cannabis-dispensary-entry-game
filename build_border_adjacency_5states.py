"""
Border-county classification, redefined per explicit user instruction (2026-08-18):
drop the 70-mile-threshold design entirely. A county is now "border" if it
GEOMETRICALLY TOUCHES a nonlegal-state county line -- literal state-line
adjacency, not proximity within some radius.

Reuses the exact geopandas approach already used elsewhere in this project
family (marijuana_new/Economic Letters/New version EL/build_border_adjacency.py):
county polygons intersected against dissolved state polygons, with a small
buffer to guard against cartographic-simplification slivers. That earlier
script ran in the OPPOSITE direction (nonlegal-state counties bordering a
legal state); this one runs for the 5 sample states' own counties against
ALL other states, then applies each neighbor's own legal status YEAR BY YEAR
(same time-varying principle as the existing 70mi design -- a neighbor that
legalizes mid-panel stops counting as "nonlegal" from its own retail date
onward), so adjacency generalizes cleanly to every neighbor rather than only
the three hardcoded pairs (CO-NM, WA-OR, OR-CA/NV) the distance version needed.
"""
import os
import pandas as pd
import geopandas as gpd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
TBL = os.path.join(BASE, "tables")
SHP = os.path.join(DATA, "us_counties_shp", "cb_2023_us_county_500k.shp")
LEGAL = os.path.join(DATA, "legal_dates_new.csv")

SAMPLE_STATES = ["CO", "WA", "OR", "MT", "NM"]


def build_adjacency_table():
    """
    Uses a projected (meters) CRS and a 1,000m buffer, not the original
    0.001-degree buffer in geographic coordinates (geopandas warns this is
    unreliable, and it verifiably was: a direct metric-distance check found
    Pacific County, WA and Clatsop County, OR (opposite banks of the Columbia
    River mouth) sit 403m apart in this shapefile -- a real, administratively-
    recognized bordering pair that a ~111m-equivalent buffer misses entirely.
    The next-nearest genuine non-touching gap anywhere among the 5 sample
    states' counties is 18.3km (Stillwater County, MT to Wyoming), so any
    buffer between roughly 500m and 18km draws the same, correct line; 1,000m
    is used with margin on both sides.
    """
    counties = gpd.read_file(SHP)[["GEOID", "STUSPS", "NAME", "geometry"]].rename(
        columns={"GEOID": "fips5", "STUSPS": "state_abbr", "NAME": "county_name"})
    counties = counties.to_crs(5070)
    states = counties.dissolve(by="state_abbr").reset_index()[["state_abbr", "geometry"]]

    sample_counties = counties[counties["state_abbr"].isin(SAMPLE_STATES)].copy()
    sample_counties["geom_buffered"] = sample_counties.geometry.buffer(1000)

    rows = []
    for _, county in sample_counties.iterrows():
        own_state = county["state_abbr"]
        touching = states[states.geometry.intersects(county["geom_buffered"]) & (states["state_abbr"] != own_state)]
        for neighbor in touching["state_abbr"]:
            rows.append({"fips5": county["fips5"], "county_name": county["county_name"],
                         "own_state": own_state, "borders_state": neighbor})

    adj = pd.DataFrame(rows)
    adj.to_csv(os.path.join(TBL, "county_border_states_5states.csv"), index=False)
    print(f"Wrote county_border_states_5states.csv: {len(adj)} county-neighbor pairs, "
          f"{adj['fips5'].nunique()} of {len(sample_counties)} sample-state counties border at least one other state")
    return adj


def neighbor_is_nonlegal(own_state, neighbor, year, retail_by_abbr):
    """
    Whether `neighbor` counts as nonlegal for `own_state` in a given `year`.
    For the three pairs where this actually varies within a sample state's
    own panel window, reuses the EXACT special-case rules already established
    and used project-wide in build_master_panel.py's apply_timevarying_border()
    (distance-based version) -- not a fresh Dec-31-cutoff derivation, to avoid
    silently diverging from that precedent:
      - CO-NM: NM counts nonlegal only for year < 2022 (NM legal 2022-04-01).
      - WA-OR: OR counts nonlegal only for year <= 2015. WA's own "year" column
        is a FISCAL year (Jul-Jun), so a plain Dec-31 calendar cutoff would
        wrongly call OR "already legal" for WA's FY2015 (which mostly precedes
        OR's actual 2015-10-01 launch) -- matching the fiscal-year convention
        here, not a calendar one.
      - OR-CA and OR-NV: both count nonlegal only for year == 2017, OR's own
        first panel year and the one year this project's existing code treats
        as still-transitional for both neighbors together (even though NV's
        own retail date, 2017-07-01, falls mid-year) -- not a per-neighbor
        date cutoff for this specific pair.
    Every other neighbor pair among the 5 sample states never changes legal
    status within the relevant state's own panel window, so a plain
    date-vs-year-end lookup against legal_dates_new.csv is exact for them.
    """
    if own_state == "CO" and neighbor == "NM":
        return year < 2022
    if own_state == "WA" and neighbor == "OR":
        return year <= 2015
    if own_state == "OR" and neighbor in ("CA", "NV"):
        return year == 2017
    cutoff = pd.Timestamp(f"{year}-12-31")
    return not (neighbor in retail_by_abbr and pd.notna(retail_by_abbr[neighbor])
                and retail_by_abbr[neighbor] <= cutoff)


def build_year_by_year_border_flag(adj):
    """
    For every sample-state county and every year 2014-2026, flag border=1 if
    the county touches at least one state that is nonlegal that year (see
    neighbor_is_nonlegal). Every sample-state county is included even if it
    has zero geometric neighbors (a landlocked-within-the-state interior
    county) -- it is simply border_adjacent=0 every year.
    """
    legal = pd.read_csv(LEGAL)
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    retail_by_abbr = dict(zip(legal["state_abbr"], legal["retail_date"]))

    counties = gpd.read_file(SHP)[["GEOID", "STUSPS"]].rename(columns={"GEOID": "fips5", "STUSPS": "state_abbr"})
    all_sample = counties[counties["state_abbr"].isin(SAMPLE_STATES)][["fips5", "state_abbr"]].drop_duplicates()

    neighbor_map = adj.groupby(["fips5", "own_state"])["borders_state"].apply(list).to_dict()

    years = list(range(2014, 2027))
    out_rows = []
    for _, row in all_sample.iterrows():
        fips5, own_state = row["fips5"], row["state_abbr"]
        neighbors = neighbor_map.get((fips5, own_state), [])
        for year in years:
            nonlegal_neighbors = [n for n in neighbors if neighbor_is_nonlegal(own_state, n, year, retail_by_abbr)]
            out_rows.append({"fips5": fips5, "state_abbr": own_state, "year": year,
                             "border_adjacent": int(len(nonlegal_neighbors) > 0),
                             "nonlegal_neighbors": ";".join(sorted(nonlegal_neighbors))})

    panel = pd.DataFrame(out_rows)
    panel.to_csv(os.path.join(TBL, "border_adjacent_by_year.csv"), index=False)
    print(f"Wrote border_adjacent_by_year.csv: {len(panel)} county-year rows")
    print("\n--- border=1 county-year counts by state-year (spot check) ---")
    print(panel[panel.border_adjacent == 1].groupby(["state_abbr", "year"]).size().unstack(fill_value=0))
    return panel


if __name__ == "__main__":
    os.makedirs(TBL, exist_ok=True)
    adj = build_adjacency_table()
    build_year_by_year_border_flag(adj)
