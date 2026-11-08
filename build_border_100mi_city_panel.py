"""
Alternative border definition, per explicit user request 2026-08-19: instead
of literal county-to-state-line geometric adjacency (border_adjacent, the
design used everywhere else in the current draft), classify a county as
"border" if it is within 100 miles of the nearest CITY (Census incorporated
place, 2023 population >= 5,000 -- same threshold and source used in the
2025-cross-section dose-response check, build_nearest_city_distance.py) that
sits in a state with NO operating recreational retail market as of that
county-year's own year.

Time-varying legal status reuses the EXACT special-case rules already
established and validated in build_border_adjacency_5states.py's
neighbor_is_nonlegal() (CO-NM: NM nonlegal iff year<2022; WA-OR: OR nonlegal
iff year<=2015, using WA's own fiscal-year panel convention; OR-CA/OR-NV:
both nonlegal iff year==2017 only) so this is not a fresh re-derivation of
those quirks. Every other (own_state, city_state) pair uses the generic
Dec-31-of-year cutoff against legal_dates_new.csv's retail_date -- and a
state that never appears in that file (e.g. ID, WY, UT, TX) is nonlegal for
every observation, which the generic rule already handles correctly.

Per explicit instruction, this is a SEPARATE appendix-only pass: outputs go
to tables/border_100mi_by_year.csv and tables_reframe/*_100mi.tex /
figures_reframe/*_100mi.png, and none of it replaces or edits the adjacency
design used in the main body of any draft.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
TBL = os.path.join(BASE, "tables")
SAMPLE_STATES = ["CO", "WA", "OR", "MT", "NM"]
POP_MIN = 5000
THRESHOLD_MI = 100
EARTH_RADIUS_MI = 3958.8


def haversine_matrix(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (miles) between every row of (lat1,lon1)
    and every row of (lat2,lon2), returned as an (len1, len2) matrix."""
    lat1r, lon1r = np.radians(lat1)[:, None], np.radians(lon1)[:, None]
    lat2r, lon2r = np.radians(lat2)[None, :], np.radians(lon2)[None, :]
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def neighbor_is_nonlegal(own_state, city_state, year, retail_by_abbr):
    """Identical rule to build_border_adjacency_5states.py's neighbor_is_nonlegal,
    generalized to any city_state (not just a geometrically adjacent one)."""
    if own_state == "CO" and city_state == "NM":
        return year < 2022
    if own_state == "WA" and city_state == "OR":
        return year <= 2015
    if own_state == "OR" and city_state in ("CA", "NV"):
        return year == 2017
    if city_state == own_state:
        return False
    cutoff = pd.Timestamp(f"{year}-12-31")
    retail_date = retail_by_abbr.get(city_state)
    return not (pd.notna(retail_date) and retail_date <= cutoff)


def load_cities():
    gaz = pd.read_csv(os.path.join(DATA, "2023_Gaz_place_national.txt"), sep="\t")
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["state_fips"] = (gaz["GEOID"] // 100000).astype(int)
    gaz["place_fips"] = (gaz["GEOID"] % 100000).astype(int)

    pop = pd.read_csv(os.path.join(DATA, "sub-est2023.csv"), encoding="latin1")
    pop = pop[pop["SUMLEV"] == 162][["STATE", "PLACE", "POPESTIMATE2023"]].copy()
    pop = pop.rename(columns={"STATE": "state_fips", "PLACE": "place_fips"})

    cities = gaz.merge(pop, on=["state_fips", "place_fips"], how="inner")
    cities = cities[cities["POPESTIMATE2023"] >= POP_MIN].copy()
    print(f"City universe: {len(cities)} incorporated places nationwide with population >= {POP_MIN}")
    return cities


def load_county_year_universe():
    pc = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    ps = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    keys = pd.concat([pc[["fips5", "state_abbr", "year"]], ps[["fips5", "state_abbr", "year"]]]).drop_duplicates()

    cc = pd.read_stata(os.path.join(DATA, "county_centroids.dta"))
    cc["fips5"] = cc["fips5"].astype(str).str.zfill(5)
    keys = keys.merge(cc[["fips5", "lat", "lon"]], on="fips5", how="left")
    n_missing = keys["lat"].isna().sum()
    print(f"County-year universe: {len(keys)} rows, {keys['fips5'].nunique()} counties, {n_missing} missing centroid")
    return keys


def main():
    os.makedirs(TBL, exist_ok=True)
    keys = load_county_year_universe()
    cities = load_cities()

    legal = pd.read_csv(os.path.join(DATA, "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    retail_by_abbr = dict(zip(legal["state_abbr"], legal["retail_date"]))

    counties = keys[["fips5", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    city_lat = cities["INTPTLAT"].to_numpy()
    city_lon = cities["INTPTLONG"].to_numpy()
    city_state = cities["USPS"].to_numpy()
    city_name = cities["NAME"].to_numpy()

    print("Computing full county-city distance matrix...")
    dist_matrix = haversine_matrix(counties["lat"].to_numpy(), counties["lon"].to_numpy(), city_lat, city_lon)
    fips_to_row = {f: i for i, f in enumerate(counties["fips5"])}

    out_rows = []
    for own_state in SAMPLE_STATES:
        sub = keys[keys.state_abbr == own_state]
        years = sorted(sub["year"].unique())
        for year in years:
            nonlegal_mask = np.array([neighbor_is_nonlegal(own_state, cs, int(year), retail_by_abbr)
                                       for cs in city_state])
            masked_dist = np.where(nonlegal_mask, dist_matrix, np.inf)
            year_counties = sub.loc[sub.year == year, "fips5"].unique()
            for fips5 in year_counties:
                row_idx = fips_to_row[fips5]
                col = np.argmin(masked_dist[row_idx])
                dist_mi = masked_dist[row_idx, col]
                out_rows.append({
                    "fips5": fips5, "state_abbr": own_state, "year": int(year),
                    "dist_nearest_nonlegal_city_mi": dist_mi,
                    "nearest_nonlegal_city": city_name[col],
                    "nearest_nonlegal_city_state": city_state[col],
                    "border_100mi": int(dist_mi <= THRESHOLD_MI),
                })

    panel = pd.DataFrame(out_rows)
    out_path = os.path.join(TBL, "border_100mi_by_year.csv")
    panel.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(panel)} county-year rows")
    print("\nborder_100mi=1 county-year counts by state-year:")
    print(panel[panel.border_100mi == 1].groupby(["state_abbr", "year"]).size().unstack(fill_value=0))

    print("\n--- Comparison: border_100mi vs border_adjacent (2025 cross-section) ---")
    adj = pd.read_csv(os.path.join(TBL, "border_adjacent_by_year.csv"), dtype={"fips5": str})
    cmp25 = panel[panel.year == 2025].merge(
        adj[adj.year == 2025][["fips5", "border_adjacent"]], on="fips5", how="left")
    print(pd.crosstab(cmp25["border_adjacent"], cmp25["border_100mi"],
                       rownames=["border_adjacent"], colnames=["border_100mi"]))


if __name__ == "__main__":
    main()
