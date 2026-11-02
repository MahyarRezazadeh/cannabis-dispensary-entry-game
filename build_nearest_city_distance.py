"""
Distance from each sample county (2025 dispensary cross-section) to the
nearest CITY located in a currently nonlegal state -- a city-based, not
county-centroid-based, potential-market measure, per explicit user request.

"City" = an incorporated place (Census SUMLEV 162: city/town/village legal
entity) with a 2023 population estimate >= 5,000, in a state that had no
operating recreational retail market as of 2025-12-31 (legal_dates_new.csv).
The 5,000 threshold is a judgment call, disclosed here: it excludes tiny
hamlets that would otherwise dominate "nearest point" results without
representing genuine market size, while still keeping real small towns.
"""
import pandas as pd
import numpy as np
import os

DATA = "data"
TBL = "tables"


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def main():
    # ---------- places (city geometry) ----------
    gaz = pd.read_csv(os.path.join(DATA, "2023_Gaz_place_national.txt"), sep="\t")
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["state_fips"] = (gaz["GEOID"] // 100000).astype(int)
    gaz["place_fips"] = (gaz["GEOID"] % 100000).astype(int)

    # ---------- places (population) ----------
    pop = pd.read_csv(os.path.join(DATA, "sub-est2023.csv"), encoding="latin1")
    pop = pop[pop["SUMLEV"] == 162][["STATE", "PLACE", "POPESTIMATE2023"]].copy()
    pop = pop.rename(columns={"STATE": "state_fips", "PLACE": "place_fips"})

    cities = gaz.merge(pop, on=["state_fips", "place_fips"], how="inner")
    print(f"Matched {len(cities)} of {len(gaz)} gazetteer places to a population estimate")

    # ---------- restrict to nonlegal-as-of-2025 states ----------
    legal = pd.read_csv(os.path.join(DATA, "legal_dates_new.csv"))
    legal["retail_date"] = pd.to_datetime(legal["retail_date"], errors="coerce")
    cutoff = pd.Timestamp("2025-12-31")
    legal_states_2025 = set(legal.loc[legal["retail_date"] <= cutoff, "state_abbr"])
    nonlegal_cities = cities[~cities["USPS"].isin(legal_states_2025)].copy()
    print(f"Nonlegal-state universe: {len(legal_states_2025)} legal states excluded, "
          f"{nonlegal_cities['USPS'].nunique()} nonlegal states remain")

    # ---------- population threshold: "a city", not every hamlet ----------
    POP_MIN = 5000
    nonlegal_cities = nonlegal_cities[nonlegal_cities["POPESTIMATE2023"] >= POP_MIN].copy()
    print(f"{len(nonlegal_cities)} nonlegal-state cities with population >= {POP_MIN}")

    # ---------- sample counties: the 2025 dispensary cross-section ----------
    ps = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    d25 = ps[ps.year == 2025][["fips5", "state_abbr", "county_name", "dispensary_count", "population"]].copy()

    cc = pd.read_stata(os.path.join(DATA, "county_centroids.dta"))
    cc["fips5"] = cc["fips5"].astype(str).str.zfill(5)
    d25 = d25.merge(cc[["fips5", "lat", "lon"]], on="fips5", how="left")
    print(f"\n2025 sample: {len(d25)} counties, {d25['lat'].isna().sum()} missing centroid")

    # ---------- nearest nonlegal city, brute-force (small N, run once) ----------
    city_lat = nonlegal_cities["INTPTLAT"].values
    city_lon = nonlegal_cities["INTPTLONG"].values
    city_name = nonlegal_cities["NAME"].values
    city_state = nonlegal_cities["USPS"].values

    nearest_dist, nearest_name, nearest_state = [], [], []
    for _, row in d25.iterrows():
        d = haversine_miles(row["lat"], row["lon"], city_lat, city_lon)
        i = np.argmin(d)
        nearest_dist.append(d[i])
        nearest_name.append(city_name[i])
        nearest_state.append(city_state[i])

    d25["dist_nearest_nonlegal_city_mi"] = nearest_dist
    d25["nearest_nonlegal_city"] = nearest_name
    d25["nearest_nonlegal_city_state"] = nearest_state

    out = os.path.join(TBL, "dist_nearest_nonlegal_city_2025.csv")
    d25.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(d25[["fips5", "state_abbr", "county_name", "dispensary_count", "population",
               "dist_nearest_nonlegal_city_mi", "nearest_nonlegal_city", "nearest_nonlegal_city_state"]]
          .sort_values("dist_nearest_nonlegal_city_mi").head(15).to_string(index=False))
    print("\nDistance summary by state:")
    print(d25.groupby("state_abbr")["dist_nearest_nonlegal_city_mi"].describe())


if __name__ == "__main__":
    main()
