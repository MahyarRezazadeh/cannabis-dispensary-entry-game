"""
Rebuild the border-exposure variable as TIME-VARYING rather than a static
2025-snapshot, for the three states whose panel window straddles a neighbor's
own retail-legalization date:
  - Colorado: New Mexico legalized retail 2022-04-01 (CO borders NM)
  - Washington: Oregon legalized retail 2015-10-01 (WA borders OR)
  - Oregon: Nevada legalized retail 2017-07-01, California legalized retail
    2018-01-01 (OR borders both) -- relevant only for OR's first panel year,
    2017, when CA was nonlegal all year and NV was nonlegal Jan-Jun.

For a given state-county-year observation, distance to the "nearest nonlegal
neighbor" is recomputed as of that year: a neighbor state counts as nonlegal
if its own retail date had not yet occurred by December 31 of that year.
Montana and New Mexico need no adjustment -- none of their neighbors change
legal status within their panel windows.
"""
import pandas as pd
import numpy as np
import os

CENTROIDS_FILE = os.path.join("data", "county_centroids.dta")
TBL = "tables"


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def min_dist_to_state(from_df, to_df):
    """For each row in from_df (lat/lon), return min haversine distance to any row in to_df."""
    out = np.full(len(from_df), np.inf)
    for _, trow in to_df.iterrows():
        d = haversine_miles(from_df["lat"].values, from_df["lon"].values, trow["lat"], trow["lon"])
        out = np.minimum(out, d)
    return out


def main():
    cc = pd.read_stata(CENTROIDS_FILE)
    cc["fips5"] = cc["fips5"].astype(str).str.zfill(5)

    co = cc[cc.state_abbr == "CO"].copy()
    wa = cc[cc.state_abbr == "WA"].copy()
    orr = cc[cc.state_abbr == "OR"].copy()
    nm_c = cc[cc.state_abbr == "NM"].copy()
    or_c = cc[cc.state_abbr == "OR"].copy()
    ca_c = cc[cc.state_abbr == "CA"].copy()
    nv_c = cc[cc.state_abbr == "NV"].copy()

    co["dist_to_NM"] = min_dist_to_state(co, nm_c)
    wa["dist_to_OR"] = min_dist_to_state(wa, or_c)
    orr["dist_to_CA"] = min_dist_to_state(orr, ca_c)
    orr["dist_to_NV"] = min_dist_to_state(orr, nv_c)

    print("CO counties within 70mi of NM (any year pre-2022 candidate):")
    print(co[co.dist_to_NM <= 70][["county_name", "dist_to_NM"]].sort_values("dist_to_NM").to_string(index=False))
    print("\nWA counties within 70mi of OR (FY2015 candidate):")
    print(wa[wa.dist_to_OR <= 70][["county_name", "dist_to_OR"]].sort_values("dist_to_OR").to_string(index=False))
    print("\nOR counties within 70mi of CA or NV (2017 candidate):")
    print(orr[(orr.dist_to_CA <= 70) | (orr.dist_to_NV <= 70)]
          [["county_name", "dist_to_CA", "dist_to_NV"]].sort_values("dist_to_CA").to_string(index=False))

    for df, name in [(co, "co"), (wa, "wa"), (orr, "or")]:
        df[["fips5", "county_name"] + [c for c in df.columns if c.startswith("dist_to_")]].to_csv(
            os.path.join(TBL, f"timevarying_dist_{name}.csv"), index=False)
        print(f"\nWrote timevarying_dist_{name}.csv")


if __name__ == "__main__":
    main()
