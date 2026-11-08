"""
Build a current-snapshot New Jersey county-level cannabis retailer count,
for the entry-density/continuous-distance margin only (New Jersey has no
county-level revenue data, and -- like Maryland -- every NJ county is within
70 miles of Pennsylvania, so there is no interior control group; only a
continuous-distance specification is usable for this state).
Source: NJ-CRC "Permitted & Licensed Cannabis Businesses" DataTables grid
(extracted via its live jQuery DataTables API -- see chat for how).
County assignment: NJ municipalities are each wholly contained in one county
(no unincorporated territory), so a municipality-name match against the
Census county-subdivision gazetteer is authoritative.
"""
import pandas as pd
import re
import os

RAW_FILE = os.path.join("data", "raw", "newjersey_licenses", "nj_permitted_businesses.csv")
COUSUB_FILE = os.path.join("data", "raw", "nj_cousubs.txt")
UNIVERSE_FILE = os.path.join("data", "county_universe_all23_with_distance.csv")
OUT_FILE = os.path.join("tables", "newjersey_entry_density_snapshot.csv")

SUFFIX_RE = re.compile(r"\s+(city|township|twp\.?|borough|boro|town|village)$", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")

# unincorporated/informal place names -> containing municipality (hand-verified)
INFORMAL_PLACE_FIX = {
    "ATCO": "WATERFORD",
    "BLACKWOOD": "GLOUCESTER",
    "CLIFFWOOD": "ABERDEEN",
    "FRANKLIN PARK": "FRANKLIN",
    "KINGSTON": "SOUTH BRUNSWICK",
    "LAKE HOPATCONG": "JEFFERSON",
    "MARLTON": "EVESHAM",
    "NEWFOUNDLAND": "WEST MILFORD",
    "OAK RIDGE": "JEFFERSON",
    "SICKLERVILLE": "WINSLOW",
    "SOMERSET": "FRANKLIN",
    "TURNERSVILLE": "WASHINGTON",
    "WARETOWN": "OCEAN",
    "HEWITT": "WEST MILFORD",
    "CARNEY'S POINT": "CARNEYS POINT",
    "SOUTH TOM'S RIVER": "SOUTH TOMS RIVER",
}


def norm(name):
    name = TAG_RE.sub("", name)
    name = name.replace("\xa0", " ").replace("&nbsp;", " ")
    name = SUFFIX_RE.sub("", name.strip())
    name = name.strip().upper()
    return INFORMAL_PLACE_FIX.get(name, name)


def build_place_to_county():
    cousub = pd.read_csv(COUSUB_FILE, delimiter="\t")
    cousub.columns = [c.strip() for c in cousub.columns]
    cousub = cousub[cousub["NAME"] != "County subdivisions not defined"]
    cousub["county_fips3"] = cousub["GEOID"].astype(str).str.zfill(10).str[2:5]
    cousub["fips5"] = "34" + cousub["county_fips3"]
    cousub["place_norm"] = cousub["NAME"].apply(norm)
    return dict(zip(cousub["place_norm"], cousub["fips5"]))


def main():
    df = pd.read_csv(RAW_FILE)
    retailers = df[df["Types"].str.contains("Retailer", na=False)].copy()
    print(f"Retailer rows: {len(retailers)} of {len(df)} total permitted businesses")

    # a business can have multiple facility locations, comma-separated
    exploded = retailers.assign(location=retailers["Locations"].str.split(",")).explode("location")
    exploded["location"] = exploded["location"].str.strip()
    print(f"Exploded to {len(exploded)} individual store locations")

    place_to_county = build_place_to_county()
    exploded["place_norm"] = exploded["location"].apply(norm)
    exploded["fips5"] = exploded["place_norm"].map(place_to_county)

    miss = exploded[exploded["fips5"].isna()]
    print(f"Unmatched locations: {len(miss)} of {len(exploded)} -- {sorted(miss['location'].unique())}")

    grp = exploded.dropna(subset=["fips5"]).groupby("fips5").size().reset_index(name="dispensary_count")

    universe = pd.read_csv(UNIVERSE_FILE, dtype={"fips5": str})
    nj_universe = universe[universe["state_abbr"] == "NJ"].copy()
    nj_universe["fips5"] = nj_universe["fips5"].astype(str).str.zfill(5)

    merged = nj_universe.merge(grp, on="fips5", how="left")
    merged["dispensary_count"] = merged["dispensary_count"].fillna(0)
    merged["dispensary_per_1000"] = 1000 * merged["dispensary_count"] / merged["population"]
    merged["border_70mi"] = (merged["dist_nonlegal_mi"] <= 70).astype(int)

    os.makedirs("tables", exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE}: {merged.shape}")
    print(f"NJ counties covered: {nj_universe.shape[0]} of 21")
    print(f"Total matched retail locations: {int(merged['dispensary_count'].sum())} of {len(exploded)}")
    print(f"Border counties (<=70mi): {merged['border_70mi'].sum()} of {len(merged)} -- i.e. ALL of NJ, no interior control group")
    print("\nCorrelation of dispensary density with distance-to-nonlegal-border (continuous check):")
    print(merged[["dist_nonlegal_mi", "dispensary_per_1000"]].corr())


if __name__ == "__main__":
    main()
