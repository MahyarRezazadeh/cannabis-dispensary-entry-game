"""
Build a current-snapshot Michigan county-level adult-use retailer count,
for the entry-density/dispensary-count margin only (Michigan has no
county-level revenue data -- see project data-audit notes).
Source: CRA Accela "Verify a License" export, Marihuana Retailer - License, all records.
"""
import pandas as pd
import os

RAW_FILE = os.path.join("data", "raw", "michigan_licenses", "michigan_retailer_licenses_20260815.csv")
ZCTA_FILE = os.path.join("data", "raw", "zcta_county_natl.txt")
UNIVERSE_FILE = os.path.join("data", "county_universe_all23_with_distance.csv")
OUT_FILE = os.path.join("tables", "michigan_entry_density_snapshot.csv")


def build_zip_to_county():
    z = pd.read_csv(ZCTA_FILE, delimiter="|", dtype=str)
    z = z.dropna(subset=["GEOID_ZCTA5_20", "GEOID_COUNTY_20"])
    z["AREALAND_PART"] = pd.to_numeric(z["AREALAND_PART"], errors="coerce").fillna(0)
    z = z.sort_values("AREALAND_PART", ascending=False).drop_duplicates(subset=["GEOID_ZCTA5_20"], keep="first")
    return dict(zip(z["GEOID_ZCTA5_20"], z["GEOID_COUNTY_20"]))


def main():
    df = pd.read_csv(RAW_FILE)
    print(f"Raw records: {len(df)}")
    print(df["Status"].value_counts())

    df = df[df["Status"].astype(str).str.strip().str.lower() == "active"].copy()
    print(f"Active retailer licenses: {len(df)}")

    df["Zip5"] = df["Address"].astype(str).str.extract(r"MI\s+(\d{5})")[0]
    zip_to_county = build_zip_to_county()
    df["county_fips"] = df["Zip5"].map(zip_to_county)

    miss = df[df["county_fips"].isna()]
    print(f"Unmatched (bad/missing zip): {len(miss)} of {len(df)}")

    grp = df.dropna(subset=["county_fips"]).groupby("county_fips").size().reset_index(name="dispensary_count")
    grp = grp.rename(columns={"county_fips": "fips5"})
    grp["fips5"] = grp["fips5"].astype(str).str.zfill(5)

    universe = pd.read_csv(UNIVERSE_FILE, dtype={"fips5": str})
    mi_universe = universe[universe["state_abbr"] == "MI"].copy()
    mi_universe["fips5"] = mi_universe["fips5"].astype(str).str.zfill(5)

    merged = mi_universe.merge(grp, on="fips5", how="left")
    merged["dispensary_count"] = merged["dispensary_count"].fillna(0)
    merged["dispensary_per_1000"] = 1000 * merged["dispensary_count"] / merged["population"]
    merged["border_70mi"] = (merged["dist_nonlegal_mi"] <= 70).astype(int)

    os.makedirs("tables", exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE}: {merged.shape}")
    print(f"MI counties covered: {mi_universe.shape[0]} of 83")
    print(f"Total active retailers matched to a county: {int(merged['dispensary_count'].sum())} of {len(df)} active statewide")
    print("\n--- Border vs interior raw comparison ---")
    print(merged.groupby("border_70mi")[["dispensary_count", "dispensary_per_1000"]].mean())
    print(f"\nBorder counties (<=70mi): {sorted(merged[merged.border_70mi==1]['county_name'].tolist())}")


if __name__ == "__main__":
    main()
