"""
Build a current-snapshot Maryland county-level dispensary count, for the
entry-density/continuous-distance margin only (Maryland has no county-level
revenue data -- only 5-region sales totals -- and every MD county is within
70 miles of a nonlegal border, so there is no interior control group; only
a continuous-distance specification is usable for this state).
Source: MCA Data Dashboard "Accessible Data Extract" (official Excel download).
"""
import pandas as pd
import os

RAW_FILE = os.path.join("data", "raw", "maryland_licenses", "md_dashboard_data_jun2026.xlsx")
UNIVERSE_FILE = os.path.join("data", "county_universe_all23_with_distance.csv")
OUT_FILE = os.path.join("tables", "maryland_entry_density_snapshot.csv")

# MCA county names -> county_universe_all23_with_distance.csv county_name values
COUNTY_RENAME = {
    "Baltimore City": "Baltimore city",
    "St. Mary's": "St. Mary's County",
    "Queen Anne's": "Queen Anne's County",
    "Prince George's": "Prince George's County",
}


def main():
    df = pd.read_excel(RAW_FILE, sheet_name="Dispensaries by County")
    df = df.rename(columns={"Physical County": "county_name", "Number of Dispensaries": "dispensary_count"})
    df["county_name"] = df["county_name"].apply(lambda c: COUNTY_RENAME.get(c, c))
    df["county_name"] = df["county_name"].apply(lambda c: c if c.endswith("County") or c.endswith("city") else c + " County")

    universe = pd.read_csv(UNIVERSE_FILE, dtype={"fips5": str})
    md_universe = universe[universe["state_abbr"] == "MD"].copy()

    merged = md_universe.merge(df, on="county_name", how="left")
    unmatched = merged[merged["dispensary_count"].isna()]
    if len(unmatched):
        print(f"WARNING: {len(unmatched)} MD counties in universe did not match dashboard county names: "
              f"{unmatched['county_name'].tolist()}")
    merged["dispensary_count"] = merged["dispensary_count"].fillna(0)
    merged["dispensary_per_1000"] = 1000 * merged["dispensary_count"] / merged["population"]
    merged["border_70mi"] = (merged["dist_nonlegal_mi"] <= 70).astype(int)

    os.makedirs("tables", exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"Wrote {OUT_FILE}: {merged.shape}")
    print(f"MD counties covered: {md_universe.shape[0]} of 24")
    print(f"Total dispensaries: {int(merged['dispensary_count'].sum())} (dashboard reports 119 addressed statewide)")
    print(f"Border counties (<=70mi): {merged['border_70mi'].sum()} of {len(merged)} -- i.e. ALL of MD, no interior control group")
    print("\nCorrelation of dispensary density with distance-to-nonlegal-border (continuous check):")
    print(merged[["dist_nonlegal_mi", "dispensary_per_1000"]].corr())


if __name__ == "__main__":
    main()
