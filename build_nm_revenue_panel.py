"""
Build New Mexico county-month cannabis revenue panel from CROP licensee-level files.
Source: crop.rld.nm.gov data catalog (downloaded via download_nm_revenue.py).
County assignment: ZIP -> county via Census 2020 ZCTA-to-county relationship file
(data/raw/zcta_county_natl.txt), majority-land-area rule for split ZCTAs.
"""
import glob
import os
import re
import pandas as pd

RAW_DIR = os.path.join("data", "raw", "newmexico_revenue")
ZCTA_FILE = os.path.join("data", "raw", "zcta_county_natl.txt")
UNIVERSE_FILE = os.path.join("data", "county_universe_all23_with_distance.csv")
OUT_PANEL = os.path.join("tables", "nm_revenue_panel.csv")
OUT_UNMATCHED = os.path.join("tables", "nm_revenue_unmatched_zips.csv")

MONTHS = {m: i + 1 for i, m in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"])}


def build_zip_to_county():
    z = pd.read_csv(ZCTA_FILE, delimiter="|", dtype=str)
    z = z.dropna(subset=["GEOID_ZCTA5_20", "GEOID_COUNTY_20"])
    z["AREALAND_PART"] = pd.to_numeric(z["AREALAND_PART"], errors="coerce").fillna(0)
    # keep the county with the largest land-area overlap for each ZCTA (handles split ZCTAs)
    z = z.sort_values("AREALAND_PART", ascending=False).drop_duplicates(subset=["GEOID_ZCTA5_20"], keep="first")
    return dict(zip(z["GEOID_ZCTA5_20"], z["GEOID_COUNTY_20"]))


def parse_year_month(filename):
    base = os.path.basename(filename)
    year_match = re.match(r"(\d{4})_", base)
    year = int(year_match.group(1))
    month = None
    for m in MONTHS:
        if m in base:
            month = MONTHS[m]
            break
    if month is None:
        raise ValueError(f"Could not parse month from {base}")
    return year, month


def main():
    zip_to_county = build_zip_to_county()
    print(f"ZIP->county crosswalk: {len(zip_to_county)} ZCTAs")

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*Licensee*")))
    print(f"{len(files)} licensee files found")

    # first pass: load everything, drop summary "Totals" rows, build a City->county fallback
    # from rows whose ZIP resolves cleanly, for the residual rows with malformed ZIPs
    raw_frames = []
    for f in files:
        year, month = parse_year_month(f)
        df = pd.read_excel(f)
        keep_cols = ["Licensee", "Address", "City", "State", "Zip", "Medical Sales", "Adult-Use Sales", "Total Sales"]
        df = df[[c for c in keep_cols if c in df.columns]].dropna(subset=["Licensee"])
        df = df[~df["Licensee"].astype(str).str.strip().str.rstrip(":").str.lower().eq("totals")]
        df["Total Sales"] = pd.to_numeric(df["Total Sales"], errors="coerce").fillna(0)
        df["year"] = year
        df["month"] = month
        df["Zip5"] = df["Zip"].astype(str).str.extract(r"(\d{5})")[0]
        df["county_fips"] = df["Zip5"].map(zip_to_county)
        raw_frames.append(df)
    full = pd.concat(raw_frames, ignore_index=True)
    grand_total_sales = full["Total Sales"].sum()

    city_lookup = (full.dropna(subset=["county_fips"])
                   .groupby([full["City"].str.strip().str.upper(), "county_fips"])
                   .size().reset_index(name="n")
                   .sort_values("n", ascending=False)
                   .drop_duplicates(subset=["City"], keep="first")
                   .set_index("City")["county_fips"].to_dict())

    still_missing = full["county_fips"].isna()
    full.loc[still_missing, "county_fips"] = full.loc[still_missing, "City"].str.strip().str.upper().map(city_lookup)

    unmatched = []
    miss = full[full["county_fips"].isna()]
    unmatched_total_sales = miss["Total Sales"].sum()
    for _, r in miss.iterrows():
        unmatched.append({"year": r["year"], "month": r["month"], "Licensee": r["Licensee"],
                           "City": r.get("City"), "Zip": r.get("Zip"), "Total Sales": r["Total Sales"]})

    rows = []
    for (year, month), df in full.groupby(["year", "month"]):

        df = df.dropna(subset=["county_fips"])
        df["Total Sales"] = pd.to_numeric(df["Total Sales"], errors="coerce").fillna(0)

        grp = df.groupby("county_fips").agg(
            total_sales=("Total Sales", "sum"),
            dispensary_count=("Licensee", "nunique"),
        ).reset_index()
        grp["year"] = year
        grp["month"] = month
        rows.append(grp)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.rename(columns={"county_fips": "fips5"})
    panel["fips5"] = panel["fips5"].astype(str)

    # aggregate to county-YEAR (matches the annual convention used for CO/WA/OR/MT panels elsewhere in this project)
    annual = panel.groupby(["fips5", "year"]).agg(
        total_sales=("total_sales", "sum"),
        dispensary_count=("dispensary_count", "max"),  # cumulative distinct-store proxy: max monthly count observed that year
    ).reset_index()

    universe = pd.read_csv(UNIVERSE_FILE, dtype={"fips5": str})
    nm_universe = universe[universe["state_abbr"] == "NM"].copy()
    nm_universe["fips5"] = nm_universe["fips5"].astype(str).str.zfill(5)
    annual["fips5"] = annual["fips5"].astype(str).str.zfill(5)

    merged = annual.merge(nm_universe, on="fips5", how="left")
    bad_fips = merged[merged["county_name"].isna()]
    if len(bad_fips):
        print(f"WARNING: dropping {len(bad_fips)} rows with fips5 not in NM county universe "
              f"(bad ZIP->county matches, likely mis-keyed ZIPs): {sorted(bad_fips['fips5'].unique())}, "
              f"total_sales dropped = ${bad_fips['total_sales'].sum():,.0f}")
        merged = merged[merged["county_name"].notna()].copy()

    merged["revenue_per_capita"] = merged["total_sales"] / merged["population"]
    merged["revenue_per_store"] = merged["total_sales"] / merged["dispensary_count"]
    merged["border_70mi"] = (merged["dist_nonlegal_mi"] <= 70).astype(int)
    merged["state_abbr"] = "NM"

    os.makedirs("tables", exist_ok=True)
    merged.to_csv(OUT_PANEL, index=False)
    print(f"Wrote {OUT_PANEL}: {merged.shape}")

    if unmatched:
        pd.DataFrame(unmatched).to_csv(OUT_UNMATCHED, index=False)
        print(f"Wrote {OUT_UNMATCHED}: {len(unmatched)} unmatched licensee-month rows")

    print("\n--- Coverage summary ---")
    print(f"Counties with data: {merged['fips5'].nunique()} of 33 NM counties")
    print(f"Years: {sorted(merged['year'].unique())}")
    print(f"Border counties (<=70mi) ever appearing: {sorted(merged[merged['border_70mi']==1]['county_name'].unique())}")
    pct_unmatched = 100 * unmatched_total_sales / grand_total_sales if grand_total_sales else 0
    print(f"\nRevenue coverage: ${grand_total_sales:,.0f} total in raw files, "
          f"${unmatched_total_sales:,.0f} ({pct_unmatched:.2f}%) dropped as unmatched-ZIP rows, "
          f"${bad_fips['total_sales'].sum():,.0f} dropped as bad-fips-match rows")


if __name__ == "__main__":
    main()
