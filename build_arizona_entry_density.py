"""
Build a current-snapshot Arizona county-level adult-use dispensary count,
for the entry-density/dispensary-count margin only (Arizona's TPT tax-collection
data is statewide by jurisdiction-type, not by county -- see project data-audit notes).
Source: AZDHS "Licensed Marijuana Establishments" PDF (Adult Use Marijuana Program).
"""
import re
import os
import pdfplumber
import pandas as pd

PDF_FILE = os.path.join("data", "raw", "arizona_licenses", "az_licensed_establishments.pdf")
ZCTA_FILE = os.path.join("data", "raw", "zcta_county_natl.txt")
UNIVERSE_FILE = os.path.join("data", "county_universe_all23_with_distance.csv")
OUT_FILE = os.path.join("tables", "arizona_entry_density_snapshot.csv")

RECORD_START = re.compile(r"(Open|Not)\s+(\d{8}ES[A-Z]{2}\d{8})")


def build_zip_to_county():
    z = pd.read_csv(ZCTA_FILE, delimiter="|", dtype=str)
    z = z.dropna(subset=["GEOID_ZCTA5_20", "GEOID_COUNTY_20"])
    z["AREALAND_PART"] = pd.to_numeric(z["AREALAND_PART"], errors="coerce").fillna(0)
    z = z.sort_values("AREALAND_PART", ascending=False).drop_duplicates(subset=["GEOID_ZCTA5_20"], keep="first")
    return dict(zip(z["GEOID_ZCTA5_20"], z["GEOID_COUNTY_20"]))


def main():
    full_text = ""
    with pdfplumber.open(PDF_FILE) as pdf:
        for page in pdf.pages:
            full_text += (page.extract_text() or "") + "\n"

    starts = list(RECORD_START.finditer(full_text))
    print(f"Found {len(starts)} candidate records")

    records = []
    for i, m in enumerate(starts):
        seg_start = m.start()
        seg_end = starts[i + 1].start() if i + 1 < len(starts) else len(full_text)
        segment = full_text[seg_start:seg_end]
        first_line = segment.split("\n", 1)[0]
        zips_in_first_line = re.findall(r"\b(\d{5})\b", first_line)
        status_word = m.group(1)
        status = "Not Operating" if status_word == "Not" else "Open"
        cert = m.group(2)
        zip5 = zips_in_first_line[-1] if zips_in_first_line else None
        records.append({"status": status, "certificate": cert, "zip": zip5, "raw_first_line": first_line})

    df = pd.DataFrame(records)
    print(df["status"].value_counts())
    print(f"Records with a parsed ZIP: {df['zip'].notna().sum()} of {len(df)}")

    zip_to_county = build_zip_to_county()
    df["county_fips"] = df["zip"].map(zip_to_county)
    print(f"Records matched to a county: {df['county_fips'].notna().sum()} of {len(df)}")

    open_df = df[df["status"] == "Open"].copy()
    grp = open_df.dropna(subset=["county_fips"]).groupby("county_fips").size().reset_index(name="dispensary_count")
    grp = grp.rename(columns={"county_fips": "fips5"})
    grp["fips5"] = grp["fips5"].astype(str).str.zfill(5)

    universe = pd.read_csv(UNIVERSE_FILE, dtype={"fips5": str})
    az_universe = universe[universe["state_abbr"] == "AZ"].copy()
    az_universe["fips5"] = az_universe["fips5"].astype(str).str.zfill(5)

    merged = az_universe.merge(grp, on="fips5", how="left")
    merged["dispensary_count"] = merged["dispensary_count"].fillna(0)
    merged["dispensary_per_1000"] = 1000 * merged["dispensary_count"] / merged["population"]
    merged["border_70mi"] = (merged["dist_nonlegal_mi"] <= 70).astype(int)

    os.makedirs("tables", exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE}: {merged.shape}")
    print(f"AZ counties covered: {az_universe.shape[0]} of 15")
    print(f"Total open dispensaries matched to a county: {int(merged['dispensary_count'].sum())} of {(open_df['status']=='Open').sum()} open statewide")
    print("\n--- Border vs interior raw comparison ---")
    print(merged.groupby("border_70mi")[["dispensary_count", "dispensary_per_1000"]].mean())
    print(f"\nBorder counties (<=70mi): {sorted(merged[merged.border_70mi==1]['county_name'].tolist())}")


if __name__ == "__main__":
    main()
