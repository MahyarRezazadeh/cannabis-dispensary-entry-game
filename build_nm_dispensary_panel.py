"""
Build a true dispensary-level (not county-aggregated) panel for New Mexico,
using the same 51 CROP licensee-level monthly files as build_nm_revenue_panel.py.

Dispensary identity = normalized street address (the physical storefront), since
a single licensee/chain can operate multiple addresses and a rebrand/ownership
change at the same address should still count as the same physical market
location. Entry month = first calendar month the address appears in the raw
files. Because the panel starts April 2022, any dispensary present in that
first month is LEFT-CENSORED (already open, true entry date unknown) --
flagged as such, not treated as a new entrant.

County / border assignment reuses the exact ZIP->county crosswalk and
county_universe_all23_with_distance.csv border_70mi / dist_nonlegal_mi fields
already validated for the county-level NM panel -- no new geocoding pipeline
is introduced, since no address-level lat/lon source exists for NM in this
project (unlike the separate CO/WA geocoded dispensary file).
"""
import glob
import os
import re
import pandas as pd

RAW_DIR = os.path.join("data", "raw", "newmexico_revenue")
ZCTA_FILE = os.path.join("data", "raw", "zcta_county_natl.txt")
UNIVERSE_FILE = os.path.join("data", "county_universe_all23_with_distance.csv")
OUT_MONTHLY = os.path.join("tables", "nm_dispensary_panel_monthly.csv")
OUT_ANNUAL = os.path.join("tables", "nm_dispensary_panel_annual.csv")

MONTHS = {m: i + 1 for i, m in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"])}

FIRST_SAMPLE_YEAR, FIRST_SAMPLE_MONTH = 2022, 4


def build_zip_to_county():
    z = pd.read_csv(ZCTA_FILE, delimiter="|", dtype=str)
    z = z.dropna(subset=["GEOID_ZCTA5_20", "GEOID_COUNTY_20"])
    z["AREALAND_PART"] = pd.to_numeric(z["AREALAND_PART"], errors="coerce").fillna(0)
    z = z.sort_values("AREALAND_PART", ascending=False).drop_duplicates(subset=["GEOID_ZCTA5_20"], keep="first")
    return dict(zip(z["GEOID_ZCTA5_20"], z["GEOID_COUNTY_20"]))


def parse_year_month(filename):
    base = os.path.basename(filename)
    year = int(re.match(r"(\d{4})_", base).group(1))
    month = None
    for m in MONTHS:
        if m in base:
            month = MONTHS[m]
            break
    if month is None:
        raise ValueError(f"Could not parse month from {base}")
    return year, month


def normalize_address(addr):
    if pd.isna(addr):
        return None
    a = str(addr).strip().upper()
    a = re.sub(r"\s+", " ", a)
    a = a.rstrip(".")
    return a


def main():
    zip_to_county = build_zip_to_county()
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*Licensee*")))
    print(f"{len(files)} licensee files found")

    frames = []
    for f in files:
        year, month = parse_year_month(f)
        df = pd.read_excel(f)
        keep = ["Licensee", "Address", "City", "State", "Zip", "Medical Sales", "Adult-Use Sales", "Total Sales"]
        df = df[[c for c in keep if c in df.columns]].dropna(subset=["Licensee"])
        df = df[~df["Licensee"].astype(str).str.strip().str.rstrip(":").str.lower().eq("totals")]
        df["Total Sales"] = pd.to_numeric(df["Total Sales"], errors="coerce").fillna(0)
        df["year"], df["month"] = year, month
        df["Zip5"] = df["Zip"].astype(str).str.extract(r"(\d{5})")[0]
        df["county_fips"] = df["Zip5"].map(zip_to_county)
        df["addr_norm"] = df["Address"].apply(normalize_address)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["addr_norm"])

    # city-based fallback county match, same convention as build_nm_revenue_panel.py
    city_lookup = (full.dropna(subset=["county_fips"])
                   .groupby([full["City"].str.strip().str.upper(), "county_fips"])
                   .size().reset_index(name="n")
                   .sort_values("n", ascending=False)
                   .drop_duplicates(subset=["City"], keep="first")
                   .set_index("City")["county_fips"].to_dict())
    still_missing = full["county_fips"].isna()
    full.loc[still_missing, "county_fips"] = full.loc[still_missing, "City"].str.strip().str.upper().map(city_lookup)
    full = full.dropna(subset=["county_fips"]).copy()
    full["county_fips"] = full["county_fips"].astype(str).str.zfill(5)

    # dispensary id = normalized address; most frequent licensee name kept as label
    disp_id = full[["addr_norm"]].drop_duplicates().reset_index(drop=True)
    disp_id["dispensary_id"] = disp_id.index
    full = full.merge(disp_id, on="addr_norm", how="left")

    label = (full.groupby(["dispensary_id", "Licensee"]).size().reset_index(name="n")
              .sort_values("n", ascending=False).drop_duplicates(subset=["dispensary_id"], keep="first")
              [["dispensary_id", "Licensee"]].rename(columns={"Licensee": "dispensary_label"}))
    full = full.merge(label, on="dispensary_id", how="left")

    monthly = (full.groupby(["dispensary_id", "dispensary_label", "addr_norm", "county_fips", "year", "month"])
               .agg(total_sales=("Total Sales", "sum"),
                    medical_sales=("Medical Sales", "sum"),
                    adultuse_sales=("Adult-Use Sales", "sum"))
               .reset_index())

    # entry month: first (year, month) this address appears in the data
    entry = (monthly.groupby("dispensary_id")
              .apply(lambda d: d.loc[(d.year * 12 + d.month).idxmin(), ["year", "month"]])
              .rename(columns={"year": "entry_year", "month": "entry_month"}).reset_index())
    monthly = monthly.merge(entry, on="dispensary_id", how="left")
    monthly["left_censored"] = ((monthly["entry_year"] == FIRST_SAMPLE_YEAR)
                                 & (monthly["entry_month"] == FIRST_SAMPLE_MONTH))

    universe = pd.read_csv(UNIVERSE_FILE, dtype={"fips5": str})
    nm_uni = universe[universe.state_abbr == "NM"][
        ["fips5", "county_name", "population", "dist_nonlegal_mi", "land_area_sqmi"]].copy()
    nm_uni["fips5"] = nm_uni["fips5"].astype(str).str.zfill(5)
    monthly = monthly.merge(nm_uni, left_on="county_fips", right_on="fips5", how="left")
    dropped = monthly["county_name"].isna().sum()
    if dropped:
        print(f"Dropping {dropped} dispensary-month rows with fips not in NM county universe")
        monthly = monthly[monthly["county_name"].notna()].copy()
    monthly["border_70mi"] = (monthly["dist_nonlegal_mi"] <= 70).astype(int)
    monthly["state_abbr"] = "NM"

    os.makedirs("tables", exist_ok=True)
    monthly.to_csv(OUT_MONTHLY, index=False)
    print(f"Wrote {OUT_MONTHLY}: {monthly.shape}, {monthly.dispensary_id.nunique()} unique dispensaries")

    annual = (monthly.groupby(["dispensary_id", "dispensary_label", "county_fips", "county_name",
                                "population", "border_70mi", "entry_year", "entry_month",
                                "left_censored", "year"])
              .agg(total_sales=("total_sales", "sum"), n_months=("month", "nunique"))
              .reset_index())
    annual["true_new_entrant"] = (~annual["left_censored"]) & (annual["year"] == annual["entry_year"])
    annual.to_csv(OUT_ANNUAL, index=False)
    print(f"Wrote {OUT_ANNUAL}: {annual.shape}")

    print("\n--- Coverage ---")
    n_disp = monthly.dispensary_id.nunique()
    n_censored = entry.merge(monthly[["dispensary_id"]].drop_duplicates(), on="dispensary_id")
    left_cens_n = monthly.drop_duplicates("dispensary_id")["left_censored"].sum()
    print(f"Unique dispensaries: {n_disp}")
    print(f"Left-censored (already open at Apr 2022): {left_cens_n} ({100*left_cens_n/n_disp:.1f}%)")
    print(f"True new entrants observed opening after Apr 2022: {n_disp - left_cens_n}")
    print(f"Border (<=70mi) dispensaries ever: {monthly[monthly.border_70mi==1].dispensary_id.nunique()}")
    entrants_by_border = (monthly.drop_duplicates('dispensary_id')
                           .groupby('border_70mi')['left_censored'].agg(['size', lambda s: (~s).sum()]))
    entrants_by_border.columns = ['n_dispensaries', 'n_true_new_entrants']
    print(entrants_by_border)


if __name__ == "__main__":
    main()
