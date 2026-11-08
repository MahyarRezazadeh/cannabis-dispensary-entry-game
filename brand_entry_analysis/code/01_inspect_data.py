"""
01_inspect_data.py  --  Venous4 / brand_entry_analysis

Read-only inspection of every input this sub-project will use. Writes a plain-text
report to logs/01_inspect_data.log. Does NOT modify or create any analysis data.

Goal: establish exactly what the NM cannabis data can and cannot identify for a
Singleton-style (2019) brand entry / exit / location analysis.
"""
import os
import re
import glob
import sys
import textwrap
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # brand_entry_analysis/
EG   = os.path.dirname(BASE)                                                # entry game-marijuana/
NMP  = os.path.join(os.path.dirname(EG), "New Mexico project")              # New Mexico project/
RAW  = os.path.join(EG, "data", "raw", "newmexico_revenue")
LOG  = os.path.join(BASE, "logs", "01_inspect_data.log")

_lines = []
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _lines.append(s)

def head_df(df, n=8, cols=None):
    d = df if cols is None else df[cols]
    p(d.head(n).to_string(max_colwidth=44))

# ----------------------------------------------------------------------------
p("=" * 90)
p("PART 1.  RAW NM CROP 'Sales by Licensee' monthly files")
p("=" * 90)
files = sorted(glob.glob(os.path.join(RAW, "*Licensee*")))
p(f"{len(files)} licensee files, {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January","February","March","April","May","June","July","August",
     "September","October","November","December"])}
def ym(fn):
    b = os.path.basename(fn)
    y = int(re.match(r"(\d{4})_", b).group(1))
    m = next(MONTHS[k] for k in MONTHS if k in b)
    return y, m

# structure of first / middle / last file
for fn in [files[0], files[len(files)//2], files[-1]]:
    y, m = ym(fn)
    df = pd.read_excel(fn)
    p("-" * 80)
    p(f"{os.path.basename(fn)}   ({y}-{m:02d})   shape={df.shape}")
    p("columns:", list(df.columns))
    p(df.head(6).to_string(max_colwidth=40))

# stack everything (Licensee-month rows) for identity/name analysis
frames = []
for fn in files:
    y, m = ym(fn)
    df = pd.read_excel(fn)
    df.columns = [str(c).strip() for c in df.columns]
    keep = [c for c in ["Licensee","Address","City","State","Zip",
                        "Medical Sales","Adult-Use Sales","Total Sales"] if c in df.columns]
    df = df[keep].copy()
    df["year"], df["month"] = y, m
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
p("-" * 80)
p(f"stacked raw licensee-month rows (all files): {len(raw)}")
p("null Licensee rows:", raw["Licensee"].isna().sum())
raw["Licensee_str"] = raw["Licensee"].astype(str).str.strip()
is_total = raw["Licensee_str"].str.rstrip(":").str.lower().isin(
    ["totals","total","grand total","grand totals"])
p("'Totals' summary rows:", int(is_total.sum()))
raw = raw[~is_total & raw["Licensee"].notna()].copy()
for c in ["Medical Sales","Adult-Use Sales","Total Sales"]:
    if c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
p("rows after dropping totals/nulls:", len(raw))
p("Total Sales: sum=${:,.0f}  min={}  max=${:,.0f}  n_zero={}  n_neg={}".format(
    raw["Total Sales"].sum(), raw["Total Sales"].min(), raw["Total Sales"].max(),
    int((raw["Total Sales"] == 0).sum()), int((raw["Total Sales"] < 0).sum())))

p("\n-- Address / Zip availability in raw --")
for c in ["Address","City","State","Zip"]:
    if c in raw.columns:
        p(f"  {c}: {raw[c].notna().mean()*100:.1f}% non-null, {raw[c].astype(str).str.strip().nunique()} distinct")
raw["zip5"] = raw["Zip"].astype(str).str.extract(r"(\d{5})")[0]
p("  zip5 parsed:", raw["zip5"].notna().mean()*100, "%   distinct zip5:", raw["zip5"].nunique())
p("  distinct (Address) :", raw["Address"].astype(str).str.strip().str.upper().nunique())

def norm_addr(a):
    if pd.isna(a): return None
    a = re.sub(r"\s+", " ", str(a).strip().upper()).rstrip(".")
    return a
raw["addr_norm"] = raw["Address"].map(norm_addr)
raw["lic_up"] = raw["Licensee_str"].str.upper()

p("\n-- NAME <-> ADDRESS structure (core for brand identification) --")
by_addr = raw.groupby("addr_norm")["lic_up"].nunique()
p(f"  addresses with >1 distinct licensee name over time: {(by_addr>1).sum()} of {by_addr.size}")
by_lic = raw.groupby("lic_up")["addr_norm"].nunique()
p(f"  licensee-name strings mapping to >1 address: {(by_lic>1).sum()} of {by_lic.size}")
p(f"  max addresses under one licensee-name string: {by_lic.max()}")
p("  top 15 licensee-name strings by #addresses:")
for nm_, k in by_lic.sort_values(ascending=False).head(15).items():
    p(f"     {k:>3}  {nm_[:70]}")

p("\n-- sample of raw licensee name strings (first appearance, 60 random) --")
uniq_names = pd.Series(sorted(raw["lic_up"].unique()))
rng = np.random.default_rng(42)
for s in uniq_names.sample(60, random_state=42).sort_values():
    p("   ", s[:100])

p("\n-- name tokens that signal DBA / branch / legal entity --")
pats = {
    "DBA":        raw["lic_up"].str.contains(r"\bDBA\b"),
    "LLC":        raw["lic_up"].str.contains(r"\bLLC\b|L\.L\.C"),
    "INC/CORP":   raw["lic_up"].str.contains(r"\bINC\b|\bCORP\b|INCORPORATED"),
    "#<num>":     raw["lic_up"].str.contains(r"#\s*\d"),
    "dash-sep":   raw["lic_up"].str.contains(r" - | / "),
    "STORE/UNIT": raw["lic_up"].str.contains(r"\bSTORE\b|\bUNIT\b|\bSUITE\b|\bLOCATION\b"),
    "digits-end": raw["lic_up"].str.contains(r"\d\s*$"),
}
for k, mask in pats.items():
    ex = raw.loc[mask, "lic_up"].drop_duplicates().head(4).tolist()
    p(f"  {k:10s}: {mask.groupby(raw['lic_up']).any().sum():4d} name-strings   e.g. {ex}")

# ----------------------------------------------------------------------------
p("\n" + "=" * 90)
p("PART 2.  Constructed dispensary panel  (entry game-marijuana/tables/nm_dispensary_panel_monthly.csv)")
p("=" * 90)
mp = os.path.join(EG, "tables", "nm_dispensary_panel_monthly.csv")
dm = pd.read_csv(mp, dtype={"county_fips": str, "fips5": str})
p("shape:", dm.shape)
p("columns:", list(dm.columns))
p(dm.dtypes.to_string())
p("\nn dispensaries:", dm.dispensary_id.nunique())
p("year-month span:", f"{dm.year.min()}-{dm.month.min()}", "..", f"{dm.year.max()}-{dm.month.max()}")
p("distinct (year,month):", dm.groupby(['year','month']).ngroups)
p("left_censored dispensaries:", dm.drop_duplicates('dispensary_id')['left_censored'].sum(),
  "of", dm.dispensary_id.nunique())
p("\nentry_year x entry_month distribution (dispensary-level):")
ent = dm.drop_duplicates('dispensary_id')
p(ent.groupby(['entry_year']).size().to_string())
p("\nborder_70mi (county-based) dispensary counts:")
p(ent.groupby('border_70mi').size().to_string())
p("\nhead:")
head_df(dm.sort_values(['dispensary_id','year','month']), 10)

# reporting-gap analysis: months present vs months between first & last
p("\n-- reporting continuity (exit vs temporary non-report) --")
dm["t"] = dm.year * 12 + dm.month
g = dm.groupby("dispensary_id")["t"].agg(["min","max","count"])
g["span"] = g["max"] - g["min"] + 1
g["gap_months"] = g["span"] - g["count"]
p(f"  dispensaries with >=1 missing interior month: {(g.gap_months>0).sum()} of {len(g)}")
p(f"  distribution of gap_months: {g.gap_months.value_counts().sort_index().head(12).to_dict()}")
panel_end_t = dm["t"].max()
g["right_censored"] = g["max"] >= panel_end_t - 1     # present in last 2 months
p(f"  present in last 2 panel months (right-censored, not exiters): {g.right_censored.sum()}")
p(f"  last obs >= 6 months before panel end (candidate exiters): {(g['max'] <= panel_end_t-6).sum()}")

# ----------------------------------------------------------------------------
p("\n" + "=" * 90)
p("PART 3.  Geocoded locations + border classification (New Mexico project/tables)")
p("=" * 90)
loc = pd.read_csv(os.path.join(NMP, "tables", "nm_dispensary_locations.csv"))
p("nm_dispensary_locations.csv shape:", loc.shape, " columns:", list(loc.columns))
p("geocode_source:\n" + loc["geocode_source"].value_counts(dropna=False).to_string())
p("has lat/lon:", loc["lat"].notna().sum(), "of", len(loc))
p(loc.head(6).to_string(max_colwidth=40))

bc = pd.read_csv(os.path.join(NMP, "tables", "nm_dispensary_border_classification.csv"))
p("\nnm_dispensary_border_classification.csv shape:", bc.shape, " columns:", list(bc.columns))
p(bc.describe().to_string())
p("border_dispensary counts:\n" + bc["border_dispensary"].value_counts().to_string())

# do the two dispensary_id spaces line up? (locations vs entry-game panel)
p("\n-- dispensary_id consistency check: locations file vs entry-game monthly panel --")
p("  loc dispensary_id range:", loc.dispensary_id.min(), "-", loc.dispensary_id.max(), " n=", loc.dispensary_id.nunique())
p("  panel dispensary_id range:", dm.dispensary_id.min(), "-", dm.dispensary_id.max(), " n=", dm.dispensary_id.nunique())
merged = dm.drop_duplicates("dispensary_id")[["dispensary_id","dispensary_label","addr_norm"]].merge(
    loc[["dispensary_id","dispensary_label","addr_norm","lat","lon"]],
    on="dispensary_id", how="left", suffixes=("_panel","_loc"))
same_label = (merged["dispensary_label_panel"].astype(str).str.upper()
              == merged["dispensary_label_loc"].astype(str).str.upper())
p("  same dispensary_label on both sides:", int(same_label.sum()), "of", len(merged))
p("  rows where labels differ (first 8):")
p(merged.loc[~same_label, ["dispensary_id","dispensary_label_panel","dispensary_label_loc"]].head(8).to_string())

# ----------------------------------------------------------------------------
p("\n" + "=" * 90)
p("PART 4.  ZIP-level scaffolding")
p("=" * 90)
for f in ["nm_dispensary_panel_monthly_zip.csv","nm_zip_border_stateline.csv",
          "nm_zip_panel_corrected.csv","nm_dispensary_quarterly_panel_v3.csv"]:
    fp = os.path.join(NMP, "tables", f)
    if os.path.exists(fp):
        d = pd.read_csv(fp, nrows=5000)
        p("-" * 70)
        p(f"{f}   (first {len(d)} rows peek)   columns:")
        p("  " + ", ".join(map(str, d.columns)))
        p(d.head(4).to_string(max_colwidth=32))
    else:
        p("MISSING:", f)

uz = os.path.join(EG, "data", "raw", "uszips_simplemaps.csv")
z = pd.read_csv(uz, dtype={"zip": str})
znm = z[z["state_id"] == "NM"]
p("-" * 70)
p(f"uszips_simplemaps.csv: {len(z)} US ZIPs, {len(znm)} NM ZIPs")
p("  NM ZIP cols:", [c for c in z.columns])
p("  NM ZIP population: sum={:,.0f}  n_with_pop={}  n_pop0={}".format(
    znm["population"].fillna(0).sum(), znm["population"].notna().sum(), int((znm["population"].fillna(0)==0).sum())))

cu = pd.read_csv(os.path.join(EG, "data", "county_universe_all23_with_distance.csv"), dtype={"fips5": str})
cunm = cu[cu["state_abbr"] == "NM"]
p("-" * 70)
p("county_universe_all23_with_distance.csv NM rows:", len(cunm), " cols:", list(cunm.columns))
p(cunm[["fips5","county_name","population","dist_nonlegal_mi"]].sort_values("dist_nonlegal_mi").to_string(index=False))

# ----------------------------------------------------------------------------
with open(LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines))
p("\n\nWROTE:", LOG)
