"""
03_build_panels.py  --  Venous4 / brand_entry_analysis

Construct the analysis panels from the raw CROP licensee files + brand crosswalk.

Frequency: calendar QUARTER.  Panel window: 2022Q2 .. 2026Q2 (17 quarters).
2026Q3 (only month 7 present in the raw data) is dropped from the balanced window and
carried in a separate flag for right-censoring logic.

Outputs (brand_entry_analysis/tables/):
  zip_xwalk.csv                 ZIP -> border, nonlegal-pop exposure @25/50/75/100mi, pop, county
  panel_storefront_quarter.csv  storefront x quarter: active/entry/exit, revenue, age,
                                competition @5/10/25mi, same-brand network, exposure
  panel_brand_quarter.csv       brand x quarter: footprint, border share, age, revenue, entry/exit
  panel_zip_quarter.csv         ZIP x quarter: n brands/stores, HHI/CR3/CR5/entropy, shares,
                                entry/exit/turnover rates
  panel_brand_zip_quarter.csv   (brand active in NM at t) x (operating ZIP) x quarter:
                                present / entry / expansion-entry / n_locations
"""
import os, re, glob, unicodedata
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EG   = os.path.dirname(BASE)
NMP  = os.path.join(os.path.dirname(EG), "New Mexico project")
RAW  = os.path.join(EG, "data", "raw", "newmexico_revenue")
TBL  = os.path.join(BASE, "tables")
LOG  = os.path.join(BASE, "logs", "03_build_panels.log")
R_EARTH = 3958.8
_lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); _lines.append(s)

def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(a))

MONTHS = {m: i+1 for i, m in enumerate(
    ["January","February","March","April","May","June","July","August",
     "September","October","November","December"])}
def ym(fn):
    b = os.path.basename(fn)
    return int(re.match(r"(\d{4})_", b).group(1)), next(MONTHS[k] for k in MONTHS if k in b)
def norm(s):
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode()
    return re.sub(r"\s+", " ", s.strip().upper())
def norm_addr(a):
    a = norm(a); return a.rstrip(".") if a else None

# ============================================================ 1. raw monthly sales
frames = []
for fn in sorted(glob.glob(os.path.join(RAW, "*Licensee*"))):
    y, m = ym(fn)
    df = pd.read_excel(fn); df.columns = [str(c).strip() for c in df.columns]
    df = df[["Licensee","Address","Total Sales","Medical Sales","Adult-Use Sales"]].dropna(subset=["Licensee"])
    df = df[~df["Licensee"].astype(str).str.strip().str.rstrip(":").str.lower().isin(
        ["totals","total","grand total","grand totals"])]
    df["Total Sales"] = pd.to_numeric(df["Total Sales"], errors="coerce").fillna(0.0)
    df["addr_norm"] = df["Address"].map(norm_addr)
    df["year"], df["month"] = y, m
    frames.append(df.dropna(subset=["addr_norm"]))
raw = pd.concat(frames, ignore_index=True)

xw = pd.read_csv(os.path.join(TBL, "brand_crosswalk.csv"))
sfm = pd.read_csv(os.path.join(TBL, "storefront_master.csv"))
panel_src = pd.read_csv(os.path.join(EG, "tables", "nm_dispensary_panel_monthly.csv"),
                        dtype={"county_fips": str, "fips5": str})
addr2id = panel_src[["addr_norm","dispensary_id"]].drop_duplicates()
addr2id["addr_norm"] = addr2id["addr_norm"].map(norm_addr)
raw = raw.merge(addr2id, on="addr_norm", how="inner").merge(xw, on="dispensary_id", how="left")

raw["q"] = (raw["month"] - 1) // 3 + 1
raw["yq"] = raw["year"].astype(str) + "Q" + raw["q"].astype(str)
raw["qt"] = raw["year"] * 4 + (raw["q"] - 1)                       # integer quarter index
QT_MIN = int(raw.loc[raw["yq"] == "2022Q2", "qt"].iloc[0])
QT_MAX = int(raw.loc[raw["yq"] == "2026Q2", "qt"].iloc[0])
raw["partial_2026q3"] = raw["qt"] > QT_MAX
p(f"raw monthly rows matched: {len(raw)};  quarters {QT_MIN}..{QT_MAX} "
  f"({QT_MAX-QT_MIN+1});  2026Q3 partial rows held out: {int(raw['partial_2026q3'].sum())}")

sqm = (raw[raw["qt"].between(QT_MIN, QT_MAX)]
       .groupby(["dispensary_id","brand_id","qt"])
       .agg(revenue=("Total Sales","sum"),
            medical=("Medical Sales", lambda s: pd.to_numeric(s, errors="coerce").sum()),
            n_months_reported=("month","nunique"))
       .reset_index())
sqm["active"] = (sqm["revenue"] > 0).astype(int)

# ============================================================ 2. ZIP crosswalk + exposure
zb = pd.read_csv(os.path.join(NMP, "tables", "nm_zip_border_stateline.csv"),
                 dtype={"zip5": str, "county_fips": str})
uz = pd.read_csv(os.path.join(EG, "data", "raw", "uszips_simplemaps.csv"), dtype={"zip": str})
uz = uz.dropna(subset=["lat","lng","population"])
legal = pd.read_csv(os.path.join(EG, "data", "legal_dates_new.csv"))
legal["rd"] = pd.to_datetime(legal["retail_date"], errors="coerce")
LEGAL = set(legal.loc[legal["rd"].notna() & (legal["rd"] <= pd.Timestamp("2023-12-31")), "state_abbr"])
nonleg = uz[~uz["state_id"].isin(LEGAL) & (uz["state_id"] != "NM")]
nl_lat, nl_lon, nl_pop = (nonleg["lat"].to_numpy(), nonleg["lng"].to_numpy(), nonleg["population"].to_numpy())
p(f"legal-retail states @2023-12-31: {len(LEGAL)};  nonlegal-state ZIPs used for exposure: {len(nonleg)}")

for R in (25, 50, 75, 100):
    vals = np.array([nl_pop[hav(la, lo, nl_lat, nl_lon) <= R].sum()
                     for la, lo in zip(zb["zip_lat"], zb["zip_lon"])])
    zb[f"nonleg_pop_{R}"] = vals
zb["border_z"] = zb["border_zip"].astype(int)          # predetermined 75mi + >=1000-pop rule
zb["log_dist_stateline"] = np.log(zb["dist_stateline_mi"].clip(lower=0.1))
zb = zb.rename(columns={"zip_population": "zip_pop"})
zb["log_zip_pop"] = np.log(zb["zip_pop"].clip(lower=1))
zb[["zip5","county_fips","county_name","zip_pop","log_zip_pop","zip_lat","zip_lon",
    "dist_stateline_mi","log_dist_stateline","border_z",
    "nonleg_pop_25","nonleg_pop_50","nonleg_pop_75","nonleg_pop_100"]
   ].to_csv(os.path.join(TBL, "zip_xwalk.csv"), index=False)
p("wrote zip_xwalk.csv")

# storefront-level exposure (own coords) -- reuse project's 75mi dist, add radii
sfm["zip5"] = sfm["zip5"].astype(str).str.extract(r"(\d{5})")[0]
geo = sfm.dropna(subset=["lat","lon"]).copy()
for R in (25, 50, 75, 100):
    geo[f"sf_nonleg_pop_{R}"] = [nl_pop[hav(la, lo, nl_lat, nl_lon) <= R].sum()
                                 for la, lo in zip(geo["lat"], geo["lon"])]
# local resident population within 25mi (all-US ZIPs)
al_lat, al_lon, al_pop = uz["lat"].to_numpy(), uz["lng"].to_numpy(), uz["population"].to_numpy()
geo["sf_localpop_25"] = [al_pop[hav(la, lo, al_lat, al_lon) <= 25].sum()
                         for la, lo in zip(geo["lat"], geo["lon"])]
geo["sf_log_dist_stateline"] = np.log(geo["dist_stateline_mi"].clip(lower=0.1))

# ============================================================ 3. storefront x quarter panel
sf_first = sqm[sqm["active"] == 1].groupby("dispensary_id")["qt"].min().rename("first_active_qt")
sf_last  = sqm[sqm["active"] == 1].groupby("dispensary_id")["qt"].max().rename("last_active_qt")
sfx = sfm.merge(sf_first, on="dispensary_id").merge(sf_last, on="dispensary_id")
sfx["left_censored"] = sfx["first_active_qt"] <= QT_MIN
sfx["exited"] = sfx["last_active_qt"] <= (QT_MAX - 2)          # >=2 silent quarters before end
sfx["exit_qt"] = np.where(sfx["exited"], sfx["last_active_qt"], np.nan)

grid = []
for r in sfx.itertuples():
    for qt in range(int(r.first_active_qt), QT_MAX + 1):
        grid.append((r.dispensary_id, qt))
sfq = pd.DataFrame(grid, columns=["dispensary_id","qt"])
sfq = sfq.merge(sqm[["dispensary_id","qt","revenue","medical","active","n_months_reported"]],
                on=["dispensary_id","qt"], how="left")
sfq[["revenue","medical"]] = sfq[["revenue","medical"]].fillna(0.0)
sfq["active"] = sfq["active"].fillna(0).astype(int)
sfq = sfq.merge(sfx[["dispensary_id","brand_id","brand_name","first_active_qt","last_active_qt",
                     "left_censored","exited","exit_qt","border_dispensary","entry_year",
                     "entry_month","link_rule","ownership_change_flag"]],
                on="dispensary_id", how="left")
sfq = sfq.merge(geo[["dispensary_id","lat","lon","county_name","fips5","zip5",
                     "dist_stateline_mi","sf_log_dist_stateline","sf_localpop_25",
                     "sf_nonleg_pop_25","sf_nonleg_pop_50","sf_nonleg_pop_75","sf_nonleg_pop_100"]],
                on="dispensary_id", how="left")

sfq["age_q"] = sfq["qt"] - sfq["first_active_qt"]
sfq["entry_this_q"] = ((sfq["qt"] == sfq["first_active_qt"]) & (~sfq["left_censored"])).astype(int)
# exit_next: this active quarter is the storefront's permanent last active quarter
sfq["exit_next"] = ((sfq["active"] == 1) & (sfq["qt"] == sfq["exit_qt"])).astype(int)
sfq["in_exit_sample"] = ((sfq["active"] == 1) & (sfq["qt"] <= QT_MAX - 2)).astype(int)
sfq["temp_gap"] = ((sfq["active"] == 0) & (sfq["qt"] > sfq["first_active_qt"])
                   & (sfq["qt"] < sfq["last_active_qt"])).astype(int)
sfq["border_dispensary"] = sfq["border_dispensary"]           # may be NaN for 61 ungeocoded
sfq["revenue_pos"] = sfq["revenue"].where(sfq["revenue"] > 0)
sfq["log_revenue"] = np.log(sfq["revenue_pos"])

# ---- competition & same-brand network, per quarter (geocoded storefronts only) ----
g = sfq[(sfq["active"] == 1) & sfq["lat"].notna()].copy()
COMP_COLS = ["n_disp_5mi","n_disp_10mi","n_disp_25mi","n_brand_5mi","n_brand_10mi","n_brand_25mi",
             "samebrand_nearest_mi","samebrand_25mi","samebrand_50mi","samebrand_100mi",
             "brand_footprint_nm","n_disp_zip","n_brand_zip"]
acc = {c: {} for c in COMP_COLS}          # (dispensary_id, qt) -> value
for qt, idx in g.groupby("qt").indices.items():
    sub = g.iloc[idx]
    la, lo = sub["lat"].to_numpy(), sub["lon"].to_numpy()
    bid = sub["brand_id"].to_numpy(); zp = sub["zip5"].to_numpy(); rid = sub["dispensary_id"].to_numpy()
    D = hav(la[:, None], lo[:, None], la[None, :], lo[None, :])
    np.fill_diagonal(D, np.inf)
    for j in range(len(sub)):
        d = D[j]; key = (int(rid[j]), int(qt))
        m5, m10, m25, m50, m100 = d <= 5, d <= 10, d <= 25, d <= 50, d <= 100
        sb = (bid == bid[j]) & (np.arange(len(sub)) != j)
        acc["n_disp_5mi"][key]  = int(m5.sum());  acc["n_disp_10mi"][key] = int(m10.sum())
        acc["n_disp_25mi"][key] = int(m25.sum())
        acc["n_brand_5mi"][key]  = int(np.unique(bid[m5]).size)
        acc["n_brand_10mi"][key] = int(np.unique(bid[m10]).size)
        acc["n_brand_25mi"][key] = int(np.unique(bid[m25]).size)
        acc["samebrand_nearest_mi"][key] = float(d[sb].min()) if sb.any() else np.nan
        acc["samebrand_25mi"][key]  = int((sb & m25).sum())
        acc["samebrand_50mi"][key]  = int((sb & m50).sum())
        acc["samebrand_100mi"][key] = int((sb & m100).sum())
        acc["brand_footprint_nm"][key] = int(sb.sum() + 1)
        acc["n_disp_zip"][key]  = int((zp == zp[j]).sum())
        acc["n_brand_zip"][key] = int(np.unique(bid[zp == zp[j]]).size)
kk = list(zip(sfq["dispensary_id"].astype(int), sfq["qt"].astype(int)))
for c in COMP_COLS:
    sfq[c] = [acc[c].get(k, np.nan) for k in kk]

sfq.to_csv(os.path.join(TBL, "panel_storefront_quarter.csv"), index=False)
p(f"wrote panel_storefront_quarter.csv  ({len(sfq)} rows, {sfq['dispensary_id'].nunique()} storefronts)")
p(f"  storefront-level: left-censored {sfx['left_censored'].sum()}, "
  f"true entrants {int((~sfx['left_censored']).sum())}, exiters {int(sfx['exited'].sum())}")

# ============================================================ 4. brand x quarter panel
brand_first = sfq[sfq["active"] == 1].groupby("brand_id")["qt"].min().rename("brand_first_qt")
bq = (sfq[sfq["active"] == 1].groupby(["brand_id","brand_name","qt"])
      .agg(n_loc=("dispensary_id","nunique"),
           n_border_loc=("border_dispensary","sum"),
           revenue=("revenue","sum"),
           n_zip=("zip5","nunique"),
           n_county=("county_name","nunique"))
      .reset_index().merge(brand_first, on="brand_id", how="left"))
bq["brand_age_q"] = bq["qt"] - bq["brand_first_qt"]
bq["border_share"] = bq["n_border_loc"] / bq["n_loc"]
bq["multi_loc_now"] = (bq["n_loc"] >= 2).astype(int)
# brand entry / exit (NM extensive margin)
blast = sfq[sfq["active"] == 1].groupby("brand_id")["qt"].max()
bq["brand_entry_q"] = (bq["qt"] == bq["brand_first_qt"]) & (bq["brand_first_qt"] > QT_MIN)
bq["brand_exit_q"]  = bq["brand_id"].map(blast).le(QT_MAX - 2) & (bq["qt"] == bq["brand_id"].map(blast))
bq.to_csv(os.path.join(TBL, "panel_brand_quarter.csv"), index=False)
p(f"wrote panel_brand_quarter.csv  ({len(bq)} rows, {bq['brand_id'].nunique()} brands)")

# ============================================================ 5. ZIP x quarter panel
sfa = sfq[sfq["active"] == 1].copy()
sfa["zip5"] = sfa["zip5"].fillna("MISSING")
bm = pd.read_csv(os.path.join(TBL, "brand_master.csv"))
lifetime_multi = dict(zip(bm["brand_id"], (bm["n_storefronts"] >= 2).astype(int)))
sfa["brand_lifetime_multi"] = sfa["brand_id"].map(lifetime_multi)

# contemporaneous multi-location status of each brand (>=2 active NM stores this quarter)
bq_nloc = bq.set_index(["brand_id","qt"])["n_loc"]
rows = []
for (z, qt), sub in sfa.groupby(["zip5","qt"]):
    rev = sub.groupby("brand_id")["revenue"].sum()
    shares = (rev / rev.sum()).values if rev.sum() > 0 else np.array([1.0])
    n_stores = sub["dispensary_id"].nunique()
    n_brands = sub["brand_id"].nunique()
    hhi = float((shares ** 2).sum())
    srt = np.sort(shares)[::-1]
    nloc_now = sub["brand_id"].map(lambda b: bq_nloc.get((b, qt), 1))
    rows.append(dict(
        zip5=z, qt=qt, n_stores=n_stores, n_brands=n_brands,
        total_revenue=float(sub["revenue"].sum()),
        loc_per_brand=n_stores / n_brands,
        hhi=hhi, hhi_10000=hhi * 1e4,
        cr1=float(srt[:1].sum()), cr3=float(srt[:3].sum()), cr5=float(srt[:5].sum()),
        entropy=float(-(shares[shares > 0] * np.log(shares[shares > 0])).sum()),
        indep_store_share=float((sub["brand_lifetime_multi"] == 0).mean()),
        chain_store_share=float((sub["brand_lifetime_multi"] == 1).mean()),
        multiloc_now_store_share=float((nloc_now >= 2).mean()),
    ))
zq = pd.DataFrame(rows)
# entry / exit / turnover at ZIP level
zb_present = {(z, qt): set(s["brand_id"]) for (z, qt), s in sfa.groupby(["zip5","qt"])}
def brands_at(z, qt):
    return zb_present.get((z, qt), set())
ent, exi = [], []
for r in zq.itertuples():
    prev, cur = brands_at(r.zip5, r.qt - 1), brands_at(r.zip5, r.qt)
    nxt = brands_at(r.zip5, r.qt + 1)
    ent.append(len(cur - prev))
    exi.append(len(cur - nxt) if r.qt <= QT_MAX - 2 else np.nan)
zq["brand_entries"] = ent
zq["brand_exits"] = exi
zq["brand_entry_rate"] = zq["brand_entries"] / zq["n_brands"]
zq["brand_exit_rate"] = zq["brand_exits"] / zq["n_brands"]
zq = zq.merge(zb.rename(columns={"zip5": "zip5"})[
    ["zip5","border_z","dist_stateline_mi","log_dist_stateline","zip_pop","log_zip_pop",
     "county_fips","county_name","nonleg_pop_25","nonleg_pop_50","nonleg_pop_75","nonleg_pop_100"]],
    on="zip5", how="left")
zq.to_csv(os.path.join(TBL, "panel_zip_quarter.csv"), index=False)
p(f"wrote panel_zip_quarter.csv  ({len(zq)} rows, {zq['zip5'].nunique()} ZIPs, "
  f"{zq[zq['border_z']==1]['zip5'].nunique()} border / {zq[zq['border_z']==0]['zip5'].nunique()} interior)")

# ============================================================ 6. brand x ZIP x quarter (presence/entry)
operating_zips = sorted(z for z in sfa["zip5"].unique() if z != "MISSING")
active_brand_q = {qt: set(s["brand_id"]) for qt, s in sfa.groupby("qt")}
present = {(b, z, qt) for b, z, qt in
          sfa.groupby(["brand_id","zip5","qt"]).groups.keys()}
brand_nm_present_q = {qt: set(s["brand_id"]) for qt, s in sfa.groupby("qt")}
# brand active-storefront coordinates by quarter (for ZIP<->network distance)
gc = g[["brand_id","qt","lat","lon"]].dropna()
brand_coords_q = {(b, qt): sub[["lat","lon"]].to_numpy()
                  for (b, qt), sub in gc.groupby(["brand_id","qt"])}
zc = zb.set_index("zip5")[["zip_lat","zip_lon"]].to_dict("index")
rows = []
for qt in range(QT_MIN, QT_MAX + 1):
    abrands = active_brand_q.get(qt, set()) | active_brand_q.get(qt - 1, set())
    for b in abrands:
        prevcoords = brand_coords_q.get((b, qt - 1))
        for z in operating_zips:
            pres = (b, z, qt) in present
            prev = (b, z, qt - 1) in present
            # network: distance from ZIP centroid to brand's nearest t-1 storefront, and counts
            nn = np.nan; n50 = 0; n100 = 0
            if prevcoords is not None and z in zc:
                d = hav(zc[z]["zip_lat"], zc[z]["zip_lon"], prevcoords[:, 0], prevcoords[:, 1])
                nn = float(d.min()); n50 = int((d <= 50).sum()); n100 = int((d <= 100).sum())
            rows.append((b, z, qt, int(pres), int(pres and not prev),
                         int(b in brand_nm_present_q.get(qt - 1, set())), nn, n50, n100))
bzq = pd.DataFrame(rows, columns=["brand_id","zip5","qt","present","entry","brand_in_nm_prevq",
                                  "brand_nearest_store_mi","brand_stores_50mi","brand_stores_100mi"])
bzq["expansion_entry"] = ((bzq["entry"] == 1) & (bzq["brand_in_nm_prevq"] == 1)).astype(int)
bzq["denovo_entry"]    = ((bzq["entry"] == 1) & (bzq["brand_in_nm_prevq"] == 0)).astype(int)
bzq = bzq.merge(zb[["zip5","border_z","log_dist_stateline","dist_stateline_mi","log_zip_pop","zip_pop",
                    "nonleg_pop_25","nonleg_pop_50","nonleg_pop_75","nonleg_pop_100",
                    "county_fips","county_name"]], on="zip5", how="left")
bm_small = bm[["brand_id","n_storefronts","link_rule"]].rename(columns={"n_storefronts":"brand_lifetime_n"})
bzq = bzq.merge(bm_small, on="brand_id", how="left")
bzq = bzq.merge(bq[["brand_id","qt","n_loc","brand_age_q","border_share","multi_loc_now"]],
                on=["brand_id","qt"], how="left")
bzq.to_csv(os.path.join(TBL, "panel_brand_zip_quarter.csv"), index=False)
p(f"wrote panel_brand_zip_quarter.csv  ({len(bzq)} rows; present rate {bzq['present'].mean():.4f}; "
  f"entries {int(bzq['entry'].sum())} [expansion {int(bzq['expansion_entry'].sum())}, "
  f"de novo {int(bzq['denovo_entry'].sum())}])")

with open(LOG, "w", encoding="utf-8") as fh: fh.write("\n".join(_lines))
p("DONE")
