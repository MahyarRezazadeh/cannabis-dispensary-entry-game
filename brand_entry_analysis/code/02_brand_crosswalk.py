"""
02_brand_crosswalk.py  --  Venous4 / brand_entry_analysis   (v2)

Storefront -> parent-brand crosswalk for NM dispensaries from raw CROP
"Sales by Licensee" name strings.  See notes/brand_cleaning_procedure.md.

Unit that is unambiguous: STOREFRONT = normalized street address (project dispensary_id).
BRAND = parent operator, inferred by union-find over three linkage rules:
  A. exact shared raw modal licensee-name string across >1 storefront     (high conf)
  B. identical algorithmically-cleaned brand key (>=1 distinctive token)   (medium)
  C. hand-curated alias map for operators A/B split or over-merge          (curated,
     every entry verified against notes/_brand_review.txt to be ONE real operator)
Default: no rule fires -> storefront is its own singleton brand.
Fuzzy similarity only PROPOSES review pairs (notes/_brand_fuzzy_candidates.csv); it
never merges automatically.

Outputs: tables/storefront_master.csv, brand_crosswalk.csv, brand_master.csv
         notes/_brand_review.txt, notes/_brand_fuzzy_candidates.csv
"""
import os, re, glob, unicodedata, difflib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EG   = os.path.dirname(BASE)
NMP  = os.path.join(os.path.dirname(EG), "New Mexico project")
RAW  = os.path.join(EG, "data", "raw", "newmexico_revenue")
TBL, NOTES = os.path.join(BASE, "tables"), os.path.join(BASE, "notes")
LOG  = os.path.join(BASE, "logs", "02_brand_crosswalk.log")
_lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); _lines.append(s)

MONTHS = {m: i+1 for i, m in enumerate(
    ["January","February","March","April","May","June","July","August",
     "September","October","November","December"])}
def ym(fn):
    b = os.path.basename(fn)
    return int(re.match(r"(\d{4})_", b).group(1)), next(MONTHS[k] for k in MONTHS if k in b)

def norm(s):
    if pd.isna(s): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.strip().upper())
def norm_addr(a):
    a = norm(a); return a.rstrip(".") if a else None

# ---------------------------------------------------------------- 1. raw stack
files = sorted(glob.glob(os.path.join(RAW, "*Licensee*")))
frames = []
for fn in files:
    y, m = ym(fn)
    df = pd.read_excel(fn); df.columns = [str(c).strip() for c in df.columns]
    df = df[["Licensee","Address","City","State","Zip","Medical Sales","Adult-Use Sales","Total Sales"]].copy()
    df = df.dropna(subset=["Licensee"])
    df = df[~df["Licensee"].astype(str).str.strip().str.rstrip(":").str.lower().isin(
        ["totals","total","grand total","grand totals"])]
    for c in ["Medical Sales","Adult-Use Sales","Total Sales"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["year"], df["month"] = y, m
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
raw["addr_norm"] = raw["Address"].map(norm_addr)
raw["name_norm"] = raw["Licensee"].map(norm)
raw["city_up"]   = raw["City"].map(norm)
raw["zip5"]      = raw["Zip"].astype(str).str.extract(r"(\d{5})")[0]
raw = raw.dropna(subset=["addr_norm"]).copy()
raw["t"] = raw["year"]*12 + raw["month"]

panel = pd.read_csv(os.path.join(EG, "tables", "nm_dispensary_panel_monthly.csv"),
                    dtype={"county_fips": str, "fips5": str})
addr2id = panel[["addr_norm","dispensary_id"]].drop_duplicates().copy()
addr2id["addr_norm"] = addr2id["addr_norm"].map(norm_addr)
raw = raw.merge(addr2id, on="addr_norm", how="left")
unm = raw.loc[raw["dispensary_id"].isna(), "addr_norm"].nunique()
p(f"raw rows {len(raw)};  unmatched-to-dispensary_id addresses: {unm} "
  f"(${raw.loc[raw['dispensary_id'].isna(),'Total Sales'].sum():,.0f}, dropped)")
raw = raw.dropna(subset=["dispensary_id"]).copy()
raw["dispensary_id"] = raw["dispensary_id"].astype(int)

# ---------------------------------------------------------------- 2. storefront master
def modal(s):
    s = s.dropna(); return s.value_counts().index[0] if len(s) else None
sf = raw.groupby("dispensary_id").agg(
    name_modal=("name_norm", modal),
    name_first=("name_norm", lambda s: raw.loc[raw.loc[s.index,"t"].idxmin(), "name_norm"]),
    n_name_variants=("name_norm", "nunique"),
    all_names=("name_norm", lambda s: " || ".join(sorted(s.unique()))),
    addr_norm=("addr_norm", "first"),
    city=("city_up", modal), zip5=("zip5", modal),
    first_t=("t", "min"), last_t=("t", "max"), n_months=("t", "nunique"),
    total_sales_life=("Total Sales", "sum"),
).reset_index()

geo = pd.read_csv(os.path.join(NMP, "tables", "nm_dispensary_locations.csv"))
sf = sf.merge(geo[["dispensary_id","lon","lat","geocode_source","county_name","fips5"]],
              on="dispensary_id", how="left")
sf = sf.merge(pd.read_csv(os.path.join(NMP, "tables", "nm_dispensary_border_classification.csv")),
              on="dispensary_id", how="left")
sf = sf.merge(panel.drop_duplicates("dispensary_id")[
    ["dispensary_id","entry_year","entry_month","left_censored"]], on="dispensary_id", how="left")
sf["left_censored"] = sf["left_censored"].fillna(False)
CITIES = set(c for c in sf["city"].dropna().unique() if c)          # NM city-name tokens

# ---------------------------------------------------------------- 3. brand key
LEGAL = re.compile(r"\b(L\.?L\.?C|INC(ORPORATED)?|CORP(ORATION)?|LLP|PLLC|LP|LC|PC|PA|"
                   r"CO(MPANY)?|LTD|HOLDINGS?|ENTERPRISES?|MANAGEMENT|GROUP|"
                   r"VENTURES?|INVESTMENTS?|PARTNERS?|FARMS?|LABS?)\b\.?")
NM_TAGS  = re.compile(r"\b(OF NEW MEXICO|NEW MEXICO|N\.? ?M\.?)\b")
TRAIL_DESC = {"RETAILER","RETAIL","RETAI","RTLR","DISPENSARIES","DISPENSARY","DISPENSARi",
              "CONSUMPTION","MICROBUSINESS","MANUFACTURER","MANUFACTURING","MFG","PRODUCERS",
              "PRODUCER","PRODUCTION","PRODUCTIONS","CULTIVATION","STORE","LOCATION","OUTLET",
              "SHOP","DISP","LLC","INC","CO","THE"}
STREET = re.compile(r"\b(ST|STREET|RD|ROAD|BLVD|BOULEVARD|AVE|AVENUE|DR|DRIVE|HWY|HIGHWAY|"
                    r"LN|LANE|CT|PKWY|PARKWAY|WAY|LOOP|TRAIL|TRL|CIR|CIRCLE|PL|PLACE|"
                    r"NE|NW|SE|SW|NORTH|SOUTH|EAST|WEST|SUITE|STE|UNIT|BLDG|US|HWY)\b")
SINGLE_TOKEN_STOP = {"GREEN","GREENS","HIGH","THE","NEW","SMOKE","HERB","HERBS","LEAF",
                     "BUD","BUDS","FARM","FARMS","CANNABIS","WEED","GRASS","TREE","TREES",
                     "GARDEN","GARDENS","FLOWER","FLOWERS","HOUSE","GOLD","SUN","SKY",
                     "ROOTS","BLOOM","HAZE","KUSH","HEMP","JOINT","DANK","LOUD","LIT",
                     "STASH","STONE","PUFF","TOKE","BLAZE","RELIEF","REMEDY","WELLNESS",
                     "HOLISTIC","ORGANIC","ORGANICS","NATURAL","NATURALS","MEDS","MED"}

def brand_key(name):
    s = norm(name)
    m = re.search(r"\bD ?/? ?B ?/? ?A ?/?\b\s*(.+)$", s)          # DBA -> trade name
    if m: s = m.group(1)
    s = re.sub(r"^\s*NMTO\s*[-:]\s*", "", s)                       # drop 'NMTO-' prefix
    s = re.sub(r"^\s*THE\s+", "", s)
    s = LEGAL.sub(" ", s)                                          # legal tokens first
    s = re.split(r"[-/\\]", s, maxsplit=1)[0]                      # branch separator (any dash/slash)
    s = re.sub(r"[.,'\"()&]", " ", s)
    s = NM_TAGS.sub(" ", s)
    toks = [t for t in s.split() if t]
    while toks and toks[-1] in TRAIL_DESC:
        toks.pop()
    # drop trailing city-name token(s) unless that empties the key
    while len(toks) > 1 and toks[-1] in CITIES:
        toks.pop()
    s = " ".join(toks)
    s = re.sub(r"^\s*\d{2,5}\b", " ", s)          # strip a leading street number only
    s = STREET.sub(" ", s)
    s = re.sub(r"\bSTE\b|\bSUITE\b", " ", s)      # keep name-digits (420, 37, 6420, 1912, ...)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# curated aliases -- verified one real multi-storefront NM operator each
ALIAS_RULES = [
    (r"\bULTRA HEALTH\b|\bNMTO\b|NEW MEXICO TOP ORGANICS|ULTRA HEALTHALAMOGORDO", "ULTRA HEALTH"),
    (r"\bR\.? ?GREEN ?LEAF\b", "R GREENLEAF ORGANICS"),
    (r"PECOS VALLEY", "PECOS VALLEY PRODUCTION"),
    (r"EVEREST APOTHECARY|EVEREST CANNABIS", "EVEREST CANNABIS CO"),
    (r"\bVERDES\b", "VERDES FOUNDATION"),
    (r"\bURBAN WELLNESS\b", "URBAN WELLNESS"),
    (r"\bPUR ?LIFE\b", "PURLIFE"),
    (r"\bSACRED GARDEN\b", "SACRED GARDEN"),
    (r"\bSAWMILL\b", "SAWMILL CANNABIS"),                          # <- separate from Sacred Garden
    (r"SOUTHWEST ORGANIC", "SOUTHWEST ORGANIC PRODUCERS"),
    (r"\bPROHIBITION 37\b", "PROHIBITION 37"),
    (r"LOVE'?S CANNABIS|\bLOVES CANNABIS\b", "LOVES CANNABIS"),
    (r"CARVER FAMILY FARM|\bCARVER'?S\b", "CARVER FAMILY FARM"),
    (r"\bMINERVA\b", "MINERVA CANNA"),
    (r"\bNATURAL RX\b", "NATURAL RX"),
    (r"BRYAN'?S GREEN ?CARE", "BRYANS GREENCARE"),
    (r"COMPASSIONATE DISTRIBUTORS", "COMPASSIONATE DISTRIBUTORS"),
    (r"FRUIT ?OF ?THE ?EARTH", "FRUIT OF THE EARTH ORGANICS"),
    (r"\bSEVEN CLOVER\b|\b7 CLOVER\b", "SEVEN CLOVER"),
    (r"\bENCHANTED BOTANICALS\b", "ENCHANTED BOTANICALS"),
    (r"\bDUKE CITY\b", "DUKE CITY CANNABIS"),
    (r"VIREO HEALTH|VIREO OF SANTA FE|\bGREEN GOODS\b", "GREEN GOODS"),
    (r"\bHIGH DESERT RELIEF\b", "HIGH DESERT RELIEF"),
    (r"\bBEST DAZE\b", "BEST DAZE"),
    (r"SANTA FE FARMS", "SANTA FE FARMS"),
    (r"MARIJUANA MEDICINE MAN|\bMEDICINE MAN\b", "MEDICINE MAN"),
    (r"\bRIO GRANDE CANNABIS\b", "RIO GRANDE CANNABIS"),
    (r"\bKURE\b", "KURE"),
    (r"GRASSROOTS GRANTS", "GRASSROOTS GRANTS"),
    (r"GRASS ?ROOTS RX", "GRASSROOTS RX"),
    (r"BUDDIES FARMS? L", "BUDDIES FARMS"),
    (r"\bLEAF AND FLOWER\b|\bLEAF & FLOWER\b", "LEAF AND FLOWER"),
    (r"\bSCORE 420\b", "SCORE 420"),
    (r"\bELKHORN\b", "ELKHORN"),
    (r"\bOCC ABQ\b", "OCC ABQ"),
    (r"\bFCI\b.*NEW MEXICO|FCI OF NM", "FCI OF NEW MEXICO"),
    (r"\bRMC NM\b", "RMC NM"),
    (r"\bACCENDO\b", "ACCENDO"),
    (r"SOUTHWEST WELLNESS", "SOUTHWEST WELLNESS CENTER"),
    (r"\b285 SOUTH\b", "285 SOUTH DISPENSARY"),
]
def alias_of(name_raw, key):
    for pat, canon in ALIAS_RULES:
        if re.search(pat, name_raw) or (key and re.search(pat, key)):
            return canon
    return None

sf["bkey"] = sf["name_modal"].map(brand_key)
sf["bkey_first"] = sf["name_first"].map(brand_key)
sf["alias"] = [alias_of(nr, bk) for nr, bk in zip(sf["name_modal"], sf["bkey"])]
# diagnostic: storefront whose *cleaned brand key* changes across reported names
# (a candidate genuine operator/ownership change, not just a descriptor variant)
def key_variants(all_names):
    ks = {brand_key(x) for x in all_names.split(" || ")}
    return len({k for k in ks if k})
sf["ownership_change_flag"] = sf["all_names"].map(key_variants) > 1

# usable clean key? (>=2 tokens, or a single distinctive token)
def usable(k):
    if not k: return False
    toks = k.split()
    if len(toks) >= 2: return True
    return len(toks) == 1 and len(toks[0]) >= 4 and toks[0] not in SINGLE_TOKEN_STOP
sf["bkey_ok"] = sf["bkey"].map(usable)

# ---------------------------------------------------------------- 4. union-find
parent = {i: i for i in sf["dispensary_id"]}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[max(ra, rb)] = min(ra, rb)

# Rule A: exact shared raw modal name
GENERIC_NAME = {"", "CANNABIS", "DISPENSARY", "RETAIL", "THE DISPENSARY", "DISPENSARY LLC"}
for nm_, g in sf.groupby("name_modal"):
    if nm_ in GENERIC_NAME or len(g) < 2: continue
    ids = g["dispensary_id"].tolist()
    for j in ids[1:]: union(ids[0], j)
# Rule B: identical usable clean key
for k, g in sf[sf["bkey_ok"]].groupby("bkey"):
    ids = g["dispensary_id"].tolist()
    for j in ids[1:]: union(ids[0], j)
# Rule C: same alias canonical
for k, g in sf[sf["alias"].notna()].groupby("alias"):
    ids = g["dispensary_id"].tolist()
    for j in ids[1:]: union(ids[0], j)

sf["comp"] = sf["dispensary_id"].map(find)

# ---------------------------------------------------------------- 5. brand display name + id
def clean_display(name):
    s = norm(name)
    m = re.search(r"\bD ?/? ?B ?/? ?A\b\s*(.+)$", s)
    if m: s = m.group(1)
    s = re.sub(r"^\s*NMTO\s*[-:]\s*", "", s)
    s = re.split(r"[-/\\]", s, maxsplit=1)[0]
    s = LEGAL.sub(" ", s); s = NM_TAGS.sub(" ", s)
    s = re.sub(r"[.,]", " ", s)
    toks = [t for t in s.split() if t]
    while toks and toks[-1] in (TRAIL_DESC - {"CO","THE"}):
        toks.pop()
    while len(toks) > 1 and toks[-1] in CITIES:
        toks.pop()
    return re.sub(r"\s+", " ", " ".join(toks)).strip() or norm(name)

rows = []
for comp, g in sf.groupby("comp"):
    if g["alias"].notna().any():
        disp = g["alias"].dropna().mode().iloc[0]
        rule = "C_alias"
    elif (g["name_modal"].map(lambda x: (sf["name_modal"] == x).sum()).max() >= 2
          and g["name_modal"].nunique() == 1):
        disp = clean_display(g["name_modal"].iloc[0]); rule = "A_sharedname"
    elif g["bkey_ok"].any():
        disp = clean_display(g.loc[g["bkey_ok"], "name_modal"].value_counts().index[0])
        rule = "B_cleankey"
    else:
        disp = clean_display(g["name_modal"].iloc[0]); rule = "D_singleton"
    rows.append((comp, disp, rule, len(g)))
comp_df = pd.DataFrame(rows, columns=["comp","brand_name","link_rule","n_sf"])
# dedupe brand_name collisions across components by suffixing
dupe = comp_df["brand_name"].duplicated(keep=False)
comp_df.loc[dupe, "brand_name"] = (comp_df.loc[dupe, "brand_name"] + " ["
                                   + comp_df.loc[dupe, "comp"].astype(str) + "]")
comp_df = comp_df.sort_values("brand_name").reset_index(drop=True)
comp_df["brand_id"] = np.arange(len(comp_df))
sf = sf.merge(comp_df[["comp","brand_id","brand_name","link_rule"]], on="comp", how="left")

# ---------------------------------------------------------------- 6. rollups + outputs
g = sf.groupby(["brand_id","brand_name","link_rule"]).agg(
    n_storefronts=("dispensary_id","nunique"),
    n_border=("border_dispensary","sum"),
    n_cities=("city","nunique"), n_zips=("zip5","nunique"),
    first_t=("first_t","min"), total_sales_life=("total_sales_life","sum"),
).reset_index()
g["multi_location"] = (g["n_storefronts"] >= 2).astype(int)
g["border_share"] = g["n_border"] / g["n_storefronts"]

p(f"\n=== BRAND SUMMARY (v2) ===")
p(f"storefronts {sf['dispensary_id'].nunique()};  brands {sf['brand_id'].nunique()};  "
  f"multi-location {(g['n_storefronts']>=2).sum()};  singleton {(g['n_storefronts']==1).sum()}")
p(f"storefronts with an ownership/brand-key change over the panel: "
  f"{int(sf['ownership_change_flag'].sum())} ({100*sf['ownership_change_flag'].mean():.1f}%)")
p("link rule (storefronts):\n" + sf['link_rule'].value_counts().to_string())
p("\ntop 30 brands by #storefronts:")
p(g.sort_values('n_storefronts', ascending=False).head(30)
   [['brand_name','n_storefronts','n_border','n_cities','link_rule']].to_string(index=False))
p("\nbrand-size distribution:\n" + g['n_storefronts'].value_counts().sort_index().to_string())

os.makedirs(TBL, exist_ok=True)
sf_cols = ["dispensary_id","brand_id","brand_name","link_rule","name_modal","name_first",
           "bkey","alias","n_name_variants","ownership_change_flag","all_names","addr_norm",
           "city","zip5","county_name","fips5","lon","lat","geocode_source",
           "dist_stateline_mi","nonlegal_pop_within_75mi","border_dispensary",
           "entry_year","entry_month","left_censored","first_t","last_t","n_months",
           "total_sales_life"]
sf[sf_cols].to_csv(os.path.join(TBL, "storefront_master.csv"), index=False)
sf[["dispensary_id","brand_id","brand_name","link_rule"]].to_csv(
    os.path.join(TBL, "brand_crosswalk.csv"), index=False)
g.to_csv(os.path.join(TBL, "brand_master.csv"), index=False)

with open(os.path.join(NOTES, "_brand_review.txt"), "w", encoding="utf-8") as fh:
    fh.write("BRAND REVIEW DUMP v2  (storefronts grouped by assigned brand)\n" + "="*100 + "\n\n")
    for multi in (1, 0):
        for _, brow in g[g["multi_location"] == multi].sort_values(
                ["n_storefronts","brand_name"], ascending=[False, True]).iterrows():
            sub = sf[sf["brand_id"] == brow["brand_id"]]
            fh.write(f"[{int(brow['brand_id'])}] {brow['brand_name']}  "
                     f"(n={int(brow['n_storefronts'])}, border={int(brow['n_border'])}, "
                     f"rule={brow['link_rule']})\n")
            for _, r in sub.sort_values("city").iterrows():
                fh.write(f"   id{r['dispensary_id']:>4} {str(r['city'])[:16]:<16} {r['zip5']} | {r['name_modal'][:72]}\n")
                if r["n_name_variants"] > 1:
                    fh.write(f"          ^variants: {r['all_names'][:150]}\n")
            fh.write("\n")

singles = g[g["n_storefronts"] == 1].merge(sf[["brand_id","name_modal","city"]], on="brand_id")
names = singles["brand_name"].tolist()
cands = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        r = difflib.SequenceMatcher(None, names[i], names[j]).ratio()
        if r >= 0.86:
            cands.append((names[i], singles.iloc[i]["city"], names[j], singles.iloc[j]["city"], round(r,3)))
pd.DataFrame(cands, columns=["name_a","city_a","name_b","city_b","ratio"]).to_csv(
    os.path.join(NOTES, "_brand_fuzzy_candidates.csv"), index=False)
p(f"\nfuzzy near-miss pairs (not merged): {len(cands)}")
with open(LOG, "w", encoding="utf-8") as fh: fh.write("\n".join(_lines))
p("WROTE storefront_master.csv, brand_crosswalk.csv, brand_master.csv, _brand_review.txt")
