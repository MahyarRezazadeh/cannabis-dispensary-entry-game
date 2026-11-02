"""
05_regressions.py  --  Venous4 / brand_entry_analysis

Stage-A reduced-form results, Singleton-inspired:
  §1  Eq(1)/Table 1 analogue  -- brand PRESENCE  (brand x ZIP x quarter LPM, cols 1-6)
  §2  Market CONCENTRATION / composition          (ZIP x quarter)
  §3  Eq(2)/Table 3 analogue  -- storefront EXIT  (storefront x quarter LPM, cols 1-4)
  §4  ENTRY margin + network interactions          (brand x ZIP x quarter LPM)
  §5  Brand SORTING                                (brand cross-section)
  §6  HETEROGENEITY by brand type

All numbers are ACTUAL ESTIMATED results. Log: logs/05_regressions.log
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import ols_cluster, panel_ols, col, stars

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TBL  = os.path.join(BASE, "tables")
LOG  = os.path.join(BASE, "logs", "05_regressions.log")
_lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); _lines.append(s)
def line(t): p("\n" + "="*92); p(t); p("="*92)
asinh = lambda x: np.log(x + np.sqrt(x**2 + 1))

# ---------------------------------------------------------------- load
bzq = pd.read_csv(os.path.join(TBL, "panel_brand_zip_quarter.csv"), dtype={"zip5": str})
zq  = pd.read_csv(os.path.join(TBL, "panel_zip_quarter.csv"), dtype={"zip5": str})
sfq = pd.read_csv(os.path.join(TBL, "panel_storefront_quarter.csv"), dtype={"zip5": str})
bm  = pd.read_csv(os.path.join(TBL, "brand_master.csv"))
sfm = pd.read_csv(os.path.join(TBL, "storefront_master.csv"), dtype={"zip5": str})

bzq = bzq[bzq["border_z"].notna()].copy()
bzq["log_zip_pop"] = np.log(bzq["zip_pop"].clip(lower=1))
bzq["asinh_nonleg75"] = asinh(bzq["nonleg_pop_75"])
bzq["brand_lifetime_multi"] = (bzq["brand_lifetime_n"] >= 2).astype(int)
bzq["log_brand_n"] = np.log(bzq["brand_lifetime_n"])
bzq["log_brand_nearest_mi"] = np.log(bzq["brand_nearest_store_mi"].clip(lower=1))
# lagged ZIP competition
zlag = zq[["zip5","qt","n_brands","n_stores"]].copy(); zlag["qt"] = zlag["qt"] + 1
bzq = bzq.merge(zlag.rename(columns={"n_brands":"zip_nbrands_lag","n_stores":"zip_nstores_lag"}),
                on=["zip5","qt"], how="left")

# ============================================================ §1 PRESENCE  (Eq 1 / Table 1)
line("§1  SINGLETON EQ.(1) / TABLE 1 ANALOGUE -- BRAND PRESENCE  (brand x ZIP x quarter LPM)")
p(f"sample: {len(bzq):,} brand-ZIP-quarter cells; mean(present) = {bzq['present'].mean():.4f}")
p("  border_z is the predetermined 75mi + >=1,000-nonlegal-resident ZIP indicator.\n")

TERMS = ["border_z","log_zip_pop","log_dist_stateline","zip_nbrands_lag",
         "border_z:brand_lifetime_multi","border_z:log_brand_n"]
p("(1) border + quarter FE")
_, t1 = ols_cluster(bzq, "present", ["border_z","C(qt)"], "zip5")
p(col(t1, ["border_z"]))
p("\n(2) + log ZIP population")
_, t2 = ols_cluster(bzq, "present", ["border_z","log_zip_pop","C(qt)"], "zip5")
p(col(t2, ["border_z","log_zip_pop"]))
p("\n(3) + log distance to nonlegal state line (predetermined market characteristic)")
_, t3 = ols_cluster(bzq, "present", ["border_z","log_zip_pop","log_dist_stateline","C(qt)"], "zip5")
p(col(t3, ["border_z","log_zip_pop","log_dist_stateline"]))
p("\n(4) + brand fixed effects  (PanelOLS: entity=brand, time=quarter; cluster ZIP)")
r4, t4 = panel_ols(bzq, "present", ["border_z","log_zip_pop","log_dist_stateline"],
                   entity="brand_id", time="qt", cluster_col="zip5")
p(col(t4, ["border_z","log_zip_pop","log_dist_stateline"]))
p("\n(5) + lagged local competition (ZIP # brands, t-1)  [brand FE, quarter FE]")
r5, t5 = panel_ols(bzq, "present", ["border_z","log_zip_pop","log_dist_stateline","zip_nbrands_lag"],
                   entity="brand_id", time="qt", cluster_col="zip5")
p(col(t5, ["border_z","log_zip_pop","log_dist_stateline","zip_nbrands_lag"]))
p("\n(6) heterogeneous border effect by brand type  [brand FE, quarter FE]")
bzq["border_x_multi"] = bzq["border_z"] * bzq["brand_lifetime_multi"]
bzq["border_x_lnn"]   = bzq["border_z"] * bzq["log_brand_n"]
r6, t6 = panel_ols(bzq, "present", ["border_z","border_x_multi","log_zip_pop","log_dist_stateline"],
                   entity="brand_id", time="qt", cluster_col="zip5")
p(col(t6, ["border_z","border_x_multi","log_zip_pop","log_dist_stateline"]))
r6b, t6b = panel_ols(bzq, "present", ["border_z","border_x_lnn","log_zip_pop","log_dist_stateline"],
                     entity="brand_id", time="qt", cluster_col="zip5")
p(col(t6b, ["border_z","border_x_lnn","log_zip_pop","log_dist_stateline"]))

p("\n-- restricted to MULTI-LOCATION brands only (brands that actually choose locations) --")
bm2 = bzq[bzq["brand_lifetime_multi"] == 1]
_, tm = ols_cluster(bm2, "present", ["border_z","log_zip_pop","log_dist_stateline","C(qt)"], "zip5")
p(col(tm, ["border_z","log_zip_pop","log_dist_stateline"]))

p("\n-- continuous exposure instead of the binary  (asinh nonlegal pop within 75mi) --")
_, tc = ols_cluster(bzq, "present", ["asinh_nonleg75","log_zip_pop","C(qt)"], "zip5")
p(col(tc, ["asinh_nonleg75","log_zip_pop"]))

# ============================================================ §2 CONCENTRATION
line("§2  MARKET CONCENTRATION / COMPOSITION  (ZIP x quarter)")
z = zq[zq["border_z"].notna()].copy()
z["log_zip_pop"] = np.log(z["zip_pop"].clip(lower=1))
z["log_nbrands"] = np.log(z["n_brands"])
z["log_revenue"] = np.log(z["total_revenue"].clip(lower=1))
z["log_rev_per_store"] = np.log((z["total_revenue"] / z["n_stores"]).clip(lower=1))
z["asinh_nonleg75"] = asinh(z["nonleg_pop_75"])
OUTC = {"n_brands":"n brands","log_nbrands":"log n brands","n_stores":"n stores",
        "hhi_10000":"HHI (0-10000)","cr1":"largest-brand share","cr3":"CR3","entropy":"entropy",
        "indep_store_share":"independent-store share","multiloc_now_store_share":"multi-loc-chain store share",
        "loc_per_brand":"locations per brand","log_revenue":"log total revenue",
        "log_rev_per_store":"log revenue per store","brand_entry_rate":"brand entry rate",
        "brand_exit_rate":"brand exit rate"}
p(f"{'outcome':30s} {'ymean':>10s} {'bBorder raw':>13s} {'bBorder+pop':>13s} "
  f"{'bLogDist+pop':>13s} {'bAsinhExp+pop':>14s}  {'N':>6s}")
p("  border_z and log_dist_stateline are collinear proxies -> reported in SEPARATE columns, "
  "never together.  cluster ZIP; quarter FE throughout.")
conc_rows = []
for y, lab in OUTC.items():
    _, a  = ols_cluster(z, y, ["border_z","C(qt)"], "zip5")
    _, b  = ols_cluster(z, y, ["border_z","log_zip_pop","C(qt)"], "zip5")
    _, d  = ols_cluster(z, y, ["log_dist_stateline","log_zip_pop","C(qt)"], "zip5")
    _, ex = ols_cluster(z, y, ["asinh_nonleg75","log_zip_pop","C(qt)"], "zip5")
    def f(t, k): return f"{t[k][0]:.4f}{stars(t[k][2])}"
    p(f"{lab:30s} {a['_ymean']:>10.3f} {f(a,'border_z'):>13s} {f(b,'border_z'):>13s} "
      f"{f(d,'log_dist_stateline'):>13s} {f(ex,'asinh_nonleg75'):>14s}  {a['_N']:>6d}")
    conc_rows.append(dict(outcome=lab, ymean=a['_ymean'],
                          b_border_raw=a['border_z'][0], p_raw=a['border_z'][2],
                          b_border_pop=b['border_z'][0], p_pop=b['border_z'][2],
                          b_logdist_pop=d['log_dist_stateline'][0], p_logdist=d['log_dist_stateline'][2],
                          b_asinhexp_pop=ex['asinh_nonleg75'][0], p_asinhexp=ex['asinh_nonleg75'][2],
                          N=a['_N']))
pd.DataFrame(conc_rows).to_csv(os.path.join(TBL, "tab_concentration.csv"), index=False)
p("\n-- county FE robustness (identifies off within-county border/interior variation; limited) --")
for y in ["log_nbrands","hhi_10000","indep_store_share"]:
    _, cc = ols_cluster(z, y, ["border_z","log_zip_pop","C(qt)","C(county_fips)"], "zip5")
    p(f"  {y:22s} border_z = {cc['border_z'][0]:.4f}{stars(cc['border_z'][2])} "
      f"({cc['border_z'][1]:.4f})   N={cc['_N']}")

# ============================================================ §3 EXIT  (Eq 2 / Table 3)
line("§3  SINGLETON EQ.(2) / TABLE 3 ANALOGUE -- STOREFRONT EXIT  (storefront x quarter LPM)")
e = sfq[(sfq["in_exit_sample"] == 1)].copy()
e["log_age"]      = np.log(e["age_q"] + 1)
e["log_rev"]      = np.log(e["revenue"].clip(lower=1))
e["log_localpop"] = np.log(e["sf_localpop_25"].clip(lower=1))
e["log_footprint"]= np.log(e["brand_footprint_nm"].clip(lower=1))
e["log_dist_sl"]  = np.log(e["dist_stateline_mi"].clip(lower=0.1))
e["asinh_nl75"]   = asinh(e["sf_nonleg_pop_75"])
e = e.merge(bm[["brand_id","n_storefronts"]].rename(columns={"n_storefronts":"brand_lt_n"}), on="brand_id", how="left")
e["log_brand_lt_n"] = np.log(e["brand_lt_n"])
e["chain"] = (e["brand_lt_n"] >= 2).astype(int)
e = e.dropna(subset=["border_dispensary","n_disp_25mi"])
p(f"exit sample: {len(e):,} active storefront-quarters; baseline Pr(exit next quarter) = {e['exit_next'].mean():.4f}")
p(f"  clustering: nearest-NM-ZIP  (fips5 fallback)\n")
e["clu"] = e["zip5"].fillna(e["fips5"].astype(str))

p("(1) border + local competition + age  [+quarter FE]")
_, x1 = ols_cluster(e, "exit_next", ["border_dispensary","n_disp_25mi","log_age","C(qt)"], "clu")
p(col(x1, ["border_dispensary","n_disp_25mi","log_age"]))
p("\n(2) + own revenue + local population")
_, x2 = ols_cluster(e, "exit_next", ["border_dispensary","n_disp_25mi","log_age","log_rev","log_localpop","C(qt)"], "clu")
p(col(x2, ["border_dispensary","n_disp_25mi","log_age","log_rev","log_localpop"]))
p("\n(3) + competing brands + own-brand network")
_, x3 = ols_cluster(e, "exit_next",
    ["border_dispensary","n_disp_25mi","n_brand_25mi","samebrand_25mi","log_footprint","log_age","log_rev","log_localpop","C(qt)"], "clu")
p(col(x3, ["border_dispensary","n_disp_25mi","n_brand_25mi","samebrand_25mi","log_footprint","log_age","log_rev","log_localpop"]))
p("\n(4) + brand size + chain dummy")
_, x4 = ols_cluster(e, "exit_next",
    ["border_dispensary","n_disp_25mi","n_brand_25mi","samebrand_25mi","log_footprint","log_brand_lt_n","chain","log_age","log_rev","log_localpop","C(qt)"], "clu")
p(col(x4, ["border_dispensary","n_disp_25mi","n_brand_25mi","samebrand_25mi","log_footprint","log_brand_lt_n","chain","log_age","log_rev","log_localpop"]))
p("\n(3c) continuous border: log distance to nonlegal state line  [spec (3) controls]")
_, xc = ols_cluster(e, "exit_next",
    ["log_dist_sl","n_disp_25mi","n_brand_25mi","samebrand_25mi","log_footprint","log_age","log_rev","log_localpop","C(qt)"], "clu")
p(col(xc, ["log_dist_sl","n_disp_25mi","n_brand_25mi","samebrand_25mi"]))
p("\n(3n) continuous border: asinh nonlegal pop within 75mi  [spec (3) controls]")
_, xn = ols_cluster(e, "exit_next",
    ["asinh_nl75","n_disp_25mi","n_brand_25mi","samebrand_25mi","log_footprint","log_age","log_rev","log_localpop","C(qt)"], "clu")
p(col(xn, ["asinh_nl75","n_disp_25mi","n_brand_25mi","samebrand_25mi"]))

# ============================================================ §4 ENTRY
line("§4  ENTRY MARGIN  (brand x ZIP x quarter LPM;  brand not present in ZIP at t-1)")
en = bzq[bzq["entry"].notna()].copy()
en = en[~((en["present"] == 1) & (en["entry"] == 0))]   # keep feasible cells (not-present last q)
en = en[en["brand_in_nm_prevq"] == 1].copy()            # potential ENTRANTS active in NM already
p(f"feasible expansion cells (brand already in NM, not yet in ZIP z): {len(en):,}; "
  f"mean(entry) = {en['entry'].mean():.5f}")
p("\n(1) border + log ZIP pop  [+quarter FE]")
_, y1 = ols_cluster(en, "entry", ["border_z","log_zip_pop","C(qt)"], "zip5")
p(col(y1, ["border_z","log_zip_pop"]))
p("\n(2) + log dist to nonlegal stateline")
_, y2 = ols_cluster(en, "entry", ["border_z","log_zip_pop","log_dist_stateline","C(qt)"], "zip5")
p(col(y2, ["border_z","log_zip_pop","log_dist_stateline"]))
p("\n(3) + brand network: log dist ZIP->nearest own store, own stores within 100mi")
_, y3 = ols_cluster(en, "entry",
    ["border_z","log_zip_pop","log_dist_stateline","log_brand_nearest_mi","brand_stores_100mi","C(qt)"], "zip5")
p(col(y3, ["border_z","log_zip_pop","log_dist_stateline","log_brand_nearest_mi","brand_stores_100mi"]))
p("\n(4) + brand FE (PanelOLS entity=brand, time=quarter, cluster ZIP)")
r4e, y4 = panel_ols(en, "entry", ["border_z","log_zip_pop","log_dist_stateline","log_brand_nearest_mi","brand_stores_100mi"],
                    entity="brand_id", time="qt", cluster_col="zip5")
p(col(y4, ["border_z","log_zip_pop","log_dist_stateline","log_brand_nearest_mi","brand_stores_100mi"]))
p("\n(5) NETWORK INTERACTION:  border_z x own-stores-within-100mi   [brand FE]")
en["border_x_net100"] = en["border_z"] * en["brand_stores_100mi"]
r5e, y5 = panel_ols(en, "entry",
    ["border_z","border_x_net100","brand_stores_100mi","log_zip_pop","log_dist_stateline","log_brand_nearest_mi"],
    entity="brand_id", time="qt", cluster_col="zip5")
p(col(y5, ["border_z","border_x_net100","brand_stores_100mi","log_zip_pop","log_dist_stateline","log_brand_nearest_mi"]))

p("\n-- DE NOVO entry (brand's first-ever NM store) : ZIP-quarter count of de novo entrants --")
dn = (bzq.groupby(["zip5","qt"]).agg(denovo=("denovo_entry","sum"),
      border_z=("border_z","first"), log_zip_pop=("log_zip_pop","first"),
      log_dist=("log_dist_stateline","first")).reset_index())
_, dnr = ols_cluster(dn, "denovo", ["border_z","log_zip_pop","log_dist","C(qt)"], "zip5")
p(col(dnr, ["border_z","log_zip_pop","log_dist"]))

# ============================================================ §5 BRAND SORTING
line("§5  BRAND SORTING  (brand cross-section: are big/old/wide-reach brands over-represented at borders?)")
sfm["border_dispensary"] = sfm["border_dispensary"]
bs = (sfm.dropna(subset=["border_dispensary"]).groupby("brand_id")
      .agg(n_sf=("dispensary_id","nunique"), n_border=("border_dispensary","sum"),
           n_cities=("city","nunique"), first_t=("first_t","min")).reset_index())
bs["border_share"] = bs["n_border"] / bs["n_sf"]
bs["log_n_sf"] = np.log(bs["n_sf"])
bs["brand_age_q"] = (bs["first_t"].max() - bs["first_t"]) / 3.0
base_rate = sfm["border_dispensary"].mean()
p(f"overall storefront border rate = {base_rate:.3f}  (a brand above this is over-represented at borders)")
p(f"\n-- all brands (n={len(bs)}) --")
_, s1 = ols_cluster(bs, "border_share", ["log_n_sf","brand_age_q","n_cities"], "brand_id")
p(col(s1, ["log_n_sf","brand_age_q","n_cities"]))
p(f"\n-- multi-location brands only (n={(bs['n_sf']>=2).sum()}) --")
_, s2 = ols_cluster(bs[bs["n_sf"] >= 2], "border_share", ["log_n_sf","brand_age_q","n_cities"], "brand_id")
p(col(s2, ["log_n_sf","brand_age_q","n_cities"]))
p("\nmean border_share by brand-size group (multi-loc brands):")
mb = bs[bs["n_sf"] >= 2].copy()
mb["sizegrp"] = np.where(mb["n_sf"] >= 6, "large(6+)", np.where(mb["n_sf"] >= 3, "mid(3-5)", "small(2)"))
p(mb.groupby("sizegrp")["border_share"].agg(["mean","median","count"]).to_string())
p("\nmean border_share: singletons vs multi-loc")
p(bs.assign(multi=(bs["n_sf"] >= 2)).groupby("multi")["border_share"].agg(["mean","count"]).to_string())

# ============================================================ §6 HETEROGENEITY
line("§6  HETEROGENEITY -- does the border response differ by brand type?")
p("-- EXIT: border_dispensary x chain, on the exit sample (spec 3 controls) --")
e["border_x_chain"] = e["border_dispensary"] * e["chain"]
_, h1 = ols_cluster(e, "exit_next",
    ["border_dispensary","border_x_chain","chain","n_disp_25mi","n_brand_25mi","log_footprint","log_age","log_rev","log_localpop","C(qt)"], "clu")
p(col(h1, ["border_dispensary","border_x_chain","chain"]))
p("\n-- ENTRY: border_z x brand_lifetime_multi (feasible expansion cells) --")
en["border_x_multi"] = en["border_z"] * en["brand_lifetime_multi"]
_, h2 = ols_cluster(en, "entry",
    ["border_z","border_x_multi","brand_lifetime_multi","log_zip_pop","log_dist_stateline","C(qt)"], "zip5")
p(col(h2, ["border_z","border_x_multi","brand_lifetime_multi"]))

with open(LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines))
p("\nWROTE logs/05_regressions.log + tab_concentration.csv")
