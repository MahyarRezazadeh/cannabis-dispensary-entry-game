"""
04_descriptives.py  --  Venous4 / brand_entry_analysis

Report section 3 (descriptive brand structure) + Singleton Table 2 analogue
(exiters vs persisters).  All ACTUAL ESTIMATED numbers.

Writes: logs/04_descriptives.log
        tables/tab_summary_stats.csv
        tables/tab_border_vs_interior_composition.csv
        tables/tab_exiters_vs_persisters.csv
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import diff_means

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TBL  = os.path.join(BASE, "tables")
LOG  = os.path.join(BASE, "logs", "04_descriptives.log")
_lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); _lines.append(s)

sfq = pd.read_csv(os.path.join(TBL, "panel_storefront_quarter.csv"))
sfm = pd.read_csv(os.path.join(TBL, "storefront_master.csv"))
bm  = pd.read_csv(os.path.join(TBL, "brand_master.csv"))
zq  = pd.read_csv(os.path.join(TBL, "panel_zip_quarter.csv"))
bq  = pd.read_csv(os.path.join(TBL, "panel_brand_quarter.csv"))

# ============================================================ A. brand structure
p("="*90); p("A.  BRAND STRUCTURE (statewide, whole panel)"); p("="*90)
n_sf, n_br = sfm["dispensary_id"].nunique(), bm["brand_id"].nunique()
p(f"storefronts {n_sf};  brands {n_br};  multi-location brands {(bm['n_storefronts']>=2).sum()} "
  f"({100*(bm['n_storefronts']>=2).mean():.1f}%);  singleton brands {(bm['n_storefronts']==1).sum()}")
p(f"storefronts operated by a multi-location brand: "
  f"{sfm['brand_id'].map(bm.set_index('brand_id')['n_storefronts']).ge(2).sum()} "
  f"({100*sfm['brand_id'].map(bm.set_index('brand_id')['n_storefronts']).ge(2).mean():.1f}%)")
rev_by_brand = sfq[sfq.active == 1].groupby("brand_id")["revenue"].sum().sort_values(ascending=False)
tot = rev_by_brand.sum()
p(f"\nstatewide revenue concentration (whole-panel $): total ${tot:,.0f}")
for k in (1, 3, 5, 10, 20):
    p(f"  top-{k:>2} brands: {100*rev_by_brand.head(k).sum()/tot:5.1f}% of revenue")
p(f"  statewide brand HHI (revenue): {1e4*((rev_by_brand/tot)**2).sum():.0f}")
p(f"\nbrand-size distribution (storefronts per brand):")
p(bm["n_storefronts"].value_counts().sort_index().to_string())
p(f"\ntop 15 brands by whole-panel revenue:")
top = rev_by_brand.head(15).reset_index().merge(bm[["brand_id","brand_name","n_storefronts","n_border"]], on="brand_id")
p(top[["brand_name","n_storefronts","n_border","revenue"]].to_string(index=False))

# ============================================================ B. summary statistics
p("\n" + "="*90); p("B.  SUMMARY STATISTICS"); p("="*90)
ss = {}
a = sfq[sfq.active == 1]
ss["storefront-quarter (active)"] = a[["revenue","medical","age_q","n_disp_5mi","n_disp_25mi",
    "n_brand_25mi","samebrand_25mi","brand_footprint_nm","sf_localpop_25","sf_nonleg_pop_75",
    "dist_stateline_mi"]].describe().T[["mean","std","25%","50%","75%"]]
ss["ZIP-quarter"] = zq[["n_brands","n_stores","loc_per_brand","hhi_10000","cr3","cr5","entropy",
    "indep_store_share","total_revenue","zip_pop","dist_stateline_mi",
    "brand_entry_rate","brand_exit_rate"]].describe().T[["mean","std","25%","50%","75%"]]
ss["brand-quarter (active)"] = bq[["n_loc","n_border_loc","border_share","brand_age_q",
    "revenue","n_zip","n_county"]].describe().T[["mean","std","25%","50%","75%"]]
for k, v in ss.items():
    p(f"\n-- {k} --"); p(v.round(2).to_string())
pd.concat({k: v for k, v in ss.items()}).round(3).to_csv(os.path.join(TBL, "tab_summary_stats.csv"))

# ============================================================ C. border vs interior composition
p("\n" + "="*90); p("C.  BORDER vs INTERIOR MARKET COMPOSITION (ZIP-quarter; unconditional)"); p("="*90)
z = zq[zq["border_z"].notna()].copy()
comp_vars = ["n_brands","n_stores","loc_per_brand","hhi_10000","cr1","cr3","cr5","entropy",
             "indep_store_share","chain_store_share","multiloc_now_store_share",
             "brand_entry_rate","brand_exit_rate","total_revenue","zip_pop"]
z["revenue_per_store"] = z["total_revenue"] / z["n_stores"]
comp_vars += ["revenue_per_store"]
tab = diff_means(z, "border_z", comp_vars)
p(tab.round(3).to_string(index=False))
tab.to_csv(os.path.join(TBL, "tab_border_vs_interior_composition.csv"), index=False)
p("\n(unconditional means; border ZIPs differ systematically in population etc. -- "
  "controlled versions in 05_regressions.py)")

# storefront-level border vs interior
p("\n-- storefront-level (one row per storefront) --")
sx = sfq.groupby("dispensary_id").agg(
    brand_id=("brand_id","first"), exited=("exited","first"),
    left_censored=("left_censored","first"), n_active_q=("active","sum"),
    border_dispensary=("border_dispensary","first"),
    dist_stateline_mi=("dist_stateline_mi","first"),
    sf_nonleg_pop_75=("sf_nonleg_pop_75","first"),
    sf_localpop_25=("sf_localpop_25","first"),
    n_disp_25mi_mean=("n_disp_25mi","mean"),
    n_brand_25mi_mean=("n_brand_25mi","mean"),
    samebrand_25mi_mean=("samebrand_25mi","mean"),
    brand_footprint_mean=("brand_footprint_nm","mean"),
    mean_pos_rev=("revenue", lambda s: s[s > 0].mean()),
    first_yr_rev=("revenue", lambda s: s.head(4).sum()),
).reset_index().merge(bm[["brand_id","n_storefronts"]].rename(columns={"n_storefronts":"brand_lifetime_n"}),
                      on="brand_id", how="left")
sx["chain_store"] = (sx["brand_lifetime_n"] >= 2).astype(int)
p(diff_means(sx[sx["border_dispensary"].notna()], "border_dispensary",
             ["n_active_q","exited","mean_pos_rev","first_yr_rev","brand_lifetime_n","chain_store",
              "n_disp_25mi_mean","n_brand_25mi_mean","samebrand_25mi_mean"]).round(3).to_string(index=False))

# ============================================================ D. Singleton Table 2 analogue
p("\n" + "="*90)
p("D.  SINGLETON TABLE 2 ANALOGUE -- EXITERS vs PERSISTERS  (storefront level)")
p("="*90)
sx["exited"] = sx["exited"].astype(int)
# persister = not exited AND not left-censored-only-briefly; keep all non-exiters as persisters
sx["age_at_last"] = sfq.groupby("dispensary_id")["age_q"].max().values
t2vars = ["border_dispensary","dist_stateline_mi","sf_nonleg_pop_75","sf_localpop_25",
          "n_disp_25mi_mean","n_brand_25mi_mean","samebrand_25mi_mean","brand_footprint_mean",
          "brand_lifetime_n","chain_store","mean_pos_rev","first_yr_rev","n_active_q","age_at_last"]
t2 = diff_means(sx, "exited", t2vars).rename(columns={"grp1_mean":"exiters","grp0_mean":"persisters"})
p(t2.round(2).to_string(index=False))
t2.to_csv(os.path.join(TBL, "tab_exiters_vs_persisters.csv"), index=False)
p(f"\nn exiters = {int(sx['exited'].sum())};  n persisters = {int((sx['exited']==0).sum())}")
p("NOTE: descriptive only -- motivates the exit model in 05, not causal.")

with open(LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines))
p("\nWROTE logs/04_descriptives.log + 4 tables")
