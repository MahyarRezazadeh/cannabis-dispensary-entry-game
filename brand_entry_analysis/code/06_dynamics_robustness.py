"""
06_dynamics_robustness.py  --  Venous4 / brand_entry_analysis

  §7  DYNAMICS   -- border vs interior market structure over 2022Q2..2026Q2 (figures)
  §8  ROBUSTNESS -- the concentration/composition result + the border x chain exit result

Writes: logs/06_dynamics_robustness.log
        figures/fig_dynamics_*.png
        robustness/tab_robustness_concentration.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import ols_cluster, stars

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TBL, FIG, ROB = (os.path.join(BASE, d) for d in ("tables", "figures", "robustness"))
LOG = os.path.join(BASE, "logs", "06_dynamics_robustness.log")
_lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); _lines.append(s)
asinh = lambda x: np.log(x + np.sqrt(x**2 + 1))

zq  = pd.read_csv(os.path.join(TBL, "panel_zip_quarter.csv"), dtype={"zip5": str})
sfq = pd.read_csv(os.path.join(TBL, "panel_storefront_quarter.csv"), dtype={"zip5": str})
bm  = pd.read_csv(os.path.join(TBL, "brand_master.csv"))
sfm = pd.read_csv(os.path.join(TBL, "storefront_master.csv"), dtype={"zip5": str})
zx  = pd.read_csv(os.path.join(TBL, "zip_xwalk.csv"), dtype={"zip5": str, "county_fips": str})

QLAB = {q: f"{q//4}Q{q%4+1}" for q in range(zq.qt.min(), zq.qt.max()+1)}

# ============================================================ §7 DYNAMICS
p("="*90); p("§7  DYNAMICS -- border vs interior, by quarter"); p("="*90)
z = zq[zq["border_z"].notna()].copy()
z["revenue_per_store"] = z["total_revenue"] / z["n_stores"]
series = {
    "n_brands": "mean # brands per ZIP",
    "n_stores": "mean # storefronts per ZIP",
    "hhi_10000": "mean HHI (0-10,000)",
    "cr1": "mean largest-brand revenue share",
    "indep_store_share": "mean independent-store share",
    "multiloc_now_store_share": "mean chain-store share",
    "brand_entry_rate": "mean brand entry rate",
    "brand_exit_rate": "mean brand exit rate",
    "revenue_per_store": "mean revenue per store ($)",
}
agg = z.groupby(["border_z", "qt"])[list(series)].mean().reset_index()
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, (k, lab) in zip(axes.flat, series.items()):
    for bz, name, c in [(1, "border ZIPs", "#c0392b"), (0, "interior ZIPs", "#2c3e50")]:
        s = agg[agg["border_z"] == bz]
        ax.plot(s["qt"], s[k], marker="o", ms=3, label=name, color=c)
    ax.set_title(lab, fontsize=10)
    ax.set_xticks(list(QLAB)[::2]); ax.set_xticklabels([QLAB[q] for q in list(QLAB)[::2]], rotation=45, fontsize=7)
    ax.grid(alpha=.25)
axes.flat[0].legend(fontsize=8)
fig.suptitle("New Mexico dispensary market structure: border vs interior ZIP codes, 2022Q2-2026Q2", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_dynamics_market_structure.png"), dpi=130)
plt.close(fig)
p("wrote figures/fig_dynamics_market_structure.png")

# statewide storefront / brand counts + entry/exit flows over time
sf_active = sfq[sfq["active"] == 1]
by_q = sf_active.groupby("qt").agg(n_stores=("dispensary_id","nunique"),
                                   n_brands=("brand_id","nunique")).reset_index()
ent_q = sfq[sfq["entry_this_q"] == 1].groupby("qt").size().reindex(by_q["qt"], fill_value=0)
exi_q = sfq[sfq["exit_next"] == 1].groupby("qt").size().reindex(by_q["qt"], fill_value=0)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(by_q["qt"], by_q["n_stores"], marker="o", label="active storefronts")
ax[0].plot(by_q["qt"], by_q["n_brands"], marker="s", label="active brands")
ax[0].legend(); ax[0].set_title("Statewide active storefronts & brands"); ax[0].grid(alpha=.25)
ax[1].bar(by_q["qt"] - .2, ent_q.values, width=.4, label="storefront entries", color="#27ae60")
ax[1].bar(by_q["qt"] + .2, exi_q.values, width=.4, label="storefront exits", color="#c0392b")
ax[1].legend(); ax[1].set_title("Storefront entry / exit flow"); ax[1].grid(alpha=.25)
for a in ax:
    a.set_xticks(list(QLAB)[::2]); a.set_xticklabels([QLAB[q] for q in list(QLAB)[::2]], rotation=45, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_dynamics_statewide.png"), dpi=130); plt.close(fig)
p("wrote figures/fig_dynamics_statewide.png")

# when did the composition gap emerge? border-interior gap in log(n_brands) and HHI, by quarter
gap = agg.pivot(index="qt", columns="border_z")
p("\nborder - interior gap by quarter (n_brands / HHI / indep_share):")
for q in QLAB:
    r = agg[agg.qt == q]
    if len(r) == 2:
        b, i = r[r.border_z == 1].iloc[0], r[r.border_z == 0].iloc[0]
        p(f"  {QLAB[q]:7s}  dNbrands={b.n_brands-i.n_brands:+5.2f}  dHHI={b.hhi_10000-i.hhi_10000:+7.0f}  "
          f"dIndepShare={b.indep_store_share-i.indep_store_share:+.3f}")

# ============================================================ §8 ROBUSTNESS
p("\n" + "="*90); p("§8  ROBUSTNESS -- composition/concentration result (headline)"); p("="*90)

# rebuild ZIP panel with extra border defs + all-368 backfill
zx["border_50"] = ((zx["dist_stateline_mi"] <= 50) & (zx["nonleg_pop_75"] >= 1000)).astype(int)
zx["border_100"] = ((zx["dist_stateline_mi"] <= 100) & (zx["nonleg_pop_75"] >= 1000)).astype(int)
zx["asinh_nl25"] = asinh(zx["nonleg_pop_25"]); zx["asinh_nl50"] = asinh(zx["nonleg_pop_50"])
zx["asinh_nl75"] = asinh(zx["nonleg_pop_75"]); zx["asinh_nl100"] = asinh(zx["nonleg_pop_100"])
zj = zq.drop(columns=[c for c in ["border_z","dist_stateline_mi","log_dist_stateline","zip_pop",
      "log_zip_pop","county_fips","county_name","nonleg_pop_25","nonleg_pop_50","nonleg_pop_75",
      "nonleg_pop_100"] if c in zq.columns]).merge(zx, on="zip5", how="left")
zj["log_zip_pop"] = np.log(zj["zip_pop"].clip(lower=1))
zj["log_nbrands"] = np.log(zj["n_brands"])
zj["log_revenue"] = np.log(zj["total_revenue"].clip(lower=1))
zj["is_abq"] = (zj["county_fips"] == "35001").astype(int)
# brand-composition subcounts by ZIP-quarter
bm_multi = dict(zip(bm["brand_id"], (bm["n_storefronts"] >= 2).astype(int)))
sfa = sfq[sfq["active"] == 1].copy()
sfa["m"] = sfa["brand_id"].map(bm_multi)
sub = sfa.groupby(["zip5","qt"]).agg(n_indep_brands=("brand_id", lambda s: s[sfa.loc[s.index,"m"]==0].nunique()),
                                     n_chain_brands=("brand_id", lambda s: s[sfa.loc[s.index,"m"]==1].nunique())
                                     ).reset_index()
zj = zj.merge(sub, on=["zip5","qt"], how="left")

def run(df, y, treat, ctrl=("log_zip_pop",), extra="", label="", cluster="zip5"):
    rhs = [treat] + list(ctrl) + ["C(qt)"] + ([extra] if extra else [])
    try:
        _, t = ols_cluster(df, y, rhs, cluster)
        b, se, pv = t[treat]
        return dict(spec=label, y=y, treat=treat, b=b, se=se, p=pv, N=t["_N"])
    except Exception as ex:
        return dict(spec=label, y=y, treat=treat, b=np.nan, se=np.nan, p=np.nan, N=0, err=str(ex)[:60])

rob = []
BASE_SAMP = zj[zj["border_z"].notna()]
for y in ["log_nbrands","hhi_10000","cr1","indep_store_share","brand_entry_rate","log_revenue"]:
    rob.append(run(BASE_SAMP, y, "border_z", label="baseline (75mi border, +logpop, cluster ZIP)"))
    rob.append(run(zj[zj["border_50"].notna()], y, "border_50", label="border def = 50mi"))
    rob.append(run(zj[zj["border_100"].notna()], y, "border_100", label="border def = 100mi"))
    rob.append(run(BASE_SAMP, y, "asinh_nl25", label="continuous: asinh nonlegal pop 25mi"))
    rob.append(run(BASE_SAMP, y, "asinh_nl50", label="continuous: asinh nonlegal pop 50mi"))
    rob.append(run(BASE_SAMP, y, "asinh_nl100", label="continuous: asinh nonlegal pop 100mi"))
    rob.append(run(BASE_SAMP, y, "log_dist_stateline", label="continuous: log dist to nonlegal stateline"))
    rob.append(run(BASE_SAMP[BASE_SAMP["is_abq"] == 0], y, "border_z", label="exclude Albuquerque (Bernalillo)"))
    rob.append(run(BASE_SAMP[BASE_SAMP["zip_pop"] >= 2500], y, "border_z", label="exclude ZIPs <2,500 pop"))
    rob.append(run(BASE_SAMP[BASE_SAMP["qt"] <= 8096], y, "border_z", label="early market (2022Q2-2023Q4)"))
    rob.append(run(BASE_SAMP[BASE_SAMP["qt"] >= 8097], y, "border_z", label="mature market (2024Q1-2026Q2)"))
    rob.append(run(BASE_SAMP, y, "border_z", label="cluster by county", cluster="county_fips"))
    rob.append(run(BASE_SAMP, y, "border_z", extra="C(county_fips)", label="+ county fixed effects"))
robdf = pd.DataFrame(rob)
robdf.to_csv(os.path.join(ROB, "tab_robustness_concentration.csv"), index=False)
p("\n(coef on the treatment variable; * .10  ** .05  *** .01)\n")
for y in robdf["y"].unique():
    p(f"--- outcome: {y} ---")
    for r in robdf[robdf["y"] == y].itertuples():
        p(f"  {r.spec:44s} {r.b:>12.4f}{stars(r.p) if pd.notna(r.p) else '':<3} ({r.se:.4f})  N={r.N}")

# indep vs chain brand counts by border
p("\n-- # independent brands vs # chain brands per ZIP (border_z, +logpop) --")
for y in ["n_indep_brands","n_chain_brands"]:
    r = run(BASE_SAMP, y, "border_z", label=y)
    p(f"  {y:18s} border_z = {r['b']:+.3f}{stars(r['p'])} ({r['se']:.3f})  N={r['N']}")

# exclude top brands (recompute HHI etc. without them) -- quick version: drop their storefront-quarters
for drop_k, lab in [(1, "drop largest brand"), (3, "drop top-3 brands"), (5, "drop top-5 brands")]:
    rev_by_b = sfa.groupby("brand_id")["revenue"].sum().sort_values(ascending=False)
    drop_ids = set(rev_by_b.head(drop_k).index)
    s2 = sfa[~sfa["brand_id"].isin(drop_ids)].copy()
    rows = []
    for (zz, qt), g2 in s2.groupby(["zip5","qt"]):
        rev = g2.groupby("brand_id")["revenue"].sum(); sh = (rev/rev.sum()).values if rev.sum()>0 else np.array([1.])
        rows.append((zz, qt, g2["brand_id"].nunique(), float((sh**2).sum())*1e4))
    zz2 = pd.DataFrame(rows, columns=["zip5","qt","n_brands","hhi_10000"]).merge(
        zx[["zip5","border_z","zip_pop"]], on="zip5", how="left")
    zz2["log_zip_pop"] = np.log(zz2["zip_pop"].clip(lower=1)); zz2["log_nbrands"] = np.log(zz2["n_brands"])
    r1 = run(zz2[zz2.border_z.notna()], "log_nbrands", "border_z", label=lab)
    r2 = run(zz2[zz2.border_z.notna()], "hhi_10000", "border_z", label=lab)
    p(f"  {lab:22s}  log_nbrands b={r1['b']:+.3f}{stars(r1['p'])}   hhi b={r2['b']:+.0f}{stars(r2['p'])}")

# ---- robustness for the border x chain exit result ----
p("\n" + "-"*60)
p("border x chain EXIT interaction robustness (storefront-quarter LPM, exit sample)")
e = sfq[sfq["in_exit_sample"] == 1].copy()
e = e.merge(bm[["brand_id","n_storefronts"]].rename(columns={"n_storefronts":"blt"}), on="brand_id", how="left")
e["chain"] = (e["blt"] >= 2).astype(int)
e["bxc"] = e["border_dispensary"] * e["chain"]
e["log_age"] = np.log(e["age_q"]+1); e["log_rev"] = np.log(e["revenue"].clip(lower=1))
e["log_localpop"] = np.log(e["sf_localpop_25"].clip(lower=1)); e["clu"] = e["zip5"].fillna(e["fips5"].astype(str))
e["oc"] = e["ownership_change_flag"].astype(int)
for lab, d in [("baseline", e),
               ("drop ownership-change storefronts", e[e["oc"] == 0]),
               ("exclude Albuquerque", e[e["fips5"] != 35001]),
               ("mature market only (qt>=8097)", e[e["qt"] >= 8097])]:
    try:
        _, t = ols_cluster(d, "exit_next",
            ["border_dispensary","bxc","chain","n_disp_25mi","n_brand_25mi","log_age","log_rev","log_localpop","C(qt)"], "clu")
        p(f"  {lab:36s} border={t['border_dispensary'][0]:+.4f}{stars(t['border_dispensary'][2])}  "
          f"border_x_chain={t['bxc'][0]:+.4f}{stars(t['bxc'][2])}  N={t['_N']}")
    except Exception as ex:
        p(f"  {lab}: ERR {ex}")

with open(LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines))
p("\nDONE -- logs/06_dynamics_robustness.log, 2 figures, tab_robustness_concentration.csv")
