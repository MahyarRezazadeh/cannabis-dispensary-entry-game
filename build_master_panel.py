"""
Build the master pooled panels for the reframed paper (border premium x entry-regime
allocation) across all five revenue states: CO, WA, OR, MT, NM.

Two outputs:
  tables/master_percapita_panel.csv  -- every state-county-year, revenue per capita
  tables/master_perstore_panel.csv   -- every state-county-year (OR/MT only as a
                                         2025 cross-section, per their data constraints)

Entry-regime coding (time-varying, state-year, CAPPED=1 / free=0). Revised per
explicit instruction: where LOCAL GOVERNMENT determines whether licensing can
happen at all (an opt-out or opt-in requirement that functions as a binding
entry restriction), the state is coded CAPPED even absent a state-level
numerical cap -- a state law saying "no cap" is not the same as free entry if
most of the state's geography can still be closed to retail by a local vote.
  CO: CAPPED the entire panel. Amendment 64 imposes no state cap, but the
      Freeman et al. county-level census found 48.4% of CO counties banned
      all marijuana facilities as of 2019 -- local government, not the state,
      is the binding constraint on entry across most of the state.
  WA: CAPPED the entire panel (I-502 state cap; level changes 334->556->
      556+52 equity slots, but never becomes free-entry; local bans exist too
      but are secondary to the state cap here).
  MT: CAPPED the entire panel. HB 701's local opt-in requirement (counties
      that voted against Initiative 190 in Nov 2020 need affirmative local
      government approval before a dispensary can locate there) is a binding
      local-government entry restriction in place from the 2022-01-01 launch
      of retail sales, not just from the May-2025 "Freeze the Footprint"
      moratorium -- so MT is capped for its whole panel, not just 2025.
  OR: kept time-varying (free 2015-2021, capped 2022-2025) for the POOLED
      spec, but per instruction analyzed SEPARATELY as its own before/after
      robustness check (analyze_oregon_regime_switch.py) rather than folded
      into the pooled capped/free contrast alone.
  NM: FREE the entire panel. NMSA 1978 Section 26-2C-6(I) is an explicit
      statutory anti-cap provision, and local bans are preempted -- the one
      state in the sample where local government does NOT determine
      licensing.
"""
import pandas as pd
import numpy as np
import os

TBL = "tables"


def entry_capped(state, year):
    year = int(year)
    if state == "CO":
        return 1
    if state == "WA":
        return 1
    if state == "OR":
        return 0 if year <= 2021 else 1
    if state == "MT":
        return 1
    if state == "NM":
        return 0
    raise ValueError(state)


def local_barrier(state, year):
    """
    Groups states by WHICH LEVEL of government imposes the binding entry
    restriction, rather than by whether one exists.
      local_barrier=1: no state cap, but local (county/city) government
        directly controls whether a dispensary can operate -- Colorado
        (county opt-out bans) and Montana (county opt-in requirement).
      local_barrier=0: any restriction, if present, comes from the state
        itself, or no restriction exists at all -- Washington (state cap,
        entire panel), Oregon (state freeze/formula cap 2022-2025, free
        2015-2021), and New Mexico (no restriction at either level; local
        bans are statutorily preempted, NMSA 1978 Section 26-2C-12(B)(2)).
    """
    year = int(year)
    if state == "CO":
        return 1
    if state == "MT":
        return 1
    if state == "WA":
        return 0
    if state == "OR":
        return 0
    if state == "NM":
        return 0
    raise ValueError(state)


def entry_capped_strict(state, year):
    """
    Alternative, narrower classification: capped = 1 only where a STATE-LEVEL
    numerical license ceiling exists, excluding cases where the binding
    restriction is purely local-government control (Colorado's county bans,
    Montana's opt-in requirement). Requested as a robustness check against the
    paper's primary entry_capped() classification.
      CO: free (no state cap; Amendment 64 leaves licensing entirely to
          local governments -- excluded here even though local bans bind in
          practice, unlike the primary classification).
      MT: free (HB 701 sets no numerical ceiling; entry barriers are
          medical-conversion-only + county opt-in, not a state cap number --
          excluded here for the same reason as Colorado).
      WA: capped the entire panel (I-502 statewide numerical cap, 334->556->
          556+52, always a hard state number).
      OR: free 2015-2021, capped 2022-2025 (same split as the primary
          classification -- bundles the Jan-2022 application freeze with the
          Mar-2024 HB 4121 per-capita formula cap into one "restricted"
          period, since the freeze already stops new entry even before the
          formula cap took effect).
      NM: free the entire panel (statutory anti-cap provision, unchanged).
    """
    year = int(year)
    if state == "CO":
        return 0
    if state == "MT":
        return 0
    if state == "WA":
        return 1
    if state == "OR":
        return 0 if year <= 2021 else 1
    if state == "NM":
        return 0
    raise ValueError(state)


def load_common(path, year_col):
    df = pd.read_csv(path)
    df = df.rename(columns={year_col: "year"})
    if "revenue_per_capita" not in df.columns:
        df["revenue_per_capita"] = df["total_sales"] / df["population"]
    return df


def load_timevarying_dists():
    co = pd.read_csv(os.path.join(TBL, "timevarying_dist_co.csv"), dtype={"fips5": str}).set_index("fips5")
    wa = pd.read_csv(os.path.join(TBL, "timevarying_dist_wa.csv"), dtype={"fips5": str}).set_index("fips5")
    orr = pd.read_csv(os.path.join(TBL, "timevarying_dist_or.csv"), dtype={"fips5": str}).set_index("fips5")
    return co, wa, orr


def apply_timevarying_border(df):
    """
    Recompute dist_nonlegal_mi / border_70mi as of each observation's own year,
    for the three states with a neighbor-legalization transition inside their
    panel window: CO (NM legal 2022-04-01), WA (OR legal 2015-10-01, WA fiscal
    year runs Jul-Jun so only FY<=2015 is entirely pre-transition), OR (CA
    legal 2018-01-01 and NV legal 2017-07-01, both bearing only on OR's 2017
    observations). All other state-year cells keep the existing static value.
    """
    co_d, wa_d, or_d = load_timevarying_dists()
    df = df.copy()
    df["dist_nonlegal_mi_tv"] = df["dist_nonlegal_mi"]

    is_co = df["state_abbr"] == "CO"
    co_idx = df.loc[is_co, "fips5"].map(co_d["dist_to_NM"])
    pre2022 = is_co & (df["year"] < 2022)
    df.loc[pre2022, "dist_nonlegal_mi_tv"] = np.minimum(
        df.loc[pre2022, "dist_nonlegal_mi"], co_idx.loc[pre2022].values)

    is_wa = df["state_abbr"] == "WA"
    wa_idx = df.loc[is_wa, "fips5"].map(wa_d["dist_to_OR"])
    pre_or_legal = is_wa & (df["year"] <= 2015)
    df.loc[pre_or_legal, "dist_nonlegal_mi_tv"] = np.minimum(
        df.loc[pre_or_legal, "dist_nonlegal_mi"], wa_idx.loc[pre_or_legal].values)

    is_or = df["state_abbr"] == "OR"
    or_ca = df.loc[is_or, "fips5"].map(or_d["dist_to_CA"])
    or_nv = df.loc[is_or, "fips5"].map(or_d["dist_to_NV"])
    y2017 = is_or & (df["year"] == 2017)
    df.loc[y2017, "dist_nonlegal_mi_tv"] = np.minimum(
        df.loc[y2017, "dist_nonlegal_mi"],
        np.minimum(or_ca.loc[y2017].values, or_nv.loc[y2017].values))

    df["border_70mi_tv"] = (df["dist_nonlegal_mi_tv"] <= 70).astype(int)
    n_changed = (df["border_70mi_tv"] != df["border_70mi"]).sum()
    print(f"  time-varying border reclassification: {n_changed} of {len(df)} rows flipped vs. static 2025 snapshot")
    return df


def main():
    cowa = load_common(os.path.join(TBL, "revenue_perstore_border70_panel_CORRECTED.csv"), "fy_year")
    or_pc = load_common(os.path.join(TBL, "oregon_revenue_percapita_panel.csv"), "fy_year")
    or_ps = load_common(os.path.join(TBL, "oregon_revenue_perstore_cross2025.csv"), "fy_year")
    mt_pc = load_common(os.path.join(TBL, "montana_revenue_percapita_panel.csv"), "fy_year")
    mt_ps = load_common(os.path.join(TBL, "montana_revenue_perstore_cross.csv"), "fy_year")
    nm = load_common(os.path.join(TBL, "nm_revenue_panel.csv"), "year")

    keep = ["fips5", "state_abbr", "county_name", "year", "population", "land_area_sqmi",
            "dist_nonlegal_mi", "border_70mi", "total_sales", "revenue_per_capita",
            "dispensary_count", "revenue_per_store"]

    # ---------- per-capita master panel: every state-county-year available ----------
    frames = []
    for name, df in [("CO+WA", cowa), ("OR", or_pc), ("MT", mt_pc), ("NM", nm)]:
        d = df.copy()
        for c in keep:
            if c not in d.columns:
                d[c] = pd.NA
        frames.append(d[keep])
    percapita = pd.concat(frames, ignore_index=True)
    percapita["fips5"] = percapita["fips5"].astype(str).str.zfill(5)
    percapita = apply_timevarying_border(percapita)
    percapita["border_70mi_static"] = percapita["border_70mi"]
    percapita["border_70mi"] = percapita["border_70mi_tv"]
    percapita["entry_capped"] = percapita.apply(lambda r: entry_capped(r["state_abbr"], r["year"]), axis=1)
    percapita["entry_capped_strict"] = percapita.apply(
        lambda r: entry_capped_strict(r["state_abbr"], r["year"]), axis=1)
    percapita["local_barrier"] = percapita.apply(
        lambda r: local_barrier(r["state_abbr"], r["year"]), axis=1)
    percapita["log_revenue_per_capita"] = pd.to_numeric(percapita["revenue_per_capita"], errors="coerce").apply(
        lambda x: None if x is None or x <= 0 else __import__("math").log(x))
    percapita["log_population"] = percapita["population"].apply(lambda x: __import__("math").log(x))
    percapita.to_csv(os.path.join(TBL, "master_percapita_panel.csv"), index=False)
    print(f"master_percapita_panel.csv: {percapita.shape}, states={sorted(percapita.state_abbr.unique())}, "
          f"years={sorted(percapita.year.unique())}")

    # ---------- per-store master panel: CO+WA+NM as true panels, OR+MT as 2025 cross-section ----------
    frames = []
    for name, df in [("CO+WA", cowa), ("NM", nm), ("OR", or_ps), ("MT", mt_ps)]:
        d = df.copy()
        d = d[d["dispensary_count"].notna() & (pd.to_numeric(d["dispensary_count"], errors="coerce") > 0)]
        d["revenue_per_store"] = d["total_sales"] / d["dispensary_count"]
        for c in keep:
            if c not in d.columns:
                d[c] = pd.NA
        frames.append(d[keep])
    perstore = pd.concat(frames, ignore_index=True)
    perstore["fips5"] = perstore["fips5"].astype(str).str.zfill(5)
    perstore = apply_timevarying_border(perstore)
    perstore["border_70mi_static"] = perstore["border_70mi"]
    perstore["border_70mi"] = perstore["border_70mi_tv"]
    perstore["entry_capped"] = perstore.apply(lambda r: entry_capped(r["state_abbr"], r["year"]), axis=1)
    perstore["entry_capped_strict"] = perstore.apply(
        lambda r: entry_capped_strict(r["state_abbr"], r["year"]), axis=1)
    perstore["local_barrier"] = perstore.apply(
        lambda r: local_barrier(r["state_abbr"], r["year"]), axis=1)
    perstore["log_revenue_per_store"] = pd.to_numeric(perstore["revenue_per_store"], errors="coerce").apply(
        lambda x: None if x is None or x <= 0 else __import__("math").log(x))
    perstore["log_population"] = perstore["population"].apply(lambda x: __import__("math").log(x))
    perstore.to_csv(os.path.join(TBL, "master_perstore_panel.csv"), index=False)
    print(f"master_perstore_panel.csv: {perstore.shape}, states={sorted(perstore.state_abbr.unique())}, "
          f"years={sorted(perstore.year.unique())}")

    print("\n--- Entry-regime coverage (per-capita panel), state x capped x n years ---")
    print(percapita.groupby(["state_abbr", "entry_capped"])["year"].agg(["min", "max", "nunique"]))


if __name__ == "__main__":
    main()
