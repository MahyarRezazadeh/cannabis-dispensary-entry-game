"""
Two remaining descriptive figures:
  fig5: distance gradient -- revenue per store against continuous distance to
        the nearest nonlegal-state county (residualized binned scatter)
  fig6: captive nonlegal-side population -- revenue per store against the
        total population of nonlegal-state counties within 70 miles, for
        every border county across all five sample states (not just CO+WA,
        which is as far as the original project took this)
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"
CENTROIDS_FILE = os.path.join("data", "county_centroids.dta")
POP_FILE = os.path.join("data", "co_est2023_alldata.csv")

# nonlegal-neighbor state list used for the captive-population construction
# (current/latest classification -- this is a cross-sectional descriptive
# figure, not a re-run of the time-varying panel regression)
NONLEGAL_NEIGHBORS = {
    "CO": ["KS", "NE", "OK", "UT", "WY"],
    "WA": ["ID"],
    "OR": ["ID"],
    "MT": ["ID", "ND", "SD", "WY"],
    "NM": ["OK", "TX"],
}


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def fig_distance_gradient(perstore):
    apply_healy_style()
    d = perstore.dropna(subset=["log_revenue_per_store", "dist_nonlegal_mi", "log_population"]).copy()
    d = d[d["dist_nonlegal_mi"] > 0]
    d["log_dist"] = np.log(d["dist_nonlegal_mi"])
    d["year"] = d["year"].astype(str)

    m = smf.ols("log_revenue_per_store ~ log_dist + log_population + C(year) + C(state_abbr)", data=d).fit(cov_type="HC1")
    b, p = m.params["log_dist"], m.pvalues["log_dist"]

    # residualize on everything except log_dist, then bin
    resid_y = d["log_revenue_per_store"] - m.predict(d) + b * d["log_dist"]
    resid_x = d["log_dist"]
    d["_ry"], d["_rx"] = resid_y, resid_x
    d["bin"] = pd.qcut(d["_rx"], 10, duplicates="drop")
    binned = d.groupby("bin", observed=True).agg(x=("_rx", "mean"), y=("_ry", "mean"), n=("_ry", "size")).reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(binned["x"], binned["y"], s=binned["n"] * 3, color=PALETTE["blue"], alpha=0.85, zorder=3)
    xs = np.linspace(d["_rx"].min(), d["_rx"].max(), 50)
    ax.plot(xs, b * xs + (resid_y - b * resid_x).mean(), color=PALETTE["red"], linewidth=1.8, zorder=2)
    ax.set_xlabel("log(distance to nearest nonlegal-state county), residualized")
    ax.set_ylabel("log(revenue per store), residualized")
    ax.set_title(f"Distance gradient, pooled 5-state sample\nslope $b$={b:+.3f}, $p$={p:.3f} (HC1)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_distance_gradient.png"))
    plt.close(fig)
    print(f"Wrote fig5_distance_gradient.png (slope={b:.4f}, p={p:.4f}, n={int(m.nobs)})")


def build_captive_population():
    cc = pd.read_stata(CENTROIDS_FILE)
    cc["fips5"] = cc["fips5"].astype(str).str.zfill(5)
    pop = pd.read_csv(POP_FILE, encoding="latin1")
    pop["fips5"] = pop["STATE"].astype(str).str.zfill(2) + pop["COUNTY"].astype(str).str.zfill(3)
    pop = pop[pop["COUNTY"] != 0][["fips5", "POPESTIMATE2023"]]
    cc = cc.merge(pop, on="fips5", how="left")

    rows = []
    for state, neighbors in NONLEGAL_NEIGHBORS.items():
        home = cc[cc.state_abbr == state]
        nonlegal = cc[cc.state_abbr.isin(neighbors)].dropna(subset=["POPESTIMATE2023"])
        for _, hrow in home.iterrows():
            d = haversine_miles(hrow["lat"], hrow["lon"], nonlegal["lat"].values, nonlegal["lon"].values)
            captive_pop = nonlegal.loc[d <= 70, "POPESTIMATE2023"].sum()
            rows.append({"fips5": hrow["fips5"], "state_abbr": state, "captive_population": captive_pop})
    return pd.DataFrame(rows)


def fig_captive_population(perstore):
    apply_healy_style()
    cap = build_captive_population()
    d = perstore.merge(cap, on=["fips5", "state_abbr"], how="left")
    d = d.dropna(subset=["log_revenue_per_store", "captive_population", "log_population"]).copy()
    d = d[d["captive_population"] > 0]
    d["log_captive_pop"] = np.log(d["captive_population"])
    d["year"] = d["year"].astype(str)

    m = smf.ols("log_revenue_per_store ~ log_captive_pop + log_population + C(year) + C(state_abbr)",
                data=d).fit(cov_type="HC1")
    m_cl = smf.ols("log_revenue_per_store ~ log_captive_pop + log_population + C(year) + C(state_abbr)",
                   data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    b, p, pcl = m.params["log_captive_pop"], m.pvalues["log_captive_pop"], m_cl.pvalues["log_captive_pop"]

    resid_y = d["log_revenue_per_store"] - m.predict(d) + b * d["log_captive_pop"]
    resid_x = d["log_captive_pop"]
    d["_ry"], d["_rx"] = resid_y, resid_x
    nbins = min(8, d["_rx"].nunique())
    d["bin"] = pd.qcut(d["_rx"], nbins, duplicates="drop")
    binned = d.groupby("bin", observed=True).agg(x=("_rx", "mean"), y=("_ry", "mean"), n=("_ry", "size")).reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(binned["x"], binned["y"], s=binned["n"] * 3, color=PALETTE["gold"], alpha=0.9, zorder=3)
    xs = np.linspace(d["_rx"].min(), d["_rx"].max(), 50)
    ax.plot(xs, b * xs + (resid_y - b * resid_x).mean(), color=PALETTE["dark"], linewidth=1.8, zorder=2)
    ax.set_xlabel("log(population within 70mi on the nonlegal side), residualized")
    ax.set_ylabel("log(revenue per store), residualized")
    ax.set_title(f"Revenue per store vs. nonlegal-side captive population\nslope $b$={b:+.3f}, "
                 f"$p_{{HC1}}$={p:.3f}, $p_{{cl}}$={pcl:.3f}, all 5 states")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_captive_population.png"))
    plt.close(fig)
    print(f"Wrote fig6_captive_population.png (slope={b:.4f}, p_HC1={p:.4f}, p_cl={pcl:.4f}, n={int(m.nobs)})")
    cap.to_csv(os.path.join(TBL, "captive_population_all5states.csv"), index=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    fig_distance_gradient(perstore)
    fig_captive_population(perstore)


if __name__ == "__main__":
    main()
