import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def add_regime_D(df):
    d = df.copy()
    is_local = d["state_abbr"].isin(["CO", "MT"])
    is_free = (d["state_abbr"] == "NM") | ((d["state_abbr"] == "OR") & (d["year"] <= 2021))
    d["regime_D"] = np.select([is_local, is_free], [1, 0], default=np.nan)
    return d[d["regime_D"].notna()].copy()


def load_with_captive(panel_file):
    df = pd.read_csv(os.path.join(TBL, panel_file), dtype={"fips5": str})
    cap = pd.read_csv(os.path.join(TBL, "captive_population_all5states.csv"), dtype={"fips5": str})
    df = df.merge(cap[["fips5", "state_abbr", "captive_population"]], on=["fips5", "state_abbr"], how="left")
    df["captive_population"] = df["captive_population"].fillna(0)
    df["log1p_captive_pop"] = np.log1p(df["captive_population"])
    df["log_dist"] = np.log(df["dist_nonlegal_mi"])
    return add_regime_D(df)


def residualize_and_plot(df, yvar, title, outfile, ylabel):
    d = df.dropna(subset=[yvar, "log_dist", "log_population", "log1p_captive_pop"]).copy()
    d = d[np.isfinite(d["log_dist"])].copy()
    d["year"] = d["year"].astype(str)

    # fit the SAME joint interacted model as analyze_continuous_distance_groupD.py
    # (one regression, not a separate per-group refit -- avoids the residualize-then-
    # refit mismatch found when sanity-checking this figure against the raw correlations)
    within_var = d.groupby("state_abbr")["regime_D"].nunique()
    state_fe = not (within_var <= 1).all()
    fe = " + C(year) + C(state_abbr)" if state_fe else " + C(year)"
    formula = f"{yvar} ~ log_dist * regime_D + log_population + log1p_captive_pop" + fe
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})

    apply_healy_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for regime, color, label in [(1, PALETTE["gold"], "Local barrier (CO+MT)"),
                                  (0, PALETTE["blue"], "Free (NM + OR 2015-2021)")]:
        sub = d[d.regime_D == regime].copy()
        # residualize on everything EXCEPT log_dist and regime (population, captive pop, year/state
        # FE, and the regime main effect), using the joint model's own coefficients, so the
        # scatter/line reflects the fitted log_dist slope for this group exactly
        pred_full = m.predict(sub)
        dist_effect = (m.params.get("log_dist", 0) + (m.params.get("log_dist:regime_D", 0) if regime == 1 else 0)) * sub["log_dist"]
        resid = sub[yvar] - (pred_full - dist_effect)
        ax.scatter(sub["log_dist"], resid, color=color, alpha=0.35, s=18)
        xs = np.linspace(sub["log_dist"].min(), sub["log_dist"].max(), 50)
        intercept = m.params["Intercept"] + (m.params.get("regime_D", 0) if regime == 1 else 0)
        slope = m.params.get("log_dist", 0) + (m.params.get("log_dist:regime_D", 0) if regime == 1 else 0)
        ys = intercept + slope * xs
        ax.plot(xs, ys, color=color, linewidth=2.2, label=f"{label} (slope={slope:+.3f})")
    ax.set_xlabel("log(distance to nearest nonlegal-state county, miles)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, outfile))
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    percapita = load_with_captive("master_percapita_panel.csv")
    perstore = load_with_captive("master_perstore_panel.csv")
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    residualize_and_plot(percapita, "log_revenue_per_capita",
                          "Revenue per capita against distance to the nonlegal border\n"
                          "Local barrier vs. free entry (Grouping D), residualized on population,\n"
                          "captive population, and year fixed effects",
                          "fig29_distgradient_groupD_percapita.png",
                          "log(revenue per capita), residualized")
    residualize_and_plot(perstore, "log_revenue_per_store",
                          "Revenue per store against distance to the nonlegal border\n"
                          "Local barrier vs. free entry (Grouping D), residualized",
                          "fig30_distgradient_groupD_perstore.png",
                          "log(revenue per store), residualized")
    residualize_and_plot(perstore, "log_dispensary_per_1000",
                          "Dispensary density against distance to the nonlegal border\n"
                          "Local barrier vs. free entry (Grouping D), residualized",
                          "fig31_distgradient_groupD_density.png",
                          "log(dispensaries per 1,000 residents), residualized")


if __name__ == "__main__":
    main()
