"""
Rebuild fig1 (premium by outcome and entry regime) and fig2 (state-specific
forest plot) under the adjacency border definition, replacing the 70mi
versions used in AppliedEconomics_dispensary_level.tex.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def load():
    pc = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    ps = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    adj = pd.read_csv(os.path.join(TBL, "border_adjacent_by_year.csv"), dtype={"fips5": str})[
        ["fips5", "year", "border_adjacent"]]
    pc = pc.drop(columns=["border_70mi_tv"], errors="ignore").merge(adj, on=["fips5", "year"], how="left")
    ps = ps.merge(adj, on=["fips5", "year"], how="left")
    ps["dispensary_per_1000"] = 1000 * pd.to_numeric(ps.dispensary_count, errors="coerce") / ps.population
    ps["log_dispensary_per_1000"] = np.log(ps.dispensary_per_1000)
    pc2 = pc[~((pc.state_abbr == "NM") & (pc.year == 2026)) & pc["revenue_per_capita"].notna()].copy()
    ps2 = ps[~((ps.state_abbr == "NM") & (ps.year == 2026)) & ps["revenue_per_store"].notna()].copy()
    return pc2, ps2


def fig1_premium_by_regime(pc2, ps2):
    specs = [("Revenue per store", ps2, "log_revenue_per_store"),
             ("Revenue per capita", pc2, "log_revenue_per_capita"),
             ("Dispensary density", ps2, "log_dispensary_per_1000")]
    rows = []
    for label, df, yvar in specs:
        for cap, regime in [(0, "Free"), (1, "Capped")]:
            d = df[df.entry_capped == cap].dropna(subset=[yvar, "border_adjacent", "log_population"]).copy()
            d["year"] = d.year.astype(str)
            formula = f"{yvar} ~ border_adjacent + log_population + C(year)" + (
                " + C(state_abbr)" if d.state_abbr.nunique() > 1 else "")
            m = smf.ols(formula, data=d).fit(cov_type="HC1")
            b, se = m.params.get("border_adjacent"), m.bse.get("border_adjacent")
            pct = (np.exp(b) - 1) * 100
            lo = (np.exp(b - 1.96 * se) - 1) * 100
            hi = (np.exp(b + 1.96 * se) - 1) * 100
            rows.append({"outcome": label, "regime": regime, "pct": pct, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)

    apply_healy_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    outcomes = [s[0] for s in specs]
    x = np.arange(len(outcomes))
    width = 0.35
    for i, (regime, color) in enumerate([("Free", PALETTE["blue"]), ("Capped", PALETTE["red"])]):
        sub = df[df.regime == regime].set_index("outcome").loc[outcomes]
        err = np.array([sub["pct"] - sub["lo"], sub["hi"] - sub["pct"]])
        ax.bar(x + (i - 0.5) * width, sub["pct"], width, yerr=err, capsize=4, label=regime, color=color, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes)
    ax.set_ylabel("Border premium (%)")
    ax.set_title("Border premium by outcome and entry regime")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_premium_by_regime_adjacency.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig1_premium_by_regime_adjacency.png")


def fig2_state_forest(pc2, ps2):
    states = ["CO", "WA", "OR", "MT", "NM"]
    specs = [("Revenue per store", ps2, "log_revenue_per_store"),
             ("Revenue per capita", pc2, "log_revenue_per_capita"),
             ("Dispensary density", ps2, "log_dispensary_per_1000")]
    rows = []
    for st in states:
        for label, df, yvar in specs:
            d = df[df.state_abbr == st].dropna(subset=[yvar, "border_adjacent", "log_population"]).copy()
            if d.border_adjacent.nunique() < 2:
                continue
            d["year"] = d.year.astype(str)
            yrs = d.year.nunique()
            formula = f"{yvar} ~ border_adjacent + log_population" + (" + C(year)" if yrs > 1 else "")
            m = smf.ols(formula, data=d).fit(cov_type="HC1")
            b, se = m.params.get("border_adjacent"), m.bse.get("border_adjacent")
            pct = (np.exp(b) - 1) * 100
            lo = (np.exp(b - 1.96 * se) - 1) * 100
            hi = (np.exp(b + 1.96 * se) - 1) * 100
            rows.append({"state": st, "outcome": label, "pct": pct, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)

    apply_healy_style()
    outcomes = [s[0] for s in specs]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, outcome in zip(axes, outcomes):
        sub = df[df.outcome == outcome].set_index("state").reindex(states).dropna()
        y = np.arange(len(sub))
        ax.errorbar(sub["pct"], y, xerr=[sub["pct"] - sub["lo"], sub["hi"] - sub["pct"]],
                     fmt="o", capsize=4, color=PALETTE["blue"])
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(np.arange(len(states)))
        ax.set_yticklabels(states)
        ax.set_title(outcome, fontsize=11)
        ax.set_xlabel("Border premium (%)")
    fig.suptitle("State-specific border coefficients by outcome, with 95% CI (HC1)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_state_forest_adjacency.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig2_state_forest_adjacency.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pc2, ps2 = load()
    fig1_premium_by_regime(pc2, ps2)
    fig2_state_forest(pc2, ps2)
