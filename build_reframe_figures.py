import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def border_coef_by_regime(df, yvar):
    """Return (b_free, se_free, p_free, n_free, b_capped, se_capped, p_capped, n_capped)."""
    out = {}
    for regime, label in [(0, "free"), (1, "capped")]:
        d = df[df.entry_capped == regime].dropna(subset=[yvar, "border_70mi", "log_population"]).copy()
        if d["border_70mi"].nunique() < 2:
            out[label] = (np.nan, np.nan, np.nan, len(d))
            continue
        d["year"] = d["year"].astype(str)
        yrs = d["year"].nunique()
        formula = f"{yvar} ~ border_70mi + log_population" + (" + C(year)" if yrs > 1 else "")
        m = smf.ols(formula, data=d).fit(cov_type="HC1")
        b = m.params["border_70mi"]
        se = m.bse["border_70mi"]
        p = m.pvalues["border_70mi"]
        out[label] = (b, se, p, int(m.nobs))
    return out


def fig_premium_by_regime(perstore, percapita):
    apply_healy_style()
    perstore = perstore.copy()
    perstore["log_dispensary_per_1000"] = np.log(1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    specs = [
        ("Revenue per store", perstore, "log_revenue_per_store"),
        ("Revenue per capita", percapita, "log_revenue_per_capita"),
        ("Dispensary density", perstore, "log_dispensary_per_1000"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(specs))
    width = 0.32

    free_pct, free_err, capped_pct, capped_err = [], [], [], []
    for label, df, yvar in specs:
        res = border_coef_by_regime(df, yvar)
        bf, sef, pf, nf = res["free"]
        bc, sec, pc, nc = res["capped"]
        free_pct.append((np.exp(bf) - 1) * 100 if pd.notna(bf) else 0)
        free_err.append(((np.exp(bf + 1.96 * sef) - 1) * 100 - (np.exp(bf) - 1) * 100) if pd.notna(bf) else 0)
        capped_pct.append((np.exp(bc) - 1) * 100 if pd.notna(bc) else 0)
        capped_err.append(((np.exp(bc + 1.96 * sec) - 1) * 100 - (np.exp(bc) - 1) * 100) if pd.notna(bc) else 0)

    ax.bar(x - width / 2, free_pct, width, yerr=free_err, label="Free entry", color=PALETTE["blue"], capsize=3)
    ax.bar(x + width / 2, capped_pct, width, yerr=capped_err, label="Capped entry", color=PALETTE["red"], capsize=3)
    ax.axhline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in specs])
    ax.set_ylabel("Border premium (%)")
    ax.set_title("Border premium by outcome and entry regime")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_premium_by_regime.png"))
    plt.close(fig)
    print("Wrote fig1_premium_by_regime.png")


def fig_state_forest(perstore, percapita):
    apply_healy_style()
    perstore = perstore.copy()
    perstore["log_dispensary_per_1000"] = np.log(1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    outcomes = [("Revenue/store", perstore, "log_revenue_per_store", PALETTE["red"]),
                ("Revenue/capita", percapita, "log_revenue_per_capita", PALETTE["blue"]),
                ("Density", perstore, "log_dispensary_per_1000", PALETTE["gold"])]
    states = ["CO", "WA", "OR", "MT", "NM"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y_base = np.arange(len(states))[::-1]
    offset = {0: 0.22, 1: 0, 2: -0.22}

    for i, (label, df, yvar, color) in enumerate(outcomes):
        ys, xs, xerrs = [], [], []
        for j, st in enumerate(states):
            d = df[df.state_abbr == st].dropna(subset=[yvar, "border_70mi", "log_population"]).copy()
            if d["border_70mi"].nunique() < 2:
                continue
            d["year"] = d["year"].astype(str)
            yrs = d["year"].nunique()
            formula = f"{yvar} ~ border_70mi + log_population" + (" + C(year)" if yrs > 1 else "")
            m = smf.ols(formula, data=d).fit(cov_type="HC1")
            b, se = m.params["border_70mi"], m.bse["border_70mi"]
            ys.append(y_base[j] + offset[i])
            xs.append((np.exp(b) - 1) * 100)
            xerrs.append(((np.exp(b + 1.96 * se) - 1) * 100 - (np.exp(b) - 1) * 100))
        ax.errorbar(xs, ys, xerr=xerrs, fmt="o", color=color, label=label, capsize=3, markersize=6)

    ax.axvline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels([f"{s}" for s in states])
    ax.set_xlabel("Border coefficient (%, 95% CI)")
    ax.set_title("State-specific border effects, by outcome")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_state_forest.png"))
    plt.close(fig)
    print("Wrote fig2_state_forest.png")


def fig_oregon_beforeafter(percapita):
    apply_healy_style()
    orr = percapita[percapita.state_abbr == "OR"].copy()
    periods = [("Free\n(2017-2021)", orr[orr.year <= 2021], PALETTE["blue"]),
               ("Capped\n(2022-2025)", orr[orr.year >= 2022], PALETTE["red"])]

    fig, ax = plt.subplots(figsize=(5, 4.6))
    xs, ys, yerrs, colors = [], [], [], []
    for i, (label, d, color) in enumerate(periods):
        d = d.dropna(subset=["log_revenue_per_capita", "border_70mi", "log_population"]).copy()
        d["year"] = d["year"].astype(str)
        m = smf.ols("log_revenue_per_capita ~ border_70mi + log_population + C(year)", data=d).fit(cov_type="HC1")
        b, se = m.params["border_70mi"], m.bse["border_70mi"]
        xs.append(i)
        ys.append((np.exp(b) - 1) * 100)
        yerrs.append(((np.exp(b + 1.96 * se) - 1) * 100 - (np.exp(b) - 1) * 100))
        colors.append(color)

    ax.bar(xs, ys, yerr=yerrs, color=colors, capsize=4, width=0.55)
    ax.axhline(0, color="#4D4D4D", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in periods])
    ax.set_ylabel("Border premium on revenue per capita (%)")
    ax.set_title("Oregon: border premium before/after\nthe 2022 entry freeze")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_oregon_beforeafter.png"))
    plt.close(fig)
    print("Wrote fig3_oregon_beforeafter.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    percapita = pd.read_csv(os.path.join(TBL, "master_percapita_panel.csv"), dtype={"fips5": str})
    perstore = pd.read_csv(os.path.join(TBL, "master_perstore_panel.csv"), dtype={"fips5": str})
    fig_premium_by_regime(perstore, percapita)
    fig_state_forest(perstore, percapita)
    fig_oregon_beforeafter(percapita)


if __name__ == "__main__":
    main()
