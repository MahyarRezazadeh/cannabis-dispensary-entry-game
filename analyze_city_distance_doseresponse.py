"""
Dispensary count as a function of distance to the nearest nonlegal-state
city (cubic polynomial), state fixed effects, and county population.
Cross-section, 2025 (the same dispensary sample used throughout this
project's map/state-specific work). Per explicit user request: report the
regression table, plot the dose-response curve, and solve for the distance
at which the fitted curve's first derivative is zero (if any exists within
the support of the data).
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Latest try", "code"))
from healy_style import apply_healy_style, PALETTE

TBL = "tables"
OUT = "figures_reframe"


def main():
    d = pd.read_csv(os.path.join(TBL, "dist_nearest_nonlegal_city_2025.csv"))
    d = d.dropna(subset=["dispensary_count", "population", "dist_nearest_nonlegal_city_mi"]).copy()
    d = d[d["dispensary_count"] > 0].copy()  # log outcome requires positive count
    print(f"Regression sample: n={len(d)} counties, {d.state_abbr.nunique()} states")

    d["log_dispensary_count"] = np.log(d["dispensary_count"])
    d["log_population"] = np.log(d["population"])
    # scale to 100-mile units: raw-miles cubic terms are numerically unstable
    # (condition number > 1e8) and produce coefficients too small to read;
    # rescaling doesn't change the fit, p-values, or the location of any
    # turning point, only the units the coefficients are expressed in.
    d["dist"] = d["dist_nearest_nonlegal_city_mi"] / 100.0
    d["dist2"] = d["dist"] ** 2
    d["dist3"] = d["dist"] ** 3

    formula = "log_dispensary_count ~ dist + dist2 + dist3 + log_population + C(state_abbr)"
    m_hc1 = smf.ols(formula, data=d).fit(cov_type="HC1")
    print(m_hc1.summary())

    ftest = m_hc1.f_test("dist = dist2 = dist3 = 0")
    print("\nJoint test that distance (linear+quad+cubic) = 0:", ftest)

    # ---------- write regression table ----------
    with open(os.path.join(TBL, "table_city_distance_doseresponse.tex"), "w") as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Dispensary count and distance to the nearest nonlegal-state city, 2025 cross-section}
\label{tab:city_distance_doseresponse}
\small
\begin{tabular}{lcc}
\toprule
Variable & Coefficient & $p$ (HC1) \\
\midrule
""")
        for term, label in [("dist", "Distance (100 mi.)"), ("dist2", "Distance$^2$"),
                             ("dist3", "Distance$^3$"), ("log_population", "log(population)")]:
            b = m_hc1.params.get(term)
            se = m_hc1.bse.get(term)
            p = m_hc1.pvalues.get(term)
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            f.write(f"{label} & ${b:.6f}${stars} ({se:.6f}) & {p:.3f} \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(f"State FE & Yes & \\\\\n")
        f.write(f"$n$ & {int(m_hc1.nobs)} & \\\\\n")
        f.write(f"$R^2$ & {m_hc1.rsquared:.3f} & \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\begin{minipage}{0.85\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Dependent variable log(dispensary count), county-level, 2025 cross-section (the same
dispensary sample used elsewhere in this project). Distance is to the nearest nonlegal-state incorporated
place with 2023 population $\geq$5,000. HC1 robust standard errors in parentheses.
$^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{minipage}
\end{table}
""")
    print("\nWrote table_city_distance_doseresponse.tex")

    # ---------- optimum distance: first derivative of fitted cubic = 0 ----------
    b1 = m_hc1.params["dist"]
    b2 = m_hc1.params["dist2"]
    b3 = m_hc1.params["dist3"]
    print(f"\nFitted cubic in distance (units of 100mi): b1={b1:.6f}  b2={b2:.6f}  b3={b3:.6f}")

    # derivative: b1 + 2*b2*x + 3*b3*x^2 = 0  (x in units of 100mi)
    dist_min_100, dist_max_100 = d["dist"].min(), d["dist"].max()
    if abs(b3) > 1e-12:
        disc = (2 * b2) ** 2 - 4 * (3 * b3) * b1
        if disc >= 0:
            roots = [(-2 * b2 + np.sqrt(disc)) / (2 * 3 * b3), (-2 * b2 - np.sqrt(disc)) / (2 * 3 * b3)]
            roots_mi = [r * 100 for r in roots]
            print("Critical points (miles):", roots_mi)
            in_range = [r for r in roots_mi if d["dist_nearest_nonlegal_city_mi"].min() <= r <= d["dist_nearest_nonlegal_city_mi"].max()]
            print("Within data support [{:.1f}, {:.1f}] miles:".format(
                d["dist_nearest_nonlegal_city_mi"].min(), d["dist_nearest_nonlegal_city_mi"].max()), in_range)
        else:
            print("No real roots -- derivative never zero (monotonic cubic over all distances)")
    else:
        if abs(b2) > 1e-12:
            root = -b1 / (2 * b2) * 100
            print(f"Quadratic critical point: {root:.2f} miles")

    # ---------- dose-response plot ----------
    apply_healy_style()
    dist_grid_mi = np.linspace(d["dist_nearest_nonlegal_city_mi"].min(), d["dist_nearest_nonlegal_city_mi"].max(), 200)
    dist_grid = dist_grid_mi / 100.0
    mean_logpop = d["log_population"].mean()
    state_shares = d["state_abbr"].value_counts(normalize=True)

    # average predicted value across states, weighting by each state's share in the sample
    pred_mean = np.zeros(len(dist_grid))
    se_mean = np.zeros(len(dist_grid))
    for i, dd in enumerate(dist_grid):
        vals, ses = [], []
        for st, share in state_shares.items():
            row = pd.DataFrame({"dist": [dd], "dist2": [dd ** 2], "dist3": [dd ** 3],
                                 "log_population": [mean_logpop], "state_abbr": [st]})
            pred = m_hc1.get_prediction(row)
            vals.append(pred.predicted_mean[0] * share)
            ses.append(pred.se_mean[0] * share)
        pred_mean[i] = sum(vals)
        se_mean[i] = np.sqrt(sum(s ** 2 for s in ses))  # approx, treating states independently

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(dist_grid_mi, pred_mean, color=PALETTE["blue"], linewidth=2)
    ax.fill_between(dist_grid_mi, pred_mean - 1.96 * se_mean, pred_mean + 1.96 * se_mean,
                     color=PALETTE["blue"], alpha=0.2)
    ax.scatter(d["dist_nearest_nonlegal_city_mi"], d["log_dispensary_count"], color=PALETTE["gray"], alpha=0.4, s=20, zorder=3)
    ax.set_xlabel("Distance to nearest nonlegal-state city (miles)")
    ax.set_ylabel("log(dispensary count)")
    ax.set_title("Dose-response: dispensary count vs. distance to nearest\nnonlegal-state city (population-share-weighted across states)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_city_distance_doseresponse.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nWrote fig_city_distance_doseresponse.png")


if __name__ == "__main__":
    main()
