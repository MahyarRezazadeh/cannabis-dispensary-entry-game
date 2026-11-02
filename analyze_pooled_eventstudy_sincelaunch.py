"""
Task 2: Pooled border-premium event study, all five states together (not
split by entry regime this time), border vs. interior counties, using
event time = years since each state's own retail legalization (the same
"years since launch" convention used throughout this project, so CO's 2014
launch and NM's 2022 launch land on a common clock). Border_70mi is the
existing time-varying indicator (already accounts for the CO-NM, WA-OR,
and OR-CA/NV neighbor-legalization switches built into the county universe).

Spec: y_ct = a + b*Border_c + sum_k theta_k(Border_c x 1{event_year=k})
      + gamma1 ln(population) + gamma2 ln(1+captive population)
      + event-year FE + state FE + eps_ct
county-clustered SEs, captive-population control, 2026 dropped.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import os

TBL = "tables"
OUT = "figures_reframe"

legal = pd.read_csv("data/legal_dates_new.csv", parse_dates=["retail_date"])
launch_map = legal.set_index("state_abbr")["retail_date"].dt.year.to_dict()


def load_with_captive(panel_file):
    df = pd.read_csv(os.path.join(TBL, panel_file), dtype={"fips5": str})
    df = df[df["year"] != 2026].copy()
    cap = pd.read_csv(os.path.join(TBL, "captive_population_all5states.csv"), dtype={"fips5": str})
    df = df.merge(cap[["fips5", "state_abbr", "captive_population"]], on=["fips5", "state_abbr"], how="left")
    df["captive_population"] = df["captive_population"].fillna(0)
    df["log1p_captive_pop"] = np.log1p(df["captive_population"])
    df["event_year"] = (df["year"] - df["state_abbr"].map(launch_map)).astype(int)
    return df


def pooled_eventstudy(df, yvar, label):
    d = df.dropna(subset=[yvar, "border_70mi", "log_population", "log1p_captive_pop"]).copy()
    event_years = sorted(d["event_year"].unique())
    for y in event_years:
        d[f"D_{y}"] = ((d["event_year"] == y) & (d["border_70mi"] == 1)).astype(int)
    ref = 0  # first year of each state's own panel
    dvars = [f"D_{y}" for y in event_years if y != ref]
    d["event_year_str"] = d["event_year"].astype(str)
    formula = (f"{yvar} ~ border_70mi + " + " + ".join(dvars)
               + " + log_population + log1p_captive_pop + C(event_year_str) + C(state_abbr)")
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["fips5"]})
    print(f"\n=== {label} (n={int(m.nobs)}, counties={d.fips5.nunique()}, "
          f"states={sorted(d.state_abbr.unique())}) ===")
    b_ref, se_ref, p_ref = m.params["border_70mi"], m.bse["border_70mi"], m.pvalues["border_70mi"]
    rows = [{"event_year": ref, "coef": b_ref, "se": se_ref, "p": p_ref, "ref": True}]
    for y in event_years:
        if y == ref:
            continue
        term = f"D_{y}"
        if term not in m.params.index:
            continue
        d_b = m.params[term]
        b_y = b_ref + d_b
        se_y = float(np.sqrt(m.cov_params().loc["border_70mi", "border_70mi"]
                              + m.cov_params().loc[term, term]
                              + 2 * m.cov_params().loc["border_70mi", term]))
        t_y = b_y / se_y
        p_y = 2 * (1 - stats.t.cdf(abs(t_y), m.df_resid))
        n_obs = int(d[d.event_year == y].shape[0])
        rows.append({"event_year": y, "coef": b_y, "se": se_y, "p": p_y, "ref": False, "n": n_obs})
    out = pd.DataFrame(rows).sort_values("event_year")
    out["pct"] = (np.exp(out["coef"]) - 1) * 100
    print(out.to_string(index=False))
    return out


def main():
    percapita = load_with_captive("master_percapita_panel.csv")
    perstore = load_with_captive("master_perstore_panel.csv")
    perstore["log_dispensary_per_1000"] = np.log(
        1000 * pd.to_numeric(perstore["dispensary_count"], errors="coerce") / perstore["population"])

    print("Event-year coverage by state (year - own launch year):")
    print(percapita.groupby("state_abbr")["event_year"].agg(["min", "max"]))

    out_pc = pooled_eventstudy(percapita, "log_revenue_per_capita", "Revenue per capita, pooled 5 states")
    out_pc.to_csv(os.path.join(TBL, "pooled_sincelaunch_eventstudy_percapita.csv"), index=False)

    out_ps = pooled_eventstudy(perstore, "log_revenue_per_store", "Revenue per store, pooled 5 states")
    out_ps.to_csv(os.path.join(TBL, "pooled_sincelaunch_eventstudy_perstore.csv"), index=False)

    out_dens = pooled_eventstudy(perstore, "log_dispensary_per_1000", "Dispensary density, pooled 5 states")
    out_dens.to_csv(os.path.join(TBL, "pooled_sincelaunch_eventstudy_density.csv"), index=False)


if __name__ == "__main__":
    main()
