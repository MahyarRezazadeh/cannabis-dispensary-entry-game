"""shared helpers for brand_entry_analysis regression scripts"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def ols_cluster(df, y, rhs, cluster, absorb=None, weights=None):
    """OLS with cluster-robust SE. `absorb` = list of columns to sweep out as FE via
    within-transform (equivalent to dummies, keeps memory sane). Returns fitted result
    plus a tidy dict of {term: (coef, se, p)}."""
    d = df.copy()
    cols = [y] + rhs + ([cluster] if isinstance(cluster, str) else list(cluster)) \
           + (absorb or []) + ([weights] if weights else [])
    d = d.dropna(subset=[c for c in set(cols) if c in d.columns])
    if absorb:
        allnum = [y] + [c for c in rhs if pd.api.types.is_numeric_dtype(d[c])]
        for g in absorb:
            means = d.groupby(g)[allnum].transform("mean")
            d[allnum] = d[allnum] - means
        d[allnum] = d[allnum] + df[allnum].mean()  # add back grand mean for interpretability
    formula = f"{y} ~ " + " + ".join(rhs) if rhs else f"{y} ~ 1"
    kw = {}
    if weights:
        kw["weights"] = d[weights]
    m = smf.ols(formula, data=d, **kw)
    if isinstance(cluster, str):
        res = m.fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    else:
        res = m.fit(cov_type="cluster", cov_kwds={"groups": d[list(cluster)]})
    tidy = {t: (res.params[t], res.bse[t], res.pvalues[t]) for t in res.params.index}
    tidy["_N"] = int(res.nobs)
    tidy["_r2"] = float(res.rsquared)
    tidy["_ymean"] = float(d[y].mean())
    return res, tidy


def panel_ols(df, y, rhs, entity, time, cluster_col=None, entity_effects=True,
              time_effects=True):
    """linearmodels PanelOLS with entity + time effects, clustered SE.
    cluster_col: column name to cluster on (default: the entity)."""
    from linearmodels.panel import PanelOLS
    need = [y] + list(rhs) + [entity, time] + ([cluster_col] if cluster_col else [])
    d = df.dropna(subset=[c for c in dict.fromkeys(need)]).copy()
    d["_e"] = d[entity].astype("category").cat.codes
    d["_t"] = d[time].astype("category").cat.codes
    dd = d.set_index(["_e", "_t"])
    mod = PanelOLS(dd[y], dd[list(rhs)], entity_effects=entity_effects, time_effects=time_effects)
    if cluster_col and cluster_col != entity:
        res = mod.fit(cov_type="clustered", clusters=dd[cluster_col].astype("category").cat.codes)
    else:
        res = mod.fit(cov_type="clustered", cluster_entity=True)
    tidy = {t: (res.params[t], res.std_errors[t], res.pvalues[t]) for t in res.params.index}
    tidy["_N"] = int(res.nobs); tidy["_r2"] = float(res.rsquared)
    tidy["_ymean"] = float(dd[y].mean())
    return res, tidy


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def col(tidy, terms):
    """format a regression 'column' as text lines for the given terms"""
    out = []
    for t in terms:
        if t in tidy:
            b, se, pv = tidy[t]
            out.append(f"  {t:32s} {b:>12.5f}{stars(pv):<3}  ({se:.5f})")
        else:
            out.append(f"  {t:32s} {'.':>12s}")
    out.append(f"  {'N':32s} {tidy['_N']:>12d}")
    out.append(f"  {'R2':32s} {tidy['_r2']:>12.4f}")
    out.append(f"  {'mean(y)':32s} {tidy['_ymean']:>12.5f}")
    return "\n".join(out)


def diff_means(df, group, cols):
    """exiters-vs-persisters style table: mean by group + diff + Welch t p-value"""
    from scipy import stats
    g0 = df[df[group] == 0]; g1 = df[df[group] == 1]
    rows = []
    for c in cols:
        a, b = g1[c].dropna(), g0[c].dropna()
        if len(a) < 2 or len(b) < 2:
            rows.append((c, a.mean(), b.mean(), np.nan, np.nan)); continue
        t, pv = stats.ttest_ind(a, b, equal_var=False)
        rows.append((c, a.mean(), b.mean(), a.mean() - b.mean(), pv))
    return pd.DataFrame(rows, columns=["var", "grp1_mean", "grp0_mean", "diff", "p"])
