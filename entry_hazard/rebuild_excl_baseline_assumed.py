"""
Diagnostic rebuild of the entry-hazard panel EXCLUDING baseline_assumed
dispensary records (CO and WA, where every single record shares the
identical placeholder open_date 2015-01-01 -- not a real observed date).
Re-runs the same Cox specs as analyze_entry_hazard_v2.py on both the
original (contaminated) panel and this corrected one, to check whether the
headline "faster 2nd entrant near the border" result survives.

Mirrors build_entry_hazard_panel_v2.py exactly except for one filter line.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from lifelines import CoxPHFitter

PROJ_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ_BASE, 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
REF_DATE = pd.Timestamp('2026-08-05')

disp = pd.read_csv(os.path.join(DATA, 'dispensaries_master_geocoded.csv'))
print("Rows before excluding baseline_assumed:", len(disp))
disp = disp[disp['date_confidence'] != 'baseline_assumed'].copy()
print("Rows after excluding baseline_assumed (CO+WA placeholder dates dropped):", len(disp))
print(disp.groupby('state_abbr' if 'state_abbr' in disp.columns else 'state').size())

disp = disp.dropna(subset=['lat', 'lon']).copy()
disp['open_date'] = pd.to_datetime(disp['open_date'], errors='coerce')
disp_timed = disp.dropna(subset=['open_date']).copy()
disp_timed = disp_timed[disp_timed['open_date'] <= REF_DATE].copy()

shp = gpd.read_file(os.path.join(DATA, 'us_counties_shp', 'cb_2023_us_county_500k.shp'))
shp = shp[['GEOID', 'geometry']].rename(columns={'GEOID': 'fips5'})
gdf = gpd.GeoDataFrame(disp_timed, geometry=[Point(xy) for xy in zip(disp_timed['lon'], disp_timed['lat'])],
                        crs='EPSG:4326').to_crs(shp.crs)
joined = gpd.sjoin(gdf, shp, how='left', predicate='within')
if joined['fips5'].isna().any():
    missing = joined[joined['fips5'].isna()][['uid', 'geometry']].set_geometry('geometry')
    nearest = gpd.sjoin_nearest(missing, shp, how='left')
    col = 'fips5_right' if 'fips5_right' in nearest.columns else 'fips5'
    fix = dict(zip(nearest['uid'], nearest[col]))
    joined.loc[joined['fips5'].isna(), 'fips5'] = joined.loc[joined['fips5'].isna(), 'uid'].map(fix)

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
universe = pd.read_csv(os.path.join(DATA, 'county_universe_with_distance.csv'), dtype={'fips5': str})
universe['border_70mi'] = (universe['dist_nonlegal_mi'] <= 70).astype(int)
universe = universe.merge(legal[['state_abbr', 'retail_date']], on='state_abbr', how='left')

joined = joined.drop(columns=[c for c in ['retail_date', 'state_abbr'] if c in joined.columns])
joined = joined.merge(universe[['fips5', 'state_abbr', 'retail_date']], on='fips5', how='left')
joined['days_since_legal'] = (joined['open_date'] - joined['retail_date']).dt.days
joined['days_since_legal'] = joined['days_since_legal'].clip(lower=0)

entry_dates = joined.groupby('fips5')['days_since_legal'].apply(lambda s: sorted(s.tolist())).rename('entry_days')
panel = universe.merge(entry_dates, on='fips5', how='left')
panel['entry_days'] = panel['entry_days'].apply(lambda x: x if isinstance(x, list) else [])
panel['n_timed_entrants'] = panel['entry_days'].apply(len)
panel['censor_days'] = (REF_DATE - panel['retail_date']).dt.days

panel['first_entry_days'] = panel['entry_days'].apply(lambda x: x[0] if len(x) >= 1 else np.nan)
panel['first_entry_event'] = panel['n_timed_entrants'].apply(lambda n: 1 if n >= 1 else 0)
panel['first_entry_time'] = panel.apply(
    lambda r: r['first_entry_days'] if r['first_entry_event'] == 1 else r['censor_days'], axis=1)


def gap_sample(panel, from_rank, to_rank):
    has_from = panel['n_timed_entrants'] >= from_rank
    event = np.where(panel.loc[has_from, 'n_timed_entrants'] >= to_rank, 1, 0)
    time = panel.loc[has_from].apply(
        lambda r: (r['entry_days'][to_rank - 1] - r['entry_days'][from_rank - 1])
        if r['n_timed_entrants'] >= to_rank
        else (r['censor_days'] - r['entry_days'][from_rank - 1]), axis=1)
    out = panel.loc[has_from].copy()
    out[f'gap_{from_rank}to{to_rank}_event'] = event
    out[f'gap_{from_rank}to{to_rank}_time'] = time.clip(lower=0)
    return out


gap12 = gap_sample(panel, 1, 2).rename(columns={'gap_1to2_event': 'second_gap_event', 'gap_1to2_time': 'second_gap_time'})
panel = panel.merge(gap12[['fips5', 'second_gap_event', 'second_gap_time']], on='fips5', how='left')

panel['log_dist_nonlegal'] = np.log(panel['dist_nonlegal_mi'])
panel['log_population'] = np.log(panel['population'])

print(f"\nCorrected panel: {len(panel)} counties")
print(f"1st entrant: {panel['first_entry_event'].sum()} events")
print(f"1->2 gap: {panel['second_gap_event'].notna().sum()} at risk, {panel['second_gap_event'].sum()} events")
print(f"share of counties with second_gap_time==0: {(panel['second_gap_time']==0).mean():.3f}")
sub = panel[panel.state_abbr.isin(['CO', 'WA'])]
print(f"CO/WA share with second_gap_time==0: {(sub['second_gap_time']==0).mean():.3f} (n={len(sub)})")

panel.drop(columns=['entry_days']).to_csv(os.path.join(OUT, 'entry_hazard_panel_v2_CORRECTED.csv'), index=False)


def run_cox(df, duration_col, event_col, covariates, label, strata=('state_abbr',)):
    cols = [duration_col, event_col] + list(strata) + covariates
    d = df[cols].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(d, duration_col=duration_col, event_col=event_col, strata=list(strata))
    print(f"\n=== {label} (n={len(d)}, events={int(d[event_col].sum())}) ===")
    print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].round(4).to_string())
    return cph


print("\n" + "#" * 70)
print("COMPARISON: 2nd-entrant gap hazard, CORRECTED panel (baseline_assumed excluded)")
print("#" * 70)
r2 = panel[panel['n_timed_entrants'] >= 1]
run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['log_dist_nonlegal', 'log_population'], 'Rank 2 (1st -> 2nd entrant), CORRECTED')
run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['border_70mi', 'log_population'], 'Rank 2 (binary border), CORRECTED')

print("\n" + "#" * 70)
print("For direct comparison, re-running the ORIGINAL (contaminated) panel's same spec")
print("#" * 70)
orig = pd.read_csv(os.path.join(OUT, 'entry_hazard_panel_v2.csv'))
r2_orig = orig[orig['n_timed_entrants'] >= 1]
run_cox(r2_orig, 'second_gap_time', 'second_gap_event',
        ['log_dist_nonlegal', 'log_population'], 'Rank 2 (1st -> 2nd entrant), ORIGINAL')
