"""
Full corrected rebuild of the entry-hazard analysis (build_entry_hazard_panel_v2.py
+ analyze_entry_hazard_v2.py combined), excluding baseline_assumed dispensary
records -- every CO and WA record in dispensaries_master_geocoded.csv shares
the identical placeholder open_date 2015-01-01 (date_confidence=baseline_assumed),
not a real observed date, and including them created degenerate zero-duration
"gaps" for 58% of CO/WA counties (vs 18% for the full contaminated sample),
since every dispensary in a county was tied at the same fake date.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from lifelines import CoxPHFitter

PROJ_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ_BASE, 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
REF_DATE = pd.Timestamp('2026-08-05')
EARTH_RADIUS_MI = 3958.8

disp = pd.read_csv(os.path.join(DATA, 'dispensaries_master_geocoded.csv'))
disp = disp[disp['date_confidence'] != 'baseline_assumed'].copy()
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
gap23 = gap_sample(panel, 2, 3)
gap34 = gap_sample(panel, 3, 4)
panel = panel.merge(gap12[['fips5', 'second_gap_event', 'second_gap_time']], on='fips5', how='left')
panel = panel.merge(gap23[['fips5', 'gap_2to3_event', 'gap_2to3_time']], on='fips5', how='left')
panel = panel.merge(gap34[['fips5', 'gap_3to4_event', 'gap_3to4_time']], on='fips5', how='left')

panel['log_dist_nonlegal'] = np.log(panel['dist_nonlegal_mi'])
panel['log_population'] = np.log(panel['population'])

# ---------- captive population (unaffected by the date issue, rebuilt for merge parity) ----------
legal_states_asof = set(legal.loc[legal['retail_date'].notna() & (legal['retail_date'] <= REF_DATE), 'state_abbr'])
centroids = pd.read_stata(os.path.join(DATA, 'county_centroids.dta'))
centroids['fips5'] = centroids['fips5'].astype(str).str.zfill(5)
pop = pd.read_csv(os.path.join(DATA, 'co_est2023_alldata.csv'), encoding='latin1')
pop = pop[pop['SUMLEV'] == 50].copy()
pop['fips5'] = pop['STATE'].astype(str).str.zfill(2) + pop['COUNTY'].astype(str).str.zfill(3)
pop = pop[['fips5', 'POPESTIMATE2023']].rename(columns={'POPESTIMATE2023': 'nl_population'})
centroids = centroids.merge(pop, on='fips5', how='left')
nonlegal = centroids[~centroids['state_abbr'].isin(legal_states_asof)].dropna(subset=['nl_population']).copy()

nonlegal_rad = np.radians(nonlegal[['lat', 'lon']].to_numpy())
tree = BallTree(nonlegal_rad, metric='haversine')
radius_rad = 70 / EARTH_RADIUS_MI
own = panel[['fips5']].merge(centroids[['fips5', 'lat', 'lon']], on='fips5', how='left')
own_rad = np.radians(own[['lat', 'lon']].to_numpy())
indices = tree.query_radius(own_rad, r=radius_rad)
captive_pop = [nonlegal.iloc[idx]['nl_population'].sum() for idx in indices]
panel['captive_nonlegal_pop_70mi'] = captive_pop
panel['log_captive_pop'] = np.log(panel['captive_nonlegal_pop_70mi'] + 1)
panel['has_captive_pop'] = (panel['captive_nonlegal_pop_70mi'] > 0).astype(int)

panel.drop(columns=['entry_days']).to_csv(os.path.join(OUT, 'entry_hazard_panel_v2_CORRECTED.csv'), index=False)
print(f"Saved entry_hazard_panel_v2_CORRECTED.csv: {len(panel)} counties")
print(f"1st entrant: {panel['first_entry_event'].sum()} events")
print(f"1->2 gap: {panel['second_gap_event'].notna().sum()} at risk, {panel['second_gap_event'].sum()} events")
print(f"2->3 gap: {panel['gap_2to3_event'].notna().sum()} at risk, {panel['gap_2to3_event'].sum()} events")
print(f"3->4 gap: {panel['gap_3to4_event'].notna().sum()} at risk, {panel['gap_3to4_event'].sum()} events")

results = []


def run_cox(df, duration_col, event_col, covariates, label, strata=('state_abbr',)):
    cols = [duration_col, event_col] + list(strata) + covariates
    d = df[cols].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(d, duration_col=duration_col, event_col=event_col, strata=list(strata))
    print(f"\n=== {label} (n={len(d)}, events={int(d[event_col].sum())}) ===")
    print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].round(4).to_string())
    for cov in covariates:
        results.append(dict(model=label, covariate=cov, coef=cph.summary.loc[cov, 'coef'],
                             hazard_ratio=cph.summary.loc[cov, 'exp(coef)'],
                             se=cph.summary.loc[cov, 'se(coef)'], p=cph.summary.loc[cov, 'p'],
                             n=len(d), events=int(d[event_col].sum())))
    return cph


print("#" * 70)
print("PART 1 (CORRECTED): hazard ratio on log(distance) across entrant rank 1,2,3,4")
print("#" * 70)
run_cox(panel, 'first_entry_time', 'first_entry_event',
        ['log_dist_nonlegal', 'log_population'], 'Rank 1 (state-launch -> 1st entrant)')
r2 = panel[panel['n_timed_entrants'] >= 1]
run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['log_dist_nonlegal', 'log_population'], 'Rank 2 (1st -> 2nd entrant)')
r3 = panel[panel['n_timed_entrants'] >= 2]
run_cox(r3, 'gap_2to3_time', 'gap_2to3_event',
        ['log_dist_nonlegal', 'log_population'], 'Rank 3 (2nd -> 3rd entrant)')
r4 = panel[panel['n_timed_entrants'] >= 3]
run_cox(r4, 'gap_3to4_time', 'gap_3to4_event',
        ['log_dist_nonlegal', 'log_population'], 'Rank 4 (3rd -> 4th entrant)')

run_cox(panel, 'first_entry_time', 'first_entry_event',
        ['border_70mi', 'log_population'], 'Rank 1 (binary border)')
run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['border_70mi', 'log_population'], 'Rank 2 (binary border)')
run_cox(r3, 'gap_2to3_time', 'gap_2to3_event',
        ['border_70mi', 'log_population'], 'Rank 3 (binary border)')
run_cox(r4, 'gap_3to4_time', 'gap_3to4_event',
        ['border_70mi', 'log_population'], 'Rank 4 (binary border)')

print("\n" + "#" * 70)
print("PART 2 (CORRECTED): captive cross-border population vs raw distance")
print("#" * 70)
run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['log_captive_pop', 'log_population'], 'Rank 2, log(captive nonlegal population)')
run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['log_dist_nonlegal', 'log_captive_pop', 'log_population'],
        'Rank 2, distance AND captive population jointly')

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT, 'entry_hazard_by_rank_and_captivepop_CORRECTED.csv'), index=False)
print("\nSaved entry_hazard_by_rank_and_captivepop_CORRECTED.csv")
