# brand_entry_analysis

Singleton (2019)-inspired supply-side analysis of brand entry / exit / location /
concentration in New Mexico's free-entry cannabis market. Offshoot of Venous4.
Nothing outside this folder is modified.

## Read first
- **`REPORT.md`** — the full 15-section report + bottom-line answers.
- `notes/01_singleton_template.md` — Singleton logic, equation-by-equation, + the mapping table.
- `notes/02_data_feasibility.md` — what the data can / cannot identify.
- `notes/brand_cleaning_procedure.md` — the storefront→brand crosswalk method + limitations.

## Reproduce (run in order, from `entry game-marijuana/`, Python 3.13 `py` launcher)
| step | script | output |
|---|---|---|
| 1 | `code/01_inspect_data.py` | `logs/01_inspect_data.log` (read-only audit) |
| 2 | `code/02_brand_crosswalk.py` | `tables/{storefront_master,brand_crosswalk,brand_master}.csv`, `notes/_brand_review.txt`, `notes/_brand_fuzzy_candidates.csv` |
| 3 | `code/03_build_panels.py` | `tables/{zip_xwalk,panel_storefront_quarter,panel_brand_quarter,panel_zip_quarter,panel_brand_zip_quarter}.csv` |
| 4 | `code/04_descriptives.py` | `tables/tab_{summary_stats,border_vs_interior_composition,exiters_vs_persisters}.csv`, `logs/04_descriptives.log` |
| 5 | `code/05_regressions.py` | `tables/tab_concentration.csv`, `logs/05_regressions.log` |
| 6 | `code/06_dynamics_robustness.py` | `figures/fig_dynamics_*.png`, `robustness/tab_robustness_concentration.csv`, `logs/06_dynamics_robustness.log` |

`code/_util.py` — shared OLS/PanelOLS + clustered-SE helpers.
Deps: pandas, numpy, statsmodels, linearmodels, geopandas, scipy, matplotlib, openpyxl/xlrd.

## Inputs (read-only, from elsewhere in the project — never modified)
- `entry game-marijuana/data/raw/newmexico_revenue/*Licensee*.xls[x]` — 51 monthly CROP files
- `entry game-marijuana/tables/nm_dispensary_panel_monthly.csv` — project storefront ids / entry months
- `New Mexico project/tables/nm_dispensary_locations.csv` — geocoding
- `New Mexico project/tables/nm_dispensary_border_classification.csv` — storefront border status
- `New Mexico project/tables/nm_zip_border_stateline.csv` — ZIP border + dist-to-stateline
- `entry game-marijuana/data/raw/uszips_simplemaps.csv` — ZIP centroids + population
- `entry game-marijuana/data/legal_dates_new.csv` — legal-retail dates (for the non-legal set)

## Headline finding
In a free-entry market a predetermined cross-border demand advantage produces **larger, LESS
concentrated, more independent-heavy** border markets — it dissipates into a competitive fringe
of small operators rather than into concentration or big-chain capture. No robust effect on
brand entry or unconditional exit; a `border × chain` survival advantage that grows as the
market matures. A structural Singleton entry game is **not** identifiable (no cost/quality data,
no time-varying policy scalar).
