# Data feasibility for a Singleton-style brand entry/exit analysis (New Mexico)

Companion to `01_singleton_template.md`. Verdict per object the task asked about.

## Source

51 monthly "Cannabis Sales by Licensee" workbooks from NM's Cannabis Reporting Online
Portal (CROP / Regulation & Licensing Division), 2022-04 … 2026-07, downloaded via
`entry game-marijuana/download_nm_revenue.py`. Each row: `Licensee` (free text),
`Address`, `City`, `State`, `Zip`, `Medical Sales`, `Adult-Use Sales`, `Total Sales`.
30,081 licensee-month rows, $2.36 B lifetime, 1,255 distinct addresses.
No license number, owner id, NAICS, headcount, transactions, price, cost, or
open/close date.

| Object | Identifiable? | How / caveat |
|---|---|---|
| **dispensary storefront** | **YES** | normalized street address = `dispensary_id`; 1,253 storefronts (1,230 with ≥1 positive-sales quarter in-window). |
| **dispensary name** | YES | `Licensee` string (modal over months). 1,377 distinct strings; 263 addresses report >1 string over time. |
| **parent brand / chain** | **YES, constructed** | union-find over (exact shared name) ∪ (cleaned key) ∪ (41-entry curated alias map); see `brand_cleaning_procedure.md`. **653 brands** (180 multi-location, 473 singleton). No ground-truth to validate against; audit dump is `_brand_review.txt`. |
| **ZIP code** | **YES** | reported per storefront-month; 139 ZIPs ever host a storefront; 129 in the built ZIP-quarter panel; 119 of those match the project's 368-ZIP border file (10 PO-box / mis-keyed ZIPs dropped). |
| **quarter** | **YES** | 2022Q2 … 2026Q2 = 17 quarters (2026Q3 partial, held out). |
| **opening quarter** | **PARTIAL** | first quarter a storefront reports positive sales. **304 storefronts already operating in 2022Q2 are LEFT-CENSORED** (true open date unknown) — flagged, excluded from entry models. 926 observed opening after 2022Q2. |
| **closing quarter** | **PARTIAL** | last quarter of a permanent run of positive sales. A storefront silent for ≥2 consecutive quarters before panel end is coded `exited` (519 storefronts); silent for 1 quarter then back = reporting gap, not exit (172 such gap-quarters). Storefronts active in the last 1–2 quarters are RIGHT-CENSORED. |
| **active status** | **YES** | positive `Total Sales` in any month of the quarter. |
| **entry** (brand opens 1st store in a ZIP) | **YES** | 1,088 brand-ZIP entries in-window: **242 expansion** (brand already had an NM store), **846 de novo** (brand's first NM store). |
| **exit** | **PARTIAL** | see closing quarter. Baseline quarterly storefront exit rate = **5.6%**. |
| **revenues** | **YES** | monthly `Total Sales` per storefront (retail + medical). This is the demand realisation. NOT profit — no cost data whatsoever. |
| **# locations of same brand** | **YES** | brand footprint per quarter from the crosswalk; time-varying. |
| **competing dispensaries** | **YES** | geocoded storefronts (1,192/1,253 have lat/lon; Census + Nominatim); counts within 5/10/25 mi, entry/exit-aware per quarter; count in own ZIP. |
| **competing brands** | **YES** | same, distinct brands. |
| **border status** | **YES, predetermined** | reuse the project's fixed 2023-12-31 rule: ZIP centroid within 75 mi of the NM line facing a **non-legal** neighbour (TX, OK, UT — CO & AZ excluded as legal) **and** ≥1,000 non-legal residents within 75 mi. Time-invariant (none of TX/OK/UT legalised). 33 border / 86 interior operating ZIPs; 436 border / 756 interior geocoded storefronts. |
| **distance to non-legal border** | **YES** | `dist_stateline_mi` (project's geopandas computation to the dissolved non-legal-state polygon), storefront- and ZIP-level. |
| **population** | **YES** | ZIP resident population (SimpleMaps/ACS); storefront local resident population within 25 mi. Single 2020-ish vintage — time-invariant. |
| **non-legal-state population exposure** | **YES** | newly computed here at **25 / 50 / 75 / 100 mi** for every ZIP and storefront, summing non-legal-state ZIP-centroid populations (haversine). |
| **effectiveness / quality** | **NO** | no analogue exists for a dispensary. |
| **variable / fixed / entry cost** | **NO** | nothing. |

## What this rules in and out

- **Rules IN:** the whole of Stage A — brand presence, concentration, sorting, entry,
  exit/survival, network expansion, heterogeneity — as reduced-form LPMs with a
  predetermined border/exposure treatment and market-size controls.
- **Rules OUT:** a literal Singleton structural estimator (Eqs 18–22). It needs offline
  demand and variable-cost estimates; we have neither. θ_i (financial responsiveness)
  would be identified only under an untestable cost assumption, and there is no policy
  scalar τ_t with time variation (NM legalised once, statewide).
- The strongest credible product is a **Singleton-inspired reduced-form paper** on how a
  predetermined cross-border demand advantage reshapes market *structure and composition*
  in a free-entry retail market.
