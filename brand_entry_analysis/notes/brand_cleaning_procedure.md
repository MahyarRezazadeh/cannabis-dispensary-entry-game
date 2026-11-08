# Brand-cleaning procedure (New Mexico dispensaries)

Implemented in `code/02_brand_crosswalk.py`. Inputs: 51 raw CROP "Sales by Licensee"
monthly files (2022-04 … 2026-07). Outputs: `tables/storefront_master.csv`,
`tables/brand_crosswalk.csv`, `tables/brand_master.csv`, plus review artefacts
`notes/_brand_review.txt` and `notes/_brand_fuzzy_candidates.csv`.

## The identification problem

The raw files give one free-text `Licensee` string per storefront per month, plus
street `Address`, `City`, `Zip`. There is **no** license number, owner id, parent-company
field, or NAICS code. Over the panel there are 1,377 distinct licensee strings for 1,255
distinct normalized addresses. The mapping is many-to-many:

* **263 of 1,255 addresses** report **>1 distinct licensee string** over time — legal-entity
  variants ("R. GREENLEAF ORGANICS DISPENSARY" → "R. GREENLEAF ORGANICS, INC." →
  "R GREENLEAF ORGANICS INC/ RETAIL APPALOOSA"), branch descriptors added/removed, and a
  minority of true operator/ownership changes at the same storefront.
* **106 licensee strings** appear at **>1 address** (e.g. "OCC ABQ LLC" at 8 addresses).
* Multi-location operators mostly use a *different* string per branch:
  "PECOS VALLEY PRODUCTION - APPALOOSA DR RETAIL", "PECOS VALLEY-DISPENSARY",
  "PECOS VALLEY - PRODUCTION", "PECOS VALLEY PRODUCTIONS INC - 2620 N LOVINGTON RETAIL".

## Unit definitions

* **Storefront** = a normalized street address (Unicode-folded, upper-cased, whitespace-
  collapsed, trailing "." removed). This is the project's existing `dispensary_id`. A
  rebrand or ownership change at the *same address* is the *same storefront*. 1,253
  storefronts. (2 raw addresses, $0.4M lifetime sales, 0.02% of the $2.36B total, do not
  join to a `dispensary_id` and are dropped.)
* **Brand** = the parent operator, *inferred* by the procedure below. 653 brands
  (180 multi-location, 473 singletons).

## Step 1 — storefront master

For each `dispensary_id`, collapse the monthly rows to: modal licensee string
(`name_modal`, the mode over months — the label that best represents the storefront's
identity across the panel), first-observed string, count and list of all variants,
modal city, modal ZIP, first/last month observed, months active, lifetime sales. Merge
the project's geocoding (`nm_dispensary_locations.csv`, 1,192/1,253 with lat/lon),
dispensary-level border classification (`nm_dispensary_border_classification.csv`), and
entry-month / left-censoring flags.

## Step 2 — brand key (algorithm)

`brand_key(name)` normalizes each `name_modal` to a comparable operator token:

1. If the string contains `DBA`/`D/B/A`, keep only the trade name after it
   ("4A FARM LLC DBA STOGIES" → "STOGIES").
2. Strip a leading `NMTO-` / `NMTO:` prefix and a leading `THE `.
3. Remove legal-entity and generic-corporate tokens as *whole words*:
   `LLC INC(ORPORATED) CORP(ORATION) LLP PLLC LP LC PC PA CO(MPANY) LTD`
   plus `HOLDINGS ENTERPRISES MANAGEMENT GROUP VENTURES INVESTMENTS PARTNERS FARMS LABS`.
4. Cut at the first branch separator — any `-`, `/`, or `\` — keeping the left side
   (the operator; the right side is a street / neighbourhood / "RETAIL" tag).
5. Drop punctuation and `&`; remove `OF NEW MEXICO` / `NEW MEXICO` / `NM`.
6. Repeatedly strip trailing descriptor tokens
   (`RETAIL(ER) DISPENSARY DISPENSARIES CONSUMPTION MICROBUSINESS MANUFACTURER
   PRODUCERS PRODUCTION CULTIVATION STORE LOCATION OUTLET SHOP DISP` …).
7. Repeatedly strip trailing **NM city-name tokens** (built from the data's own `City`
   column), unless that would empty the key.
8. Remove residual street words (`ST RD BLVD AVE DR HWY … NE NW SE SW SUITE STE UNIT US`)
   and a *leading* 2–5-digit street number. **Name-digits are kept** ("SCORE 420",
   "285 SOUTH", "1912 CANNABIS", "PROHIBITION 37", "ELEVATION 6420" ≠ "ELEVATION 6969").
9. Keep `CANNABIS`/`MARIJUANA` as a token (it makes keys *more* specific, e.g.
   "RELEAF CANNABIS" vs a bare "RELEAF").

A key is **usable for automatic linkage** only if it has ≥2 tokens, or a single token of
≥4 characters that is not in a 60-word generic stop-list (`GREEN HIGH LEAF BUD FARM
GARDEN FLOWER HOUSE ROOTS RELIEF REMEDY WELLNESS ORGANIC NATURAL MEDS …`).

## Step 3 — linkage (union-find over three rules)

Storefronts are merged into brands by union-find. An edge is added when:

| Rule | Condition | Confidence |
|---|---|---|
| **A** `A_sharedname` | ≥2 storefronts share the **exact** raw modal string (excluding a short generic list: "", CANNABIS, DISPENSARY, RETAIL, THE DISPENSARY) | high |
| **B** `B_cleankey` | ≥2 storefronts share an identical **usable** cleaned key | medium |
| **C** `C_alias` | storefronts match the same entry in a **hand-curated alias map** | curated |

If none fires, the storefront is its own singleton brand (`D_singleton`).

**Fuzzy string similarity is used only to *propose* review pairs** — all singleton-brand
name pairs with `difflib` ratio ≥ 0.86 are written to
`notes/_brand_fuzzy_candidates.csv` for manual inspection. Nothing is merged on
similarity alone. (After curation, 5 near-miss pairs remain unmerged, all judged to be
genuinely different operators in different cities: FIELDS OF DREAMS / FIELDS TO DREAMS,
GANJA GARDENS / GANJIKA GARDENS, GO GREEN CANNABIS / GREEN CANNABIS,
HIGH PLAINS CANNABIS / HIGHLINE CANNABIS, two single-store "GH LLC".)

## Step 4 — the curated alias map

41 entries in `ALIAS_RULES` (regex on the raw name **and** the cleaned key → canonical
brand). Each entry was checked against `notes/_brand_review.txt` to confirm it refers to
**one real multi-storefront NM operator**. It is used to:

* **Unite** name families the algorithm splits, e.g.
  `ULTRA HEALTH` ⊇ {NMTO-ULTRA HEALTH, NEW MEXICO TOP ORGANICS ULTRA HEALTH INC,
  ULTRA HEALTH RETAIL, NMTO- ULTRA HEALTHALAMOGORDO}; `GREEN GOODS` ⊇ {VIREO HEALTH OF LC,
  VIREO OF SANTA FE, GREEN GOODS} (Vireo rebranded to Green Goods);
  `EVEREST CANNABIS CO` ⊇ {EVEREST APOTHECARY, EVEREST CANNABIS}.
* **Prevent** over-merges the algorithm would otherwise make. The clearest example:
  the first pass linked `SACRED GARDEN` and `SAWMILL CANNABIS COMPANY` because a single
  regex alternation `\bSAWMILL\b|SACRED GARDEN` was written on an unverified assumption
  that they share ownership. `_brand_review.txt` showed they are two distinct
  Albuquerque operators; the rule was split into two separate canonicals. This is exactly
  the failure mode the task warned about ("do not assume two stores belong to the same
  brand merely because their names are similar") and it is why every alias entry is
  individually justified rather than pattern-generated.

## Diagnostics carried forward

* `link_rule` per storefront: A_sharedname 57, B_cleankey 761, C_alias 395, D_singleton 40.
* `ownership_change_flag`: 134 storefronts (10.7%) report names whose *cleaned key* changes
  over the panel — candidate genuine operator changes (e.g. "MJ EXPRESS-O" storefronts
  later reporting as "PURLIFE"). The panel uses a **time-invariant, modal-based** brand
  assignment for stability; results are re-run dropping these 134 storefronts as a
  robustness check (`code/12_robustness.py`).
* `geocode_source`: Census 1,080, Nominatim fallback 112, unmatched 61. The 61
  un-geocoded storefronts cannot receive a distance-based border/exposure value and are
  dropped from border-design regressions (flagged, never imputed).

## Known limitations (report §1)

1. No ground-truth ownership registry ⇒ the crosswalk cannot be externally validated; it
   is defensible, not certified. The review dump is the audit trail.
2. Brand is time-invariant; ~11% of storefronts had an identity change the panel does not
   track. Robustness drops them.
3. Vertically-integrated operators file "MANUFACTURER"/"PRODUCTION" rows at some
   addresses; these are treated as storefronts of the same brand (they carry retail
   revenue in most months) — a minor over-count of a brand's *retail* footprint at the
   margin.
4. Franchise-like arrangements (common name, independent owners) would be wrongly merged;
   none was identified in the review but the possibility cannot be excluded.
