# Cross-border demand and the supply side of a free-entry cannabis market
### A Singleton (2019)-inspired analysis of brand entry, exit, location, and concentration in New Mexico

*Venous4 / `brand_entry_analysis/`. All regressions run on NM CROP licensee data, 2022Q2–2026Q2.
Numbers labelled **[ACTUAL]** are estimated from the data; **[PROPOSED]** items need data we do not have.
Reproduce with `code/01…06` in order (see `code/README` logic in each file header).*

---

## 0. One-paragraph answer

In New Mexico's free-entry recreational cannabis market, a **predetermined cross-border demand
advantage does change *who* supplies the market, not only *how much* is supplied — but in the
opposite direction to the "captive demand → rents → concentration" intuition.** Border ZIP codes
support **~48% more brands** and **~86% more revenue** than interior ZIPs of the same population,
and are **substantially *less* concentrated** (HHI lower by ~1,400 points on a 0–10,000 scale;
largest-brand revenue share lower by ~12 pp). The extra brands are disproportionately
**independent / small operators**: large statewide chains are *under*-represented at the border
(border share 0.27 for 6+-store brands vs 0.42 for 2-store brands and independents). There is
**no** robust effect of border exposure on the *extensive margin* of brand presence or brand
entry once market size is controlled, and **no** evidence that established chains use their
geographic networks to expand into border markets. Storefront **survival** is not
unconditionally higher at the border, but **chain-operated** border storefronts survive
significantly better than chain storefronts in the interior, increasingly so as the market
matures. The composition gap **emerged gradually** over 2023–2025, not at legalization. A
literal Singleton structural entry game is **not identifiable** here (no cost data, no
time-varying policy scalar); the credible product is a reduced-form paper on **market-structure
composition under a spatial demand advantage in a free-entry industry.**

---

## 1. Data feasibility

*(Full detail: `notes/02_data_feasibility.md`.)*

**Can brands, entry, exit, and choice sets be identified reliably?**

| | Verdict |
|---|---|
| Storefront (address), ZIP, quarter, active status, revenue | **Yes** — clean. |
| **Parent brand / chain** | **Yes, constructed.** 653 brands (180 multi-location) from a documented union-find crosswalk over exact-name, cleaned-key, and a 41-entry hand-curated alias map (`notes/brand_cleaning_procedure.md`, `tables/brand_crosswalk.csv`). No ownership registry exists to validate against; `notes/_brand_review.txt` is the audit trail. ~11% of storefronts had an identity/ownership change over the panel; the crosswalk assigns a time-invariant modal brand and robustness drops them. |
| **Entry** | **Yes**, but the panel opens 2022Q2 with **304 of 1,230 storefronts already operating (left-censored)** — true open dates unknown, excluded from entry models. 926 storefronts and 1,088 brand-ZIP cells enter in-window (242 expansion + 846 de novo). |
| **Exit** | **Partial.** Defined as ≥2 consecutive silent quarters before the panel end (a 1-quarter gap is a reporting gap, not exit — 172 such gap-quarters). 519 storefronts exit; right-censoring in the last 1–2 quarters is handled by restricting the exit sample to `qt ≤ T−2`. Quarterly baseline exit hazard **5.6%**. |
| **Realistic choice sets for a structural location model** | **Weak.** Potential entrants, their information sets, timing, and entry costs are not observed. Choice-set proxy (operating ZIPs; brand's existing footprint) supports a *reduced-form* / conditional-logit location analysis only. |
| **Cost, quality, price, quantity** | **None.** Revenue is the only demand realisation; it is **not** profit. |
| **Border / exposure** | **Strong.** Predetermined 75-mi + ≥1,000-non-legal-resident ZIP rule (reused from the existing paper, fixed at 2023-12-31, genuinely time-invariant since TX/OK/UT never legalised) **plus** newly-computed continuous distance-to-non-legal-stateline and non-legal-state population within 25/50/75/100 mi. |

**Bottom line:** everything needed for a Singleton-*inspired* reduced-form supply-side paper is
present; nothing needed for a Singleton *structural* replication is.

---

## 2. Mapping Singleton to cannabis — what transfers, what does not

*(Full table + equation-by-equation logic: `notes/01_singleton_template.md`.)*

| Singleton object | NM cannabis analogue | Transfers? |
|---|---|---|
| Charter school (school-year) | Dispensary storefront (address, quarter) | ✅ |
| Charter operator / management org | **Brand / parent operator** (constructed) | ✅ (constructed) |
| Charter *type* (No Excuses / for-profit / other) | **Brand type by scale/reach**: chain vs independent, # locations, age, # cities. *No* mission/for-profit analogue. | ⚠️ scale only |
| Census tract | ZIP code | ✅ (coarser) |
| School year | Calendar quarter | ✅ |
| Charter presence `Charters^k_{jt}` | `brand_present_{bzt}` | ✅ |
| Local households / `x^D` | ZIP population **+ non-legal-state population within 25–100 mi** | ✅ |
| Public + charter competitors by distance band | Other storefronts / brands within 5/10/25 mi, entry-exit-aware | ✅ |
| Enrollment `D_{ij}` | Storefront **revenue** (market-size realisation, *not* a headcount) | ⚠️ revenue only |
| **Flat per-pupil funding rate τ_t** | **Cross-border captive demand** — a location-linked revenue shifter the firm does not internalise as a cost. *Conceptual* analogue: τ_t is a known policy scalar with time variation; border demand is an estimated, fixed geographic attribute our own prior work says raises profitability. | ⚠️ conceptual, and it is the treatment, **not an instrument** |
| Entry / exit | Brand opens 1st store in ZIP / storefront closes permanently | ✅ (censoring) |
| School-type heterogeneity | Brand heterogeneity by scale | ✅ |
| Effectiveness μ_{it} (value-added / vertical quality) | **none** | ❌ |
| Variable cost `VC(D, μ, Z, x^C)` | **none** | ❌ |
| Fixed / entry cost | **none** (fees are public & modest; real cost unobserved) | ❌ |
| Funding-policy counterfactual | revenue-side accounting decomposition only | ⚠️ weak |

**Where the analogy holds:** free entry + revealed preference (NM has statutory anti-cap
language, no local retail opt-out); location pins down the served market and cannot be cheaply
moved; spatial competition with distance decay; heterogeneous operators whose objective weights
are an empirical question; entry and exit as *distinct* margins (τ_t / border demand nets out of
a pure location comparison but enters the enter-vs-stay-out and incumbent-exit margins —
Singleton p. 2592).

**Where it breaks (state plainly in any paper):** (1) **no quality object** — cannot ask
whether border demand sustains "bad" firms in a quality sense, only "small" ones; the
contribution must be about *market structure*, not welfare-via-quality. (2) **No cost data** —
revenue ≠ profit; cannot separate a demand effect from a margin effect; θ_i unidentified. (3)
**τ_t is a policy scalar; border exposure is a fixed geographic attribute** our own research
argues raises profitability → it is the treatment, has no exogenous within-NM time variation,
and is **not a valid instrument** for anything (task §13). (4) **Hundreds of brands, not 3
types** → a conditional-logit location model, not a 3-type equilibrium game. (5)
**Left/right-censoring** absent from Singleton's 2006–2013 window.

---

## 3. Descriptive brand structure (border vs interior) — **[ACTUAL]**

*(`logs/04_descriptives.log`, `tables/tab_border_vs_interior_composition.csv`,
`tables/tab_summary_stats.csv`.)*

**Statewide, the market is fragmented and unconcentrated** — consistent with genuine free
entry: 653 brands; the top-5 brands earn only **26.9%** of revenue, top-10 **42.2%**; statewide
revenue HHI = **229**. 27.6% of brands are multi-location but they run 62.3% of storefronts.
Largest brands: Pecos Valley Production (29 stores, $166 M), R Greenleaf (22, $137 M),
OCC ABQ (17), PurLife (27), Ultra Health (37, $84 M). One single-store border operator
(Fields of Dreams, Sunland Park) earned **$35.7 M** on its own — the "border megastore" pattern.

**Border vs interior ZIP-quarter (unconditional means, `border_z` = 1):**

| | border | interior | diff | p |
|---|---:|---:|---:|---:|
| # brands | 6.85 | 5.26 | **+1.59** | <0.001 |
| # storefronts | 7.11 | 5.55 | **+1.55** | <0.001 |
| HHI (0–10,000) | 4,741 | 5,922 | **−1,181** | <0.001 |
| largest-brand revenue share | 0.558 | 0.663 | **−0.105** | <0.001 |
| CR3 | 0.825 | 0.875 | −0.050 | <0.001 |
| entropy | 1.156 | 0.876 | +0.281 | <0.001 |
| independent-store share | 0.473 | 0.378 | +0.094 | <0.001 |
| contemporaneous chain-store share | 0.385 | 0.509 | −0.124 | <0.001 |
| total revenue ($) | 1.65 M | 1.15 M | +0.50 M | <0.001 |
| **revenue per store ($)** | 198 k | 185 k | +14 k | 0.14 (ns) |
| ZIP population | 15,968 | 18,571 | −2,603 | 0.002 |

**Storefront-level:** border storefronts belong to *smaller* brands (5.8 vs 7.1 lifetime
storefronts, p=0.02), are less likely chain (0.57 vs 0.65, p=0.006), face far less local
competition (38 vs 136 dispensaries within 25 mi — mostly a rural/ABQ artifact), and **exit
less** (0.38 vs 0.45, p=0.02).

**Reading:** border markets are **bigger, less concentrated, and more independent-heavy** — and
per-store revenue is *not* different, so the extra revenue is an extensive-margin (more stores)
phenomenon. This already tells us border markets are **not simply scaled-up interior markets**;
their *composition* differs.

---

## 4. Singleton Eq. (1) / Table 1 analogue — **brand presence** — **[ACTUAL]**

*(`logs/05_regressions.log §1`; LPM on 767,550 brand-ZIP-quarter cells, mean(present)=0.0125,
cluster ZIP.)*

| | (1) +Q FE | (2) +log pop | (3) +log dist | (4) +brand FE | (5) +lag competition | (6) ×brand type |
|---|---:|---:|---:|---:|---:|---:|
| `border_z` | 0.0042 (0.0031) | **0.0045\*\*** (0.0023) | 0.0007 (0.0039) | 0.0007 (0.0039) | −0.0002 (0.0003) | 0.0010 (0.0038) |
| `log_zip_pop` | | 0.0066\*\*\* | 0.0068\*\*\* | 0.0068\*\*\* | 0.0003\*\*\* | 0.0068\*\*\* |
| `log_dist_stateline` | | | −0.0019 | −0.0019 | −0.0003\*\*\* | −0.0019 |
| `border_z × multi-loc` | | | | | | −0.0008 (ns) |

Multi-location brands only: `border_z` = −0.003 (ns). Continuous asinh(non-legal pop 75 mi):
+0.00036\* (marginal).

**Finding:** with just population controlled, border ZIPs host ~36% more brands per potential
brand (0.0045 on a 0.0125 base). That gap **vanishes** once distance-to-stateline or brand FE
enter — the raw association is a market-size / rurality artifact. **No robust
extensive-margin brand-presence effect of border exposure.** This mirrors Singleton's Table 1,
where the raw correlation with public-school presence flips once `log density` is added.

---

## 5. Singleton Table 2 analogue — **exiters vs persisters** — **[ACTUAL]**

*(`logs/04_descriptives.log §D`, `tables/tab_exiters_vs_persisters.csv`; storefront level,
519 exiters vs 711 persisters/censored. Descriptive — motivates §6, not causal.)*

| variable | exiters | persisters | diff | p |
|---|---:|---:|---:|---:|
| border storefront | 0.33 | 0.40 | −0.07 | 0.02 |
| distance to non-legal stateline (mi) | 116.3 | 103.4 | +12.9 | <0.01 |
| local resident pop within 25 mi | 408,737 | 356,248 | +52,488 | 0.01 |
| dispensaries within 25 mi (mean) | 113.1 | 91.0 | +22.1 | <0.001 |
| competing brands within 25 mi (mean) | 77.5 | 63.0 | +14.5 | <0.001 |
| brand lifetime # storefronts | 5.7 | 7.4 | −1.7 | <0.001 |
| brand footprint (mean, quarters active) | 4.2 | 5.4 | −1.2 | <0.001 |
| chain (brand ≥2 stores) | 0.62 | 0.62 | +0.01 | 0.82 |
| mean positive-month revenue ($) | 96,829 | 214,081 | **−117,252** | <0.001 |
| first-year revenue ($) | 453,837 | 852,960 | −399,123 | <0.001 |
| quarters active | 6.5 | 10.2 | −3.7 | <0.001 |

**Reading:** exiters are lower-revenue, in **denser and more competitive** local markets, belong
to **smaller** brands, and are **less likely to be at the border**. Unlike Singleton (where
competition did *not* predict exit), here local competition is strongly associated with exit.
Cross-border exposure (`sf_nonleg_pop_75`) shows no raw difference because it is extremely
right-skewed; the binary border indicator is the informative one.

---

## 6. Singleton Eq. (2) / Table 3 analogue — **storefront exit** — **[ACTUAL]**

*(`logs/05_regressions.log §3`; LPM on 8,913 active storefront-quarters, baseline
Pr(exit next quarter) = **0.056**, cluster nearest-ZIP.)*

| | (1) | (2) | (3) | (4) |
|---|---:|---:|---:|---:|
| `border_dispensary` | −0.0000 (0.0060) | −0.0003 (0.0053) | −0.0025 (0.0067) | 0.0001 (0.0066) |
| dispensaries within 25 mi | 0.0001\*\*\* | −0.0001\* | −0.0006\* | −0.0005 |
| competing brands within 25 mi | | | 0.0008 | 0.0007 |
| same-brand stores within 25 mi | | | 0.0027\* | 0.0022 |
| log brand footprint | | | 0.0070\*\* | 0.0079 |
| log brand lifetime # stores | | | | −0.0113 |
| **chain dummy** | | | | **0.0426\*\*\*** |
| log(age+1) | −0.0047 | 0.0442\*\*\* | 0.0414\*\*\* | 0.0422\*\*\* |
| log revenue | | **−0.0437\*\*\*** | −0.0451\*\*\* | −0.0456\*\*\* |
| log local pop (25 mi) | | 0.0053\* | 0.0040 | 0.0038 |
| R² | 0.013 | 0.124 | 0.127 | 0.130 |

Continuous: **log distance to non-legal stateline = −0.0038\*\*** (spec-3 controls) →
storefronts *farther* from the border exit *less*, conditional on competition and revenue
(the raw pattern — farther = more exit — is driven by ABQ density and reverses on controls).
asinh(non-legal pop 75 mi) = +0.0004 (ns).

**Findings:**
- The **binary** `border_dispensary` does **not** predict exit on its own in any specification.
- **Revenue** (−0.044\*\*\*) and **age** dominate. The positive age coefficient (conditional on
  revenue) is a young-market artifact: almost no NM storefront is "old", and a mature store
  with a young store's revenue is under-performing its cohort → exits. Report with this caveat;
  it differs from Singleton's −0.033.
- **Own-brand crowding raises exit**: `same-brand stores within 25 mi` +0.0027\* and
  `log brand footprint` +0.0070\*\* — over-expanded chains cannibalise and shed stores.
- **Chain storefronts exit *more* than independents conditional on size (+0.043\*\*\*)** — but
  see the heterogeneity result in §8: this reverses at the border.
- Marginal effects are LPM coefficients directly; relative to the 5.6% base, a 1-log-point
  revenue increase (≈2.7×) lowers the quarterly exit probability by ~4.5 pp (≈80% of baseline).

---

## 7. Entry margin — **[ACTUAL]**

*(`logs/05_regressions.log §4`.)*

**Expansion entry** (676,321 feasible brand-ZIP-quarter cells where the brand is already in NM
and not yet in ZIP z; mean(entry)=0.00035):

| | (1) | (2) | (3) | (4) +brand FE | (5) ×network |
|---|---:|---:|---:|---:|---:|
| `border_z` | 0.00009 (0.00008) | −0.00002 | −0.00002 | −0.00003 | 0.00003 |
| `log_zip_pop` | 0.00020\*\*\* | 0.00021\*\*\* | 0.00017\*\*\* | 0.00017\*\*\* | 0.00017\*\*\* |
| log dist ZIP→nearest own store | | | **−0.00030\*\*\*** | −0.00046\*\*\* | −0.00049\*\*\* |
| own stores within 100 mi | | | 0.00036\*\*\* | −0.00007 (ns) | −0.00006 |
| `border_z × own-stores-100mi` | | | | | −0.00023 (ns) |

**De novo entry** (brand's first-ever NM store; ZIP-quarter count): `border_z` = 0.013 (ns),
`log_zip_pop` = 0.223\*\*\*.

**Findings:** **border exposure does not increase brand entry** — expansion or de novo, with or
without brand FE. Entry is driven by **ZIP population** and **proximity to the brand's own
existing network** (closer → more likely to expand there; the raw "own stores within 100 mi"
effect is brand size, gone with brand FE). The border × network interaction is null — no
evidence that established networks *specifically* target border markets.

---

## 8. Brand heterogeneity & sorting — **[ACTUAL]**

*(`logs/05_regressions.log §5–6`.)*

**Sorting (brand cross-section, `border_share_b` = share of a brand's storefronts that are border;
overall storefront border rate = 0.366):**

| | all brands (n=632) | multi-location only (n=172) |
|---|---:|---:|
| `log(# storefronts)` | −0.031 (0.040) | **−0.161\*\*** (0.066) |
| brand age (quarters) | −0.012\*\*\* | −0.007 (ns) |
| `# cities` | 0.005 (ns) | 0.024\*\* |

Mean `border_share` by brand size (multi-location brands): **large (6+ stores) 0.27**, mid
(3–5) 0.36, **small (2) 0.42**; singletons 0.42, multi-location 0.38.

**Heterogeneity of the border response:**
- **Exit**: `border_dispensary × chain` = **−0.017\*** (spec-3 controls); `border_dispensary`
  alone (≈ independents) = +0.011 (ns). So **at the border, chain storefronts exit less;
  independents no different.** In the mature market (2024Q1+) the interaction is **−0.037\*\*\***.
- **Entry**: `border_z × multi-location` = +0.0002 (ns) — no differential expansion response.
- **Presence**: `border_z × multi-location` = −0.0008 (ns); `border_z × log(brand size)` = ns.

**Findings:** **the big-chain-captures-the-border hypothesis is rejected.** Large statewide
chains *under*-locate at the border; small operators and independents *over*-locate there.
The one place chains do better at the border is **survival** — a border chain storefront is
markedly less likely to close than an interior chain storefront, increasingly so over time.

---

## 9. Network effects — **[ACTUAL]**

*(`logs/05_regressions.log §4 (3)–(5)`.)*

- **Proximity to a brand's existing network strongly predicts where it expands**: a ZIP one
  log-point farther from the brand's nearest current storefront is ~0.0003–0.0005 less likely
  to receive that brand's next store (on a 0.00035 base — a large relative effect). This is a
  clean, robust "diffusion from the existing footprint" result (cf. Holmes 2011 on Wal-Mart).
- **But that diffusion is not tilted toward the border**: `border_z × own-stores-within-100mi`
  is a precise null. Established networks expand outward from where they are; the border is not
  a special draw for them.
- Own-brand crowding *raises* storefront exit (§6) — networks that over-densify shed stores.

**Implication:** the "network expansion into border markets" mechanism (task §17, mechanism F)
is **not** what is happening. If anything the border is filled by *new / small* entrants, not by
incumbents leveraging a network.

---

## 10. Concentration & composition — the headline result — **[ACTUAL]**

*(`logs/05_regressions.log §2`, `tables/tab_concentration.csv`,
`robustness/tab_robustness_concentration.csv`, `logs/06_dynamics_robustness.log §8`.)*

ZIP-quarter LPM/OLS, quarter FE, cluster ZIP. `border_z` and `log_dist_stateline` are collinear
border proxies → reported in **separate** columns, never together.

| outcome (mean) | +log pop, `border_z` | +log pop, `log_dist` | +log pop, asinh non-legal pop 75 mi |
|---|---:|---:|---:|
| log # brands (1.21) | **+0.394\*\*\*** | **−0.188\*\*\*** | +0.030\*\*\* |
| # brands (5.71) | +1.89\* | −0.93\*\* | +0.148\* |
| **HHI, 0–10,000 (5,583)** | **−1,397\*\*\*** | **+685\*\*\*** | −103\*\*\* |
| largest-brand share (0.633) | **−0.124\*\*\*** | +0.061\*\*\* | −0.009\*\*\* |
| CR3 (0.861) | −0.059\*\* | +0.023\*\* | −0.004\* |
| entropy (0.956) | +0.332\*\*\* | −0.155\*\*\* | +0.025\*\*\* |
| independent-store share (0.405) | +0.084 (p≈0.14) | −0.003 (ns) | +0.010\* |
| contemporaneous chain-store share (0.474) | −0.112\*\* | +0.016 (ns) | −0.008\*\* |
| log total revenue (12.96) | +0.616\*\* | −0.382\*\*\* | +0.051\*\*\* |
| log revenue per store (11.72) | +0.228 (ns) | −0.191\*\*\* | +0.022\* |
| brand entry rate (0.129) | +0.025\*\* | −0.011\*\*\* | +0.003\*\*\* |
| brand exit rate (0.043) | −0.002 (ns) | +0.003 (ns) | −0.000 (ns) |

`e^{0.394} − 1 ≈ +48%` more brands; `e^{0.616} − 1 ≈ +85%` more revenue.

**Compositional decomposition:** border ZIPs get **+1.26\*\*\* more independent brands** but only
**+0.63 (ns) more chain brands** — the extra brands at the border are disproportionately
independents.

**Robustness (headline results — `robustness/tab_robustness_concentration.csv`):** log-#-brands
and HHI effects survive: border defined at 50 mi (+0.43\*\*\* / −1,469\*\*\*); continuous
non-legal-pop exposure at all four radii (\*\*/\*\*\*); excluding Albuquerque (+0.48\*\*\* /
−1,456\*\*\*); excluding ZIPs < 2,500 pop (+0.41\*\* / −1,396\*\*); early vs mature sub-periods;
**dropping the largest / top-3 / top-5 brands** (+0.37–0.39\*\*\* / −1,322 to −1,431\*\*\*);
**clustering by county instead of ZIP** (all headline coefficients stay \*\*/\*\*\* — log-#-brands
+0.39\*\*\*, HHI −1,397\*\*\*, largest-brand share −0.124\*\*\*, revenue +0.62\*\*\*); adding **county
fixed effects** (log-#-brands +0.55\*\*, though HHI loses power, −1,256, p≈0.29, as expected with
limited within-county border variation). The `revenue-per-store` and
`independent-store-share` effects are the weakest — the continuous distance measure is
significant for revenue-per-store (−0.19\*\*\*) but the binary is not, and independent-share is
only marginal.

**So border markets are: (1) larger; (2) more competitive / less concentrated; NOT (3) more
concentrated; and served disproportionately by (4) particular firms — small and independent
ones, not big chains.**

---

## 11. Robustness — **[ACTUAL]**

*(`code/06_dynamics_robustness.py`, `logs/06_dynamics_robustness.log`,
`robustness/tab_robustness_concentration.csv`.)*

Run and passed for the headline concentration/composition results: alternative border
definitions (50/75/100 mi); continuous distance-to-stateline; non-legal population exposure at
25/50/75/100 mi; excluding Albuquerque (Bernalillo county); excluding ZIPs under 2,500
population; excluding the largest / top-3 / top-5 brands; early (2022Q2–2023Q4) vs mature
(2024Q1–2026Q2); county fixed effects; and (for the exit interaction) dropping the 134
ownership-change storefronts and restricting to the mature market. **Not yet run / open items:**
alternative competitor radii for the presence and entry models; a full conditional-logit
location-choice specification; a placebo using distance to the *legal* CO/AZ line (the existing
project's ZIP-level version of this placebo passed — a within-project cross-check); ZIP fixed
effects are **not usable** — they absorb the time-invariant `border_z` and the identifying
variation (noted, per task §20).

---

## 12. Structural feasibility

| Singleton element | Adaptable? | Why |
|---|---|---|
| Eq (1) presence / Table 1 | **YES** | done — §4. |
| Table 2 exiters vs persisters | **YES** | done — §5. |
| Eq (2) / Table 3 exit LPM | **YES** | done — §6. |
| Eq (4)–(5) operating utility `U(Π, D, μ)`, `Π = R − VC − FC` | **PARTIALLY** | can write a *latent-profit index* `Π_{bzt} = R_{bzt} − VC_{bzt} − FC_{bzt}` conceptually and estimate a reduced-form profit proxy from revenue + competition + demographics; **cannot populate VC, FC, or μ**. |
| Eq (9) incumbent continue-vs-exit | **PARTIALLY** | reduced-form exit hazard yes (§6); a structural continuation value needs forward-looking VC and an option value we cannot form. |
| Eq (10) entrant location choice | **PARTIALLY** | a reduced-form / conditional-logit choice among operating ZIPs is feasible (the entry regressions in §7 are the linear version); a structural version with entry cost `EC_t` and equilibrium beliefs is not. |
| Eq (18)–(22) revealed-preference entry-game estimation of θ_i | **NO** | requires offline demand + variable-cost estimates that do not exist for NM dispensaries; θ_i would be identified only under an unfalsifiable cost assumption; the market-unobservable ξ_t is collinear with a policy scalar that, here, has no time variation. |

**Verdict: YES for the reduced-form skeleton (Eqs 1–2, Tables 1–3), PARTIALLY for the
conceptual profit/location framing, NO for the structural estimator.**

**Strongest reduced-form model.** A ZIP-quarter panel of market-structure outcomes (# brands,
HHI, largest-brand share, entropy, independent-brand count, entry/exit rates, revenue) on the
predetermined border indicator *and* the continuous non-legal-population-exposure gradient, with
`log(population)` and quarter fixed effects, clustered by ZIP; plus a storefront-quarter exit
LPM with the `border × chain` interaction; plus the brand cross-section `border_share_b`
sorting regression; plus the entry LPM showing the null border effect and the positive
own-network-proximity effect. That is a complete, internally consistent supply-side paper.

---

## 13. Literature & novelty

**Closest methodological templates (verified references):**
- **Singleton, J. D. 2019.** "Incentives and the Supply of Effective Charter Schools." *AER*
  109(7): 2568–2612. — the framing: location & survival as revealed preference under a
  location-linked funding distortion; separate entry/exit margins; type-heterogeneous
  financial responsiveness.
- **Bresnahan, T. & P. Reiss. 1990/1991.** *RES* / *JPE*. — entry thresholds; competitive
  conduct changes most with the 2nd–3rd entrant.
- **Seim, K. 2006.** "An Empirical Model of Firm Entry with Endogenous Product-Type Choices."
  *RAND J. Econ.* 37(3): 619–640. — first to endogenise store location in an entry game;
  semi-structural spatial profit.
- **Jia, P. 2008.** "What Happens When Wal-Mart Comes to Town." *Econometrica* 76(6):
  1263–1316. — chain scale economies; chains vs independent discounters.
- **Nishida, M. 2015 (*Mgmt. Sci.*) / 2025 (*IER*).** Convenience-store chains, Okinawa;
  multi-store entry; entry order, performance and geography.
- **Holmes, T. 2011.** "The Diffusion of Wal-Mart and Economies of Density." *Econometrica*
  79(1): 253–302. — network diffusion from an existing footprint (our §9).
- **Datta, S. & K. Sudhir.** "Does Reducing Spatial Differentiation Increase Product
  Differentiation? Effects of Zoning on Retail Entry and Format Variety." *QME*. — regulation
  → *composition/format* variety, not just store count (directly parallel to our "composition,
  not just scale" point).
- **Aguirregabiria, V. & J. Suzuki.** "Empirical Games of Market Entry and Spatial Competition
  in Retail Industries." *Handbook* chapter. — the methodological review closest to our
  problem; a better structural map than Singleton *if* a structural version is ever attempted.
- Toivanen & Waterson 2005 (*RAND*); Igami & Yang 2016 (*Quant. Econ.*) — dynamic rival-presence
  and geographic chain entry.

**Cross-border demand (verified):**
- **Hansen, B., K. Miller & C. Weber. 2020.** "Federalism, Partial Prohibition, and
  Cross-Border Sales." *JPubE* 187: 104159. — WA retailers on the OR border lost 36% of sales
  when OR's market opened; the border premium lives in *revenue at existing stores*. Our result
  is the supply-side complement: in a **free-entry** state it shows up in *store and brand
  counts*, not per-store revenue.
- **Hollenbeck, B. & K. Uetake. 2021.** "Taxation and Market Power in the Legal Marijuana
  Industry." *RAND J. Econ.* 52(3): 559–595. — WA's entry *cap* produced local-monopoly market
  power and ~54% retail margins. NM is the mirror image: free entry, fragmentation, HHI 229.
- Asplund, Friberg & Wilander 2007 (*JPubE*); Chandra, Head & Tappata 2014 (*ReStat*);
  Friberg et al. 2024 (*Scand. J. Econ.*) — cross-border shopping in alcohol/tobacco; border
  areas have more retailers.
- Cannabis market structure: Thomas & Tian 2021 (*RSUE*); Dong & Tyndall 2024 (*Ann. Reg.
  Sci.*); Secretariat 2023 industry report on cannabis market structure (grey literature).
- Ferreyra & Kosenok 2018 — charter demand and entry in Washington DC (cited by Singleton).

**The precise novelty question — has prior research studied how a *predetermined geographic
demand advantage* (such as cross-border demand) affects **firm/brand composition, entry, exit,
and concentration in a free-entry retail market**?**

- Demand-side cross-border leakage in cannabis: **established** (Hansen–Miller–Weber; the
  existing Venous4 border-premium paper).
- Supply-side entry-game / spatial-competition IO: **established** (Bresnahan–Reiss, Seim, Jia,
  Nishida, Aguirregabiria).
- Cannabis entry & market power under **caps**: **established** (Hollenbeck–Uetake).
- **The specific combination — a spatial, predetermined demand advantage reshaping the
  *composition* (concentration, chain-vs-independent mix, survival heterogeneity) of a
  *free-entry* retail market — has not, to our knowledge, been studied.** The nearest work
  (Datta–Sudhir on zoning → format variety) is about a regulatory constraint, not a demand
  advantage, and not a cross-border one. This is a genuine, fillable gap. It is an *incremental*
  contribution methodologically (reduced-form, a known IO regularity applied in a new setting),
  but a *novel* empirical fact: cross-border demand in a free-entry market **dissipates into a
  larger competitive fringe of small/independent firms rather than into concentration or
  chain capture.**

---

## 14. Publication potential

**Assessment.** Identification: predetermined geography + continuous exposure gradient, no
reliance on within-NM time variation; the border indicator is the treatment, correctly *not*
used as an IV. Data quality: a clean census of licensee revenue, a defensible but
un-validatable brand crosswalk, left/right censoring, 4-year young market, only 33 border ZIPs.
Results: the concentration/composition result is strong and robust; the presence, entry, and
unconditional-exit nulls are clean; the `border × chain` survival result and the sorting result
are suggestive-to-solid. Mechanism: coherent (A scale + D competitive expansion + reversed-B
sorting + conditional-E survival). Contribution: a novel empirical fact in a small,
policy-relevant literature; incremental method.

**This is best positioned as a *major extension of the existing border-premium paper*** (a
"who supplies the border premium, and does it concentrate?" section or companion), **or a
standalone field / regional / applied paper**, not a top-5 or top-field IO paper (no structural
model, single state, 4 years, brand crosswalk not externally validated).

| Journal | Orientation | Fit | Major strength | Major weakness | Needed before submission | Prob. |
|---|---|---|---|---|---|---|
| **Journal of Regional Analysis & Policy** (the project's current target) | regional/applied | high — border geography, market structure, policy | the project's own established venue; the composition result is genuinely new | not a high-visibility outlet | fold into the existing draft or write a tight standalone; add the legal-border placebo | **Realistic–Strong, ~65%** |
| **Journal of Cannabis Research** | interdisciplinary cannabis policy | high | strong topical fit; free-entry vs capped-market framing travels well | limited economics readership | frame around policy (what free entry does to market structure); light on structural jargon | **Strong, ~65%** |
| **Regional Science and Urban Economics** | spatial/urban econ | medium–high | spatial demand advantage → firm composition is squarely RSUE territory (cf. Thomas–Tian) | referees will want either a structural model or a sharper identification story than "predetermined geography" | add conditional-logit location choice; tighten the exposure gradient; second state (CO border counties) as external validation | **Realistic, ~35–45%** |
| **Journal of Economic Geography** | econ geography | medium | "captive demand dissipates into a competitive fringe" is a clean geography narrative | methods may read as thin for JEG | more theory framing; connect to agglomeration / selection literature | **Realistic, ~30–40%** |
| **International Journal of Industrial Organization** | IO | low–medium | the entry/exit/composition question is IO-core | reduced-form only; one state; no welfare | would realistically need the structural entry model (Aguirregabiria template) — a large lift | **High-risk, ~15–25%** |
| **Journal of Public Economics** | public econ | low | cross-border / federalism angle (cf. Hansen–Miller–Weber) | no tax/welfare quantification; supply-side composition is not the usual JPubE object | add a fiscal/welfare decomposition; needs the structural model for counterfactuals | **High-risk, <20%** |
| **Quantitative Marketing & Economics** | quant marketing | low–medium | brand entry / assortment / chain-vs-independent is QME-adjacent (Seim, Datta–Sudhir here) | no consumer demand model, no brand-choice data | build the demand side (need transaction/scanner data we do not have) | **High-risk, ~20%** |

Do **not** inflate: RAND J. Econ. / *AEJ:Micro* / *Econometrica* are out of range for a
single-state reduced-form paper.

---

## 15. The economic mechanism

Border cross-border demand in NM's free-entry market operates through:

- **A. Scale — YES (strong).** Border ZIPs are bigger markets: +48% brands, +55% storefronts,
  +85% revenue, per unit of resident population.
- **B. Brand sorting — YES, but *reversed*.** Small and independent operators sort *toward*
  the border; large statewide chains sort *away* (border share 0.27 for 6+-store brands vs
  0.42 for independents). Cross-border demand selects a *particular* type of firm — the small
  one.
- **C. Concentration — NO.** Border markets are *less* concentrated (HHI −1,400, largest-brand
  share −12 pp, higher entropy). Captive demand is not captured as rents by dominant firms.
- **D. Competitive expansion — YES (strong).** Free entry converts the border demand into a
  larger, more fragmented field: more brands, more independent brands specifically, higher
  entropy, (early) higher brand entry rate, lower chain-store share.
- **E. Survival — PARTIAL / conditional.** No robust unconditional border survival effect; a
  marginal protective distance gradient conditional on competition; and a clear **border ×
  chain** survival advantage that strengthens as the market matures.
- **F. Network expansion — NO.** Established chains do not use their networks to move into the
  border; the border × own-network interaction is a precise null, and chains under-locate
  there.
- **G. Combination — the story is A + D + reversed-B + conditional-E.**

**Most consistent single interpretation:** in a free-entry market, a predetermined spatial
demand advantage does not accrue to incumbents or large chains as market power; it is
**competed away** by a wave of small, independent entrants, producing border markets that are
larger *and* structurally more competitive than interior markets of the same size. The chains
that *do* locate at the border enjoy better survival — consistent with the border demand
raising the profitability floor for a well-run store — but they do not dominate the market or
crowd out the fringe.

---

## BOTTOM LINE — direct answers

1. **Are border markets compositionally different from interior cannabis markets?**
   **Yes.** Same population, but border ZIPs have ~48% more brands, HHI lower by ~1,400 points,
   largest-brand share lower by ~12 pp, higher entropy, more independent brands. Robust to
   border definition, continuous exposure, excluding ABQ, excluding small ZIPs, dropping the
   top brands, and county FE (for brand counts). Not merely scaled up — structurally different.

2. **Are particular brands disproportionately represented at borders?**
   **Yes — small and independent ones.** Border share is 0.42 for 2-store brands and
   independents vs 0.27 for 6+-store chains; `log(brand size)` coefficient on border share is
   −0.16\*\* among multi-location brands.

3. **Does border exposure increase brand entry?**
   **No.** Null in every presence and entry specification (expansion and de novo) once
   population is controlled. Entry is driven by ZIP population and proximity to the brand's own
   network.

4. **Does border exposure reduce exit?**
   **Not on its own.** The binary border indicator does not predict exit. But (a) the
   continuous distance-to-stateline gradient is marginally protective conditional on
   competition (−0.0038\*\*), and (b) **chain** storefronts at the border exit significantly
   less than interior chain storefronts (`border × chain` = −0.017\*, −0.037\*\*\* in the
   mature market).

5. **Are border markets more concentrated or more competitive?**
   **More competitive / less concentrated**, unambiguously.

6. **Are large chains more border-responsive than independents?**
   **No — less.** Large chains under-locate at the border. The only chain-specific border
   response is *survival*, not entry.

7. **Do existing brand networks predict border expansion?**
   Networks predict expansion **in general** (proximity to a brand's footprint is a strong
   positive predictor of where it opens next), **but not toward the border specifically** —
   the border × network interaction is a precise null.

8. **Does the effect survive population and market controls?**
   The **composition/concentration** effects: yes, robustly (and confirmed by the continuous
   distance and exposure gradients). The **presence and entry** effects: no — they are
   population artifacts.

9. **Does controlling for brand heterogeneity weaken or strengthen the border effect?**
   For **presence/entry** it does not matter (already null). For **concentration** the effect
   is *about* brand heterogeneity (it is a statement about the brand mix) and survives dropping
   the largest brands. For **exit**, adding brand type reveals the `border × chain` interaction
   that the pooled coefficient hides.

10. **Can we credibly estimate a Singleton-style structural entry model?**
    **No.** No cost data (θ_i unidentified), no quality object, no time-varying policy scalar,
    hundreds of brands rather than 3 types, and left/right censoring.

11. **If not, what is the strongest reduced-form model?**
    A ZIP-quarter panel of market-structure outcomes on the predetermined border indicator and
    the continuous non-legal-population exposure gradient, with `log(population)` + quarter FE,
    clustered by ZIP; **plus** the storefront-quarter exit LPM with `border × chain`; **plus**
    the brand cross-section `border_share_b` sorting regression; **plus** the entry LPM
    (null border, positive own-network proximity). See §12.

12. **Is the contribution novel?**
    **Yes, as an empirical fact** — a spatial, predetermined demand advantage reshaping the
    *composition* (concentration, chain-vs-independent mix, survival heterogeneity) of a
    *free-entry* retail market has not been studied; the nearest work is on caps
    (Hollenbeck–Uetake) or zoning (Datta–Sudhir). **Incremental as method** (reduced-form, a
    known IO regularity in a new setting).

13. **Is there a publishable paper here?**
    **Yes** — as a major extension of the existing border-premium paper, or a standalone
    regional/applied/cannabis-policy paper. Not a top-field IO paper without a structural model.

14. **Which journal family fits better: economics/IO or quantitative marketing?**
    **Economics — specifically regional / applied / public-economics-adjacent**, and
    cannabis-policy. Quantitative marketing needs a consumer-demand and brand-choice model that
    the data cannot support. Pure IO needs the structural entry game.

15. **What are the next three analyses to prioritise?**
    1. **Conditional-logit location choice** for expansion entrants (choice set = operating
       ZIPs; utility = f(border, non-legal exposure, population, distance to own network,
       local competition)) — the proper Singleton Eq. (10) analogue, and what an IO referee
       will ask for.
    2. **Legal-border placebo** at the storefront/ZIP level (distance to the *legal* CO/AZ
       line) — the ZIP-level version passed in the existing project; redo it on the brand /
       concentration outcomes to show the composition effect is cross-border-demand-specific,
       not a generic "near any state line" artifact.
    3. **Second-market external validation**: build the same brand crosswalk and
       ZIP/county-quarter composition panel for Colorado's non-legal-border counties (UT / OK
       / KS / NE / WY borders), where the project already has dispensary and revenue data — a
       free-entry-ish comparison that would show whether the "fragmentation, not concentration"
       result generalises.
