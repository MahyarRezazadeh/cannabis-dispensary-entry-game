# Singleton (2019, AER 109(7):2568–2612) as a methodological template

Research note for `brand_entry_analysis/`. Purpose: distill the economic logic of the
parts of Singleton we can adapt, translate them to the New Mexico cannabis setting, and
mark what transfers vs. what does not.

--------------------------------------------------------------------------------
## 1. What Singleton does, in one paragraph

Florida does not cap charter schools and approves them on minimal criteria, so where a
charter opens and whether it survives can be read as **revealed preference**. Charters are
publicly funded by a **flat per-pupil formula** that does not adjust for the cost of
educating disadvantaged students. Because charters cannot screen students but *self-select
a location*, and student composition (hence variable cost) is pinned down by location, the
flat formula creates a strategic incentive to locate where the served population is
cheap to educate ("cost-based cream-skimming"). Singleton (i) estimates school
effectiveness (value-added) from a test-score panel; (ii) estimates demand and a variable
cost function; (iii) plugs these "offline" objects into a **two-step (Hotz–Miller) dynamic
entry game** to recover charter operators' objective weights on net income vs. other goals,
separately by operator type (No Excuses / for-profit / other); (iv) runs funding
counterfactuals.

--------------------------------------------------------------------------------
## 2. The specific objects the task asks us to reproduce

### Preliminary / reduced-form evidence

**Equation (1) — determinants of charter-type presence (Table 1).**
Stacked LPM over (tract j, type k, year t):
`Charters^k_{jt} = β^k X_{jt} + ε^k_{jt}`
`Charters^k_{jt}` = 1 if ≥1 charter of type k operates in tract j at t.
`X_{jt}` = local public-school test performance (`Public z̄`), competition counts by
distance band (# publics <1mi, 1–3mi, 3–5mi), `log population`, `log density`,
`log HH income`, `% black`, `% Hispanic`, and type interactions. District×year FE.
66,213 tract×type×year obs. Findings: charter presence is *negatively* related to nearby
public-school performance (strategic entry + differential exit); positively related to
%black and %Hispanic; the raw positive correlation with public-school *presence* is a
market-size artifact that flips once `log density` is added; for-profits are the most
cost-oriented, No Excuses the least.

**Table 2 — characteristics of exiters vs. persisters.** Not a regression per se: for each
variable, regress it on exiter / persister / entrant dummies + age + year FE using the full
panel, and report conditional means. Location characteristics (density, HH income, % white,
% black, …), student characteristics (enrollment, % FRP lunch, race shares), school
characteristics (expenditure/pupil, math/reading z, No Excuses, for-profit). Exiters sit in
denser, ~$16k-lower-income tracts, serve +21pp FRP-lunch and +37pp black students, spend
~$900 (~10%) more per pupil, and are ~1σ lower value-added.

**Equation (2) — probability of exit (Table 3).** LPM, *not* IV:
`Exit_{it} = β f(Z_{it}) + γ X_{it} + ε_{it}`
`Exit_{it}` = 1 if school i exits between t and t+1. `Z_{it}` = student-body composition
(race shares, FRP-lunch), entered with nonlinearities (quadratics/interactions). `X_{it}` =
competition (# publics/charters nearby), `log enrollment`, reading z, `log(age+1)`, tract
density & demographics. District & year FE. Columns add controls progressively: (1)
competition + student body + age; (2) + enrollment; (3) + reading z; (4) + school type.
Findings: competition does **not** predict exit; larger and higher-performing charters exit
less; **% black remains a positive, stable predictor of exit across all specs** — read as a
cost proxy that the flat formula does not compensate. Baseline Pr(Exit) ≈ 0.046.

### Structural framework (Section III)

**III.B Charter School Utility.**
`U_{ij}(s_t) = U_i( Π_{ij}(s_t), D_{ij}(s_t), μ_{it} )`   (4)
Latent utility of operator i running a school in location j at state `s_t`, as a function of
net income Π, the vector of student-type enrollments D, and effectiveness μ. Utility is
**operator-type-specific** (`U_i`), so responsiveness to money is an empirical question.

`Π_{ij}(s_t) = τ_t D_{ij}(s_t) − VC_{ij}(s_t) − FC_{jt}`   (5)
Net income = (flat funding rate τ_t × enrollment) − variable cost − commonly-known fixed
cost. VC depends on enrollment, effectiveness, **student composition Z**, and location
characteristics `x^C_{jt}` (eq 8) — this composition-dependence of cost is the whole
mechanism.

Demand (eq 6–7): enrollment of household type z at school i is `D^z_{ij} =
D^z_i(μ_{it}, x^D_{jt}, a_{-it}, μ_{-it})` — depends on own effectiveness, local market
size, and the locations/effectiveness of competitors (spatial competition).

**III.C Entry, Location and Exit.**
Incomplete-information simultaneous-move game each (district, year). Incumbents decide
exit; potential entrants decide whether and where to enter; then households choose.

Incumbent continues in location j iff
`E[U_j(a_{-i}, s_i, s_{-i}, x_t)] + ε_{ijt} ≥ ε_{i0t}`   (9)
(utility of exiting normalised to 0).

Potential entrant solves
`max{ E[U_j(...)] − EC_t + ε_{ijt}, …, E[U_{j'}(...)] − EC_t + ε_{ij't}, ε_{i0t} }`   (10)
i.e. pick the best location net of entry cost `EC_t`, or stay out.

**IV.B Entry-game estimation (eqs 18–22).**
Utility made quasi-linear in net income per pupil:
`U_{ij}(s_t) = θ_i Π_{ij}(s_t)/D_{ij}(s_t) + g_i( D_{ij}, Z_{ij}, μ_{it} )`   (18)
`θ_i` = marginal utility of net income per pupil, type-specific → *the key parameter*.
Rewriting with (5): `U_{ij} = θ_i( τ_t − VC_{ij}/D_{ij} ) + ḡ_i( FC, D, Z, μ )`   (19).
(20): the incumbent's continue-vs-exit comparison in expected-utility terms; τ_t shifts it.
Identification argument (page 2592): for a *location* choice between j and j′, τ_t **drops
out**, so θ_i is identified only from the *sensitivity of entrants' location choice to
expected variable-cost differences*. From the *enter-vs-stay-out* and *incumbent
exit* margins, τ_t does **not** drop out and can help identify θ_i. Market-level
unobservables ξ_t are **collinear with τ_t** — so with unobserved heterogeneity, only
expected VC/pupil identifies financial responsiveness; hence the offline cost estimates
matter.
Type-I EV choice shocks ⇒ closed forms:
`P^I_{ij} = exp(ū_{ij}+ξ) / (1 + exp(ū_{ij}+ξ))`   (21)   (incumbent continuation)
`P^E_{ij} = exp(−EC+ū_{ij}+ξ) / (1 + Σ_j exp(−EC+ū_{ij}+ξ))`   (22)  (entrant location)
Two-step Hotz–Miller: step 1 recovers CCPs `P̂_t` (multinomial logit with flexible
polynomial in state vars + district×year FE); step 2 simulates beliefs and estimates the
utility/entry-cost parameters by matching choice probabilities. Pooled over 26 districts
with ≥3 entries; entrant choice set = all tracts in the assigned district; potential-entrant
pool sized to (# observed entrants + ¾·# observed, ≥3), split ⅓ to each type.

--------------------------------------------------------------------------------
## 3. Mapping table: Singleton object → New Mexico cannabis analogue

| Singleton object | Cannabis analogue (this project) | Feasible with our data? |
|---|---|---|
| Charter school (physical school-year) | Dispensary = a physical retail storefront, identified by normalized street address, observed at monthly frequency | **Yes.** 1,253 storefronts, Apr-2022–May-2026, monthly revenue. |
| Charter operator / management org | **Brand / parent operator** — the entity that owns one or more storefronts. Must be constructed from messy "Licensee" name strings (see `brand_cleaning_procedure.md`). | **Partially.** No official ownership registry; constructed from name strings + address + hand-curated aliases. ~[N] brands, of which ~[M] multi-location. |
| Charter *type* k (No Excuses / for-profit / other) | **Brand type** — statewide chain vs. independent; large vs. small; brand age; number of NM locations; multi-city footprint. No ideology/for-profit analogue; heterogeneity is by *scale and reach*, which is what the data supports. | **Partially.** Scale/reach/age: yes. "Objective type" (mission vs. profit): no. |
| Census tract j | **ZIP code z** (368 populated NM ZIPs; ~116–139 ever have dispensary activity). Coarser than a tract but it is the finest predetermined market unit with population + border geography already built in this project. | **Yes**, with the caveat that ZIP ≠ a clean "local market"; robustness uses continuous distance and 25/50/75/100-mi exposure. |
| School year t | **Calendar quarter t** (2022Q2 … 2026Q2, 17–18 quarters). Monthly data aggregated to quarters to cut reporting noise. | **Yes.** |
| Charter presence `Charters^k_{jt}` | `brand_present_{bzt}` = 1 if brand b operates ≥1 storefront in ZIP z in quarter t. | **Yes.** |
| Local households / demand shifters `x^D_{jt}` | ZIP resident population; **plus** the cross-border piece: nonlegal-state population within 25/50/75/100 mi; distance to the nearest nonlegal state line. | **Yes** (population from Census/SimpleMaps; nonlegal exposure computed from nonlegal-state ZIP centroids). |
| Public + charter competitors, by distance band | Other dispensaries and other brands within 5/10/25 mi; count in own ZIP; count in adjacent ZIPs. | **Yes** (from geocoded storefronts, entry/exit-aware per quarter). |
| Enrollment `D_{ij}` (demand / market-size realisation) | **Storefront revenue** (`total_sales`), and market revenue at ZIP level. Used as a *market-size / demand realisation* proxy, **not** as a structural enrollment object. | **Yes for revenue.** No headcount/transactions/quantity/price. |
| Flat per-pupil funding rate τ_t | **Cross-border demand advantage.** There is no cannabis "funding formula"; the analogue of "a location-linked revenue subsidy the firm does not fully internalise in its cost" is the *captive out-of-state demand* that accrues to border storefronts. Border exposure raises revenue per store (established in the existing paper) without a commensurate cost difference. | **Conceptual analogue only.** τ_t is a known policy scalar; border demand is an estimated, location-specific revenue shifter. This is the biggest divergence — see §4. |
| Entry: charter opens in a tract | **Brand entry** = brand opens its *first* storefront in ZIP z (or: brand's first NM storefront ever, for the extensive margin). | **Yes**, subject to left-censoring at 2022Q2. |
| Exit: charter closes | **Storefront exit** = storefront stops reporting sales and does not resume (with a min-gap rule so a 1-quarter reporting gap ≠ exit); **brand exit** = brand's last NM storefront closes. | **Partially.** Right-censored at panel end; "exit" vs "prolonged non-report" is a judgement call, handled explicitly. |
| Effectiveness μ_{it} (value-added, vertical quality) | **No analogue.** There is no product-quality measure for a dispensary (no reviews, no lab-potency panel, no price index). Revenue conflates quality, size and demand. | **No.** This is a real loss — Singleton's whole "does equity cost effectiveness?" question has no counterpart. |
| Variable cost function VC(D, μ, Z, x^C) | **Not observed.** No expenditure, wage, rent, COGS, or margin data for NM dispensaries. | **No.** Kills a literal structural replication (see §5). |
| Fixed cost FC_{jt} | Not observed; at best proxied by commercial-rent indices by area (not in project). | **No / weak proxy only.** |
| Entry cost EC_t | Latent. NM application + license fees are public and modest; real entry cost (buildout, security, real estate) unobserved. Could be a *latent* parameter in a Rust/BR-style model but not separately measured. | **Latent only.** |
| Funding-formula counterfactual | "What if border demand were taxed away / redistributed" — a counterfactual on the *revenue* side only, since we cannot simulate quality or cost responses. | **Weak.** Possible as an accounting decomposition, not a welfare statement. |

--------------------------------------------------------------------------------
## 4. Where the analogy holds

- **Free entry + revealed preference.** NM is *the* clean free-entry recreational market
  (statutory anti-cap language, NMSA 1978 §26-2C-6(J); no local opt-out for retail). Like
  Florida charters, NM dispensary location and survival can be read as revealed preference
  with respect to a spatial profit motive. This is a *better* institutional fit than most
  cannabis settings and is the reason Singleton is a natural template.
- **Location pins down the served market, and firms cannot move it cheaply.** A dispensary,
  like a charter, serves a spatially local (travel-cost-bounded) population; the border
  variant of that population (out-of-state consumers) is the analogue of Singleton's
  "cheap-to-serve" population that the flat formula over-rewards.
- **Spatial competition with distance decay.** Both settings have imperfectly competitive
  spatial markets where a competitor 1 mile away matters and one 5 miles away barely does
  (Singleton Table 5). We can build the same distance-banded competitor counts.
- **Heterogeneous operators.** Singleton's headline methodological move — let the objective
  function vary by operator type and recover the weights by revealed preference — maps
  directly onto "do statewide chains respond to border demand differently from
  independents?". We cannot recover a structural θ_i, but we *can* estimate
  type-interacted reduced-form responses, which is the same economic question.
- **Entry vs. exit as distinct margins.** Singleton separates incumbents' continuation
  (eq 9) from entrants' location choice (eq 10). We keep that separation: a storefront-exit
  hazard and a brand-entry model, estimated separately, with the explicit point that τ_t
  (here: border demand) enters the exit and enter-vs-stay-out margins but nets out of a
  pure location comparison.

## 4b. Where the analogy breaks down (state this plainly in the paper)

1. **No quality object.** Singleton *measures* effectiveness from test scores and makes
   "equity vs. aggregate effectiveness" the central trade-off. Cannabis retail has no
   value-added analogue. We cannot ask whether border demand sustains "bad" firms in a
   quality sense — only "small" or "independent" ones. The contribution has to be about
   **market structure / composition**, not welfare-via-quality.
2. **No cost data at all.** Singleton estimates VC(·) from audited charter financials and
   this is what makes θ_i identified under unobserved heterogeneity. We observe revenue
   only. Revenue ≠ profit; we cannot separate a demand effect from a margin effect. Any
   structural entry parameter would be identified only up to an untested cost assumption.
3. **τ_t is a policy scalar; border demand is an estimated, endogenous-looking shifter.**
   Singleton's funding rate is set by statute, common knowledge, and varies over time in a
   way the firm does not choose. "Border exposure" is a fixed geographic attribute that our
   *own prior work argues directly raises profitability*. It is therefore **not an
   instrument** for anything (see task §13); it is the treatment. There is no exogenous
   time variation in it within NM (NM legalised once, statewide, 2022-04-01).
4. **Hundreds of "types", not three.** Singleton has 3 charter types and ~350 tracts per
   district. We have ~[N] brands. A literal type-specific utility with brand fixed effects
   is a high-dimensional conditional-logit location model, not a 3-type game. Feasible as a
   descriptive location-choice model; a full nested-fixed-point equilibrium solve is not
   warranted by the data.
5. **Left- and right-censoring.** Singleton observes 2006–2013 with known entry/exit years.
   Our panel opens 2022Q2 with 231/1,253 storefronts already operating (true entry date
   unknown) and ends mid-2026 with most storefronts still active (exit unobserved). Entry
   models condition on non-left-censored storefronts; exit models handle right-censoring.
6. **Demand-side abstraction is safer for us, not riskier.** Singleton needs household
   demand to get enrollment/composition. Our question is purely supply-side (who supplies,
   where, for how long), and we have a *revenue* measure of realised demand, so we can stay
   reduced-form on the demand side without losing the object of interest. Per task
   instruction: no consumer demand model unless a structural parameter needs it.

--------------------------------------------------------------------------------
## 5. Structural feasibility verdict (preview of report §11)

| Singleton element | Adaptable to NM cannabis? | Why |
|---|---|---|
| Eq (1) presence / Table 1 | **YES** | ZIP×brand×quarter presence LPM is a direct translation. |
| Table 2 exiters vs persisters | **YES** | Descriptive conditional means, full panel. |
| Eq (2) / Table 3 exit LPM | **YES** | Storefront-quarter exit LPM with progressive controls. |
| Eq (4)–(5) operating utility = f(Π, D, μ) | **PARTIALLY** | Can write `Π = R − VC − FC` conceptually and estimate a *latent-profit* index; cannot populate VC, FC, or μ. |
| Eq (9) incumbent continue-vs-exit | **PARTIALLY** | Reduced-form exit hazard yes; structural continuation value no (needs forward-looking VC + option value). |
| Eq (10) entrant location choice | **PARTIALLY** | Reduced-form / conditional-logit location choice among "active" ZIPs: yes. Structural with EC_t and equilibrium beliefs: no. |
| Eq (18)–(22) revealed-preference entry-game estimation | **NO** | Requires offline demand + VC estimates that we cannot produce; θ_i would be identified only under an unfalsifiable cost assumption; ξ_t collinear with the (here non-existent) policy scalar. |

**Bottom line:** the strongest credible product is a **Singleton-*inspired* reduced-form
paper** on brand presence, concentration, entry, exit/survival, sorting, and network
expansion in a free-entry market with a predetermined cross-border demand advantage — using
Singleton's *framing* (location & survival as revealed preference; separate entry and exit
margins; type-heterogeneous responsiveness) but not its structural estimator.
