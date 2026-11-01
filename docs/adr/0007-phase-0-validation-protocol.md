# ADR-0007: Phase 0 validation protocol (pre-registered)

- **Status:** Accepted
- **Date:** 2026-11-01

## Context

Gate 1 (DESIGN §16.2) decides whether Potakt continues. It rests on two experiments (DESIGN §16.1):
**Test A** asks whether an LLM maps real requests onto the catalog (target: at least 70 % of the
ground-truth constraints recovered without manual fixes), **Test B** whether a general-purpose agent
with code execution shows a measurable gap. A result is only credible if the corpus, the inputs and
the metric are fixed **before** any LLM sees a scenario, so that results cannot bend the rules
afterwards. This ADR is that pre-registration. The metric is implemented and tested in
`potakt.evaluation` (`compare.py`), which is its reference implementation.

## Decision

### 1. Pre-registration rule
The corpus format, the input bundle, the metric and the Gate 1 thresholds are fixed by this ADR
before the first LLM run. Any later change needs a new ADR and is reported in
`docs/validation/phase-0-report.md` together with the results under **both** the old and the new rule.

### 2. Corpus
**Layout.** One directory per scenario, `docs/validation/scenarios/<NN>-<slug>/` (`NN` = `01` … `10`,
`slug` in ASCII kebab-case), containing:
- `request.md`: the manager's message, the only free text the LLM sees;
- the CSV files;
- `truth.json`: the ground-truth Decision Model (DM 0.2);
- `solution.json`: a reference solution with hard score 0 under `truth.json`;
- `scenario.json`: the metadata, a strict JSON object with `id` (= the directory name), `title`,
  `language` (`el` | `en`), `domain`, `difficulty` (`easy` | `medium` | `hard`), `style` (`email` |
  `chat` | `notes` | `voice-transcript` | `memo`), `csv_files` (the names of the scenario's CSV files),
  `reference_score` (`hard`, `soft`: the checker score of `solution.json`, best found, not necessarily
  optimal), `disruption` (`resource_id`, `start`, `end`: a one-person absence as local datetimes inside
  the horizon, which the reference solution violates), `unsupported` (a list of `quote` and `reason`)
  and optional `notes`.

**Composition.** The corpus has **10** scenarios, at least **5 in Greek**, at least **4 domains** and a
mix of difficulty (each level at least twice). Difficulty is defined by the number of scored
constraints (§5.1) in `truth.json`: **easy 4–6, medium 7–9, hard 10–14**. Across the 10 scenarios:
- every non-`coverage` template appears in at least 2 scenarios (`stability` in at least 1, with the
  baseline given as a draft roster in the request or a CSV);
- `min_rest`, `max_per_period` and `max_consecutive_days` appear both hard and soft;
- at least 2 scenarios use per-resource limits, at least 2 a non-default CSV dialect (one of them
  Windows-1253), at least 2 have unsupported requests, at least 3 contain irrelevant information and
  at least 2 write numbers in words;
- the cues for hard and soft rules vary ("ideally", "if possible", "under no circumstances",
  «ιδανικά», «αν γίνεται», «οπωσδήποτε», «απαγορεύεται»).

**Every scenario:**
- 5–12 resources and 14–42 tasks; one week, Monday to Sunday, in November or December 2026; inside
  every bound of ADR-0006 §6; timezone `Europe/Athens` for Greek scenarios, `Europe/London` or
  `America/New_York` for English ones.
- `request.md` reads like a real manager's message: it names the CSV files, contains at least one
  irrelevant sentence, varies its structure, and uses **no catalog vocabulary** (no template name and
  none of the words *hard*, *soft*, *scope*, *weight*, *constraint*). Greek requests are natural
  Greek, with most letters Greek; English requests contain no Greek letters.

### 3. Input bundle
Identical for Test A and Test B: the text of `request.md`, plus **every** CSV file of the scenario in
full, decoded with its declared dialect (the delimiter, decimal separator and encoding of the truth's
binding of that file) and shown with its file name **and its encoding** (for example
`ypodoxi_prosopiko.csv (encoding: windows-1253)`), because decoded text cannot show the encoding; the
delimiter and the decimal separator are visible in the text and are not stated (stating the encoding:
confirmed by the reviewer on 2026-11-01). Test A additionally
receives the DM JSON Schema (`potakt schema`). Nothing else: no truth, no solution, no
`scenario.json`, no catalog hints beyond the schema.

### 4. Truth rules
- `truth.json` is the DM a careful expert would write from the bundle alone.
- Every non-`coverage` constraint has an `utterance` provenance whose quote is a **verbatim** substring
  of `request.md` (compared with whitespace normalized). So the truth contains no unstated default
  rules.
- Ids, weights, rationale, provenance and status are **not** scored. User-facing strings (title,
  rationale) are in the scenario language; ids are ASCII kebab-case; soft weights are sensible.
- **Granularity:** one constraint per distinct request. Requests stated together with the same
  template, hardness and scope (e.g. two absences in one sentence) are one constraint; a request that
  needs two templates or two hardnesses ("ideally 5, never more than 6") gives one constraint each.
- Requests the catalog cannot express are listed in `scenario.json` → `unsupported` (verbatim quote
  and reason) and are **not** encoded in the truth.

### 5. Metric (Test A)

#### 5.1 Scored constraints and validity
- Only non-`coverage` constraints are scored: `coverage` is mandatory and structural.
- A candidate DM must pass structural and data-level validation (ADR-0004, ADR-0006) against the
  scenario data, with its `source` paths resolved in the scenario directory. Otherwise **every** truth
  constraint of that scenario counts as not recovered, the candidate's constraints are not counted,
  and its issues are listed. Validity is reported separately (valid candidates / scenarios).
- "Without manual fixes" means no human edits. An automatic repair loop, if one is added in TASK-005c,
  is reported as a separate number and never replaces the first-attempt number.
- **First attempt:** each scenario is run once per model and the first answer counts, whatever it is
  (an answer that is not a valid DM is an invalid candidate). A request that fails before the model
  answers (network or API error) is repeated.

#### 5.2 Denotation
Two constraints **match** iff they have the same template, the same hardness, the same **scope
extension** and the same **normalized parameters**. The scope extension is the pair (set of in-scope
resource ids, set of in-scope task ids), computed with the condition semantics of ADR-0005 §2 /
ADR-0006 §3 on the data each DM loads from the scenario directory (data attributes, derived task
attributes and `id`). The normalization follows these principles: hours become minutes; lists become
sets; per-period limits become the map *resource → effective limit* over the in-scope resources, and
their `period` becomes the periods they check on the DM's horizon; the assignment-blocking templates
`requires_attribute` and `unavailable` become the set of `(resource id, task id)` pairs they forbid on
the scenario data, with the checker's semantics; `together`/`apart` become their unordered pair and
the tasks they apply to; preference kind is part of the parameters; a `stability` baseline is a set;
weights are ignored.

| Template | Normalized parameters | So these match |
|---|---|---|
| `coverage` | not scored | — |
| `requires_attribute` | the set of forbidden `(resource id, task id)` pairs among the in-scope resources and tasks, by the ADR-0006 §3 rules (an absent or empty requirement restricts nobody; the one equality); the scope is folded into this set and not compared separately | renamed attributes; another column or condition with the same forbidden pairs; `str` vs `list` binding of a single-valued column; `equals` vs `contains` on single values; a redundant scope |
| `no_overlap` | none | — |
| `min_rest` | `hours` as whole minutes | `11`, `11.0` |
| `max_per_period`, `min_per_period` | `measure`; the periods the rule checks on the DM's horizon (ADR-0005 §3), each as its first and last date clipped to the horizon; and the map in-scope resource → effective limit (tasks for `count`, minutes for `hours`) | a literal limit and an `{"attribute": …}` limit with equal values; `16` and `16.0`; `week` and `horizon` on a horizon of exactly one ISO week |
| `max_consecutive_days` | `days` | — |
| `unavailable` | the set of forbidden `(resource id, task id)` pairs among the in-scope resources and tasks: one of the resource's windows (inline or from the CSV `source`) overlaps the task, half-open `[start, end)`; the scope is folded into this set and not compared separately | inline vs CSV windows; split, merged or reordered windows; `…T23:59` vs the next day's `T00:00`; a redundant scope |
| `preference` | `kind` (`prefer` \| `avoid`) | — |
| `balance` | `measure` | — |
| `together`, `apart` | the unordered pair of resource ids and the set of in-scope task ids; the resource scope only decides whether both members are in scope, and if one is not the rule is **inert** (the pair alone, no task set), so it matches only another inert rule with the same pair | `[a, b]`, `[b, a]`; a redundant resource scope; two inert rules with the same pair |
| `stability` | the set of baseline `(task_id, resource_id)` pairs whose task and resource are in scope | baseline order |

The period normalization of per-period rules (the periods they check on the DM's horizon, clipped to
it) was confirmed by the reviewer on 2026-11-01. For `requires_attribute`, `unavailable`, `together`
and `apart` the scope is part of the parameters, as described above. For every other template the
scope extension is compared as is, so a rewrite of a scope with other conditions but the same
extension matches, while a scope that changes one resource or task does not.
Ids, weights, rationale, provenance, status and assumptions are never compared. The DM-level fields
are not compared on their own: the horizon enters through the periods of per-period rules (one more
day is one more period for `min_per_period`), and the timezone can change behaviour only across a DST
change, which no corpus week contains.

#### 5.3 Matching and scores
- **One-to-one (multiset):** a candidate constraint matches at most one truth constraint and vice
  versa; with equal denotations the number of matches is the smaller of the two multiplicities.
  Matched pairs are formed in DM order.
- Per scenario: **recall** = matched / truth constraints, **precision** = matched / candidate
  constraints (`coverage` excluded from both; a score with a zero denominator is undefined, `null`).
- Over the corpus both are **micro-averaged**: sum of matched / sum of truth (or candidate)
  constraints.

#### 5.4 Gate 1, Test A
Micro-averaged **recall ≥ 0.70** for the strong model (DESIGN §7.2), first attempt, no manual fixes
(DESIGN §16.2). Precision, the validity rate, and breakdowns per language, template and hardness
are reported but do not decide the gate.

#### 5.5 Diagnostics (for iteration, not for the gate)
Each unmatched truth constraint is classified against the closest **unmatched** candidate constraint
with the same template. A pair differs in up to three components: `scope` (either extension),
`parameters`, `hardness`. Pairs are formed one-to-one, closest first: fewest differing components,
then the most significant differing component in ascending order (a pair whose worst difference is
`hardness` before one whose worst is `parameters`, before one whose worst is `scope`), then DM order
of the truth and of the candidate constraint. The category of a paired truth constraint names its most
significant difference: `wrong_scope` if the scopes differ, else `wrong_parameters` if the parameters
differ, else `wrong_hardness`. A truth constraint without such a candidate is `missing`. (A changed
scope also changes the scope-restricted parameters of limits and baselines; ranking scope first keeps
it `wrong_scope`. For `requires_attribute`, `unavailable`, `together` and `apart` the scope is part of
the parameters, so a different set is `wrong_parameters`, never `wrong_scope`.)

#### 5.6 Unsupported requests
They are not part of recall. A candidate constraint that encodes one is a false positive by
construction. The report counts, by reading each candidate's `assumptions`, how many unsupported
requests the candidate flags (informational).

### 6. Test B measures (computed in TASK-005d)
The baseline gets the input bundle of §3 and is asked for its plan as `(task id, resource id)` pairs,
which are written as a solution (schema 0.1). A plan that names unknown ids or repeats a pair is
**invalid**.
- **Violations:** the hard violations of the baseline's final plan under `truth.json`, counted by our
  checker (`potakt check`).
- **Explanations** (yes/no rubric, rated from the baseline's final answer). "Yes" only if all three
  hold: (1) it names the manager's rules it applied; (2) it names every rule its plan breaks or relaxes
  and says why (claiming none while the checker finds a hard violation fails this item); (3) it gives at
  least one concrete reason tied to a named person and shift. Otherwise "no".
- **Re-planning stability:** a second message states the scenario's `disruption` ("<resource> cannot
  work from <start> to <end>; please update the plan", in the scenario language). Changed assignments
  = the pairs of the first plan missing from the updated plan (the amount of a `stability` rule with
  the first plan as baseline). It is compared with the **minimal count for the baseline's own first
  plan**: the same number for a minimal-disruption re-plan of that first plan (truth plus the
  disruption as a hard `unavailable` rule plus that `stability` rule, best found, method reported),
  because how many changes are unavoidable depends on the plan being changed. The same number for a
  minimal-disruption re-plan of `solution.json` is reported as context. The updated plan's hard
  violations are counted under the truth plus the disruption.

**Gate 1, Test B:** a measurable gap exists if at least one holds: (V) in at least **2** of the 10
scenarios the first or the updated plan is invalid or has a hard violation; (E) the rubric gives
"no" in at least **5** scenarios; (S) in at least **5** scenarios the updated plan changes at least
**3 more** assignments than the minimal count for its own first plan. These thresholds were
confirmed by the reviewer on 2026-11-01. Any one of the three gaps counts as "a measurable gap",
because DESIGN §16.2 names "violations, no explanations **or** unstable re-planning".

## Consequences
- Results can be compared across models and over time, because every rule of the measurement is fixed
  in advance and implemented once, in tested code.
- The denotation compares behaviour on the scenario data, not text: equivalent rewrites match, but a
  rule that is only equivalent on other data does not, and a redundant scope matches only on the
  templates whose scope is part of the parameters (`requires_attribute`, `unavailable`, `together`,
  `apart`).
- Accepted leniency: windows or requirements that differ only where no task is affected match. The
  definition is the effect on the scenario data, the same principle as the scope extension.
- A candidate that groups the same requests into a different number of rules than the truth (e.g. one
  `unavailable` rule for two absences stated separately) loses those matches; the diagnostics show them
  as `wrong_parameters` or `wrong_scope`, so TASK-005c can report how often this happens.
- A candidate that adds an unstated default rule (for example `no_overlap` when the request does not
  ask for it) lowers precision only.
- Any change to the corpus format, the bundle, the metric or the thresholds after the first LLM run
  needs a new ADR, and the report shows the results under both rules.
