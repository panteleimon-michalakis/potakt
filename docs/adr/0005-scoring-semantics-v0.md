# ADR-0005: Scoring semantics v0

- **Status:** Accepted
- Amended by ADR-0006.
- **Date:** 2026-10-31

## Context

The independent checker is the reference semantics of Potakt (DESIGN §8.3): it scores any solution
against a Decision Model without trusting any solver, and every later component (CP-SAT compiler,
Timefold service, conformance suite, evals, the Phase 0 baseline comparison) is judged by it.
ADR-0004 fixed the vocabulary, the time model and the parameters of catalog v0, but deferred the exact
scoring of each template. This ADR is that definition: what counts as one violation, how large it is
and how violations add up to a score (DESIGN §5.4, §9).

## Decision

A solution is a **set** of assignments `(task_id, resource_id)` (duplicates are rejected when the
solution is validated). Each rule is scored independently; the score is the lexicographic pair
`(hard, soft)`.

### 1. Time and intervals
- All comparisons use **timezone-aware instants**: local naive times are interpreted in the DM
  timezone as in ADR-0004 (ambiguous times mean the first occurrence). Durations and gaps are real
  elapsed **integer minutes**, so a shift across a DST change has its real length.
- Intervals are **half-open `[start, end)`**. Two intervals overlap iff
  `a.start < b.end and b.start < a.end`. A shift ending at 15:00 overlaps neither a shift nor an
  unavailability window starting at 15:00.
- The rest gap between tasks `a`, `b` of the same resource with `a.end <= b.start` is
  `b.start − a.end`.
- Hour parameters are converted to whole minutes (`round(hours × 60)`; ADR-0004 guarantees a whole
  number of minutes).
- A task counts in full in the period of its local start, even if it ends after midnight or after the
  horizon (no clipping).

### 2. Scope
- `scope.resources` and `scope.tasks` select the resources and tasks a rule applies to (ADR-0004
  operators). A rule only ever looks at assignments whose resource **and** task are both in scope;
  everything else is invisible to it. For pairwise rules (`no_overlap`, `min_rest`) both tasks must be
  in scope.
- Derived task attributes (`weekday`, `start_hour`, `duration_hours`, `date`) come from the local
  start; `id` is the entity id.
- Condition semantics:
  - A condition on an attribute the entity does not have is **false** for every operator, including
    `ne` and `not_in`.
  - `eq` / `ne`: equality where a boolean only equals a boolean (`true` ≠ `1`) and numbers compare
    numerically (`2` = `2.0`). `in` / `not_in`: the attribute equals / equals none of the listed values.
  - `contains`: the attribute (a list; a scalar counts as a one-item list) contains the value.
  - `gte` / `lte`: number vs number, or string vs string (ISO dates compare chronologically); any
    other combination is false.
- `together` and `apart` name their two resources in `params`; they apply only if **both** named
  resources are in `scope.resources` (otherwise the rule's amount is 0). Their task scope applies as
  usual (for `apart` too).
- `stability` ignores baseline entries whose task or resource is out of scope.

### 3. Periods
- `day` = the local calendar days of the horizon, labelled `2026-11-04`.
- `week` = the ISO weeks (Mon–Sun) that intersect the horizon, labelled with the ISO year and week,
  `2026-W45`.
- `horizon` = the whole horizon, labelled as an ISO interval `2026-11-02/2026-11-08`.
- A task belongs to the period of its local start.
- For `min_per_period` every in-scope resource is checked in **every** period, including periods in
  which it works nothing.
- **Known v0 limitation:** partial weeks at the horizon edges are treated like full weeks (a horizon
  starting on a Thursday still demands the full weekly minimum in that first week).

### 4. Violation amounts
All amounts are non-negative integers. A violation with amount 0 is not listed.

| Template | Unit of violation | `amount` |
|---|---|---|
| `coverage` | per in-scope task | `abs(assigned − demand)`, `assigned` = number of in-scope resources assigned to the task |
| `requires_attribute` | per assignment whose task has `task_attribute` and whose resource does not match (see below) | 1 |
| `no_overlap` | per resource, per unordered pair of overlapping assigned in-scope tasks | 1 |
| `min_rest` | per resource, per ordered pair `a`, `b` with `0 <= gap < hours` | 1 |
| `max_per_period` | per resource, per period with value > limit | `count`: excess; `hours`: **started hours** of excess = `ceil(excess_minutes / 60)` |
| `min_per_period` | per resource, per period with value < limit | `count`: shortfall; `hours`: `ceil(shortfall_minutes / 60)` |
| `max_consecutive_days` | per resource, per maximal run of consecutive working days of length `L > days` | `L − days` |
| `unavailable` | per (resource, assigned task) overlapping one of that resource's windows (the task counts once even if several windows overlap) | 1 |
| `preference` | per assignment of an in-scope resource to an in-scope task | 1 (sign in §5) |
| `balance` | once per rule | `count`: `max − min` over **all** in-scope resources (zeros included); `hours`: `ceil((max − min) / 60)` with values in minutes |
| `together` | per in-scope task where exactly one of the two resources is assigned | 1 |
| `apart` | per in-scope task where both resources are assigned | 1 |
| `stability` | per in-scope baseline `(task_id, resource_id)` missing from the solution (a pair listed twice counts once) | 1 |

Details:
- `requires_attribute`: tasks without `task_attribute` are unrestricted. Both values are read as
  lists (a scalar counts as a one-item list) and compared with the `eq` equality of §2.
  `match = contains`: the resource's list contains **every** required item. `match = equals`: the two
  lists have the same items, order and repetitions ignored (so a scalar requirement equals a
  one-item list). A resource without `resource_attribute` matches only an empty requirement.
- `min_rest` counts **every** ordered pair within the gap, not only consecutive tasks: it is the
  pairwise reading that solver compilers encode directly, and for non-overlapping tasks it is zero
  exactly when every consecutive gap is long enough. Overlapping pairs (gap < 0) belong to
  `no_overlap`, not to `min_rest`.
- `max_per_period` / `min_per_period`: the value is the number of in-scope tasks (`count`) or their
  total real duration in minutes (`hours`) of the resource in the period.
- `max_consecutive_days`: a working day is a horizon day on which the resource **starts** at least
  one in-scope task. Days before and after the horizon are unknown and not counted.
- `balance`: with no in-scope resources the amount is 0.

### 5. Score and result
- `hard` = sum of `amount` over all **hard** rules. A solution is **feasible** iff `hard == 0`.
- `soft` = sum over **soft** rules of `weight × amount`, except `preference` with `kind = prefer`,
  which contributes `−weight × amount` (a reward). `soft` may therefore be negative.
- Scores compare lexicographically: lower `hard` wins, then lower `soft`.
- The result lists **every** rule of the DM (also those without violations) with `rule_id`,
  `template`, `hardness`, `weight` (null for hard rules), total `amount`, `contribution` (to `hard`
  for hard rules, to `soft` for soft rules) and its violations. Each violation has `resource_id`
  (null if the violation has no single resource: `coverage`, `balance`, `apart`; for `together` it is
  the resource that works the task alone), `task_ids` (sorted), `period` (null if none; for
  `max_consecutive_days` the run as an ISO interval `2026-11-02/2026-11-07`), `amount` and a short
  English `message` built from a fixed template (no free text). Messages are explanatory, not
  normative.
- Determinism: rules appear in DM order; violations are sorted by resource id (null first), period,
  task ids. The same inputs always produce byte-identical JSON output.

## Consequences
- The checker, the CP-SAT compiler and the Timefold compiler have one precise target; the
  conformance suite compares their scores against the checker.
- Amounts are integers, so solver objectives can use integer arithmetic; `hours` measures round
  excess up to started hours, which slightly over-penalizes small excesses.
- Partial weeks at the horizon edges are scored like full weeks in v0; a later ADR may prorate them.
- Changing any rule above changes scores of existing solutions and needs a new ADR.
