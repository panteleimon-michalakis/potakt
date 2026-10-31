# ADR-0004: Decision Model v0 — assignment archetype, catalog, time model

- **Status:** Accepted
- **Date:** 2026-10-30

## Context

Every later component (checker, compilers, agent, evals, the Phase 0 validation experiment) reads or
writes the Decision Model (DM) (DESIGN §5). ADR-0002 fixes that the LLM emits DM JSON only. This ADR
is the semantic reference for DM v0: vocabulary, time model, data bindings, constraint catalog and
validation. Exact scoring semantics of each template follow in TASK-003.

## Decision

### Scope and vocabulary
- Archetype **assignment**: *tasks* (e.g. shifts) are given as data; the decision is which
  *resources* (e.g. employees) work each task. No task generation in v0.
- The core speaks only of `resources` and `tasks`. An optional `vocabulary` block
  (`{"resource": "employee", "task": "shift"}`) is used for display only.

### Time model
- Each DM has an IANA `timezone` (must be a key of the bundled tz database, `tzdata` is a runtime
  dependency) and a `horizon` with inclusive `start` and `end` dates (`end >= start`).
- Task and window `start`/`end` are **local naive ISO-8601 datetimes** in the DM timezone, with no
  UTC offset and whole minutes (seconds = 0).
- Durations are computed **timezone-aware** (zoneinfo, via UTC), so shifts across DST changes have
  their real length. Ambiguous local times (DST fall-back) mean the first occurrence; local times that
  do not exist (DST spring-forward gap) are invalid.
- A task needs `end > start`, duration ≤ 24 h, and a local start date within the horizon. Tasks may
  cross midnight and may end after the horizon's last day.
- Period membership: a task belongs to the **day** and the **ISO week (Mon–Sun)** of its local start;
  `horizon` is the whole horizon.
- Rule parameters use **hours** (decimals allowed, must be a whole number of minutes); internally
  durations are **integer minutes**.
- Derived task attributes for filters: `weekday` (`mon`…`sun`), `start_hour` (0–23),
  `duration_hours` (float) and `date` (ISO date string), all from the local start.

### Data model and bindings
- **Resource:** `id` + `attributes`. **Task:** `id`, `start`, `end`, `demand` (int ≥ 1) + `attributes`.
- Attribute values: `str | int | float | bool | list[str]`. Attribute names are `snake_case`
  (`^[a-z][a-z0-9_]*$`) and must not be reserved: `id`, `start`, `end`, `demand`, `weekday`,
  `start_hour`, `duration_hours`, `date`.
- The `data` block binds `resources` and `tasks` to CSV files with an explicit column mapping
  (`id`, and for tasks `start`, `end`, `demand`) and typed `attributes`
  (`{"column": ..., "type": "str"|"int"|"float"|"bool"|"list", "separator": ";"}`;
  `separator` only for `list`, default `;`).
- `source` paths are relative to the DM file's directory and must stay inside it (no absolute paths,
  no `..`): a DM is untrusted input (DESIGN §12).
- `source` must not be a Windows-reserved device name (checked on every platform).
- CSV: UTF-8 (BOM tolerated), header row required, extra columns ignored, cells trimmed.
  `int`/`float` use a decimal point (no `nan`/`inf`); `bool` accepts `true/false/yes/no/1/0`
  (case-insensitive); `list` splits by `separator`, trims and drops empty items.
  An empty cell means the attribute is **absent** for that row (`list`: empty list).
- A loaded dataset is a **ProblemInstance** — `resources`, `tasks` and the unavailability windows
  of each `unavailable` rule (inline and CSV-loaded) — and is immutable after loading.

### Constraint instances
Common fields: `id` (unique), `template`, `hardness` (`hard` | `soft`), `weight` (int ≥ 1,
**required iff soft**, forbidden if hard), `scope`, `params`, `rationale` (non-empty short text),
`provenance`, `status` (`proposed` | `confirmed`, default `proposed`).

- **Provenance:** `kind` ∈ `utterance` | `data` | `assumption` | `default`, optional `turn_id`,
  `quote`, `source`. `utterance` requires `quote`; `data` requires `source`. For `assumption` and
  `default` the reason is stated in `rationale`.
- **Scope:** `scope.resources` and `scope.tasks`, each a list of conditions combined with AND;
  empty = all. A condition is `{attribute, op, value}`, `op` ∈ `eq, ne, in, not_in, contains, gte,
  lte`. `attribute` is `id`, a data attribute or (tasks only) a derived attribute. Value shapes:
  `in`/`not_in` a list; `contains` a scalar (tests membership in a list attribute); `gte`/`lte` a
  number or a string (ISO dates compare chronologically); `eq`/`ne` any attribute value.

### Catalog v0
| Template | params | Meaning | Hardness |
|---|---|---|---|
| `coverage` | — | each in-scope task gets exactly `demand` resources | hard only |
| `requires_attribute` | `task_attribute`, `resource_attribute`, `match` (`contains`\|`equals`) | a resource may work a task only if its attribute matches the task's requirement; tasks without the attribute are unrestricted | hard/soft |
| `no_overlap` | — | a resource cannot work two time-overlapping tasks | hard/soft |
| `min_rest` | `hours` > 0 | gap between consecutive tasks of a resource ≥ hours | hard/soft |
| `max_per_period` | `measure` (`count`\|`hours`), `period` (`day`\|`week`\|`horizon`), `limit` ≥ 0 | upper bound per resource per period | hard/soft |
| `min_per_period` | as above | lower bound per resource per period | hard/soft |
| `max_consecutive_days` | `days` ≥ 1 | at most N consecutive working days | hard/soft |
| `unavailable` | `windows` (`{resource_id, start, end}`) and/or `source` (CSV binding with those columns) | resource cannot work tasks overlapping the windows | hard/soft |
| `preference` | `kind` (`prefer`\|`avoid`) | scoped resources prefer/avoid scoped tasks | soft only |
| `balance` | `measure` (`count`\|`hours`) | minimize max − min of the measure over scoped resources, counting scoped tasks | soft only |
| `together` | `resource_ids` (2 distinct) | the two should work the same in-scope tasks | soft only |
| `apart` | `resource_ids` (2 distinct) | the two should not work the same task | hard/soft |
| `stability` | `baseline` (`[{task_id, resource_id}]`) | penalty per baseline assignment that changes | soft only |

`limit` is an integer for `count` and a whole number of minutes for `hours`. `unavailable` needs at
least one of `windows` / `source`; a window needs `end > start`. Templates are a discriminated union
on `template`; new templates require an ADR.

### Decision Model root
`schema_version` (`"0.1"`), `id`, `version` (int ≥ 1), `parent_version` (optional, < `version`),
`status` (`draft` | `confirmed` | `approved`), `title`, `language` (`en` | `el`), `archetype`
(`"assignment"`), `vocabulary` (optional), `timezone`, `horizon`, `data`, `constraints` (≥ 1, exactly
one `coverage`), `assumptions` (`[{id, text, status}]`, unique ids, status `proposed` | `confirmed`),
`objective` (v0 fixed: `"lexicographic_hard_then_soft"`, the default).

### Validation
- Parsing is **strict** (no type coercion: `"5"` is not a number) and unknown fields are rejected,
  because the DM is model output (ADR-0002).
- **Structural** (Pydantic): types, enums, required fields, hardness/weight rules, per-template params,
  unique ids, exactly one `coverage`, valid timezone, horizon order, binding names.
- **Data-level** (after loading): unique resource/task ids, task times valid and inside the horizon,
  windows valid, every attribute referenced by scopes/params exists in the bindings (or is
  `id`/derived), every `resource_id` / `task_id` in params exists in the data.
- Every error is reported as a `ValidationIssue` with a JSON-pointer `path` into the DM (RFC 6901,
  `/` for the root) and a human-readable `message`; CSV row errors name the file and line.
- The JSON Schema is generated from the Pydantic models and committed at
  `spec/decision-model.schema.json`; a test fails when it drifts. Cross-field rules (uniqueness,
  weight iff soft, data-level checks) are not expressible in the schema and are enforced in code.

## Consequences
- The checker and both compilers share one precise, typed input; templates are validated before any
  solver sees them.
- Per-person limits that differ by resource (e.g. contract hours) need one rule per group in v0,
  because parameters are constants.
- A schema version bump and an ADR are needed for any new template, field or semantic change.
- Scoring semantics: see ADR-0005.
