# ADR-0006: Decision Model 0.2 and scoring amendments

- **Status:** Accepted
- **Date:** 2026-10-31

## Context

The Phase 0 validation experiment (DESIGN §16.1) measures how well an LLM maps everyday requests onto
the catalog. Two everyday needs cannot be expressed well in DM v0 (ADR-0004): **per-person limits**
such as contract hours (v0 needs one rule per group of people) and **CSV files saved by a
Greek-locale Excel** (`;` delimiter, decimal comma, Windows-1253). The reviews of TASK-002/003 also
left open decisions about the CLI exit codes and output bytes, Windows device names, empty
requirements and list equality, and found inputs that are accepted although they are meaningless or
crash later. This ADR records those decisions. It amends ADR-0004 and ADR-0005 where stated; anything
not mentioned here is unchanged.

## Decision

### 1. CLI contract (A1, A2)
- `potakt check` exits **0** when the solution is feasible, **1** when an input is invalid (issues on
  stderr) and **3** when it is infeasible. Exit code **2** stays the usage error of the CLI framework
  (Click/Typer: unknown option, missing argument), so "infeasible" can never be confused with "called
  wrongly". `potakt validate` exits 0 or 1 and, like `check`, prints issues on **stderr**; its `OK: …`
  line goes to stdout.
- Everything the CLI writes to stdout and stderr is **UTF-8 with `\n` line endings** on every platform,
  also when redirected to a file or a pipe, so `--json` output is byte-identical on Linux and Windows.
  The streams are reconfigured by the console-script entry point before any argument is parsed, so
  help, `--version` and usage errors follow the same rule; the existing error handler of each stream
  is kept. (An interactive Windows console is written through the console API by Click; that is not a
  byte stream and is outside this guarantee.)

### 2. Data sources (A3)
Amends ADR-0004 "Data model and bindings": the reserved device names are `CON`, `PRN`, `AUX`, `NUL`,
`CONIN$`, `CONOUT$`, and `COM`/`LPT` followed by one of `0`–`9`, `¹`, `²`, `³`. They are recognised
case-insensitively on every platform, with or without extensions, and with trailing dots, spaces or
colons (`nul:`, `Aux.`, `con .csv`). A colon is treated like a dot: the base name ends at the first
`.` or `:` (`nul:`, `nul:x` and `nul:.csv` all name the `NUL` device on Windows).

### 3. Scope and rule semantics (A4, A5)
Amends ADR-0005 §2 and §4:
- **One equality everywhere.** Two attribute values are equal when, if either is a list, both read
  as lists (a scalar counts as a one-item list) have the **same items, order and repetitions
  ignored**; otherwise they are equal as before (a boolean only equals a boolean, numbers compare
  numerically). `eq` / `ne` use this equality; `in` / `not_in` test whether the attribute equals one /
  none of the listed values with it; `requires_attribute` with `match = equals` uses it too. So
  `["als", "opening"] eq ["opening", "als"]` and `["als"] eq "als"` are true.
- **Empty requirement = no requirement.** In `requires_attribute`, a task whose `task_attribute` is
  absent **or an empty list** is unrestricted, for both `match = contains` and `match = equals`. This
  replaces "a resource without `resource_attribute` matches only an empty requirement".

### 4. Structural and data-level validation (B1, B3, B4)
Amends ADR-0004 "Time model" and "Validation":
- **Horizon length** (structural, at `/horizon`): `end − start + 1` is at most **366 days**, so a whole
  leap year fits and absurd horizons (thousands of day periods) are rejected before any scoring.
  This does not bound the calendar dates themselves; see Consequences.
- **Stability baseline** (structural): a `(task_id, resource_id)` pair may appear only once in
  `stability.baseline`; a repeated pair is an error at the repeated entry. This replaces "a pair listed
  twice counts once" in ADR-0005 §4.
- **Empty datasets** (data-level): a dataset with zero resources or zero tasks is an error at the
  binding's `source` (`/data/resources/source`, `/data/tasks/source`).
- **Pathological JSON** in a DM or solution file (for example nesting so deep that the JSON decoder
  overflows, or an integer with thousands of digits) is reported as a normal issue at `/`, never as
  a crash.

### 5. Decision Model 0.2 (C1–C3)

#### Per-resource limits (C1)
Amends ADR-0004 "Catalog v0" and ADR-0005 §4 for `max_per_period` and `min_per_period`: `limit` is
either a number ≥ 0 (as before) **or** `{"attribute": "<resource attribute>"}`, meaning that the limit
of each resource is the value of that attribute.
- The attribute must be a bound resource attribute of type `int` or `float` (data-level, at
  `…/params/limit/attribute`, like every other attribute reference).
- Every resource **in the rule's scope** must have a value (data-level, one issue per resource
  without one, at `…/params/limit`). Resources are never skipped silently: to exempt someone, scope
  the rule. Values of out-of-scope resources are not read.
- Each in-scope value obeys the rules of a literal limit: it is ≥ 0; for `measure: hours` it is in
  hours and a whole number of minutes; for `measure: count` it is a whole number (a `float` column
  value such as `16.0` counts as whole, exactly like a literal `16.0`).
- The checker compares each resource with its own limit; amounts and messages are computed exactly as
  for a literal limit (`… worked, minimum 16 h`). A rule whose every in-scope resource has value `L`
  scores exactly like the same rule with the literal limit `L`.

#### CSV dialect per binding (C2)
Amends ADR-0004 "Data model and bindings": the `resources` and `tasks` bindings and the `source` of an
`unavailable` rule accept an optional `csv` block

```json
{"delimiter": "," | ";" | "\t", "decimal": "." | ",", "encoding": "utf-8" | "windows-1253" | "iso-8859-7"}
```

with defaults `","`, `"."` and `"utf-8"` (a UTF-8 BOM is tolerated, as before).
- `decimal: ","` requires a delimiter other than `","` (structural, at `…/csv/decimal`).
- The decimal setting applies to `float` cells only: under `decimal: ","` the cell `32,5` is 32.5 and
  a cell containing `.` is rejected, because in those locales `.` is the thousands separator and
  `1.234` must not silently become 1.234. Thousands separators are never accepted. `int` cells
  (including `demand`) never accept a decimal separator.
- Bytes that are invalid in the declared encoding give the usual "cannot read" issue naming that
  encoding (`cannot read staff.csv: not valid utf-8`). CRLF and LF line endings are both accepted.
  The encodings are mapped to Python codecs by Potakt itself, not looked up from the operating system.
- This covers what "Save as CSV" produces in a Greek-locale Excel (`;`, `,`, Windows-1253, CRLF) and
  its "CSV UTF-8" option (UTF-8 with BOM).

#### Version (C3)
The DM `schema_version` is **`"0.2"`** and only `"0.2"` is accepted (there are no users of 0.1, so
there is no migration code). The solution `schema_version` stays `"0.1"`. The committed schemas in
`spec/` are regenerated from the models.

## Consequences
- Contract hours and similar per-person limits are one rule bound to a data column instead of one
  rule per group (ADR-0004 "Consequences" no longer applies to them); the LLM only has to name the
  column.
- Greek Excel exports load without manual conversion; other locales need a new `encoding` value,
  which is a schema change.
- Condition semantics change for list attributes (order no longer matters) and for empty
  requirements; scores of existing solutions can change only for models that used those cases.
- Scope evaluation is needed by data-level validation (C1: "every in-scope resource"), so the
  condition semantics live in `potakt.model` and the checker uses them from there; there is still one
  implementation.
- The 366-day cap stops absurd horizons but not dates at the very edge of the calendar (for example a
  horizon in the last week of the year 9999, or a window near 9999-12-31 in a timezone behind UTC);
  bounding the calendar range is left to a later decision.
- Any further change to these rules needs a new ADR.
