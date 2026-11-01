# Phase 0 validation corpus

Ten shift-scheduling scenarios (six in Greek) for the Phase 0 experiment (DESIGN §16.1). The
corpus format, the input bundle and the metric are pre-registered in
[ADR-0007](../adr/0007-phase-0-validation-protocol.md); the metric's reference implementation is
`potakt.evaluation`.

Each scenario is one directory, `scenarios/<NN>-<slug>/`:
- `request.md`: the manager's message, the only free text the LLM sees;
- the CSV files named in the request;
- `truth.json`: the ground-truth Decision Model (DM 0.2);
- `solution.json`: a reference solution with hard score 0 under the truth;
- `scenario.json`: metadata (reference score, disruption, unsupported requests, tags, notes).

To check a scenario, from `python/`:
`uv run potakt validate ../docs/validation/scenarios/<dir>/truth.json` and
`uv run potakt check ../docs/validation/scenarios/<dir>/truth.json ../docs/validation/scenarios/<dir>/solution.json`.
The corpus tests in `python/tests/test_validation_corpus.py` check every ADR-0007 rule for all
scenarios (`uv run pytest tests/test_validation_corpus.py`).

The corpus is frozen after the first LLM run: any later change needs a new ADR (ADR-0007 §1).

All scenarios are fictional: every person, document number and plan is invented; organisations, places and public facts appear only for realism, with sources in each `scenario.json` → `notes`.
