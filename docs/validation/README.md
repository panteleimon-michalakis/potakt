# Phase 0 validation corpus

Ten shift-scheduling scenarios (six in Greek) for the Phase 0 experiment (DESIGN §16.1). The
corpus format, the input bundle and the metric are pre-registered in
[ADR-0007](../adr/0007-phase-0-validation-protocol.md); the metric's reference implementation is
`potakt.evaluation`. The Test A harness is pre-registered in ADR-0007 §7; its instruction text is
[`prompts/test-a.md`](prompts/test-a.md), and `potakt eval prompt` shows the exact request.

The owner runs Test A with `potakt eval run`, step by step as described in
[`test-a-runbook.md`](test-a-runbook.md): without `--yes` it prints the plan and the cost upper
bound and sends nothing; with `--yes` it reads the key from `POTAKT_ANTHROPIC_API_KEY` or
`POTAKT_GEMINI_API_KEY`, sends every scenario without a recorded answer (Claude models in one
Message Batch, Gemini one request at a time), records the answers and prints the summary.
`claude-opus-5-5` runs only after `claude-haiku-5-5` has a complete run; the same command again
resumes. Exit code 0: all recorded or the plan printed; 1: the run stopped (the reason on stderr),
a missing or invalid key, the order refused, a missing directory or an interruption; 2: usage
error, also for an unregistered model id.

Test A records go to `runs/test-a/<model>/<scenario id>/` (ADR-0007 §7 and its amendment):
`request.json`, `response.json` (the raw answer, written first and never overwritten),
`candidate.txt` and `comparison.json`. For a Claude model, `runs/test-a/<model>/` also holds
`batches.json` (each batch's id, scenarios, creation time and state) and
`batch-<id>.results.jsonl` (the raw batch results); the summary and `eval rescore` ignore these
files. `candidate.txt` and `comparison.json` are derived from `response.json` alone, so
`uv run potakt eval rescore ../docs/validation/runs/test-a/<model>` recomputes them offline after
a harness fix, without a new request; it prints `unchanged`, `updated (<files>)` or
`skipped (no response.json)` per scenario and exits 1 for a missing directory, a directory not
named after a registered model or a scenario not in the corpus. `potakt eval summary` reports a
model's results; the Test A gate is read only from a complete summary of `claude-opus-5-5`.

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
