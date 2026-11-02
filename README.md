# Potakt

[![CI](https://github.com/panteleimon-michalakis/potakt/actions/workflows/ci.yml/badge.svg)](https://github.com/panteleimon-michalakis/potakt/actions/workflows/ci.yml)

Potakt is an open-source platform for *decision agents*: AI agents that turn requirements written in
plain language (English or Greek) into a verified, explainable decision model, solve it with real
optimization solvers, explain the result, answer "what-if" questions, and publish a decision only after
a human approves it. **The LLM understands, the solver guarantees, the human decides.**

## Status

**Early development (Phase 0).** What exists today:

- the **Decision Model** (schema version 0.2): typed JSON validated structurally and against the CSV
  data it binds, with every issue reported as a JSON pointer and a message;
- the **independent checker**, which scores any solution against a model (hard and soft score, per-rule
  violations) without trusting whoever produced it;
- the CLI commands **`potakt validate`**, **`potakt check`** (exit code 0 feasible, 1 invalid input,
  3 infeasible) and **`potakt schema`** (`--solution` for the solution schema).

A pre-registered validation corpus for the Phase 0 experiment (10 scenarios, six of them in Greek) and
its comparison metric exist under [`docs/validation/`](docs/validation/) and in
[ADR-0007](docs/adr/0007-phase-0-validation-protocol.md); no LLM results exist yet.

**Phase 0 evaluation.** The Test A harness (ADR-0007 §7) is pre-registered. `potakt eval run` is
the only command that calls the model APIs; the owner runs it by following
[`docs/validation/test-a-runbook.md`](docs/validation/test-a-runbook.md):
- `potakt eval run --model ID [--runs DIR] [--corpus DIR] [--poll-seconds N] [--yes]
  [--release BATCH_ID]` without `--yes` prints the plan (pending and recorded scenarios, an open
  or held batch or an unconfirmed create request, a lock, the cost upper bound) and sends
  nothing; with `--yes` it reads the key from
  `POTAKT_ANTHROPIC_API_KEY` (Claude models, Message Batches API) or `POTAKT_GEMINI_API_KEY`
  (`gemini-3.8-flash`), records every pending
  scenario under `docs/validation/runs/test-a/<model>/` (`request.json`, `response.json`,
  `candidate.txt`, `comparison.json`, and for Claude models `batches.json` and
  `batch-<id>.results.jsonl`) and prints the summary; `claude-opus-5-5` runs only after
  `claude-haiku-5-5` has a complete run, and the same command again resumes, also after an
  interrupted batch creation, which it reconciles before sending anything; a batch whose results
  need the reviewer is held, and every run stops until `--yes --release <id>` after the
  reviewer's decision; one `--yes` run at a time (the lock file `<runs>/.eval-run.lock`); after
  a Claude run that called the API it runs a check for unrecorded batches after the run (up to
  five minutes) and names any batch that no run recorded, without changing the exit code (exit
  code 0 all recorded or plan printed, 1 run stopped, a held batch, missing or invalid key, order
  refused, another run active, a `--release` of a batch that is not held, missing directory, a
  corpus without a scenario, an unexpected error (its type only) or interrupted, 2 usage error,
  also for an unregistered model id or `--release` without `--yes`).

The other commands are offline: none of them calls a model or needs credentials:
- `potakt eval prompt <scenario dir> [--model ID]` prints the exact JSON request payload for one
  scenario; the registered models are `claude-haiku-5-5`, `gemini-3.8-flash` and `claude-opus-5-5`
  (exit code 0 printed, 1 invalid scenario, 2 usage error, also for an unregistered model id);
- `potakt eval compare <scenario dir> <candidate file>` validates a candidate against the
  scenario's CSV files only and prints its comparison with the truth (exit code 0 for valid and
  invalid candidates alike, 1 invalid scenario or unreadable file);
- `potakt eval summary <model runs dir> [--corpus DIR] [--json]` summarizes one model's recorded
  runs at the batch prices, with how the answers stopped and the scenarios still missing; it is
  complete only when all ten are scored (exit code 0 printed, 1 missing directory);
- `potakt eval rescore <model runs dir> [--corpus DIR]` recomputes each run's `candidate.txt` and
  `comparison.json` from its `response.json` alone, sending no request, and prints one line per
  scenario: `unchanged`, `updated (<files>)` or `skipped (no response.json)` (exit code 0 all
  processed, 1 missing directory, a directory not named after a registered model, or a scenario
  not in the corpus, whose line is printed while processing continues).

There is no agent, solver or server yet; the capabilities below are planned.

## Try it

```
# from the repository root (needs uv: https://docs.astral.sh/uv/)
cd python
uv sync
uv run potakt validate ../examples/clinic-el/model.json
uv run potakt check ../examples/clinic-el/model.json ../examples/clinic-el/solution.json
```

## Planned capabilities

- **Reliable modeling** — conversation + tabular data into a typed, validated, versioned Decision Model;
  the LLM never writes solver code.
- **Ask only what matters** — clarifying questions only when the answer changes the decision.
- **Trust by construction** — every solution re-checked by an independent checker, every constraint
  traceable to its source.
- **Explain and re-plan** — per-constraint score breakdown, "why" answers, counterfactuals and
  minimal-disruption re-optimization.
- **Human control** — approval gates, roles, audit log, cost budget per session.
- **Open and portable** — model-agnostic, solver-agnostic (OR-Tools CP-SAT, Timefold), built on MCP,
  A2A and OpenTelemetry.
- **Measurable** — an evaluation suite with a simulated client and public benchmarks.
- **Extensible** — new domains added as domain packs without touching the core.

## Design

See [`docs/DESIGN.md`](docs/DESIGN.md) and the architecture decision records in [`docs/adr/`](docs/adr/).

## License

Apache-2.0 — Copyright 2026 Panteleimon Michalakis. See [`LICENSE`](LICENSE).
