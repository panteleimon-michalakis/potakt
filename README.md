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

**Phase 0 evaluation.** The Test A harness (ADR-0007 §7) is pre-registered, and its offline part
exists; none of these commands calls a model or needs credentials:
- `potakt eval prompt <scenario dir> [--model ID]` prints the exact JSON request payload for one
  scenario (exit code 0 printed, 1 invalid scenario, 2 usage error, also for a Claude model without
  pre-registered settings);
- `potakt eval compare <scenario dir> <candidate file>` validates a candidate against the
  scenario's CSV files only and prints its comparison with the truth (exit code 0 for valid and
  invalid candidates alike, 1 invalid scenario or unreadable file);
- `potakt eval summary <model runs dir> [--corpus DIR] [--json]` summarizes one model's recorded
  runs (exit code 0 printed, 1 missing directory).

There is no LLM, solver, agent or server yet; the capabilities below are planned.

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
