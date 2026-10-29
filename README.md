# Potakt

[![CI](https://github.com/panteleimon-michalakis/potakt/actions/workflows/ci.yml/badge.svg)](https://github.com/panteleimon-michalakis/potakt/actions/workflows/ci.yml)

Potakt is an open-source platform for *decision agents*: AI agents that turn requirements written in
plain language (English or Greek) into a verified, explainable decision model, solve it with real
optimization solvers, explain the result, answer "what-if" questions, and publish a decision only after
a human approves it. **The LLM understands, the solver guarantees, the human decides.**

**Status: early development (Phase 0).** Nothing below is implemented yet beyond the project skeleton.

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
