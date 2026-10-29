# ADR-0001: Monorepo with Python core and Java solver service

- **Status:** Accepted
- **Date:** 2026-10-30

## Context

Potakt needs an AI-agent layer and real optimization solvers (DESIGN §6, §8). Agents and AI tooling
live mostly in Python; Timefold and enterprise integration live on the JVM. The project is built by a
small team and needs a single place for code, docs, evals and CI (DESIGN §15).

## Decision

Use one monorepo with:

- `python/` — the `potakt` package (uv project): Decision Model, catalog, validator, checker, diff,
  CP-SAT compiler, explainer, agent, MCP server and REST API.
- `solver-service/` — a Java service (Java 25, Spring Boot 4, Timefold, Maven) exposing a Timefold
  backend over REST with asynchronous jobs (Phase 2).
- `web/` (React + TypeScript, Phase 4), `evals/`, `examples/`, `docs/` and `.github/workflows/`.

The boundary between Python and Java is a small, versioned REST contract that carries the Decision
Model JSON. CP-SAT (OR-Tools) runs in-process in Python and is the default backend.

## Consequences

- Either side of the REST contract can evolve independently; the DM JSON is the shared language.
- One CI pipeline, one issue tracker and atomic cross-component changes.
- Two toolchains (uv and Maven) must be maintained; the Java service is optional and only started
  when needed (8 GB RAM dev machine).
- Two solver backends can drift; this is mitigated by shared fixtures, a conformance suite and the
  independent checker as reference semantics (DESIGN §8.3, §17).
