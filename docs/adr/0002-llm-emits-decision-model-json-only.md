# ADR-0002: LLM emits Decision Model JSON only — no generated code

- **Status:** Accepted
- **Date:** 2026-10-30

## Context

LLMs produce plausible but often wrong combinatorial decisions, and "LLM writes solver code" systems
are hard to validate, explain and secure (DESIGN §2). Executing model-generated code is a whole class
of agent risk (DESIGN §12).

## Decision

The LLM **only** produces or edits Decision Model (DM) JSON, clarifying questions and phrasing of
explanations — never executable code (DESIGN §7.2).

- The DM is a typed, solver-agnostic description: entities, decision, constraints, objective,
  assumptions and metadata (DESIGN §5.1).
- Constraints are instances of typed templates from the **constraint catalog**; the LLM selects and
  parameterizes them instead of writing free-form math (DESIGN §5.2).
- DM JSON is validated against a JSON Schema generated from Pydantic models before use.
- Every constraint carries provenance (DESIGN §5.3).
- Explanations are generated from structured results; the LLM only phrases them (DESIGN §9).

## Consequences

- Models are validatable, diffable, explainable and safe by construction.
- Expressiveness is limited to the catalog; new templates (and any future expression escape hatch)
  require an ADR.
- Each template's semantics must be implemented consistently in the checker and each solver
  compiler, kept aligned by a conformance test suite.
