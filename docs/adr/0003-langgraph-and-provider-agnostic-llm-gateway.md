# ADR-0003: LangGraph as agent framework; provider-agnostic LLM gateway

- **Status:** Accepted
- **Date:** 2026-10-30

## Context

The decision lifecycle (intake → extract → lint → clarify → confirm → solve → check → explain →
approve → publish, with what-if loops) needs human-in-the-loop pauses that can last hours
(DESIGN §7.1). Potakt must stay model-agnostic (Goal G6) and within a small LLM budget (DESIGN §7.2).
Agent frameworks change quickly (DESIGN §17).

## Decision

- Use **LangGraph 1.x** to orchestrate the agent workflow in `potakt.agent`, with `interrupt()` for
  human-in-the-loop steps and a persistent checkpointer (SQLite in dev, PostgreSQL later).
- Access LLMs through a **provider-agnostic LLM gateway** (Claude / GPT / Gemini via config) with
  model routing — a cheap model for language detection, column mapping and summaries; a strong model
  for extraction and clarification — plus caching and a per-session token/cost budget.
- **Framework independence:** the core library (`potakt.model`, `catalog`, `checker`, `compilers`,
  `explain`) has no dependency on LangGraph or any LLM SDK (DESIGN §6).

## Consequences

- Sessions can pause and resume across human interactions.
- Swapping LLM providers or models is a configuration change, not a code change.
- If LangGraph is replaced, only `potakt.agent` changes; the Decision Model, solvers, checker and
  evals survive unchanged.
- The gateway is an extra abstraction layer to build and maintain.
