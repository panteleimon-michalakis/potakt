# Potakt — Design Document

> **Status:** Draft v0.2 · **Date:** 2026-10-30 · **Owner:** Panteleimon Michalakis
> **Working name:** *Potakt* (provisional — may be renamed before v1.0)
> **License:** Apache-2.0

This document is the single source of truth for *what* Potakt is, *why* it exists and *how* it is built.
Any change to a decision recorded here goes through an ADR in `docs/adr/` first.

---

## 1. Summary

**Potakt is an open-source platform for *decision agents*: AI agents that turn requirements written in
plain language (English or Greek) into a *verified, explainable decision model*, solve it with real
optimization solvers, explain the result, answer "what-if" questions, and publish a decision only after
a human approves it.**

It is exposed as an **MCP server** and an **A2A agent**, so any AI assistant or enterprise agent can
delegate planning and scheduling problems to it, and as a **CLI**, a **REST API** and a **web UI**.

The core idea in one line: **the LLM understands, the solver guarantees, the human decides.**

**Positioning.** Potakt is *infrastructure for reliable agent decisions*, not a rostering app. Staff
scheduling is only the first domain pack (§13) — the value is the verified decision lifecycle that any
agent or system can use.

---

## 2. Problem and motivation

1. **LLMs are bad at combinatorial decisions.** Asked to build a staff roster or a delivery plan, an
   LLM produces something plausible that silently violates constraints, is far from optimal, and cannot
   be re-planned consistently.
2. **Optimization solvers are excellent at exactly this, but expensive to use.** Writing a correct
   model (CP-SAT, Timefold, MIP) needs an expert and weeks of work; small and medium businesses never get there.
3. **Existing "LLM + solver" work stops at one-shot modeling.** Research systems (OptiMUS, ORLM,
   OptimAI) translate a description into solver code in one go and are evaluated on benchmarks where up
   to 54% of instances contain errors. Tool integrations such as *MCP-Solver* are research prototypes
   without explanation, what-if, re-optimization or approval. Vendor offerings (NVIDIA cuOpt skills,
   Timefold's agentic intake) are tied to one vendor.
4. **Enterprises need control, not just answers.** Agent projects get cancelled for unclear value,
   cost and weak risk controls. Decisions about people's schedules need traceability, approval and an
   audit trail (in the spirit of the EU AI Act).

**What is missing is the full decision lifecycle:**
`understand → clarify → model → verify → solve → check → explain → what-if → approve → publish → re-plan`.
Potakt implements that lifecycle on open standards.

---

## 3. Goals and non-goals

### Goals
- **G1 — Reliable modeling:** convert a conversation + tabular data into a typed *Decision Model* that is
  validated, versioned and human-readable. The LLM never writes solver code.
- **G2 — Ask only what matters:** detect ambiguity and ask a question only when its answer changes the
  decision (*solver-informed clarification*, §7.3).
- **G3 — Trust by construction:** every solution is re-checked by an independent checker; every
  constraint carries provenance (which user sentence it came from).
- **G4 — Explain and re-plan:** per-constraint score breakdown, "why" answers, counterfactuals and
  *minimal-disruption* re-optimization.
- **G5 — Human control:** approval gates, roles, audit log, cost budget per session.
- **G6 — Open and portable:** model-agnostic (Claude / GPT / Gemini via config), solver-agnostic
  (OR-Tools CP-SAT and Timefold), standards-based (MCP, A2A, OpenTelemetry).
- **G7 — Measurable:** an evaluation suite with a simulated client and public benchmarks, run in CI.
- **G8 — Extensible:** new domains are added as *domain packs* without touching the core.

### Non-goals (for v1)
- Arbitrary mathematical programming (free-form MIP/NLP). v1 covers the **assignment archetype**; routing
  is the second archetype (§13).
- Training or fine-tuning LLMs.
- Running LLMs locally (the dev laptop has 8 GB RAM; hosted models are used).
- Real-time / streaming optimization at industrial scale.

---

## 4. Users and first use case

| Persona | Need |
|---|---|
| **Operations manager** (clinic, hotel, retail, call center) | "Build next week's roster that respects rest rules and people's preferences, and fix it when someone calls in sick." |
| **AI assistant / enterprise agent** | Delegate a planning sub-task through MCP/A2A and get back a verified plan + explanation. |
| **Developer / integrator** | Embed decision-making in an existing system through REST or MCP. |

**First domain pack: staff shift scheduling (rostering).** Chosen because it is universally understood
in a demo, needed in almost every industry, small enough for an MVP and a direct instance of the general
*assignment* archetype, which later also covers timetabling and task allocation.

---

## 5. Core concepts

### 5.1 Decision Model (DM)
A typed, solver-agnostic description of a decision problem — *Decision-as-Code*. It is to optimization
what SQL is to data access. A DM contains:

- **Entities** with attributes (e.g. `employees`, `shifts`), bound to data sources (CSV/XLSX/DB).
- **Decision**: what is being chosen (e.g. *assign each shift slot to one employee*).
- **Constraints**: instances of templates from the **constraint catalog** (§5.2), each with
  `hardness` (`hard` | `soft`), `weight`, `scope` filters, parameters, `rationale` and **provenance**.
- **Objective**: lexicographic — minimize hard violations, then the weighted sum of soft penalties.
- **Assumptions**: things the agent assumed and still needs to confirm.
- **Metadata**: id, version, parent version, created by, status (`draft` | `confirmed` | `approved`).

DMs are stored as JSON, validated by a JSON Schema generated from Pydantic models, versioned, and
diffable ("what changed since yesterday").

### 5.2 Constraint catalog
Instead of free-form math, the LLM **selects and parameterizes typed templates**. This makes models
validatable, explainable and safe (no code execution). Templates have precise semantics implemented
three times and kept consistent by a **conformance test suite**: in the checker (§8.3), the CP-SAT
compiler and the Timefold compiler.

**Assignment archetype — catalog v0:**

| Template | Meaning | Typical hardness |
|---|---|---|
| `coverage` | each shift gets exactly `demand` employees (structural) | hard (always) |
| `requires_attribute` | employee must have the attribute value(s) the shift requires (e.g. skill) | hard |
| `no_overlap` | an employee cannot work two time-overlapping shifts | hard |
| `min_rest` | at least `hours` between consecutive shifts of the same employee | hard / soft |
| `max_per_period` | max shifts or hours per `day` / `week` / `horizon` | hard / soft |
| `min_per_period` | min shifts or hours per period (e.g. contract hours) | soft |
| `max_consecutive_days` | at most `n` consecutive working days | hard / soft |
| `unavailable` | employee cannot work in given time windows | hard |
| `preference` | employee prefers / avoids windows or shift types | soft |
| `balance` | minimize deviation of a measure (hours, nights, weekends) across a group | soft |
| `together` / `apart` | pairs that should / should not share shifts | soft |
| `stability` | penalty per assignment changed vs. a baseline plan (added automatically in what-if) | soft |

New templates (and later an *expression escape hatch* for custom rules) are added through ADRs.

### 5.3 Provenance
Every constraint records the user utterance (turn id + quote) or data column it came from, or
`"assumption"`. Explanations and audit logs use it: *"Rule `min_rest(11h)` — from: 'nobody should work a
morning shift right after a night shift'."*

### 5.4 Solution and Score
A solution = assignments + a **score breakdown**: hard violations and soft penalty per constraint
instance, per entity. The same breakdown structure is produced by both solvers and by the checker.

---

## 6. Architecture

```
   AI assistants / enterprise agents                 Web UI (React + TypeScript)
        │ MCP            │ A2A                        chat · model view · roster · diff · approve
        ▼                ▼                                       │ REST
 ┌────────────────────────────── Decision Agent (Python) ──────────────────────────────┐
 │ LangGraph workflow (§7) · human-in-the-loop interrupts · checkpointer                │
 │ Decision Model (Pydantic) · Constraint catalog · Validator · Diff · Versioning       │
 │ Checker (independent) · Explainer · What-if · Approval · Cost budget                 │
 │ LLM gateway: provider-agnostic, model routing (cheap vs strong), caching             │
 │ Telemetry: OpenTelemetry (GenAI semantic conventions)                                │
 └───────────────┬──────────────────────────────────────────────┬──────────────────────┘
                 ▼                                              ▼
   CP-SAT compiler + solver (Python, OR-Tools)     Solver Service (Java 25 · Spring Boot 4 · Timefold)
   default backend, in-process                     async jobs · score analysis · pinning/minimal disruption
                 │                                              │
                 └──────────────► Storage: SQLite (dev) → PostgreSQL ◄──────────────┘
                                  models, versions, solutions, sessions, audit log
```

**Components**

| Component | Language | Responsibility |
|---|---|---|
| `potakt` core library | Python | DM types, catalog, validator, checker, diff, CP-SAT compiler, explainer |
| `potakt.agent` | Python | LangGraph workflow, prompts, LLM gateway, budget |
| `potakt.mcp` | Python | MCP server (tools + resources) |
| `potakt.api` | Python | FastAPI REST + A2A endpoint |
| `solver-service` | Java | Timefold backend over REST, async solving, score analysis |
| `web` | TypeScript | React UI |
| `evals` | Python | scenario suite, simulated client, benchmark runners, reports |

**Why two languages:** agents and AI tooling live in Python (most demanded in the market); Timefold and
enterprise integration live on the JVM. The boundary is a small, versioned REST contract that carries
the DM JSON, so either side can evolve independently.

**Framework independence (durability rule):** the core library (`potakt.model`, `catalog`, `checker`,
`compilers`, `explain`) has **no dependency on LangGraph or on any LLM SDK**. LangGraph only orchestrates
the workflow in `potakt.agent`. If the agent framework is replaced in the future, only that package
changes; the Decision Model, the solvers, the checker and the evals survive unchanged.

---

## 7. Agent workflow (LangGraph)

### 7.1 Graph
```
intake ─► extract ─► lint ─► clarify? ──(human)──► extract (loop)
                         │ no
                         ▼
                 confirm-by-example ──(human)──► fix model (loop)
                         │ ok
                         ▼
                      solve ─► check ─► explain ─► approve? ──(human)──► publish
                                         ▲               │ reject / what-if
                                         └── what-if ◄───┘
```

- **intake:** collect the description and attached data files; detect language.
- **extract:** structured output (Pydantic) into a *draft DM*: entities, data bindings, constraint
  instances with provenance, assumptions, and open questions with candidate interpretations.
- **lint:** schema validation + semantic checks (unknown attributes, contradictory rules, impossible
  demand vs. capacity → early infeasibility warning).
- **clarify:** ask only *decision-relevant* questions (§7.3), at most a few per turn, ranked by impact.
- **confirm-by-example:** show tiny concrete cases that the model allows or forbids and ask the human
  to confirm (§7.4).
- **solve / check:** run the selected backend; the independent checker re-scores the solution.
- **explain:** score breakdown, "why" answers, counterfactuals (§9).
- **approve / publish:** human approval interrupt; publishing writes an immutable approved version + audit entry.
- **what-if:** modify data or constraints → re-solve with `stability` (minimal disruption) → show diff.

Human-in-the-loop uses LangGraph `interrupt()` with a persistent checkpointer (SQLite in dev,
PostgreSQL later), so a session can pause for hours and resume.

### 7.2 LLM usage policy
- The LLM **only** produces/edits DM JSON, questions and explanations — never executable code.
- **Model routing:** a cheap model for language detection, data-column mapping and summaries; a strong
  model for extraction and clarification. Configured, not hard-coded.
- **Budget:** each session has a token/cost budget; the agent degrades gracefully (asks the user)
  instead of overspending. Development budget target: ≤ €20/month.

### 7.3 Solver-informed clarification (key differentiator)
An ambiguity is only worth a question if it changes the decision. For each open question with
interpretations *A* and *B*:
1. build `DM_A` and `DM_B`;
2. solve both with a short time limit;
3. if the solutions are equivalent (same assignments or same score profile) → **do not ask**; record
   the choice as an *assumption*;
4. otherwise compute an **impact** (e.g. number of assignments that differ, score delta) and ask the
   highest-impact questions first.

This keeps conversations short and focused, and it is measurable (§10).

### 7.4 Confirm-by-example (counterexample-guided)
For each constraint the agent is unsure about, temporarily relax it, find a small assignment that
violates it, and ask: *"Should this be allowed?"* A "yes" means the constraint is wrong or too strict.
This catches models that look right but encode the wrong rule.

---

## 8. Solving

### 8.1 Backends
- **CP-SAT (OR-Tools, Python)** — default, in-process, ideal for MVP sizes (tens of employees, weeks of horizon).
  Boolean `x[e,s]` for employee *e* on shift *s*; hard constraints as constraints or as penalized
  violations (so infeasible requests still return the "least bad" plan plus a diagnosis);
  soft constraints as weighted penalty terms.
- **Timefold (Java, solver service)** — for larger instances, continuous planning, pinning and rich
  score analysis (`ScoreAnalysis` / constraint matches). Asynchronous jobs:
  `POST /solve` → job id → status/progress → result.

### 8.2 Backend selection
Configured per request, with an automatic rule later (size, time budget, need for pinning).

### 8.3 Independent checker
A plain-Python evaluator of the DM semantics that **does not trust any solver**. It recomputes every
constraint and the full score breakdown. Used after every solve, in property-based tests (Hypothesis)
and in the compiler conformance suite: *same DM + data ⇒ both backends' solutions are valid and their
checker scores agree with what each backend reports.*

### 8.4 Infeasibility
When hard constraints cannot all be satisfied, report the *smallest set of conflicting rules* found by
relaxation, in plain language with provenance, and propose relaxations for the human to choose.

---

## 9. Explanation and what-if
- **Breakdown:** per constraint and per entity ("Maria: 2 soft penalties — 1 avoided night shift, 1 weekend imbalance").
- **Why X?** Counterfactual: force the alternative ("why not Nikos on Tue morning?"), re-solve, and report
  what would break or how much worse the plan gets.
- **What-if:** change data or rules → re-solve with `stability` against the current plan → show a diff
  (assignments added/removed, score delta, people affected).
- Explanations are generated from structured results; the LLM only phrases them (and cites provenance),
  it does not invent reasons.

---

## 10. Evaluation strategy
| Layer | What | When |
|---|---|---|
| Unit + property tests | DM validation, catalog semantics, compilers, checker (Hypothesis) | every commit (CI) |
| Conformance suite | CP-SAT vs Timefold vs checker on shared fixtures | every commit (from Phase 2) |
| Agent scenario evals | YAML scenarios with hidden ground-truth DM; a **simulated client** LLM answers questions consistently with the hidden truth | small subset on PRs (cost-capped), full set per release |
| Public benchmarks | adapted subsets of cleaned optimization-modeling benchmarks where they fit the archetype | per release |
| **Baseline comparison** | the same scenarios given to a *general-purpose agent with code execution* (no Potakt); we measure constraint violations, explanation quality, re-planning stability and cost | Phase 0 (validation), then per release |

**Metrics:** constraint precision/recall (template + params + hardness), final-plan validity under the
hidden truth, number of questions asked, dialogue turns, tokens and cost, latency. Results are published
in the repo as a leaderboard per LLM — rerun when a new model ships.

Tests that call real LLMs are opt-in (`pytest -m llm`); default CI uses recorded fixtures and costs nothing.

---

## 11. Interfaces
- **CLI** (`potakt`): `chat`, `validate`, `solve`, `explain`, `whatif`, `eval`.
- **MCP server** (official Python SDK): tools `create_model`, `update_model`, `validate_model`, `solve`,
  `explain`, `what_if`, `get_plan`; resources for DM JSON, schema and catalog docs.
- **REST API** (FastAPI) with OpenAPI; used by the web UI.
- **A2A agent card + endpoint** so other agents can delegate planning tasks (Phase 4).
- **Web UI** (React + TypeScript + Vite): chat, model view with provenance, roster grid, diff view, approval.

---

## 12. Security, governance and observability
- **No code execution from model output** — the LLM emits validated JSON only. This removes a whole
  class of agent risks by design.
- Secrets only via environment variables; `.env` never committed; least-privilege API keys.
- MCP tools are read/compute-only by default; `publish` requires an approval token.
- Audit log: who/what/when for every model version, solve, approval and publish.
- Prompt-injection hygiene: uploaded data and tool outputs are treated as data, never as instructions.
- Telemetry via OpenTelemetry (GenAI semantic conventions) to a local or hosted trace viewer
  (choice recorded in an ADR in Phase 3); cost per session in every trace.

---

## 13. Domain packs and extensibility
A domain pack = vocabulary + entity templates + default constraints + data importers + example
scenarios + eval scenarios. Planned order:
1. **Shift scheduling** (v1)
2. **Timetabling / exam scheduling** (assignment archetype, new templates for rooms/curricula)
3. **Vehicle routing** (second archetype: sequencing + time windows)
4. **Task allocation / project scheduling**
5. Ideas: Greek-language public-sector or SMB processes, healthcare on-call rosters.

---

## 14. Technology stack (pinned at bootstrap, verified 2026-10-30)
| Area | Choice |
|---|---|
| Python | 3.14 (supported range 3.12–3.14), managed with **uv** |
| Agent framework | **LangGraph 1.x** (+ `langchain-core`, provider packages, `langchain-mcp-adapters` where useful) |
| Data models | Pydantic 2 |
| Solver (Python) | OR-Tools 9.15 (CP-SAT) |
| MCP | official `mcp` Python SDK 2.x |
| A2A | `a2a-sdk` 1.x |
| API | FastAPI |
| CLI | Typer |
| Quality | Ruff (lint + format), Pyright (types), pytest, Hypothesis, coverage |
| Java service | Java 25 LTS, Spring Boot 4.1, Timefold Solver 2.x, Maven |
| Web | React + TypeScript + Vite |
| Storage | SQLite (dev) → PostgreSQL |
| Telemetry | OpenTelemetry SDK |
| CI | GitHub Actions (Ubuntu + Windows) |
| Containers | Docker / Compose (from Phase 2) |

---

## 15. Repository layout (monorepo)
```
potakt/
├─ README.md · LICENSE · CLAUDE.md · .gitignore · .gitattributes · .editorconfig
├─ docs/
│  ├─ DESIGN.md            ← this document
│  ├─ adr/                 ← architecture decision records (0001-…)
│  └─ validation/          ← Phase 0 experiment report and data (§16.1)
├─ python/                 ← uv project, package `potakt`
│  ├─ pyproject.toml
│  ├─ src/potakt/          ← model/, catalog/, checker/, compilers/, agent/, mcp/, api/, cli.py
│  └─ tests/
├─ solver-service/         ← Java (Phase 2)
├─ web/                    ← React/TS (Phase 4)
├─ evals/                  ← scenarios, simulated client, reports (Phase 3)
├─ examples/               ← sample data (CSV) and models
└─ .github/workflows/      ← CI
```

---

## 16. Roadmap (≈26 weeks)
| Weeks | Phase | Exit criteria |
|---|---|---|
| 1–2 | **0 · Foundations + validation** | Repo public, CI green on Ubuntu + Windows, ADRs 0001–0004, DM schema v0 with examples, **validation experiment completed (§16.1) and Gate 1 decided**. |
| 3–8 | **1 · Vertical slice (MVP) — must-have** | Shift scheduling end-to-end from CLI and MCP: conversation → DM → lint → clarify → CP-SAT → checker → explanation → what-if. ≥ 10 eval scenarios. Demo GIF in README. **Gate 2.** |
| 9–13 | **2 · Java solver service — optional** | Timefold backend behind REST, async jobs, score analysis, minimal-disruption re-solve, conformance suite green, Docker Compose. |
| 14–18 | **3 · Measurement** | Simulated-client eval harness, leaderboard for ≥ 3 LLMs + baseline, regression in CI, tracing + cost reports. **Gate 3.** |
| 19–23 | **4 · Product — UI optional** | Web UI (chat, model, roster, diff, approve), audit log + roles, A2A endpoint, second domain pack. |
| 24–26 | **5 · Release** | v1.0, docs site, demo video, deployment, write-up; optional paper draft (solver-informed clarification + benchmark). |

**Must-have vs optional.** Phase 1 (the MVP) and Phase 3 (measurement) are the core of the project.
The Java solver service (Phase 2) and the web UI (Phase 4) are valuable but optional; Gate 2 decides
their order or whether a second domain pack comes first.

### 16.1 Validation experiment (Phase 0, before any product code)
Goal: try to **disprove** the idea as cheaply as possible (budget ≈ €5 of API usage).

1. Write **10 realistic shift-scheduling descriptions** (clinic, hotel, retail, call center…), at least
   half in Greek, each with a small CSV dataset and a hand-written **ground-truth Decision Model**.
2. **Test A — extraction:** can an LLM map each description to the catalog (§5.2) correctly?
   Metric: share of ground-truth constraints recovered with the right template, parameters and hardness.
3. **Test B — baseline:** give the same problems to a *general-purpose agent with code execution* (no
   Potakt). Measure: hard-constraint violations in its plans (checked by our checker against the ground
   truth), whether it explains decisions, and how many assignments change when one person becomes
   unavailable (re-planning stability).
4. Record everything in `docs/validation/phase-0-report.md` (prompts, models, raw outputs, costs).

### 16.2 Decision gates
| Gate | When | Go if… | Otherwise |
|---|---|---|---|
| **1 · Idea** | end of week 2 | Test A recovers **≥ 70%** of constraints without manual fixes **and** Test B shows a measurable gap (violations, no explanations or unstable re-planning) | If Test A < 70%: iterate on catalog/prompts for one more week, then re-decide. If Test B shows **no** gap: stop or pivot (e.g. reposition as an evaluation/benchmark project). |
| **2 · MVP** | week 8 | Demo shown to 2–3 people (incl. one person who builds rosters in real life); they find the explanation/what-if useful | Re-scope Phase 2–4 based on feedback. |
| **3 · Evidence** | week 18 | Benchmark shows Potakt beats the baseline on validity/stability at acceptable cost | Publish the benchmark as the main contribution; consider a paper either way. |

### 16.3 Upstream contributions
When work on Potakt reveals a bug or gap in an upstream project (LangGraph, MCP SDK, OR-Tools, Timefold,
Spring AI), prefer a small, well-tested pull request upstream over a local workaround, and link it in
the README. This builds a public track record independent of Potakt's own success.

---

## 17. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Scope creep | Archetype-first catalog; strict phase exit criteria; must-have vs optional phases; ADRs for any expansion. |
| General-purpose agents (LLM + code execution) catch up on one-shot modeling | Validate early against them (§16.1); keep the durable parts in the core (checker, provenance, approval, minimal-disruption re-planning, benchmark); expose Potakt as an MCP tool those agents can call. |
| Agent framework churn | Core library is framework-independent (§6); only `potakt.agent` depends on LangGraph. |
| LLM extraction errors | Typed catalog, lint, confirm-by-example, independent checker, evals. |
| API cost | Model routing, caching, recorded fixtures in CI, per-session budget. |
| Two solver backends drift | Shared fixtures + conformance suite + single checker as the reference semantics. |
| 8 GB RAM dev machine | Hosted LLMs; CP-SAT in-process; Java service only when needed. |
| Name collision | Provisional name; rename before v1.0 if needed. |

---

## 18. Open questions (to be closed by ADRs)
1. Trace viewer: self-hosted (Langfuse / Phoenix) vs hosted free tier (Phase 3).
2. Exact time model: discrete slots vs continuous intervals in DM v0 (ADR-0004, Phase 0).
3. Data import formats beyond CSV/XLSX.
4. Deployment target for the public demo.

---

## 19. ADR index (planned)
- **ADR-0001** Monorepo with Python core and Java solver service
- **ADR-0002** LLM emits Decision Model JSON only — no generated code
- **ADR-0003** LangGraph as agent framework; provider-agnostic LLM gateway
- **ADR-0004** Decision Model v0: assignment archetype, catalog, time model

---

## 20. References
- MCP-Solver: Integrating Language Models with Constraint Programming Systems — arXiv:2501.00539
- A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions — arXiv:2508.10047
- OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents — arXiv:2504.16918
- TRACE-CS: Explainable course scheduling using LLMs and logic — AAAI
- Model Context Protocol and Agent2Agent (A2A) — Agentic AI Foundation, Linux Foundation
- OpenTelemetry semantic conventions for generative AI
