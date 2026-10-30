# CLAUDE.md — working agreements for Potakt

Read this file and `docs/DESIGN.md` at the start of every session. `docs/DESIGN.md` is the source of
truth; if a task seems to contradict it, **stop and ask** instead of improvising.

## Roles
- **You (Claude Code)** implement one task at a time from `.tasks/TASK-XXX-*.md`.
- **The reviewer** (a separate Claude conversation) writes the tasks, reviews diffs and verifies pushes.
- **The owner (Pantelis)** runs the gates, reviews, and does every `git push` manually.

## Golden rules
1. **One task per session/run.** Do exactly what the task file says. No drive-by refactors, no extra features.
   If you notice something worth doing, list it under "Suggestions" in your final report.
2. **Never push.** Never run `git push`, never create or modify remotes, never rewrite history
   (`--amend` only on your own unpushed commit and only if the task allows it).
3. **Never read or write secrets.** No `.env` contents in output, code, tests or logs. Secrets come
   only from environment variables.
4. **Commits:** Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`, `refactor:`),
   English, imperative mood, small and logical. **Do not add `Co-Authored-By` or any AI/tool attribution
   trailers or footers.** Stage files explicitly by path (no `git add -A` / `git add .`).
5. **Design changes need an ADR.** Do not change decisions in `docs/DESIGN.md` on your own. Propose an
   ADR in your report instead.
6. **The LLM never generates executable code in the product.** Model output is validated Decision Model
   JSON only (ADR-0002).
7. **Language:** code, comments, docs, commit messages and identifiers in English.

## Workflow for every task
1. Read the task file fully. Restate the goal and the acceptance criteria in 3–6 bullets before coding.
2. Work test-first where code is involved (red → green → refactor). Use the superpowers TDD and
   verification skills; skip brainstorming/planning skills when the task is already fully specified.
3. Run all gates before reporting (from `python/`):
   ```
   uv run ruff format --check .
   uv run ruff check .
   uv run pyright
   uv run pytest -q
   ```
   All must pass. Never weaken a gate (no blanket `# type: ignore`, no disabling rules, no skipping
   tests) to make it pass — if you think a rule is wrong, say so in the report.
4. Commit in logical steps (see rule 4).
5. **Stop and report** using this template:
   ```
   ## Report — TASK-XXX
   ### Done
   ### Acceptance criteria (each: ✅/❌ + evidence)
   ### Gates (paste the last lines of each command)
   ### Commits (git log --oneline <base>..HEAD)
   ### Files changed (git diff --stat <base>..HEAD)
   ### Deviations from the task (and why)
   ### Open questions / Suggestions
   ```

## Environment
- Windows 10, PowerShell. Python via **uv** only (`uv add`, `uv run`, `uv sync`); never `pip install`
  globally. Node 26 / npm, Git, GitHub CLI are installed. Java is not installed yet (Phase 2).
- Machine has 8 GB RAM: avoid starting heavy services unless the task needs them.
- Line endings are normalized by `.gitattributes`; do not change them.

## Testing and cost
- Tests must not call real LLM APIs by default. Real-LLM tests are marked `@pytest.mark.llm` and are
  opt-in (`uv run pytest -m llm`).
- Prefer small, deterministic fixtures. Solver tests use small instances and explicit time limits.
- Monthly LLM budget for development is about €20: use the cheapest model that works, and say in the
  report if a task required real LLM calls and roughly how many.

## Project facts
- Product: open-source platform for *decision agents* (see `docs/DESIGN.md` §1).
- Python package: `potakt` in `python/src/potakt`. Java service in `solver-service/` (Phase 2).
  Web UI in `web/` (Phase 4).
- License: Apache-2.0.

## Tools, plugins and MCP
Plugins are installed at user level and load automatically; you do not need to enable them. Use them
deliberately:
- **context7 (MCP):** before writing code against any third-party API (Pydantic, Typer, LangGraph,
  OR-Tools, MCP SDK, FastAPI, Timefold, Spring), fetch the docs for the version pinned in `uv.lock` /
  `pom.xml`. Do not rely on memory for APIs that change between versions.
- **superpowers:** use test-driven-development, systematic-debugging and verification-before-completion.
  Skip brainstorming / writing-plans when the task file already specifies the plan. Do not spawn
  parallel subagents unless the task asks for it (usage limits).
- **security-guidance:** runs automatically on edits; treat its warnings as blocking unless clearly a
  false positive (explain it in the report).
- **Self-review:** for tasks that add or change more than ~200 lines of code, review your own diff
  (superpowers requesting-code-review) before the report and fix confirmed issues.
- **mcp-server-dev:** only for tasks that build the MCP server.
- **playwright, frontend-design, typescript-lsp, playground:** only for web UI tasks (Phase 4).
- **agent-sdk-dev:** not used; Potakt uses LangGraph (ADR-0003).
Each task file may name additional tools; follow it.
