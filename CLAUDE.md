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
   Run the preflight (Definition of done, item 1), open the work log and write both as its first entry.
2. Work test-first where code is involved (red → green → refactor). Use the superpowers TDD and
   verification skills; skip brainstorming/planning skills when the task is already fully specified.
3. Run all six gates before reporting (from `python/`), equivalent to CI. CI runs `uv sync --locked`
   and a separate Python 3.12 job; locally `uv lock --check` and `--isolated` give the same checks.
   ```
   uv lock --check
   uv run ruff format --check .
   uv run ruff check .
   uv run pyright
   uv run pytest -q --cov=potakt
   uv run --locked --isolated --python 3.12 pytest -q --cov=potakt   # as in CI; leaves .venv on 3.14
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
   ### Definition of done (items 1–9: ✅/❌ + evidence)
   ### Traceability (task item → tests)
   ### Tools, plugins and MCP used (tool → what for)
   ### Work log (path, number of entries)
   ### Open questions / Suggestions
   ```

## Work log
Every task keeps a work log at `.tasks/logs/TASK-XXX.md` (local only: `.tasks/` is gitignored, never
commit it). Create it before the first change and append an entry after every step (each TDD cycle,
each commit, each gate run, each problem). Each entry has: local time; the step; files touched;
commands run with their outcome (last lines only, e.g. `318 passed`); **decisions taken and the
reason in one or two sentences**, plus alternatives rejected; tools, plugins and MCP used and what
for; problems found and how they were resolved; deviations from the task. Keep it concise and
factual: an engineering log, not a transcript. Never put secrets or `.env` values in it. The final
report summarises this log.

## Definition of done
Every task; the evidence goes in the report.
1. **Preflight** before the first change: clean tree, HEAD = the task's base, up to date with
   `origin/main`; otherwise stop and report.
2. **Gates** (Workflow step 3) green on both Pythons.
3. **Smoke test:** run the CLI on both examples exactly as a user would (`validate` and `check` for
   `examples/clinic-el` and `examples/retail-en`) and paste the score lines and exit codes.
4. **Traceability:** a table "task item → test(s) that prove it". Every decision and acceptance
   criterion has at least one test, or a stated reason why it cannot be tested.
5. **Regression tests:** every bug fix has a test that failed before the fix (red output in the log).
6. **Platform independence:** behaviour must not depend on the OS or OS version (local Windows 10,
   CI Windows and Linux). Our code decides; tests never rely on what an OS does with special names,
   encodings, line endings or locale. Tests never depend on environment variables (colour, terminal
   width, CI); tests/conftest.py resets them. Before reporting, run the suite once with
   `GITHUB_ACTIONS=true FORCE_COLOR=1 COLUMNS=40 uv run pytest -q` (Git Bash). Tests never depend
   on interpreter limits (recursion depth, C stack size, hash seed) or on the Python patch version:
   CI uses the latest 3.12.x and 3.14.x. Explicit limits are enforced by our code before calling a
   library, and a test that needs another machine's behaviour reproduces it (for example with a
   large-stack thread).
7. **Docs consistency:** help text, README, ADRs and examples match the behaviour (exit codes,
   messages, schema versions).
8. **Repository hygiene:** `git ls-files .tasks .claude` prints nothing; no secrets; `git ls-files --eol`
   shows no `i/crlf` except `*.ps1`, `*.cmd`, `*.bat`; `git log --format=%(trailers) <base>..HEAD |
   grep -c .` prints 0 (the git command alone prints one empty line per commit; grep exits 1 when
   nothing matches, which is the expected result).
9. **Postflight:** `git status` clean and "ahead of 'origin/main' by N commits", N = the commits in the
   report; nothing pushed.

## Editing and shell rules
- Change source, test and doc files with the **Edit/Write tools**, never `sed -i`/`awk`.
- Never run a command that waits on stdin (inline `python -` heredocs on Windows, `cat > file`). Put
  helper scripts in the session scratchpad and run them with `uv run python -I <file>`.
- Temporary mutations for red evidence: copies **inside the scratchpad** (never `$TEMP`), restore, then
  prove nothing is left with `git diff --stat`.
- Scratch files never go into the repository.
- A PreToolUse hook on the owner's machine blocks in-place `sed`/`perl`/`awk`, shell redirection into
  repository source files and Python reading a script from stdin. If a command is blocked, use the
  Edit/Write tools or a scratchpad file instead; never work around the hook.

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
Use plugins, skills and MCP servers proactively, at any step where they help. The list below is a
minimum, not a whitelist. Each task file names the tools it expects. Log every use (tool and purpose)
in the work log and list them in the report.

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
  (superpowers requesting-code-review) before the report and fix confirmed issues: dispatch exactly
  one reviewer subagent with `model: sonnet` (unless the task says otherwise); give it the task file,
  the relevant ADRs and your list of interpretations; if it stops on a usage limit, resume the same
  agent.
- **mcp-server-dev:** only for tasks that build the MCP server.
- **playwright, frontend-design, typescript-lsp, playground:** only for web UI tasks (Phase 4).
- **agent-sdk-dev:** not used; Potakt uses LangGraph (ADR-0003).
Each task file may name additional tools; follow it.
