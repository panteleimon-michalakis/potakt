# Development runs runbook (owner only)

How the owner runs the development set (ADR-0008 D1 and its amendment) with
`potakt eval run --dev`. The development scenarios `91-swimming-pool`, `92-museum-front-desk` and
`93-fuel-station` (`docs/validation/dev/`) test the harness end to end and may lead to a revision
of the instruction, the schema text or the settings before the freeze; they never count for a
gate. Every step that sends a request costs money or quota and is done by the owner alone, in a
terminal of their own, never by Claude Code. `<repo>` is the local clone of this repository.

Every rule of the [Test A runbook](test-a-runbook.md) applies here, with three differences:
- every `eval run`, `eval rescore`, `eval summary`, `eval repairs` and `eval draws` command
  carries `--dev`, and none carries `--corpus` or `--runs`;
- the records go to `docs/validation/runs/dev/round-1/` instead of `docs/validation/runs/test-a/`
  (also the lock, `acknowledged-batches.txt` and every path of its §10);
- the Anthropic key belongs to the workspace `potakt-dev`, never to `potakt-phase0`.

Until the commit tagged `phase0-freeze`, a run without `--dev` refuses to send anything: no corpus
scenario is sent before the freeze (ADR-0008 D1).

## 1. Workspace and keys

1. In the Claude Console, Settings, Workspaces, create a workspace `potakt-dev`.
2. Select it, click "Spend limits" and set its monthly limit to $15 (ADR-0008 D13 and its
   amendment). A batch may go slightly over a workspace's limit; the cost upper bounds of all the
   development runs together stay well under it (at most about $11, repairs included).
3. Create a personal API key as in §1 of the Test A runbook, **restricted to the workspace
   `potakt-dev`**, expiring in 7 days. In the Console's API keys list, check that the key's
   workspace is `potakt-dev`.
4. In Billing, check that the credit balance is at least $1 before Haiku and $3 before each Opus
   run.
5. Gemini: the key and the **Free** project of §1 of the Test A runbook. In Google AI Studio, look
   up the project's daily quota for `gemini-3.8-flash`; the development runs send at most six
   Gemini requests (three first requests, at most three repairs).
6. Set the variables in a new PowerShell window as in §2 of the Test A runbook (the `potakt-dev`
   key in `POTAKT_ANTHROPIC_API_KEY`), then run the preflights and send the three outputs to the
   reviewer before any `--yes`:

   ```powershell
   uv run potakt eval preflight --model claude-haiku-5-5
   uv run potakt eval preflight --model claude-opus-5-5
   uv run potakt eval preflight --model gemini-3.8-flash
   ```

## 2. Haiku

```powershell
uv run potakt eval run --model claude-haiku-5-5 --dev
uv run potakt eval run --model claude-haiku-5-5 --dev --yes
```

- The dry run's first line is `Test A run plan: claude-haiku-5-5 (development set)`; its
  `Runs directory:` ends with `runs\dev\round-1\claude-haiku-5-5`, `Pending (3)` names the three
  development scenarios and the cost upper bound is `$0.0569`.
- The `--yes` run works like §3 of the Test A runbook: the clock check, the waits, one batch, the
  check for batches that no run recorded, then the summary.
- Do the Console check of §3 of the Test A runbook in the workspace `potakt-dev`, against
  `docs/validation/runs/dev/round-1/claude-haiku-5-5/batches.json`.

## 3. Check the Haiku records

```powershell
uv run potakt eval rescore --dev ../docs/validation/runs/dev/round-1/claude-haiku-5-5
uv run potakt eval summary --dev ../docs/validation/runs/dev/round-1/claude-haiku-5-5
```

- `eval rescore` prints only `unchanged` lines, and `skipped (no response.json)` for a scenario
  given up after five batches (ADR-0008 D6).
- `eval summary` shows `Complete: yes`. Send the summary to the reviewer, then commit the records
  (§8).

## 4. Gemini

```powershell
uv run potakt eval run --model gemini-3.8-flash --dev
uv run potakt eval run --model gemini-3.8-flash --dev --yes
uv run potakt eval rescore --dev ../docs/validation/runs/dev/round-1/gemini-3.8-flash
uv run potakt eval summary --dev ../docs/validation/runs/dev/round-1/gemini-3.8-flash
```

The same checks as §3; send the summary to the reviewer, then commit the records (§8). A 429
that stops the run: run the same command the next day after 10:00 Athens time.

## 5. Opus and the output budget (ADR-0008 D4)

Only after the reviewer has confirmed §3:

```powershell
uv run potakt eval run --model claude-opus-5-5 --dev
uv run potakt eval run --model claude-opus-5-5 --dev --yes
uv run potakt eval rescore --dev ../docs/validation/runs/dev/round-1/claude-opus-5-5
uv run potakt eval summary --dev ../docs/validation/runs/dev/round-1/claude-opus-5-5
```

- Run `uv run potakt eval preflight --model claude-opus-5-5` first if the key or the day changed.
  The dry run's cost upper bound is `$2.2776`.
- Send the summary to the reviewer. In each scenario row the reviewer reads the stop and the
  output tokens: a stop `max_tokens`, or more than 48 000 output tokens, in any development run of
  `claude-opus-5-5` (this one and the repeats of §7) means that the output budget is decided again
  by an amendment to ADR-0008 before the freeze (D4).
- Commit the records (§8).

## 6. Repairs (ADR-0008 D7)

In the order Haiku, Gemini, Opus, each only after its first development run is checked and
committed:

```powershell
uv run potakt eval run --model claude-haiku-5-5 --dev --repair
uv run potakt eval run --model claude-haiku-5-5 --dev --repair --yes
uv run potakt eval rescore --dev ../docs/validation/runs/dev/round-1/repair/claude-haiku-5-5
uv run potakt eval repairs --model claude-haiku-5-5 --dev
```

- The dry run's first line is `Test A run plan: claude-haiku-5-5 (development set, repair)`. If
  its `Pending` is `none`, skip the `--yes` command and its `eval rescore`: `eval repairs` gives
  the report.
- `eval rescore` prints only `unchanged` lines, and `skipped (no response.json)` for a repair
  given up after five batches. Send the report of `eval repairs` to the reviewer and commit the
  repair records (§8).
- Then the same for `gemini-3.8-flash` and `claude-opus-5-5`.

## 7. Repeats (ADR-0008 D8)

Only after the first Opus development run is checked and committed:

```powershell
uv run potakt eval run --model claude-opus-5-5 --dev --repeat 2
uv run potakt eval run --model claude-opus-5-5 --dev --repeat 2 --yes
uv run potakt eval rescore --dev ../docs/validation/runs/dev/round-1/repeat-2/claude-opus-5-5
uv run potakt eval summary --dev ../docs/validation/runs/dev/round-1/repeat-2/claude-opus-5-5
```

- Check and commit as in §5 (the output budget check included), then the same with `--repeat 3`
  and `repeat-3` in both paths.
- Then run `uv run potakt eval draws --dev` and send its output to the reviewer.

## 8. Commit the records

As §9 of the Test A runbook, with `docs/validation/runs/dev` in place of
`docs/validation/runs/test-a`: from the repository root, `git status` and
`git status --ignored docs/validation/runs/dev` may show only files under
`docs/validation/runs/dev/` and no `*.tmp` file; then

```powershell
git add docs/validation/runs/dev
git commit -m "docs(validation): record development runs (<model>)"
```

and push the same day to the review branch the reviewer names.

## 9. Close the development runs

Only when the reviewer says that the development runs are closed (every record committed and
checked, and the revisions they caused decided):
1. Note the total cost of `potakt-dev` on the Console's Cost page (every month) and send it to the
   reviewer.
2. In the Claude Console, Settings, Workspaces, open the menu (three dots) of `potakt-dev` and
   select "Archive workspace". Archiving deactivates the workspace and archives its keys, keeps
   its history for reporting, and cannot be undone.
3. Close the PowerShell window of the development runs. The corpus runs use a new window and a
   key of `potakt-phase0` (Test A runbook §1 and §2).

## 10. If something goes wrong

- The stops and warnings of §10 of the Test A runbook apply, with
  `docs/validation/runs/dev/round-1` in place of `docs/validation/runs/test-a` and the workspace
  `potakt-dev` in place of `potakt-phase0`.
- `no corpus scenario is sent before the commit tagged phase0-freeze (ADR-0008 D1); run the
  development set with --dev; nothing was sent`: `--dev` is missing from the command.
- `<scenarios>: corpus scenarios never run with --dev (ADR-0008 D1); nothing was sent` or
  `<runs> is under <path>, where --dev runs never record (ADR-0008 D1); nothing was sent`: a
  `--corpus` or `--runs` option was given; run the commands of this runbook exactly.
- `the workspace has batch <ids> that no batches.json in <runs> records …`: the key may belong to
  `potakt-phase0`, or the workspace holds batches that no development record knows. Do not run
  `--yes` again; send the message to the reviewer.
