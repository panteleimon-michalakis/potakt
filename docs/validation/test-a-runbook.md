# Test A runbook (owner only)

How the owner runs Test A (ADR-0007 §7 and its amendment) with `potakt eval run`. Every step that
sends a request costs money or quota and is done by the owner alone, in a terminal of their own,
never by Claude Code. `<repo>` is the local clone of this repository.

Rules for the whole runbook:
- A key is never pasted into Claude Code, a chat, a file or a commit, and never typed on a command
  line; it lives only in the environment of one PowerShell window, for one session.
- `potakt eval run` without `--yes` is a dry run: it prints the plan and the cost upper bound,
  sends nothing and reads no key. Always run it first.
- A scenario with a recorded answer (`response.json`) is never sent again; running the same
  command again resumes where it stopped.

## 1. Keys

**Anthropic**
1. In the Claude Console, create a workspace `potakt-phase0`.
2. Set its monthly spend limit to $10 (a non-default workspace: "Spend limits"). When the limit is
   reached the API answers with a 400 error, and the run stops at once with no record.
3. Create an API key scoped to that workspace.

**Gemini**
1. Create a key in Google AI Studio.
2. Restrict it to the Gemini API ("Restrict to Gemini API only").

Never paste a key into Claude Code, a chat, a file or a commit.

## 2. Session-only variables

Open a **new Windows PowerShell 5.1 window**, not the one running Claude Code, and type:

```powershell
cd <repo>\python
$secure = Read-Host "Anthropic API key" -AsSecureString
$env:POTAKT_ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
Remove-Variable secure
```

Then the same block for the Gemini key:

```powershell
$secure = Read-Host "Gemini API key" -AsSecureString
$env:POTAKT_GEMINI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
Remove-Variable secure
```

The variables exist only in this window. At the end of the session, run:

```powershell
Remove-Item Env:POTAKT_ANTHROPIC_API_KEY, Env:POTAKT_GEMINI_API_KEY
```

then close the window.

## 3. Haiku

```powershell
uv run potakt eval run --model claude-haiku-5-5
uv run potakt eval run --model claude-haiku-5-5 --yes
```

The first command prints the plan and the cost upper bound. The second sends one Message Batch and
polls it every 60 seconds until it ends (usually well under an hour), then records the answers and
prints the summary. This first batch also confirms the API version (`anthropic-version:
2026-10-21`) and the `output_config` field: a 400 here stops the run with no charge and no record,
and the reason is printed.

## 4. Check the Haiku records

```powershell
uv run potakt eval rescore ../docs/validation/runs/test-a/claude-haiku-5-5
uv run potakt eval summary ../docs/validation/runs/test-a/claude-haiku-5-5
```

- `eval rescore` must print only `unchanged` lines.
- `eval summary` must show `Complete: yes`.
- Send the summary to the reviewer.

## 5. Gemini

```powershell
uv run potakt eval run --model gemini-3.8-flash
uv run potakt eval run --model gemini-3.8-flash --yes
```

The free tier may answer 503 (overloaded) or 429 (quota); the run retries and then stops. Run the
same command again later: only the scenarios still pending are sent.

## 6. Opus

Only after the reviewer has confirmed step 4:

```powershell
uv run potakt eval run --model claude-opus-5-5
uv run potakt eval run --model claude-opus-5-5 --yes
```

The dry run shows the cost upper bound. `claude-opus-5-5` is refused until `claude-haiku-5-5` has a
complete run. Its first answer per scenario is the one that counts for the gate: there is no
second attempt.

## 7. Commit the records

For each model, from the repository root:

```powershell
git add docs/validation/runs/test-a
git commit -m "docs(validation): record Test A runs (<model>)"
```

Push only after the reviewer's check of a fresh clone.

## 8. If something goes wrong

- **Interrupted run** (Ctrl+C, a closed window, a lost connection): run the same command again; it
  resumes the open batch or sends only the pending scenarios. Nothing that was recorded is lost.
  One exception: if a Claude run is interrupted after it starts sending a batch but before it
  prints `batch <id> created for N scenarios`, the batch may exist without being recorded, and the
  next run would create another. In that case, first look at the Claude Console (Batches): cancel
  any batch created in the last minutes, then run again.
- **A malformed results line** stops the run after the other lines are recorded; the raw results
  stay in `batch-<id>.results.jsonl`. Send that file to the reviewer before running again,
  because running again sends the affected scenarios in a new, paid batch.
- **Several batches could be the one a lost request created** (the run stops and names them):
  none of them is recorded in `batches.json`, so the next run would not use them. Cancel every
  batch the message names in the Claude Console (Batches), then run the same command again; it
  sends one new batch. Requests that a cancelled batch already processed are billed, the others
  are not.
- **After Phase 0:** delete both keys (Claude Console and Google AI Studio).
