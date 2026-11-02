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
- One run at a time: never run `potakt eval run` in two windows at once, not even for two
  different models. A second `--yes` run is refused while the first one holds the lock (§8).
- Keep the laptop awake and plugged in while a batch runs. If it sleeps anyway, run the same
  command again.

## 1. Keys

**Anthropic**
1. In the Claude Console, create a workspace `potakt-phase0`.
2. Set its monthly spend limit to $10 (a non-default workspace: "Spend limits"). When the limit is
   reached the API answers with a 400 error, and the run stops at once with no record.
3. Create a **personal** API key (Claude Console, API keys, Create key):
   - Linked account: yourself;
   - **restricted to the workspace `potakt-phase0`**;
   - expiry: **7 days**.

   Workspace keys are deprecated (the page calls them legacy); a personal key restricted to one
   workspace replaces them. A key that is not restricted to a workspace makes every call fail
   with a 400 about `anthropic-workspace-id`, and the run stops with no batch. An expired key
   gives a 401 `authentication_error`: create a new one. The run sends the key as
   `Authorization: Bearer <key>`, only to `api.anthropic.com`. See the Claude authentication
   page, <https://platform.claude.com/docs/en/manage-claude/authentication> (checked on
   2026-11-02).

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
prints the summary. While the batch is in progress, the line `batch <id>: in_progress (…)` is
printed when it changes and again after every 10 polls without a change (every 10 minutes), so the
window shows that the run is alive. This first batch also confirms the API version
(`anthropic-version: 2026-10-21`), the key (§1) and the `output_config` field: a 400 or a 401
here stops the run with no charge and no record, and the reason is printed.

If the answer to the batch creation is lost or unclear (no answer, a redirect, a 408 or a 5xx),
the run may pause about two minutes; this is normal. It first prints
`waiting <n> s before looking for a batch the create request may have made` (at most 180 s), then
looks the batch up in the Claude batch list once, page by page back to the request, adopts it if
it was created, and sends nothing twice. Do not press Ctrl+C during that wait; if you do, the next
run waits again (up to three minutes) before it sends anything.

After every Claude run that called the API, the run checks once more for batches that no run
recorded: it prints `waiting <n> s before checking for batches that no run recorded` (at most
five minutes after the last batch creation), lists the recent batches and only then prints the
stop message or the summary, also after an unexpected error. This check never changes the
records or the exit code. It looks back to the model's first batch creation, so a batch it once
named is named again by every later run until the reviewer has decided about it. If it
prints `warning: batch <id> (<n> requests, created <time>) is recorded by no run: …`, see §8.

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

- **Interrupted run** (Ctrl+C, a closed window, a lost connection, a crash, the laptop sleeping):
  run the same command again; it resumes the open batch or sends only the pending scenarios.
  Nothing that was recorded is lost, and no batch is paid for twice: before a Claude run sends a
  batch it records an unconfirmed create request in `batches.json`, and the next run first looks
  for that batch in the Claude batch list.
- **What these lines mean:**
  - `adopted batch <id> created by an earlier create request`: an earlier create request (in
    this run, when its answer was lost, a redirect, a 408 or a 5xx, or in an interrupted run) did
    create a batch; the run polls that batch and records its answers instead of sending a new one.
  - `no batch was created by the earlier create request (started <time>)`: the batch list was
    checked once, two minutes after the answer (three minutes after the request, for a request
    of an interrupted run), and no batch was found; the run continues and sends the pending
    scenarios.
  - `waiting <n> s before looking for a batch the create request may have made`: the pause
    before that check; it is at most 180 s.
  - `Unconfirmed create request: started <time> (<n> scenarios); --yes checks for its batch
    before sending anything` (dry run): an earlier run was interrupted while creating a batch.
    The dry run itself sends nothing; the next `--yes` run does the check above first.
- **`another eval run is active (<model> <time>), or one was cut off: if no other window runs
  eval run, delete <path> and run again`**: every `--yes` run holds the lock file
  `docs/validation/runs/test-a/.eval-run.lock` while it runs and removes it when it ends,
  also after a stop or Ctrl+C. Check every PowerShell window: if one still runs `eval run`, let
  it finish. If none does (the laptop crashed or a window was killed), delete the file named in
  the message and run the same command again; nothing is lost. The dry run shows an existing lock
  as `Lock: <model> <time> (another run may be active)`. The lock file is never committed. If a
  run ends with `could not remove the lock <path> (…): delete it before the next run`, delete
  that file once the window is idle; a run never removes a lock that another run took over.
- **Several batches could be the one created by an earlier create request** (the run stops and
  names them; for example after two windows ran at once): in the Claude Console (Batches) cancel
  all of them but the oldest, then run the same command again; it adopts the one left. Requests
  that a cancelled batch already processed are billed, the others are not.
- **A held batch.** Some results need the reviewer: `batch <id>: no result for <scenario>`, a
  malformed results line, or `<scenario>: invalid_request_error: …` (the API refused that request
  itself, so it would fail again). The run records every other line, keeps the raw results in
  `batch-<id>.results.jsonl` and stops with a message that ends with
  `batch <id> is held: send batch-<id>.results.jsonl and this message to the reviewer`. The batch
  is now **held**: no scenario of it is sent again, and every run, also the same command, stops
  at once with `batch <id> is held for the reviewer: <problems>; after the reviewer's decision,
  run the same command with --release <id>`, sending nothing. At most 20 problems are kept and
  shown; any others are counted in a last line `… and <n> more problems (see
  batch-<id>.results.jsonl)`. Step by step:
  1. Send `batch-<id>.results.jsonl` and the message to the reviewer.
  2. Wait for the reviewer's decision.
  3. Only if the reviewer says so, run the same command with `--yes --release <id>`. It prints
     `batch <id> released after review` and continues: the scenarios still pending are sent in a
     new batch.
- **`warning: batch <id> (<n> requests, created <time>) is recorded by no run: it may be a paid
  duplicate; send this warning to the reviewer, who decides whether to cancel it in the Claude
  Console (Batches)`**: a batch exists in the workspace that no run recorded, for example a
  second batch made by a proxy that repeated one create request and that appeared in the list
  only after the run had adopted the first. Its requests are being paid for. Do not cancel it
  yourself and do not run `--yes` again: send the warning to the reviewer, who decides whether
  to cancel it in the Claude Console (Batches). A batch that a still unconfirmed create request
  may adopt is never named: the next run adopts it.
- **`warning: could not check for batches that no run recorded (<reason>): look in the Claude
  Console (Batches)`** or **`audit interrupted: look in the Claude Console (Batches)`**: the
  check could not finish (it failed, or Ctrl+C during its wait). The run's own outcome stands.
  Open the Claude Console (Batches), and send the reviewer the list of batches of the last hour
  with this message.
- **`batches.json has an entry started at <time>, in the future: the clock or the file is wrong;
  send batches.json to the reviewer`**: the laptop's clock moved back, or `batches.json` was
  edited. Nothing was listed or sent. Check the Windows clock (Settings, Time, "Sync now"), do not
  edit `batches.json`, and send it with the message to the reviewer.
- **`more than 1000 batches were created recently in this workspace; look in the Claude Console
  (Batches) and send this message to the reviewer`**: the run could not read the whole recent
  batch list, so it stops instead of guessing that no batch exists; nothing more was sent.
- **The same open batch fails on every run** (for example `polling batch <id> failed: HTTP 404`
  or `downloading the results of batch <id> failed`, because the batch was deleted or its results
  are no longer kept): do not edit `batches.json`. Send the message and `batches.json` to the
  reviewer, who decides how to continue.
- **`<scenario>: could not be recorded (<type>)`**: one result could not be saved (for example a
  full disk). In a Claude batch it is one of the problems of a held batch (above). For Gemini the
  run stops with `<scenario>: could not be recorded (<type>); send this message to the
  reviewer`. Only the type of the error is shown, never its text. Send the message to the
  reviewer.
- **`<scenario>: the record could not be completed (<type>); send this message to the
  reviewer`**: an earlier answer is saved but its score could not be derived; nothing was sent.
  Send the message to the reviewer.
- **`no scenario in <corpus>`**: the `--corpus` directory holds no scenario; check the path.
  Nothing was read or sent.
- **`eval run stopped by an unexpected error (<type>): run the dry run and send its output and
  this message to the reviewer`**: a failure the runner does not know. The records on disk are
  kept and the lock is removed. Run the same command without `--yes` and send both outputs to
  the reviewer; do not run `--yes` again until the reviewer answers.
- **`malformed` in the summary's `Stops` line**: an answer that could not be read, for example
  text that is not valid Unicode; it counts as an invalid candidate, like any answer that did not
  end normally, and its `response.json` keeps it exactly. Nothing to do.
- **`<id>: record completed (…)`** at the start of a run means an earlier run was interrupted
  after saving an answer and before scoring it; the score is derived from the saved answer, and
  nothing is sent for that scenario.
- **After Phase 0:** delete both keys (Claude Console and Google AI Studio).
