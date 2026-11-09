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
- The run compares the laptop clock with the API's and stops if they differ by more than two
  minutes. If it stops for that, sync the Windows clock (Settings, Time & language, Date & time,
  "Sync now") and run again.
- Never delete, move or edit anything under `docs/validation/runs/test-a/` (in particular
  `batches.json` and any `.tmp` file), never run `git clean` or `git stash -u` before the
  records are committed, and never run a model from a second clone or with another `--runs`:
  the run would send its pending scenarios again, and no check would see the duplicate.

## 1. Keys

**Anthropic**
1. In the Claude Console, create a workspace `potakt-phase0`.
2. Set its monthly spend limit to $15 for the development runs; raise it to $45 before the corpus
   Test A runs and to $60 before Test B (ADR-0008 D13; a non-default workspace: "Spend limits"). If
   the limit is already reached when a batch is created, the API answers with a 400 error and the
   run stops with no record. If it is reached while a batch runs, the documentation does not say
   how the remaining requests end: send the run's output to the reviewer. The dry runs' cost upper
   bounds fit under the limit.
3. Create a **personal** API key (Claude Console, API keys, Create key):
   - Linked account: yourself;
   - **restricted to the workspace `potakt-phase0`**;
   - expiry: **7 days**.

   Workspace keys are **legacy**: the page says "Workspace API keys work but are legacy."; a
   personal key restricted to one workspace replaces them. A key that is not restricted to a
   workspace makes every call fail with a 400 about `anthropic-workspace-id`, and the run stops
   with no batch. An expired key gives a 401 `authentication_error`: create a new one. The run
   sends the key as `Authorization: Bearer <key>`, only to `api.anthropic.com`; the page calls
   the other header legacy too: "You can also send the key in the legacy `x-api-key:
   YOUR_API_KEY` header in place of `Authorization`." See the Claude authentication page,
   <https://platform.claude.com/docs/en/manage-claude/authentication> (checked on 2026-11-06).
4. In the Console's API keys list, check that the key's workspace is `potakt-phase0`. In
   Billing, check that the credit balance is at least $1 before Haiku and $8 before Opus.

**Gemini**
1. Create a key in Google AI Studio.
2. Restrict it to the Gemini API ("Restrict to Gemini API only").
3. In Google AI Studio, Projects, check that the key's project shows **Free** in the Billing
   Tier column. A project linked to a Cloud Billing account is on a paid tier: every request is
   billed and a repeated one twice, although the dry run says $0. If it is not Free, stop and ask
   the reviewer.

Never paste a key into Claude Code, a chat, a file or a commit.

**Check the keys** (last step, once the variables of §2 are set). In the PowerShell window of §2,
run:

```powershell
uv run potakt eval preflight --model claude-haiku-5-5
uv run potakt eval preflight --model claude-opus-5-5
uv run potakt eval preflight --model gemini-3.8-flash
```

Each command makes only free, read-only calls (the model's metadata and, for Claude, one page of
the batch list): no prompt is sent, nothing is written and nothing is charged. It prints one line
per check, starting with `key: accepted by api.anthropic.com` (or
`generativelanguage.googleapis.com`), and ends with
`preflight passed: no prompt was sent and nothing was charged`. For a Claude model the second
line compares the laptop clock with the API's: `clock: within <n> s of api.anthropic.com`; more
than 120 s fails the check `clock` (sync the Windows clock: Settings, Time & language, "Sync
now"). A field shown as `not reported` is not a failure. A line
`preflight failed: <check>: <reason>` (exit code 1) is a key or model problem found for free;
an HTTP 401 or 400 there is the key (§1). Send the three outputs to the reviewer before any
`--yes`.

## 2. Session-only variables

Open a **new Windows PowerShell 5.1 window**, not the one running Claude Code, and type:

```powershell
cd <repo>\python
$secure = Read-Host "Anthropic API key" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$env:POTAKT_ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
Remove-Variable secure, bstr
```

Then the same block for the Gemini key:

```powershell
$secure = Read-Host "Gemini API key" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$env:POTAKT_GEMINI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
Remove-Variable secure, bstr
```

Copy messages from the window itself; do not redirect the output with `>` (Windows PowerShell
5.1 garbles non-ASCII characters in files).

The variables exist only in this window. At the end of the session, run:

```powershell
Remove-Item Env:POTAKT_ANTHROPIC_API_KEY, Env:POTAKT_GEMINI_API_KEY
```

then close the window.

## 3. Haiku

If the key or the day changed since the check of §1, run the preflight again first:

```powershell
uv run potakt eval preflight --model claude-haiku-5-5
uv run potakt eval run --model claude-haiku-5-5
uv run potakt eval run --model claude-haiku-5-5 --yes
```

The dry run prints the plan and the cost upper bound. The `--yes` run sends one Message Batch and
polls it every 60 seconds until it ends (usually well under an hour), then records the answers and
prints the summary. While the batch is in progress, the line `batch <id>: in_progress (…)` is
printed when it changes and again after every 10 polls without a change (every 10 minutes), so the
window shows that the run is alive. This first batch also confirms the `output_config` field. The
API checks each request of a batch only while it processes the batch ("Validation of the
`params` object for each message request is performed asynchronously"), so a refused field comes
back as `<scenario>: invalid_request_error: …` for every scenario and the batch is held (§8);
such results are not billed. A wrong API version or key is refused at once with a 400 or a 401,
before any batch exists: the run stops with no charge and no record, and the reason is printed.
A key problem that slips past the preflight stops the run at its first call, the clock check:
`listing batches failed: HTTP 401 (…)` (or a 400), with nothing sent and no check of the batch
list after it. A 400 at the batch creation itself (for example the spend limit, §1) is printed
only after the check for unrecorded batches below, up to five minutes later: this is normal;
wait for the message.

If the answer to the batch creation is lost or unclear (no answer, a redirect, a 408 or a 5xx),
the run may pause about two minutes; this is normal. It first prints
`waiting <n> s before looking for a batch the create request may have made` (at most 180 s), then
looks the batch up in the Claude batch list once, page by page back to the request, adopts it if
it was created, and sends nothing twice. Do not press Ctrl+C during that wait; if you do, the next
run waits again (up to three minutes) before it sends anything.

Before the first batch of each Claude model, the run waits until two minutes after it started,
printing `waiting <n> s before checking the workspace for batches that no run recorded` (at most
120 s), and lists every batch of the workspace: a batch that no `batches.json` records stops the
run before anything is sent (§8). Do not press Ctrl+C during that wait.

After every Claude run that made an API call other than the clock check, the run checks once more
for batches that no run recorded: it prints `waiting <n> s before checking for batches that no
run recorded` (at most five minutes after the last batch creation), lists the recent batches and
only then prints the stop message or the summary, also after an unexpected error. This check
never changes the records or the exit code. It looks back to the first batch creation of any
model, so a batch it once named is named again by every later Claude run that made an API call
other than the clock check, unless it is cancelled or acknowledged. Only a batch still in
progress can be cancelled ("Batches may be canceled any time before processing ends."); a batch
that has already ended stays in the batch list. Once the reviewer has decided about a named
batch, the reviewer may acknowledge it in acknowledged-batches.txt (§8), and no run names it
again; until then its warning on later runs is expected: send a warning again only if it names a
new batch id. If it prints
`warning: batch <id> (<n> requests, created <time>) is recorded by no run: …`, see §8.

**Check the Console after each model's run** (here and in §6):
1. Open the Claude Console, Batches, for the workspace `potakt-phase0`.
2. Compare the batch ids listed there with the ids in
   `docs/validation/runs/test-a/<model dir>/batches.json` (every model run so far; for example
   `Get-Content ../docs/validation/runs/test-a/claude-haiku-5-5/batches.json`).
3. Send the reviewer that list.

A batch listed there that no `batches.json` records is a paid duplicate, whatever the audit said.

## 4. Check the Haiku records

```powershell
uv run potakt eval rescore ../docs/validation/runs/test-a/claude-haiku-5-5
uv run potakt eval summary ../docs/validation/runs/test-a/claude-haiku-5-5
```

- `eval rescore` must print only `unchanged` lines.
- `eval summary` must show `Complete: yes`.
- Send the summary to the reviewer.

## 5. Gemini

If the key or the day changed since the check of §1, run
`uv run potakt eval preflight --model gemini-3.8-flash` again first. For Gemini the preflight
proves only that the key exists: Google's documentation says that read-only metadata calls "only
validate that your API key exists and don't check project tier eligibility or billing status".

```powershell
uv run potakt eval run --model gemini-3.8-flash
uv run potakt eval run --model gemini-3.8-flash --yes
```

The free tier may answer 503 (overloaded) or 429 (quota); the run retries and then stops. Run the
same command again later: only the scenarios still pending are sent.

A 403 (for example `PERMISSION_DENIED` with `FREE_TIER_VERIFICATION_REQUIRED`), a 400
`FAILED_PRECONDITION` or a 402 stops the run at once with no record: the project cannot use the
free tier as it is. Fix it in AI Studio (for example, verify the account) or send the message to
the reviewer. Daily quotas reset at midnight Pacific time (10:00 in Athens in November and
December).

## 6. Opus

Only after the reviewer has confirmed step 4:

```powershell
uv run potakt eval preflight --model claude-opus-5-5
uv run potakt eval run --model claude-opus-5-5
uv run potakt eval run --model claude-opus-5-5 --yes
```

Run the preflight first if the key or the day changed since the check of §1. The dry run shows
the cost upper bound. `claude-opus-5-5` is refused until `claude-haiku-5-5` has a complete run.
Its first answer per scenario is the one that counts for the gate: there is no second attempt.
After the run, do the Console check of §3 for both `batches.json` files.

## 7. Commit the records

Commit each model's records right after its run and its checks, before the next model:
- from `python/`, go to the repository root (`cd ..`) and run `git status`;
- only files under `docs/validation/runs/test-a/` may appear, and no `*.tmp` file: git ignores
  a leftover `.tmp`, so also run `git status --ignored docs/validation/runs/test-a`; if it shows
  a `.tmp` file, do not commit, and see §8;
- then commit them:

  ```powershell
  git add docs/validation/runs/test-a
  git commit -m "docs(validation): record Test A runs (<model>)"
  ```

- push the commits the same day to the review branch the reviewer names, for the check of a fresh
  clone; push to `main` only after that check;
- until a commit is pushed, the records exist only on this laptop (Anthropic keeps batch results
  for 29 days after the batch was created).

## 8. If something goes wrong

Any stop or warning that is not listed below: do not run `--yes` again; send the whole output to
the reviewer.

- **Interrupted run** (Ctrl+C, a closed window, a lost connection, a crash, the laptop sleeping):
  run the same command again; it resumes the open batch or sends only the pending scenarios.
  Nothing that was recorded is lost, and no batch is paid for twice: before a Claude run sends a
  batch it records an unconfirmed create request in `batches.json`, and the next run first looks
  for that batch in the Claude batch list. On Windows, Ctrl+C takes effect only when the current
  network call returns: up to a minute while a batch is created, polled or listed, up to five
  minutes while results download, up to ten minutes during a Gemini request. Wait for
  `interrupted: run the same command again to resume (nothing is lost)`; do not close the window
  instead (a closed window leaves the lock). A Gemini answer that arrives during that wait is
  discarded, and that scenario is sent again by the next run. If saving a Gemini answer fails
  before its temporary file is complete, the answer is lost and the next run sends that scenario
  again (free tier: no charge).
- **What these lines mean:**
  - `adopted batch <id> created by an earlier create request`: an earlier create request (in
    this run, when its answer was lost, a redirect, a 408 or a 5xx, or in an interrupted run) did
    create a batch; the run polls that batch and records its answers instead of sending a new one.
  - `no batch was created by the earlier create request (started <time>)`: the create request
    was made by an earlier run that was interrupted; the batch list was checked, page by page, at
    least three minutes after that request, and no batch was found; the run continues and sends
    the pending scenarios.
  - `waiting <n> s before looking for a batch the create request may have made`: the pause
    before that check; it is at most 180 s.
  - `waiting <n> s before checking the workspace for batches that no run recorded`: the pause
    before the first batch of a Claude model (§3), at most 120 s, so that a batch created before
    this run started is listed; then the run lists every batch of the workspace. Do not press
    Ctrl+C.
  - `batch <id> was already released`: the same `--release` command, run again after a stop that
    came after the release (for example the clock check); nothing to do, the run continues.
  - `<scenario>: a leftover response.json.tmp will be replaced by the answer of batch <id>`
    (also in the dry run): an earlier run was interrupted while it saved this answer; the run
    downloads that batch's results again and records the answer from them. Nothing to do.
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
- `the lock <path> now belongs to another run (…); it was left in place`: two runs overlapped. Do
  the Console check of §3 for every model, and send the output of both windows to the reviewer.
- **Several batches could be the one created by an earlier create request** (the run stops and
  names them; for example after two windows ran at once): in the Claude Console (Batches) cancel
  all of them but the oldest, then run the same command again; it adopts the one left. A batch
  that has already ended cannot be cancelled: then send the message to the reviewer, who decides
  which batches to acknowledge («Acknowledged batches»). Requests that a cancelled batch already
  processed are billed, the others are not.
- **A held batch.** Some results need the reviewer: `batch <id>: no result for <scenario>`, a
  malformed results line, `<scenario>: invalid_request_error: …` (the API refused that request
  itself, so it would fail again) or `<scenario>: errored (<type>); it would fail again` (an
  error that a repeat cannot fix, such as `billing_error`, or an unknown one). The run records
  every other line, keeps the raw results in
  `batch-<id>.results.jsonl` and stops with a message that ends with
  `batch <id> is held: send batch-<id>.results.jsonl and this message to the reviewer`. The batch
  is now **held**: no scenario of it is sent again, and every run, also the same command, stops
  at once with `batch <id> is held for the reviewer: <problems>; after the reviewer's decision,
  run the same command with --release <id>`, sending nothing. At most 20 problems are kept and
  shown; any others are counted in a last line `… and <n> more problems (see
  batch-<id>.results.jsonl)`. Step by step:
  1. Send `batch-<id>.results.jsonl` and the message to the reviewer.
  2. Wait for the reviewer's decision.
  3. Only if the reviewer says so, run the same command with `--yes --release <id>`. It first
     records, from `batch-<id>.results.jsonl`, each scenario of that batch that is still pending
     and has exactly one line there, a succeeded one, with nothing sent (a scenario with two
     lines, or with an errored, expired or unreadable one, is sent again):
     `<scenario>: recorded from the kept results of batch <id>`. Then it prints
     `batch <id> released after review` and continues: the scenarios still pending are sent in a
     new batch. If one of those recordings fails, it stops with
     `<scenario>: could not be recorded (<type>); batch <id> is still held: send this message to
     the reviewer` (or, if the answer itself was saved, `<scenario>: recorded, but its record
     could not be completed (<type>); it is completed when the batch is released`): the batch
     stays held and nothing was sent; send the message to the reviewer.
  4. If that file is missing or cannot be read, the release stops with
     `cannot read the kept results <path>: <reason>; batch <id> is still held: send this message
     to the reviewer`; if it has lines but none of them is a result of that batch (for example a
     download in another shape), with
     `<path> holds no result of batch <id>; batch <id> is still held: send this message to the
     reviewer`. Nothing was recorded or sent and the batch stays held: send the message to the
     reviewer. Only on the reviewer's decision, one of these ways out, then the same
     `--yes --release <id>` command again:
     - restore the file from the Console's download of the batch results (Claude Console,
       Batches, the batch); the reviewer first checks it: one `{"custom_id": …, "result": …}`
       object per line;
     - only if the reviewer decides so, an empty file at that path, from the repository root
       (the release then sends every pending scenario of that batch again):

       ```powershell
       $file = "docs/validation/runs/test-a/<model dir>/batch-<id>.results.jsonl"
       New-Item -ItemType File -Force -Path $file
       ```
- **Errored, expired and canceled results.** `<scenario>: errored (<type>); pending` (only the
  four transient types), `<scenario>: expired; pending` and `<scenario>: canceled; pending`:
  - that request got no answer and is not billed;
  - the run sends it again in a new batch after a short pause.

  `<scenario>: errored (<type>); it would fail again`: the batch is held; send the message to the
  reviewer.
- **The 4-batch cap.** `<scenario> was sent in 4 batches without an answer; it is not sent
  again`: do not edit `batches.json`; send the message and `batches.json` to the reviewer.
- **Four failures in a row.** `the request failed 4 times; last: …`, `creating a batch failed 4
  times; last: …` and `listing batches failed: …`:
  - the network or the API failed four times in a row;
  - nothing was lost and nothing is paid twice;
  - wait a few minutes and run the same command again. If it fails again, send the message to
    the reviewer.
- **A refused batch.** `creating a batch failed: HTTP <status> (<type>: …)`: the API refused the
  batch, and nothing was created or charged.
  - A 400 or a 401 is the key or the spend limit (§1).
  - A 402 `billing_error` is the organisation's credit balance.
  - For anything else, send the message to the reviewer.
- **The key variable.** `POTAKT_ANTHROPIC_API_KEY is not set in this shell` (or
  `POTAKT_GEMINI_API_KEY`): this window has no key, so run the block of §2 in this window. `… is
  not a valid key`: the pasted value holds a space, a line break or a character that is not plain
  ASCII, so set it again (§2).
- **The batch state.** `<path> is not a valid batch state; it was not changed`: do not edit it;
  send it with the message to the reviewer.
- **The lock file.** `cannot create <path>: …` or `cannot write <path>: …`: the runs directory
  cannot be written (a full disk, or a folder the antivirus protects); fix that and run again.
- **The laptop clock.** `the laptop clock differs from the API's by <n> s (more than 120 s): sync
  the Windows clock (Settings, Time & language, Date & time, Sync now) and run again; nothing was
  sent`: a laptop clock that is wrong by more than two minutes would misplace the checks that
  keep a batch from being paid twice. Sync the clock and run the same command again.
  `the API answer has no readable Date header, so the laptop clock cannot be checked; send this
  message to the reviewer`: nothing was sent; send the message to the reviewer.
- **An answer that could not be saved.** `<scenario>: an answer that could not be saved is kept
  in <path>; nothing was sent; send this message to the reviewer and do not delete or rename the
  file`: an earlier run received this answer but Windows refused to save it (another program held
  the file for longer than the run retries, about 3 seconds). The answer is kept in that `.tmp`
  file; sending the scenario again would pay twice and lose it, so every run stops first, the dry
  run too. Send the message to the reviewer, who decides whether the file holds a complete
  answer, and leave the file where it is. (A leftover of a scenario of the open batch does not
  stop the run: see `… will be replaced by the answer of batch <id>` above.)
- **A changed prompt or schema.** `the prompt file differs from the one ADR-0007 §7 pins (SHA256
  <sha256>); nothing was sent` or `the schema section differs from the one ADR-0007 §7 pins
  (SHA256 <sha256>); nothing was sent` (every run, the dry run too): the prompt or the code is
  not the one pre-registered, so no answer would count. Run `git status` from the repository
  root, do not run `--yes`, and send both outputs to the reviewer.
- **A changed request.** `<scenario>: the request for <model> differs from the one the harness
  pins (SHA256 <sha256>); nothing was sent` or `<scenario>: no request is pinned for <model>;
  nothing was sent` (every run, the dry run too): the request of that scenario, as rendered from
  its CSV files, the prompt and the schema, is not the pre-registered one; nothing was sent. Run
  `git status` from the repository root, do not run `--yes`, and send both outputs to the
  reviewer.
- **`the workspace has batch <ids> that no batches.json in <runs> records and
  acknowledged-batches.txt does not name; nothing was sent: …`** (before the first batch of a
  Claude model; `batches` when it names several): the workspace has a batch that the records on
  this laptop do not know. If the records of an earlier run were lost (for example after
  `git clean`, in a second clone or with another `--runs`), run
  `git status docs/validation/runs/test-a` from the repository root; if it shows the files of
  `<model dir>` as deleted, restore them with
  `git restore docs/validation/runs/test-a/<model dir>` and run the same command again.
  Otherwise do not run `--yes` again: send the message and that output to the reviewer.
- **`the workspace lists more than 1000 batches: this check cannot see them all; nothing was
  sent; send this message to the reviewer`**: send the message to the reviewer.
- **`<scenario>: its answer is in batch-<id>.results.jsonl but its records are missing: restore
  them from git before running again; nothing was sent`**: the files of that scenario were
  deleted, but its answer is in the kept results of a finished batch; sending it again would pay
  twice. From the repository root, `git status docs/validation/runs/test-a` shows them as
  deleted; `git restore docs/validation/runs/test-a/<model dir>/<scenario>` brings them back;
  then run the same command again. If git does not show them, send the message to the reviewer.
  `cannot read the kept results <path>: <reason>; nothing was sent: restore it from git or send
  this message to the reviewer`: the same check could not read that file; if
  `git status docs/validation/runs/test-a` shows it as deleted, `git restore <path>` brings it
  back; otherwise send the message to the reviewer.
- **The acknowledgement file.** `<path> is not UTF-8: write it with the command in the runbook;
  it was not changed`, `<path> line <n> is not a batch id; it was not changed` or
  `cannot read <path>: <reason>` (for `acknowledged-batches.txt`): nothing was sent. Do not edit
  the file; send the message to the reviewer, who decides how to rewrite it (only with the
  command of «Acknowledged batches»). For `cannot read`, first close any program that has the
  file open (an editor) and run the same command again.
- **Acknowledged batches.** Only on the reviewer's decision, from the repository root, once for
  each batch id the reviewer names:
  `Add-Content -Path docs/validation/runs/test-a/acknowledged-batches.txt -Value "<id>  # <reason>"
  -Encoding Ascii`
  Then commit the file with the records. A run never adopts or names an acknowledged batch, and
  the check before a model's first batch does not stop on it. Do not write the file in any other
  way: `Set-Content`, `Out-File` and `>` change its encoding. The same command, on two short
  lines to copy:

  ```powershell
  $file = "docs/validation/runs/test-a/acknowledged-batches.txt"
  Add-Content -Path $file -Value "<id>  # <reason>" -Encoding Ascii
  ```
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
  reviewer`, or, if the answer itself was saved, with `<scenario>: recorded, but its record could
  not be completed (<type>); the next run completes it; send this message to the reviewer`. Only
  the type of the error is shown, never its text. Send the message to the reviewer.
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
- **`cannot summarize <dir>: <reason>`** after a run: every answer is recorded and the exit code
  is 0; only the summary could not be shown (for example a file locked by another program). Run
  `uv run potakt eval summary <dir>` once the file is free and send the output to the reviewer.
- **`preflight failed: <check>: <reason>`**: a check of `eval preflight` found a definite
  problem with the key or the model; nothing was sent or charged. An HTTP 401 or 400 is the key
  (§1); anything else, send the output to the reviewer. Do not run `--yes` until it passes.
- **`interrupted: the preflight sent no prompt and recorded nothing; run it again`** or
  **`eval preflight stopped by an unexpected error (<type>): send this message to the
  reviewer`**: run the preflight again, or send the message to the reviewer.
- **After Phase 0:** delete both keys (Claude Console and Google AI Studio).
