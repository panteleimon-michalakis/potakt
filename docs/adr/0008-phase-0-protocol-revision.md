# ADR-0008: Phase 0 protocol revision before the first LLM run

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

TASK-005r1 reviewed the whole project before the first paid run; no model had seen any prompt. It
found that the experiment of ADR-0007 could still decide Gate 1 for reasons other than the one it
measures: the schema section did not show many rules that make a Decision Model invalid, Test B's
baseline was not fixed, one invalid answer can move the gate number by up to 15 points, and two
truth rules miss encodings with the same effect. ADR-0007 already carries six amendments; this ADR
records the revision in one place. Where it changes ADR-0007, it says so, and ADR-0007 points here.

## Decision

### D1. Development set, freeze, and "before any LLM run"
- Development scenarios live in `docs/validation/dev/`, in the corpus format of ADR-0007 §2, and
  never count for a gate. Before the freeze, models run on them only: to test the harness end to end
  and to revise the instruction, the schema text and the settings. Their records are committed under
  `docs/validation/runs/dev/`, and the report lists every revision they caused.
- No corpus scenario is sent to any model before the freeze. "Before the first LLM run" and "before
  any LLM run" in ADR-0007 mean before the first run on a corpus scenario.
- The freeze is the commit tagged `phase0-freeze`, an annotated tag pushed to GitHub before the
  first corpus run. It fixes the corpus, the instruction texts, the schema section, the payload
  pins, the Test B and judge prompts, and every rule of ADR-0007, this ADR and ADR-0009. After it no
  request byte changes; between the `claude-haiku-5-5` and the `claude-opus-5-5` corpus runs only
  changes that leave every request byte unchanged are allowed (a fix found by `eval rescore`, a
  runbook fix). Any other later change needs a new ADR, and the results are reported under both the
  old and the new rule (ADR-0007 §1).

### D2. The schema section shows every validation rule
- The schema section (ADR-0007 §3 and §7 «Prompt») states every rule that `potakt validate`
  enforces on a Decision Model and on its bindings to the data: by a JSON Schema draft 2020-12
  keyword where one expresses the rule exactly, by a description otherwise. An inventory test maps
  every validation message in the code to the place where the schema shows its rule. The set of
  valid Decision Models does not change, with one exception: a horizon date written as a
  Unix-timestamp string (for example "1794787200"), which the parser accepted by accident and
  ADR-0004 never allowed, is now rejected. A differential test checks, both ways, that the schema
  and the validator disagree only on rules a description states.
- The instruction asks for a model that "conforms to the schema and to the rules its descriptions
  state", and to "bind the data files you need by their file names exactly as given (no folder)".
- The pins in force are listed in Appendix A; every change before the freeze updates them.

### D3. The gate number and its uncertainty (Test A)
The summary of ADR-0007 §5.4 also shows, without changing the gate:
- (a) one row per scenario: language, difficulty, truth count, validity, stop reason, output and
  thinking tokens, estimated cost, matched count, recall and precision and, for an invalid
  candidate, the class of its first issue (JSON or fence, schema, cross-field rule, data file,
  data-level id or attribute, stop reason);
- (b) the micro recall over the valid candidates only, labelled "not the gate";
- (c) a 95 % percentile bootstrap interval of the micro recall: the scenarios are resampled with
  replacement 10 000 times with the seed 20261109, and Σ matched / Σ truth is computed in each
  resample.

A result whose interval contains 0.70 is decided by its point value like any other, and the report
says that it lies within the uncertainty of a ten-scenario corpus. No decision band is added.

### D4. Output budget
`max_tokens` stays 64000 (ADR-0007 §7, amendment of 2026-11-07), unless a development run of
`claude-opus-5-5` stops with `max_tokens` or uses more than 48 000 output tokens. In that case the
output budget is decided again by an amendment to this ADR before the freeze, with new pins.

### D5. Records
`batches.json` also keeps each ended batch's `ended_at` as the API reports it. A Gemini run writes
`received.json` (`{"received_at": <UTC instant>}`) next to `response.json`. Neither changes a score.

### D6. A scenario that never gets an answer
A corpus scenario that has no answer after four batches is sent again in one more run, at least 24
hours later. If it still has no answer, it counts as an invalid candidate for the gate (every truth
constraint of that scenario missed), and the report says so.

### D7. One repair turn, reported apart
After a model's corpus run is complete, each invalid candidate gets one repair request. It is the
same request plus the answer as the assistant turn and one user message: `The Decision Model is not
valid:`, then one issue per line (`<path>: <message>`), then `Answer with the corrected JSON object
only.` The repaired candidate is scored like the first and reported as "recall after one repair". It
never decides the gate.

### D8. Draw-to-draw variation
After the first `claude-opus-5-5` corpus run is complete, the same ten requests are sent twice more
and recorded under `docs/validation/runs/test-a-repeats/claude-opus-5-5/<k>/` (k = 2, 3). The report
gives each scenario's recall over the three draws (mean, minimum and maximum) and the three micro
recalls. These runs never change the gate or a first-attempt record.

### D9. Look-alike ids
Ids are compared exactly. Two ids are look-alikes when they become equal after every Greek capital
letter that has a Latin look-alike (Α Β Ε Ζ Η Ι Κ Μ Ν Ο Ρ Τ Υ Χ) is replaced by that Latin letter
(A B E Z H I K M N O P T Y X).
- A validation message about an unknown id names the one data id that is its look-alike, if exactly
  one is.
- The report also gives, without changing the gate, the recall after every unknown id in each
  invalid candidate is replaced by its unique look-alike data id, and the number of candidates this
  makes valid.

### D10. Equal effect under the truth's hard pair rules (ADR-0007 §5.2)
Let F be the set of (resource, task) pairs that the truth's hard `requires_attribute` and
`unavailable` rules forbid on the truth's data (`forbidden_pairs`). Before matching, in both the
truth and the candidate:
- a `stability` baseline loses its pairs in F;
- a `together` rule's in-scope tasks lose the tasks that F forbids to both of its resources;
- an `apart` rule's in-scope tasks lose the tasks that F forbids to at least one of its resources.

On every plan that keeps the truth's hard pair rules, these differences change no `together` penalty
and no `apart` violation. They shift every plan's `stability` penalty by the same amount, so they
rank those plans identically. The rule applies to the primary and to the merged recall (ADR-0007
§5.3 and §5.7).

### D11. Diagnostics (ADR-0007 §5.5)
A new informational category, `other_template_same_effect`, covers an unmatched truth constraint
whose hardness and set of penalized or forbidden (resource, task) pairs on the plan "every resource
works every task", computed with the checker, equal those of an unmatched candidate constraint of
another template. It never changes recall or the gate.

### D12. Flagged unsupported requests (ADR-0007 §5.6)
The summary prints each candidate's assumption texts next to the scenario's `unsupported` quotes. An
assumption flags an unsupported request if a reader of the scenario language finds in it the
request's subject: the people, the attribute and the time it names. The owner rates, the reviewer
checks, and the count is reported per model.

### D13. Budget
The Phase 0 API budget is a hard cap of US$70, enforced by the Anthropic workspace spend limit and
raised in stages so that each stage covers the worst case of every run before it: $15 for the
development runs, $50 before the corpus Test A runs and $70 before Test B. The upper bounds are
about $15 for the development runs, about $31 for Test A (first attempt $7.63, two repeats $15.26,
plus repairs and Haiku) and about $13 for Test B with its judge. This replaces DESIGN §16.1's
"≈ €5", and the actual costs are reported. DESIGN's development budget target for later phases
(≤ €20/month) is unchanged.

### D14. Test B
ADR-0009 fixes the full protocol before the freeze. This ADR fixes:
- **Baseline:** `claude-opus-5-5` through the Message Batches API, with only the code execution tool
  (`code_execution_20260521`, no web access), effort `high`, no sampling parameters.
- **Messages:** the first message is a pinned instruction plus the bundle of ADR-0007 §3 without the
  schema. The second message is the disruption of ADR-0007 §6, in the scenario language, in the same
  container.
- **Plan and attempts:** the plan is the last `json` code block of the answer; one first attempt per
  scenario and turn. (V) is reported split into "no plan or invalid plan" and "hard violation".
- **Rubric:** the owner and one LLM judge of another model family rate it. The judge is pinned and
  checked against the owner's ratings on the development set. The verdict is "no" only if both say
  "no"; the raw agreement and Cohen's κ are reported.
- **Minimal count:** computed with OR-Tools CP-SAT, pinned in `uv.lock`, in at most 600 s per
  scenario; a value not proven optimal is flagged.

### D15. Schedule
Gate 1 is decided after the frozen runs, not at the "end of week 2" of DESIGN §16.2. The report
gives the dates.

## Consequences
- Every analysis rule exists before any corpus result. The development runs absorb harness and
  wording problems that would otherwise surface first on the test set.
- More engineering before Gate 1: development scenarios and runs, the reporting code, the repair and
  repeat runs, and the Test B harness. The cost cap rises from about €5 to US$70.
- The metric stays strict on meaning and lenient only on encodings whose effect is provably equal
  (D10). Other equal-effect encodings are reported (D11). A plan-level check, solving each candidate
  and scoring the plan under the truth, is left to Phase 1.

## Appendix A — pins in force
- Instruction `docs/validation/prompts/test-a.md`: SHA256 `2dcb497384a35ae227ccc8df46349d6179162687093ac16e0cd48be9f406263c`.
- Schema section (`potakt schema`): SHA256 `afd54559d52d492c8ce49914a886be49b2f377187d11098f8fff7395dec9c02c`.
- Payloads: the 30 values of `PAYLOAD_SHA256` in `python/src/potakt/evaluation/runner.py` at the
  commit that last changed this appendix.
