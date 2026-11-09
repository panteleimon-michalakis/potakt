"""The Test A request payloads (ADR-0007 §7 «Models», «Settings»; TASK-005c1 D5).

Field names from the official API references, looked up on 2026-11-02: Anthropic Messages API
(``model``, ``max_tokens``, ``system``, ``messages``, ``output_config.effort``) and Gemini
``generateContent`` (``systemInstruction``, ``contents``, ``generationConfig.maxOutputTokens``).
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import potakt.evaluation.runner as runner_module
import potakt.evaluation.test_a as test_a_module
from potakt.evaluation import (
    CORPUS_PATH,
    INSTRUCTION_PATH,
    REGISTERED_MODELS,
    STRONG_MODEL,
    Prompt,
    UnknownModelError,
    load_scenario,
    render_payload,
    render_test_a,
    request_payload,
)
from potakt.evaluation.test_a import ANTHROPIC_MAX_TOKENS, GEMINI_MAX_OUTPUT_TOKENS

PROMPT = Prompt(system="Answer with JSON.\n", user="## Request\n\nPlan the week, Μαρία.\n")


def test_the_strong_model_is_opus() -> None:
    assert STRONG_MODEL == "claude-opus-5-5"


def test_opus_payload() -> None:
    assert request_payload(PROMPT, "claude-opus-5-5") == {
        "model": "claude-opus-5-5",
        "max_tokens": 64000,
        "system": PROMPT.system,
        "messages": [{"role": "user", "content": PROMPT.user}],
        "output_config": {"effort": "high"},
    }


def test_haiku_payload_has_the_same_limit_and_the_documented_effort() -> None:
    # Anthropic documents effort for claude-haiku-5-5, so ADR §7 sends it ("as above": high).
    # TASK-005c3g G8: the same output budget for every model (thinking shares it).
    assert request_payload(PROMPT, "claude-haiku-5-5") == {
        "model": "claude-haiku-5-5",
        "max_tokens": 64000,
        "system": PROMPT.system,
        "messages": [{"role": "user", "content": PROMPT.user}],
        "output_config": {"effort": "high"},
    }


def test_gemini_gets_the_gemini_payload() -> None:
    # The Gemini model is part of the URL, not of the body.
    assert request_payload(PROMPT, "gemini-3.8-flash") == {
        "systemInstruction": {"parts": [{"text": PROMPT.system}]},
        "contents": [{"role": "user", "parts": [{"text": PROMPT.user}]}],
        "generationConfig": {"maxOutputTokens": 64000},
    }


def test_every_model_has_the_same_output_budget() -> None:
    assert ANTHROPIC_MAX_TOKENS == {"claude-opus-5-5": 64000, "claude-haiku-5-5": 64000}
    assert GEMINI_MAX_OUTPUT_TOKENS == 64000


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-5-5", "claude-opus-5", "claude-", "gemini-2.5-flash", "any-other-id", ""],
)
def test_a_model_that_is_not_registered_is_refused(model: str) -> None:
    # TASK-005c2 E3: only the three pre-registered ids, listed in run order in the message.
    with pytest.raises(UnknownModelError) as caught:
        request_payload(PROMPT, model)

    assert str(caught.value) == (
        f"no pre-registered Test A settings for {model!r}; "
        "use claude-haiku-5-5, gemini-3.8-flash or claude-opus-5-5"
    )


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-haiku-5-5", "gemini-3.8-flash"])
def test_payloads_carry_no_credentials_tools_or_sampling(model: str) -> None:
    text = render_payload(request_payload(PROMPT, model)).lower()

    for word in ("key", 'token"', "authorization", "tools", "temperature", "top_p", "top_k"):
        assert word not in text, word
    assert "thinking" not in text  # left to the API default


def test_render_payload_is_sorted_indented_utf8_json_with_a_final_newline() -> None:
    text = render_payload(request_payload(PROMPT, "claude-opus-5-5"))

    assert text.endswith("}\n")
    assert "Μαρία" in text  # not escaped
    assert json.loads(text) == request_payload(PROMPT, "claude-opus-5-5")
    assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


# --- TASK-005c3g G8: all 30 request.json payloads pinned -------------------------------------
# The SHA-256 of render_payload(request_payload(render_test_a(scenario), model)) as UTF-8, the
# exact bytes of each request.json, re-rendered on 2026-11-10 (TASK-005e1 follow-up: the schema
# also states integer literals, empty list cells and the horizon date shape; 10
# scenarios x 3 models). A changed byte in the prompt, a CSV, the
# schema or the code that builds a payload changes one of them.

PINS = """
claude-haiku-5-5
  01-pharmacy 3702e6913ec0d626028c283e3e03d9a364d5835de17213c1219dc00ecfda0d0d
  02-hotel-reception dfec88ac78989f21f921bfcc1c7bff086f6f035ff5e107bb70ebbefe6446d4e0
  03-call-centre 8b8761dd4362d66c8b7185dcfea97b22fc0ba517ba1b66ff37664ba93449de79
  04-restaurant dc4e68bb1f8cdc61698431094aa083a3ac4a75680540bac5f127fcedb1f682cb
  05-supermarket-checkouts 6c2259c82ff4fcb4e867d85bd611f9f15cffb2d867eb1cda0eb007eb8bc5bd50
  06-ambulance-crews 91569eb9d7241983bbe4ea6bd44a3970570b81b73462fa33a0c2c8e0aef5828f
  07-security-guards 47f6d6cf8c4675cb570c6ce03bef8be40ca8da04b598c0c100b07361222df053
  08-warehouse 1525adf054f55a200f021c08e2b40f76d9d5755d6f6063465f048cdc416b1f68
  09-care-home 7f8283ca6b26c8cddc5019dfa1e166044f88617f62a5a3349a90578104d9fca5
  10-municipal-nursery 3a719e8884209c4d0b6989bc1aaaf10596b3b84f1788bad38aaf83fbae29014e
gemini-3.8-flash
  01-pharmacy 86b758384fb8811e0ecfff73509c6b38aee54fd5cda975f3652bbca558b025b1
  02-hotel-reception 50fac1af77904284a0643e1dca2cdcef4d5119c9c802d0afc996a5dedd1a2a4c
  03-call-centre 551192810c8e6cf7bd9c6cb0a9d96b209ca1725b8bb3e781f9fb41b4dcfd0836
  04-restaurant 6a899c76a5231f254fbba306c4cb08bcbbea6615e30a5c4f13ccb967d64f8241
  05-supermarket-checkouts 7d20478e0a6cda78ee79031a4f379225eb926f8908424a51cdc2cd0b312182d0
  06-ambulance-crews 79717ecc42927fbcc52872092150c0ff80399668f427f2789a1bd6c2e617cafd
  07-security-guards a50462e093d1c3d1eb9f0801d53fdb7badbf4950e3c6bd41209e806bb97662b3
  08-warehouse 30c6533b32e759d035ec5cdfe853068bf3fdf216036d80621533fe0e0d9333cd
  09-care-home 3a386574e09650fef04513c3712881887d4685e440e299bb0283441dd97df62c
  10-municipal-nursery e819d76a8b487840f15a70a32449bc5c7ae65c2b1fa066b168d6bfcd351f8ee5
claude-opus-5-5
  01-pharmacy 6ffdad29d405c47c577321813e46264bfb69ee9705f0d8445dbfd716b2d7418a
  02-hotel-reception 97d1a0c22fddabced95674ff6b9560430d1f5d83e294777e0eeae37539375f96
  03-call-centre 6edfc3d28f5018c4529b1696ed7f29f4594425bbba56f8ede3ca4c2dbd266877
  04-restaurant c43339b196bce5fc1f30684490456760485aebc1a54d08b07ae926e53f1e36ec
  05-supermarket-checkouts 0e45865310df919f9ca74152250765c4ec405bc51241b1bec2da52279a564507
  06-ambulance-crews bceca2befc110f5be82600dad0074cd08b1e463f6c04a5883d734580d278391c
  07-security-guards 180ae4c2c7dab4f47ddd553dd15e81fe8fbf954140595806266e88da27935d7b
  08-warehouse d515ac77bfe5e9bf49e847e0f21507be1610c851347f1baadbf341590487859c
  09-care-home 5c1ca20accfbddf970e7d1378a8f040e52893659586023ff60bb6522c7b7728d
  10-municipal-nursery b2dc1d24eba5c709c5d298cb2968dc6d5b3ec1ecd99ec8ae16f23e9c77966a30
"""


def _pins(text: str) -> dict[tuple[str, str], str]:
    """``PINS`` read: a model line, then its indented ``<scenario> <sha256>`` lines."""
    pins: dict[tuple[str, str], str] = {}
    model = ""
    for line in text.strip().splitlines():
        if not line.startswith(" "):
            model = line
            continue
        name, digest = line.split()
        pins[(model, name)] = digest
    return pins


PINNED_PAYLOADS = _pins(PINS)


def payload_sha256(corpus: Path, name: str, model: str) -> str:
    """The SHA-256 of the request.json bytes of ``name`` for ``model`` (``eval prompt``)."""
    payload = request_payload(render_test_a(load_scenario(corpus / name)), model)
    return hashlib.sha256(render_payload(payload).encode("utf-8")).hexdigest()


def test_the_pins_cover_every_scenario_and_model() -> None:
    names = sorted(
        (path.name for path in CORPUS_PATH.iterdir() if (path / "scenario.json").is_file()),
        key=lambda name: name.encode("utf-8"),
    )
    assert sorted(PINNED_PAYLOADS) == sorted(
        (model, name) for model in REGISTERED_MODELS for name in names
    )
    assert len(PINNED_PAYLOADS) == 30


def test_the_runner_checks_requests_against_exactly_this_table() -> None:
    # TASK-005c3h H7: the runner's pins (checked when each request is built) are these 30.
    assert dict(runner_module.PAYLOAD_SHA256) == PINNED_PAYLOADS  # both ways: dict equality
    assert len(runner_module.PAYLOAD_SHA256) == 30


@pytest.mark.parametrize(("model", "name"), sorted(PINNED_PAYLOADS))
def test_each_payload_is_pinned(model: str, name: str) -> None:
    assert payload_sha256(CORPUS_PATH, name, model) == PINNED_PAYLOADS[(model, name)]


def test_a_changed_byte_in_a_csv_changes_the_pinned_payload(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS_PATH / "01-pharmacy", corpus / "01-pharmacy")
    csv = sorted((corpus / "01-pharmacy").glob("*.csv"))[0]
    data = bytearray(csv.read_bytes())
    data[-2] = data[-2] ^ 1  # one bit of the last character before the final line break
    csv.write_bytes(bytes(data))

    for model in REGISTERED_MODELS:
        assert (
            payload_sha256(corpus, "01-pharmacy", model) != PINNED_PAYLOADS[(model, "01-pharmacy")]
        )


def test_a_changed_byte_in_the_prompt_changes_the_pinned_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = tmp_path / "test-a.md"
    prompt.write_bytes(INSTRUCTION_PATH.read_bytes().replace(b"JSON", b"JSOM", 1))
    assert prompt.read_bytes() != INSTRUCTION_PATH.read_bytes()
    monkeypatch.setattr(test_a_module, "INSTRUCTION_PATH", prompt)

    for model in REGISTERED_MODELS:
        assert (
            payload_sha256(CORPUS_PATH, "01-pharmacy", model)
            != PINNED_PAYLOADS[(model, "01-pharmacy")]
        )
