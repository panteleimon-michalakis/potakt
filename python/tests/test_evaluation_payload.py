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
# exact bytes of each request.json, re-rendered on 2026-11-09 (TASK-005e1: the schema shows
# every validation rule, and the instruction asks for the data files by their exact names; 10
# scenarios x 3 models). A changed byte in the prompt, a CSV, the
# schema or the code that builds a payload changes one of them.

PINS = """
claude-haiku-5-5
  01-pharmacy 9e4b268398b255b7aedbd3c157dc37c418fca415ba5cf0e2a3ec62ade2096692
  02-hotel-reception 33c5e582f68fa6020f2575cbbf5c3114b1d4cb18fbfd4565f787c693b99ff778
  03-call-centre aa35242f938e0c7c2ce1fa55644111235fb3ffd09c3e6632f548bbca487a21fc
  04-restaurant df2b704bb01c9187e41e4220224c013e31303e6a90cb80d25d457f53d719939b
  05-supermarket-checkouts 4ecaa78f97f673d841308c89708904a1e41755f2f1cd8244677e7cb6863d8e3a
  06-ambulance-crews e339d57622a494bd11e0ee757a8339cb3808c0a0446f912aa423156fd63e1af3
  07-security-guards 0815d610bfb77bea368740490be55c687d18d97bf3c1fa9d6281b83d2a6ea62a
  08-warehouse 58c18338f157bbaa04c62f630ca2407ba2aeb2ed4f83b81e2b85057c9a6d07b3
  09-care-home 911c50070c758bcf876c68a938f2cf595645d356fb5bd796d04df0eb0d211daa
  10-municipal-nursery dbeeebe6e5a29c1338ea6c3716b6e6eb333f36256c0693e4534853aaa98766a0
gemini-3.8-flash
  01-pharmacy 9f72fb57c4e25dab878583fa95c2b04ee997b925c5a1effddd72d9544efba193
  02-hotel-reception 777251e24d646a83df0a459b0b5c65c5ef8a1876fcce15a6a90b61008272a850
  03-call-centre b7e6b18933de912c43890231d8255163743da3248d09be77f74f93467632b307
  04-restaurant e9e6580ea0adf79242d54ee181b907efe1ec406407d02d6c791ec234c0e3ef6e
  05-supermarket-checkouts b194dc595ba8f12f89ce40216315b13f985aa95988b6b19f8d1033770c4cf9e3
  06-ambulance-crews 1c664390a9975e5f9e98703abe9172cc5a27c3acafeecd6f4a088045d260fd22
  07-security-guards 2face974010b134802e78e7ac652b4e0b3af343f2831ad876246985e7c70c86e
  08-warehouse 61e78a888cdb8f2d2a7237f9b63ad8efc4180f57ba3e9dc1d93dfa7964c4e103
  09-care-home a9df05f8898807162327ecc2f78367ee739f5c1844a00e28f51bcbe8e6b067ef
  10-municipal-nursery c4ec07849b03a818ab8e047d6b0ff6200eb9e648e809ff7dfcf68c708b32df16
claude-opus-5-5
  01-pharmacy 47807e3828d48375587aa694c89e73b8887e726d757b8c202861c4368f4ae159
  02-hotel-reception 74e8600aa6b0b963dae8be593c7fceef2a6f57c560c536f099d629f06ad002f1
  03-call-centre ae28d2be035286791131d62d118926cc2e4ef776b7700a0eac83d8d64a8228c7
  04-restaurant 3f347598214653d48ef37b7166f87b9fa3ef72cab69b65eb464d17ab4c738211
  05-supermarket-checkouts 81708be040aca347609cbb83888326408bc8767c4e0a2378f1bf1a0344f3675d
  06-ambulance-crews ca327b949f210bca0097a0ed8bf6275c6531ea9ba1475d5ca65fdefafea74ace
  07-security-guards 06e88920a717886e56df8d9ed885b405ffa679531b9f39a545f49ef60a000d54
  08-warehouse 8770a2cba8a70862cc5eca18d02db8ad65057ec7c891f0963601f25b4857184c
  09-care-home 6d3af04b01500af4010d02d78146275a33fee0b781de9b6627ebdc7f84434e4b
  10-municipal-nursery a987b7f0e7268b02379493302c4f5bb7119b1a1b4c21a7ae35fef755a22be071
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
