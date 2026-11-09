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
# every validation rule; 10 scenarios x 3 models). A changed byte in the prompt, a CSV, the
# schema or the code that builds a payload changes one of them.

PINS = """
claude-haiku-5-5
  01-pharmacy 7c7ca5e4824fe36874a63bd9add3fd5a67bc3cc654832183acf53df6309dc73e
  02-hotel-reception 73f074a6f4a7933decf9d2352a8d4eaf4804a2934b0d2c99dda8eaec0eae83d1
  03-call-centre f641425806789efc47d0ce767712890780a06e1769d6412e83f7d173ae2254e9
  04-restaurant e3a4bc0886546ace3ad1e889af9954e8131f6ddb39c28b321c9f6bd4c4b295f6
  05-supermarket-checkouts 21558257d71ba732cc23375d6c393bc766c1f767859e4a33101a64c073a56363
  06-ambulance-crews 11ca5561d952e6e9ad0db339a0af4e337df4db8352ad514e8074cbd2d541e91e
  07-security-guards 3668ac0f2c906686c115b605a2f7c39ef6d1097211243bfbabf3f5ccee548912
  08-warehouse 1b21603e8041f5f504df17139e8b568f235a96afc49aa5d2b0722594df75b7cb
  09-care-home d5b08794ceb152431b4030af3f02dc12f931dba2934117b10f1dc0c31d92e481
  10-municipal-nursery 267c5022afaaec98441f44220ba8f887f4cab44a55d7ee6ee92bd775b51185a3
gemini-3.8-flash
  01-pharmacy a9d167abfd270c166c410753282b69cfe0afe81b89bc6ff0ff5631336f92c2a0
  02-hotel-reception acad2c5119380996d15756012e69ab20d2c2f9e0f8dcb60ca30c1c9cc34e7e8a
  03-call-centre 6672726ec2331eafe82b4c8f03e6a10c8f3250d6873738290b8d42c4b5385a6d
  04-restaurant f864c2f43108231f4fb2d97870e188f3a49955e4bfd191802249a8c9779e9dec
  05-supermarket-checkouts df3d64bd1d9fd25ee4e26ab8bda571da6b72c3265876960ae46ab975dae92caa
  06-ambulance-crews 57410ec16a6e75ff08615a224eff2e1805d596b0c80703ebd525f970f436eab7
  07-security-guards 766491cadbbc0a46750c76537b57b84393773f26b3260ae48535f7750a49cfae
  08-warehouse d693a78fd7616d7cd3424f5761031dc5b7cddba348895b105c64fafcdad60ad4
  09-care-home 465f596b8e9aaea7572aa804cf431f91de127e60034687840a76f2a2e883a9a0
  10-municipal-nursery faa1572855e851288a5639fa7e58a6a961e2e5ff244c9e520c724d5b299c7863
claude-opus-5-5
  01-pharmacy bb15710ef23f6de5cdc00ec541cad83238e5bb8cb04d9c52958f70074bbc1cca
  02-hotel-reception d3cfe6f0fec7067d9e26345cd5f6559b76dcd9f5043e9faf4a0be01b8ef40f94
  03-call-centre 87cd1a945af9165dd6a7415e8f004890085c45acf5382b392d8f48640afe9ffb
  04-restaurant f7c7b2b01f6d366fff6038a7ace9109e5b99217f1bd8e0366c3399578199f4fe
  05-supermarket-checkouts 7b31693f5af0205c4cf98c02b537cca37fb0a6442c61880fec126afccbd0e440
  06-ambulance-crews 6f63e75248254284ceb04dd2698258cfb636177653bf07bd66f6fea5e18e9851
  07-security-guards dcf8af24abefb10551cac1296fc20b9cd90681fbc77ff4d8b1d53ac2c79496d8
  08-warehouse c6f068a0913527d284d565277be85b2dd85fcbbbe9d98079560eb9d4389bcc7a
  09-care-home 97029720558a72925ff4aa8e3d448e160f17d825064904cf8f2bacea982cbbd5
  10-municipal-nursery 0ae7adbd41bc59c5e8a9a498b4a768f48a15f428f00e97498a775614aacfd541
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
