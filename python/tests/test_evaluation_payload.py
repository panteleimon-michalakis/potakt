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
# exact bytes of each request.json, rendered on 2026-11-07 (10 scenarios x 3 models). A changed
# byte in the prompt, a CSV, the schema or the code that builds a payload changes one of them.

PINS = """
claude-haiku-5-5
  01-pharmacy faef5841bbd85ecd626384e960c4b7c46f346ea07daa2a50b6fe5612f734fd05
  02-hotel-reception 81a80663d0336591ddd7ba5accc16daa01ff1b6808f0ff89f4b1e6f66295982d
  03-call-centre 421350f0c28470490bdb3c064dc2c6b39ff4c896fe01c796c7c1562150626be2
  04-restaurant 1428fa1b37659f82fb63f0edab27ab41c2fef5358b6835eacf78c276770ca0cb
  05-supermarket-checkouts 479ad540b308c41f8608cea99e1d851c713ac0e3a3efcb6a7720ff349d9a2647
  06-ambulance-crews 1c0369b9904296acaa28d945929783b9462d0cdbdfaadf8d3b6cb30c87e8ca54
  07-security-guards 10cbdc212bb05ecc0eda59666810d44141dc2e6ebdb69a92a5c331b5608357ab
  08-warehouse f62fa4f41920ce1914a41ec6c515fcede6aa72bd2d9e5f8b5b632976842da639
  09-care-home c3afab5992ee1ec636118d47d1c142303ff87e9c649745c2ce59095fbc10fe83
  10-municipal-nursery d8a3334b2e5bc839942f042816e44c160887daaf6d4c581b50b618875c1ddee4
gemini-3.8-flash
  01-pharmacy d961421eb43735a6b34f3ca22c4463bd7a1e2c3f44e2ebd92356b68f5f4a87d4
  02-hotel-reception 0ba5f16b35e1ba8713112eff81565fefd5e2fb51fe3f2cbbd90a450148d1538f
  03-call-centre b6df81386b396daedf779e3e97a5096a9e5259af7c48402a621e1625f6ab9354
  04-restaurant 1df724151443e1c11887b51bf4d394e4b57ffcfded3de462be34121a812f3b21
  05-supermarket-checkouts 2d514bc139b46ab484133dbac3d92da0abc479457c5f49ca3980c2b743ba9a55
  06-ambulance-crews b3a08cfc864e59da11f6a161480ffc9eba3d15659c9cc6730943a19c1827e184
  07-security-guards 4badf090dfc0930cece1ca5964e0b00cecee4084471e999364b416c9ec38f93a
  08-warehouse 1129739c397f4a90a2253c64408a706065f948c64b648cb51a0ad1300c5957f2
  09-care-home 760b6f9e78a0d0ad555ba61375e0a34a9a2bf07e705bea53bcc4c9d82394de39
  10-municipal-nursery 228286c61c2bc1e53609507c130137cfe80bdc84f8971a265ea2644f695f3047
claude-opus-5-5
  01-pharmacy ad2a0b5ed17b6bc6060dca73e25f78465ae44ed5835e3a685dbf170d31481770
  02-hotel-reception ae284a47d48a85852d12173ad3dfba5745d0372db4abf630d58759b5bd1885a4
  03-call-centre fd42f6b8ff7fe4dadc3c27b06a91b6b243654ec7e8acc966146e722c0c6c36b2
  04-restaurant 41e862ca7928a5dbc47446e85ac62a891085d526dc387ece761b258dedf3d729
  05-supermarket-checkouts 1a459150d622e7e95792c4faace69911f99641394df1332306289dfe9c0a098b
  06-ambulance-crews c1567196b7aff31143022c71ea6493a89e03303a9d90d8f49e75670c49a6a611
  07-security-guards ff891de4da228abaec98642240e382e559f81d6872fd7b56cc3121f391c46e1c
  08-warehouse 159330b7222341544111acefdfe860d1fb1719c1c546f925bb81ebdd538aaf59
  09-care-home acd34f6e055bda192f44cc2881f7c42c2e332f1a16568e2101b5b9869d79a6d5
  10-municipal-nursery 8c6c04064ce699e25a53eadb1b1a902f4349fb9145a22c2c1a058f58b024a8ba
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
