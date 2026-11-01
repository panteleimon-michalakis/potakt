"""The Test A request payloads (ADR-0007 §7 «Models», «Settings»; TASK-005c1 D5).

Field names from the official API references, looked up on 2026-11-02: Anthropic Messages API
(``model``, ``max_tokens``, ``system``, ``messages``, ``output_config.effort``) and Gemini
``generateContent`` (``systemInstruction``, ``contents``, ``generationConfig.maxOutputTokens``).
"""

import json

import pytest

from potakt.evaluation import (
    STRONG_MODEL,
    Prompt,
    UnknownModelError,
    render_payload,
    request_payload,
)

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


def test_haiku_payload_has_its_own_limit_and_the_documented_effort() -> None:
    # Anthropic documents effort for claude-haiku-5-5, so ADR §7 sends it ("as above": high).
    assert request_payload(PROMPT, "claude-haiku-5-5") == {
        "model": "claude-haiku-5-5",
        "max_tokens": 32000,
        "system": PROMPT.system,
        "messages": [{"role": "user", "content": PROMPT.user}],
        "output_config": {"effort": "high"},
    }


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-2.5-flash", "any-other-id"])
def test_every_other_model_gets_the_gemini_payload(model: str) -> None:
    # The Gemini model is part of the URL, not of the body.
    assert request_payload(PROMPT, model) == {
        "systemInstruction": {"parts": [{"text": PROMPT.system}]},
        "contents": [{"role": "user", "parts": [{"text": PROMPT.user}]}],
        "generationConfig": {"maxOutputTokens": 32000},
    }


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-5", "claude-"])
def test_a_claude_model_without_pre_registered_settings_is_refused(model: str) -> None:
    with pytest.raises(UnknownModelError, match="no pre-registered Test A settings"):
        request_payload(PROMPT, model)


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
