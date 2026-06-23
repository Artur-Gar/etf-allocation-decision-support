from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import llm.client as client_module
from llm.client import (
    ETFClassification,
    OutputValidationError,
    get_client,
    load_env_file,
    parse_batch_results,
    validate_classification_output,
)


def _payload(style_focus: str = "Growth") -> dict[str, object]:
    """Build a minimal ETF classification payload."""
    return {
        "primary_region": "US",
        "developed_or_emerging": "Developed",
        "style_focus": style_focus,
        "sector_focus": "Broad Market",
        "confidence": 0.91,
        "reasoning_short": "The ETF targets broad U.S. equities.",
    }


def _batch_line(custom_id: str, payload: dict[str, object]) -> str:
    """Build one fake Batch API output line."""
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": json.dumps(payload)}}]},
            },
        }
    )


def test_load_env_file_supports_standard_and_legacy_keys(
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load config from .env and map legacy placeholder keys to the active names."""
    env_path = workspace_tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                'API_KEY_ENV_VAR="sk-test-from-legacy"',
                'MODEL_ENV_VAR="gpt-5.4-mini"',
            ]
        ),
        encoding="utf-8",
    )

    for key in (
        "API_KEY_ENV_VAR",
        "MODEL_ENV_VAR",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    loaded_path = load_env_file(env_path)

    assert loaded_path == env_path
    assert os.getenv("OPENAI_API_KEY") == "sk-test-from-legacy"
    assert os.getenv("OPENAI_MODEL") == "gpt-5.4-mini"


def test_get_client_uses_api_key_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the OpenAI client from the configured API key."""
    captured_kwargs: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: str) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(client_module, "load_env_file", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    get_client()

    assert captured_kwargs == {
        "api_key": "sk-test",
    }


def test_validate_classification_output_accepts_valid_payload() -> None:
    """Accept a payload that matches the strict ETF classification schema."""
    result = validate_classification_output(_payload())

    assert isinstance(result, ETFClassification)
    assert result.primary_region == "US"
    assert result.confidence == 0.91


def test_validate_classification_output_rejects_invalid_values() -> None:
    """Reject a payload when one of the enum values is outside the schema."""
    with pytest.raises(OutputValidationError, match="Invalid style_focus"):
        validate_classification_output(_payload(style_focus="Blend"))


def test_parse_batch_results_splits_successes_and_errors() -> None:
    """Parse successful batch lines and preserve invalid responses as errors."""
    output_text = "\n".join([_batch_line("ok-1", _payload()), _batch_line("bad-1", _payload("Blend"))])
    error_text = json.dumps({"custom_id": "http-1", "error": {"message": "Request failed"}})

    results, errors = parse_batch_results(output_text=output_text, error_text=error_text)

    assert "ok-1" in results
    assert results["ok-1"].style_focus == "Growth"
    assert "bad-1" in errors
    assert "http-1" in errors

