from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from openai import OpenAI

from config import BASE_DIR
from llm.schema import (
    CLASSIFICATION_RESPONSE_FORMAT,
    DEVELOPED_OR_EMERGING_VALUES,
    ETFClassification,
    OutputValidationError,
    PRIMARY_REGION_VALUES,
    SECTOR_FOCUS_VALUES,
    STYLE_FOCUS_VALUES,
    coerce_content,
    parse_model_output,
    validate_classification_output,
)


API_KEY_ENV_VAR = "OPENAI_API_KEY"
MODEL_ENV_VAR = "OPENAI_MODEL"
DEFAULT_ENV_PATH = BASE_DIR / ".env"
ENV_VAR_ALIASES = {"API_KEY_ENV_VAR": API_KEY_ENV_VAR, "MODEL_ENV_VAR": MODEL_ENV_VAR}


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load environment variables from a local .env file if present."""
    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and (override or key not in os.environ):
            os.environ[key] = value

    for alias, target in ENV_VAR_ALIASES.items():
        alias_value = os.getenv(alias, "").strip()
        if alias_value and (override or not os.getenv(target, "").strip()):
            os.environ[target] = alias_value
    return env_path


def has_live_openai_config() -> bool:
    """Check whether a usable OpenAI API key and model are configured."""
    try:
        get_api_key()
        get_model_name()
    except RuntimeError:
        return False
    return True


def get_api_key() -> str:
    """Read the OpenAI API key from the environment."""
    load_env_file()
    api_key = os.getenv(API_KEY_ENV_VAR, "").strip()
    if _is_missing_like(api_key):
        raise RuntimeError(f"Missing {API_KEY_ENV_VAR} environment variable.")
    return api_key


def get_model_name(model: str | None = None) -> str:
    """Resolve the model name from an override or the environment."""
    if model:
        return model
    load_env_file()
    env_model = os.getenv(MODEL_ENV_VAR, "").strip()
    if _is_missing_like(env_model):
        raise RuntimeError(f"Missing {MODEL_ENV_VAR} environment variable.")
    return env_model


def get_client(api_key: str | None = None) -> OpenAI:
    """Create an OpenAI client using the configured API key."""
    load_env_file()
    return OpenAI(api_key=api_key or get_api_key())


def build_messages(prompt: str) -> list[dict[str, str]]:
    """Build the chat messages for ETF classification."""
    return [
        {"role": "system", "content": "You classify ETFs from limited metadata. Return only JSON that matches the required schema."},
        {"role": "user", "content": prompt},
    ]


def build_single_classification_request(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Build a synchronous chat completion request body."""
    return {
        "model": get_model_name(model),
        "messages": build_messages(prompt),
        "temperature": 0,
        "response_format": CLASSIFICATION_RESPONSE_FORMAT,
    }


def build_batch_request(custom_id: str, prompt: str, model: str | None = None) -> dict[str, Any]:
    """Build a single JSONL line for the Batch API."""
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": build_single_classification_request(prompt, model)}


def build_batch_requests(prompts_by_id: Mapping[str, str], model: str | None = None) -> list[dict[str, Any]]:
    """Build Batch API requests for multiple prompts."""
    return [build_batch_request(custom_id, prompt, model=model) for custom_id, prompt in prompts_by_id.items()]


def save_batch_input(requests: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Save batch requests as a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(dict(request), ensure_ascii=False) + "\n")
    return output_path


def submit_batch_job(
    client: OpenAI,
    input_path: str | Path,
    *,
    completion_window: str = "24h",
    metadata: Mapping[str, str] | None = None,
) -> Any:
    """Upload the JSONL file and create a batch job."""
    with Path(input_path).open("rb") as handle:
        uploaded_file = client.files.create(file=handle, purpose="batch")
    return client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata=dict(metadata or {}),
    )


def check_batch_status(client: OpenAI, batch_id: str) -> Any:
    """Fetch the latest status for a batch job."""
    return client.batches.retrieve(batch_id)


def classify_prompt(client: OpenAI, prompt: str, model: str | None = None) -> ETFClassification:
    """Run a synchronous classification request."""
    response = client.chat.completions.create(**build_single_classification_request(prompt, model))
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused the request: {message.refusal}")
    return parse_model_output(coerce_content(message.content))


def retrieve_batch_results(client: OpenAI, batch_id: str) -> tuple[Any, dict[str, ETFClassification], dict[str, str]]:
    """Download, parse, and validate completed batch results."""
    batch = check_batch_status(client, batch_id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch_id} is not ready yet. Current status: {batch.status}")
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch_id} completed without an output file.")

    output_text = client.files.content(batch.output_file_id).text
    error_text = client.files.content(batch.error_file_id).text if getattr(batch, "error_file_id", None) else None
    results, errors = parse_batch_results(output_text=output_text, error_text=error_text)
    return batch, results, errors


def parse_batch_results(
    *,
    output_text: str,
    error_text: str | None = None,
) -> tuple[dict[str, ETFClassification], dict[str, str]]:
    """Parse success and error lines returned by the Batch API."""
    results: dict[str, ETFClassification] = {}
    errors: dict[str, str] = {}

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        custom_id = str(item.get("custom_id", ""))
        response = item.get("response") or {}
        if response.get("status_code") != 200:
            errors[custom_id] = json.dumps(response, ensure_ascii=False)
            continue

        message = (((response.get("body") or {}).get("choices") or [{}])[0]).get("message") or {}
        if message.get("refusal"):
            errors[custom_id] = str(message["refusal"])
            continue
        try:
            results[custom_id] = parse_model_output(coerce_content(message.get("content")))
        except Exception as exc:
            errors[custom_id] = str(exc)

    if error_text:
        _merge_batch_error_lines(error_text, errors)
    return results, errors


def _merge_batch_error_lines(error_text: str, errors: dict[str, str]) -> None:
    """Merge failed Batch API JSONL rows into the error dictionary."""
    for raw_line in error_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        custom_id = str(item.get("custom_id", item.get("id", "")))
        errors[custom_id] = json.dumps(item.get("error") or item, ensure_ascii=False)


def _is_missing_like(value: object) -> bool:
    """Treat blank and placeholder-like values as missing configuration."""
    text = str(value or "").strip()
    if not text:
        return True
    return text.lower() in {"insert your key here", "your_api_key", "your key here", "changeme", "set-me"}

