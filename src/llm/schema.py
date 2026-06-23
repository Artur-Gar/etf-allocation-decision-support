from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _values(text: str) -> list[str]:
    """Split compact pipe-delimited enum definitions."""
    return [value.strip() for value in text.split("|")]


PRIMARY_REGION_VALUES = _values(
    "US|Europe|Japan|Emerging Markets|China|India|Latin America|Developed ex-US|Global|"
    "Global ex-US|Asia Pacific ex-Japan|Australia|Austria|Belgium|Brazil|Canada|Chile|"
    "Denmark|Finland|France|Germany|Hong Kong|Indonesia|Israel|Italy|Kuwait|Malaysia|"
    "Mexico|Netherlands|New Zealand|Norway|Philippines|Poland|Qatar|Saudi Arabia|"
    "Singapore|South Africa|South Korea|Spain|Sweden|Switzerland|Taiwan|Thailand|"
    "Turkey|United Kingdom|Other|No information"
)
DEVELOPED_OR_EMERGING_VALUES = _values("Developed|Emerging|Mixed|Frontier|No information")
STYLE_FOCUS_VALUES = _values(
    "Core|Value|Growth|Momentum|Quality|Dividend|Low Volatility|Multi-Factor|ESG|Thematic|Other|No information"
)
SECTOR_FOCUS_VALUES = _values(
    "Broad Market|Technology|Healthcare|Financials|Energy|Industrials|Consumer|Real Estate|Utilities|Thematic|Other|No information"
)
MODEL_OUTPUT_FIELDS = ["primary_region", "developed_or_emerging", "style_focus", "sector_focus", "confidence", "reasoning_short"]

CLASSIFICATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "etf_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "primary_region": {"type": "string", "enum": PRIMARY_REGION_VALUES},
                "developed_or_emerging": {"type": "string", "enum": DEVELOPED_OR_EMERGING_VALUES},
                "style_focus": {"type": "string", "enum": STYLE_FOCUS_VALUES},
                "sector_focus": {"type": "string", "enum": SECTOR_FOCUS_VALUES},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning_short": {"type": "string", "minLength": 1, "maxLength": 240},
            },
            "required": MODEL_OUTPUT_FIELDS,
        },
    },
}


class OutputValidationError(ValueError):
    """Raised when the model output does not match the expected schema."""


@dataclass(slots=True)
class ETFClassification:
    """Validated ETF classification produced by the model."""

    primary_region: str
    developed_or_emerging: str
    style_focus: str
    sector_focus: str
    confidence: float
    reasoning_short: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a plain dictionary."""
        return asdict(self)


def validate_classification_output(payload: Mapping[str, Any]) -> ETFClassification:
    """Validate and normalize the model output."""
    if set(payload.keys()) != set(MODEL_OUTPUT_FIELDS):
        raise OutputValidationError(f"Expected keys {sorted(MODEL_OUTPUT_FIELDS)}, got {sorted(payload.keys())}.")

    values = {field: str(payload[field]).strip() for field in MODEL_OUTPUT_FIELDS if field != "confidence"}
    _validate_enum("primary_region", values["primary_region"], PRIMARY_REGION_VALUES)
    _validate_enum("developed_or_emerging", values["developed_or_emerging"], DEVELOPED_OR_EMERGING_VALUES)
    _validate_enum("style_focus", values["style_focus"], STYLE_FOCUS_VALUES)
    _validate_enum("sector_focus", values["sector_focus"], SECTOR_FOCUS_VALUES)
    if not values["reasoning_short"]:
        raise OutputValidationError("reasoning_short must not be empty.")

    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError) as exc:
        raise OutputValidationError("confidence must be numeric.") from exc
    if not 0.0 <= confidence <= 1.0:
        raise OutputValidationError("confidence must be between 0 and 1.")

    return ETFClassification(confidence=confidence, **values)


def parse_model_output(content: str) -> ETFClassification:
    """Parse and validate a JSON string returned by the model."""
    try:
        return validate_classification_output(json.loads(content))
    except json.JSONDecodeError as exc:
        raise OutputValidationError("Model output was not valid JSON.") from exc


def coerce_content(content: Any) -> str:
    """Turn the SDK content payload into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [str(item.get("text")) for item in content if isinstance(item, dict) and item.get("text")]
        if text_parts:
            return "".join(text_parts)
    raise OutputValidationError("Could not read the model response content.")


def _validate_enum(field: str, value: str, allowed: list[str]) -> None:
    """Validate one string enum field."""
    if value not in allowed:
        raise OutputValidationError(f"Invalid {field}: {value}")
