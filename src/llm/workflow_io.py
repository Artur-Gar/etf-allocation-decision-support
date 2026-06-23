from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from config import PROCESSED_ETF_CLASSIFIED_PATH, RAW_ETF_UNIVERSE_PATH
from llm.schema import ETFClassification
from llm.workflow_config import (
    DEFAULT_JOB_ROOT,
    MANIFEST_NAME,
    REQUIRED_COLUMNS,
    TRIGGER_FIELDS,
    UPDATED_FIELDS,
)


def load_input_frame(path: Path) -> pd.DataFrame:
    """Read and validate the ETF input file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {', '.join(missing_columns)}")
    return df


def load_template(path: Path) -> Template:
    """Load the Jinja template used for ETF classification."""
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")

    environment = Environment(
        loader=FileSystemLoader(str(path.parent)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return environment.get_template(path.name)


def render_prompt(template: Template, row: pd.Series | dict[str, Any]) -> str:
    """Render a prompt for one ETF row."""
    return template.render(
        ticker=clean_cell(row["ticker"]),
        etf_name=clean_cell(row["etf_name"]),
        provider=clean_cell(row["provider"]),
        description=clean_cell(row["description"]),
    )


def build_target_mask(df: pd.DataFrame) -> pd.Series:
    """Find rows that still need LLM review."""
    mask = pd.Series(False, index=df.index)
    for column in TRIGGER_FIELDS:
        mask = mask | df[column].fillna("").astype(str).str.strip().eq("Other")
    return mask


def apply_classification(df: pd.DataFrame, row_index: int, result: ETFClassification) -> None:
    """Update the model-returned classification fields in place."""
    for field in UPDATED_FIELDS:
        df.at[row_index, field] = getattr(result, field)


def validate_context_row(df: pd.DataFrame, row_index: int, context_row: dict[str, Any]) -> None:
    """Sanity-check that the row index still points to the same ETF ticker."""
    if row_index not in df.index:
        raise IndexError(f"Row index {row_index} from the batch context is missing in the input file.")

    current_ticker = clean_cell(df.at[row_index, "ticker"])
    expected_ticker = clean_cell(context_row["ticker"])
    if current_ticker != expected_ticker:
        raise ValueError(f"Ticker mismatch at row {row_index}: expected {expected_ticker}, found {current_ticker}.")


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    """Resolve the classified output CSV path."""
    if output_path is not None:
        return output_path.resolve()
    if input_path.resolve() == RAW_ETF_UNIVERSE_PATH.resolve():
        return PROCESSED_ETF_CLASSIFIED_PATH.resolve()
    return input_path.with_name(f"{input_path.stem}_classified.csv").resolve()


def resolve_job_dir(*, input_path: Path, job_dir: Path | None) -> Path:
    """Resolve the folder used for batch artifacts."""
    if job_dir is not None:
        return job_dir.resolve()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_JOB_ROOT / f"{input_path.stem}_{timestamp}").resolve()


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Save the updated ETF classification dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_manifest(job_dir: Path, manifest: dict[str, Any]) -> None:
    """Save batch metadata to disk."""
    (job_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def load_manifest(job_dir: Path) -> dict[str, Any]:
    """Load batch metadata from disk."""
    manifest_path = job_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Batch manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_custom_id(*, row_index: int, ticker: Any) -> str:
    """Create a stable custom ID for batch requests."""
    return f"row::{row_index}::{clean_cell(ticker)}"


def clean_cell(value: Any) -> str:
    """Convert NaN-like values to an empty string and normalize text cells."""
    if pd.isna(value):
        return ""
    return str(value).strip()

