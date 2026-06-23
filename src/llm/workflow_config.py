from __future__ import annotations

from pathlib import Path

from config import OUTPUT_DIR, RAW_ETF_UNIVERSE_PATH


DEFAULT_INPUT_PATH = RAW_ETF_UNIVERSE_PATH
DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().with_name("classify_etf.j2")
DEFAULT_JOB_ROOT = OUTPUT_DIR / "llm_jobs"
MANIFEST_NAME = "job_manifest.json"

PROMPT_FIELDS = ["ticker", "etf_name", "provider", "description"]
TRIGGER_FIELDS = ["primary_region", "style_focus", "size_focus", "sector_focus"]
UPDATED_FIELDS = ["primary_region", "developed_or_emerging", "style_focus", "sector_focus"]
REQUIRED_COLUMNS = sorted(set(PROMPT_FIELDS + TRIGGER_FIELDS + ["developed_or_emerging"]))
CONTEXT_COLUMNS = [
    "custom_id",
    "row_index",
    "ticker",
    "etf_name",
    "provider",
    "primary_region",
    "developed_or_emerging",
    "style_focus",
    "size_focus",
    "sector_focus",
]

