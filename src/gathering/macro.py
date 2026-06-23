from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from config import MACRO_MAX_WORKERS, PROCESSED_ETF_CLASSIFIED_PATH, RAW_ETF_UNIVERSE_PATH, RAW_MACRO_PATH
from downloaders.macro import build_macro_series_definitions, download_macro_indicators
from utils import ensure_directories, first_existing_path, save_table


DEFAULT_INPUT_PATH = first_existing_path(PROCESSED_ETF_CLASSIFIED_PATH, RAW_ETF_UNIVERSE_PATH)
DEFAULT_OUTPUT_PATH = RAW_MACRO_PATH


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for macro-only data gathering."""
    parser = argparse.ArgumentParser(description="Download monthly macro indicators from FRED.")
    parser.add_argument("--max-workers", type=int, default=MACRO_MAX_WORKERS)
    return parser.parse_args()


def run_macro_gathering(
    max_workers: int = MACRO_MAX_WORKERS,
) -> dict[str, Any]:
    """Download macro indicators from FRED for the classified ETF country universe."""
    ensure_directories()

    resolved_input = first_existing_path(
        DEFAULT_INPUT_PATH.resolve(),
        RAW_ETF_UNIVERSE_PATH.resolve(),
    )
    resolved_output = DEFAULT_OUTPUT_PATH.resolve()

    if not resolved_input.exists():
        raise FileNotFoundError(f"ETF characteristics file not found: {resolved_input}")

    characteristics = pd.read_csv(resolved_input)
    macro = download_macro_indicators(characteristics, max_workers=max_workers)

    save_table(macro, resolved_output)

    series_definitions = build_macro_series_definitions(characteristics)
    return {
        "input_path": str(resolved_input),
        "output_path": str(resolved_output),
        "max_workers": max_workers,
        "country_series_requested": len(series_definitions),
        "country_labels_requested": len({item["country_or_region"] for item in series_definitions}),
        "downloaded_rows": len(macro),
        "downloaded_series": int(macro["series_id"].nunique()) if not macro.empty else 0,
        "downloaded_country_labels": int(macro["country_or_region"].nunique()) if not macro.empty else 0,
    }


def main() -> None:
    """Run macro-only gathering from the command line."""
    args = parse_args()
    summary = run_macro_gathering(max_workers=args.max_workers)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

