from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from config import (
    TABLEAU_RELATIONAL_OUTPUT_PATH,
)
from tableau_model.build import (
    build_relational_tables_with_summary,
)
from utils import ensure_directories, save_excel


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Tableau relational export."""
    return argparse.ArgumentParser(
        description="Build a normalized Tableau relational workbook from the ETF project outputs."
    ).parse_args()


def run_tableau_relational_export() -> dict[str, Any]:
    """Build the normalized Tableau workbook and save each table on its own sheet."""
    ensure_directories()

    from config import (
        PROCESSED_ETF_CLASSIFIED_PATH,
        PROCESSED_ETF_MONTHLY_RETURNS_PATH,
        PROCESSED_MACRO_MONTHLY_PATH,
    )

    resolved_characteristics = PROCESSED_ETF_CLASSIFIED_PATH.resolve()
    resolved_market = PROCESSED_ETF_MONTHLY_RETURNS_PATH.resolve()
    resolved_macro = PROCESSED_MACRO_MONTHLY_PATH.resolve()
    resolved_output = TABLEAU_RELATIONAL_OUTPUT_PATH.resolve()

    characteristics = pd.read_csv(resolved_characteristics)
    market = pd.read_csv(resolved_market)
    macro = pd.read_csv(resolved_macro)
    tables, summary = build_relational_tables_with_summary(
        characteristics=characteristics,
        etf_market_source=market,
        macro_monthly_source=macro,
    )
    save_excel(tables, resolved_output)

    result = {
        "characteristics_path": str(resolved_characteristics),
        "market_path": str(resolved_market),
        "macro_path": str(resolved_macro),
        "output_path": str(resolved_output),
        "table_count": len(tables),
        "tables": {name: len(frame) for name, frame in tables.items()},
    }
    result.update(summary)
    return result


def main() -> None:
    """Run the Tableau relational export from the command line."""
    parse_args()
    summary = run_tableau_relational_export()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

