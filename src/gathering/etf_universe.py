from __future__ import annotations

import json
from typing import Any

from config import RAW_ETF_UNIVERSE_PATH
from downloaders.etf_characteristics import download_etf_characteristics
from utils import ensure_directories, save_table


def run_etf_universe_gathering() -> dict[str, Any]:
    """Download ETF metadata from iShares and save one raw CSV."""
    ensure_directories()
    resolved_output = RAW_ETF_UNIVERSE_PATH.resolve()

    etf_universe = download_etf_characteristics()
    save_table(etf_universe, resolved_output)

    return {
        "output_path": str(resolved_output),
        "rows": len(etf_universe),
        "tickers": int(etf_universe["ticker"].nunique()) if "ticker" in etf_universe else 0,
    }


def main() -> None:
    """Run ETF universe gathering from the command line."""
    summary = run_etf_universe_gathering()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

