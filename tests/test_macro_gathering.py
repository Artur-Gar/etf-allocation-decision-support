from __future__ import annotations

from pathlib import Path

import pandas as pd

import gathering.macro as macro_gathering


def test_run_macro_gathering_uses_classified_input_and_saves_outputs(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """Run macro-only gathering against a tiny classified ETF input."""
    input_path = workspace_tmp_path / "etf_universe_classified.csv"
    output_path = workspace_tmp_path / "macro_indicators_fred.csv"

    pd.DataFrame({"primary_region": ["US", "Canada", "Global"]}).to_csv(input_path, index=False)

    saved = {}

    monkeypatch.setattr(macro_gathering, "DEFAULT_INPUT_PATH", input_path)
    monkeypatch.setattr(macro_gathering, "DEFAULT_OUTPUT_PATH", output_path)
    monkeypatch.setattr(macro_gathering, "RAW_ETF_UNIVERSE_PATH", input_path)
    monkeypatch.setattr(macro_gathering, "ensure_directories", lambda: None)
    monkeypatch.setattr(
        macro_gathering,
        "download_macro_indicators",
        lambda characteristics, max_workers: pd.DataFrame(
            [
                {
                    "date": "2024-01-31",
                    "country_or_region": "US",
                    "indicator_name": "Interest Rate",
                    "indicator_value": 5.0,
                    "unit": "Percent",
                    "series_id": "FEDFUNDS",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        macro_gathering,
        "save_table",
        lambda df, path: saved.update({"csv_path": path, "csv_rows": len(df)}),
    )

    summary = macro_gathering.run_macro_gathering(max_workers=4)

    assert saved["csv_path"] == output_path.resolve()
    assert saved["csv_rows"] == 1
    assert summary["downloaded_rows"] == 1
    assert summary["country_labels_requested"] == 2
    assert summary["max_workers"] == 4

