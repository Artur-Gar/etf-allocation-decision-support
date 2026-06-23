from __future__ import annotations

from pathlib import Path

import pandas as pd

import gathering.prices as price_gathering


def test_run_price_gathering_reads_tickers_and_saves_prices(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """Download Yahoo prices for the ETF tickers listed in the input universe file."""
    input_path = workspace_tmp_path / "etf_universe_classified.csv"
    output_path = workspace_tmp_path / "etf_prices_yahoo.csv"
    input_frame = pd.DataFrame({"ticker": ["IVV", "AAXJ", None]})
    input_frame.to_csv(input_path, index=False)

    saved: dict[str, object] = {}

    monkeypatch.setattr(price_gathering, "PROCESSED_ETF_CLASSIFIED_PATH", input_path)
    monkeypatch.setattr(price_gathering, "RAW_ETF_UNIVERSE_PATH", input_path)
    monkeypatch.setattr(price_gathering, "RAW_ETF_PRICES_PATH", output_path)
    monkeypatch.setattr(price_gathering, "ensure_directories", lambda: None)
    monkeypatch.setattr(
        price_gathering,
        "download_etf_prices",
        lambda tickers, start_date, end_date: pd.DataFrame(
            {
                "ticker": tickers,
                "date": ["2024-01-31"] * len(tickers),
                "adjusted_close": [100.0] * len(tickers),
            }
        ),
    )
    monkeypatch.setattr(
        price_gathering,
        "save_table",
        lambda df, path: saved.update({"path": path, "rows": len(df), "tickers": sorted(df["ticker"].tolist())}),
    )

    summary = price_gathering.run_price_gathering(start_date="2024-01-01", end_date="2024-12-31")

    assert saved["path"] == output_path.resolve()
    assert saved["rows"] == 2
    assert saved["tickers"] == ["AAXJ", "IVV"]
    assert summary["tickers_requested"] == 2
    assert summary["tickers_downloaded"] == 2

