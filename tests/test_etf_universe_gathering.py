from __future__ import annotations

from pathlib import Path

import pandas as pd

import gathering.etf_universe as etf_universe_gathering


def test_run_etf_universe_gathering_saves_one_csv(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """Download the iShares ETF universe and save it to one target file."""
    output_path = workspace_tmp_path / "etf_universe_ishares.csv"
    saved: dict[str, object] = {}

    monkeypatch.setattr(etf_universe_gathering, "RAW_ETF_UNIVERSE_PATH", output_path)
    monkeypatch.setattr(etf_universe_gathering, "ensure_directories", lambda: None)
    monkeypatch.setattr(
        etf_universe_gathering,
        "download_etf_characteristics",
        lambda: pd.DataFrame({"ticker": ["IVV", "AAXJ"], "etf_name": ["iShares Core S&P 500 ETF", "iShares MSCI AC Asia ex Japan ETF"]}),
    )
    monkeypatch.setattr(
        etf_universe_gathering,
        "save_table",
        lambda df, path: saved.update({"path": path, "rows": len(df)}),
    )

    summary = etf_universe_gathering.run_etf_universe_gathering()

    assert saved["path"] == output_path.resolve()
    assert saved["rows"] == 2
    assert summary["rows"] == 2
    assert summary["tickers"] == 2

