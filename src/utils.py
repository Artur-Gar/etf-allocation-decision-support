from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR

EXCEL_MAX_ROWS = 1_048_575


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def first_existing_path(primary: Path, *fallbacks: Path) -> Path:
    """Return the first existing path, otherwise the primary target path."""
    for path in (primary, *fallbacks):
        if path.exists():
            return path
    return primary


def _sheet_chunks(table: pd.DataFrame) -> list[pd.DataFrame]:
    if len(table) <= EXCEL_MAX_ROWS:
        return [table]
    return [table.iloc[start:start + EXCEL_MAX_ROWS].copy() for start in range(0, len(table), EXCEL_MAX_ROWS)]


def save_excel(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            chunks = _sheet_chunks(table)
            if len(chunks) == 1:
                chunks[0].to_excel(writer, sheet_name=sheet_name[:31], index=False)
                continue

            for index, chunk in enumerate(chunks, start=1):
                chunk_name = f"{sheet_name}_{index}"
                chunk.to_excel(writer, sheet_name=chunk_name[:31], index=False)

