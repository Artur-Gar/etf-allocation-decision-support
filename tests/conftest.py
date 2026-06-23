from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
LOCAL_TMP_DIR = ROOT_DIR / ".tmp"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def workspace_tmp_path() -> Path:
    """Provide a repo-local temp directory without relying on pytest tmp_path."""
    LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="test_", dir=LOCAL_TMP_DIR))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

