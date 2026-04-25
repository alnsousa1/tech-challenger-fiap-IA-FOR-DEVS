from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

os.environ.setdefault("MPLBACKEND", "Agg")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def trained_metadata():
    from womens_health_ai.modeling.training import ensure_training_artifacts

    return ensure_training_artifacts(force=False)
