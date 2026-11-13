import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, TESTS, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import types

import pytest

from pipeline import make_dataset
from synthetic import fake_ae_fit

# E2SC imports nmslib at module level; the tests never run E2SC, so a stub lets the
# runner be imported on machines without nmslib (it is installed in the Docker image).
try:
    import nmslib  # noqa: F401
except ImportError:
    sys.modules["nmslib"] = types.ModuleType("nmslib")


@pytest.fixture
def fake_ae(monkeypatch):
    """Replace autoencoder training with a deterministic score (no torch training in tests)."""
    from src.iSel import autoencoder_is
    monkeypatch.setattr(autoencoder_is.AutoencoderIS, "fit", fake_ae_fit)


@pytest.fixture
def dataset(tmp_path):
    """A toy dataset under tmp_path in the layout run_generateSplit.py reads."""
    make_dataset(tmp_path)
    return tmp_path
