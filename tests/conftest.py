import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluator import evaluate  # noqa: E402

SEED_PATH = os.path.join(ROOT, "seeds", "binary_euclid.py")


@pytest.fixture(scope="session")
def seed_src():
    with open(SEED_PATH) as f:
        return f.read()


@pytest.fixture(scope="session")
def seed_result(seed_src):
    return evaluate(seed_src)
