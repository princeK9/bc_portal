"""
Shared pytest fixtures and import path setup.

The repo is a flat set of modules rather than an installed package, so the repo
root and scripts/ both need to be importable before any test module is
collected.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def sample_resumes_dir() -> Path:
    """Directory holding the committed synthetic resume corpus."""
    directory = REPO_ROOT / "sample_data" / "resumes"
    if not directory.is_dir():
        pytest.skip("sample_data/resumes is not present")
    return directory


@pytest.fixture(scope="session")
def sample_docx(sample_resumes_dir: Path) -> Path:
    """A .docx resume from the corpus, for text-extraction tests."""
    candidates = sorted(sample_resumes_dir.glob("*.docx"))
    if not candidates:
        pytest.skip("no .docx fixture available")
    return candidates[0]


@pytest.fixture(scope="session")
def sample_png(sample_resumes_dir: Path) -> Path:
    """A .png resume from the corpus, for multimodal-routing tests."""
    candidates = sorted(sample_resumes_dir.glob("*.png"))
    if not candidates:
        pytest.skip("no .png fixture available")
    return candidates[0]
