"""Guards the backend/frontend dependency split.

The split is what keeps each image from installing the union of both sides (see CLAUDE.md,
"Dependency layout"). It is a convention `uv add` does not enforce: a bare `uv add <pkg>` lands in
the shared list and silently re-fattens both images by whatever that package drags in. These tests
turn that convention into a failure.
"""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# Only packages imported by BOTH app/ and frontend/ belong here. Adding to this set is a real
# decision: it enlarges every image.
EXPECTED_SHARED = {"pydantic-settings", "python-dotenv"}

# Sentinels: heavy trees that must stay on one side. streamlit drags in pyarrow/pandas/numpy
# (~250MB); the agent stack is comparably heavy in the other direction.
BACKEND_ONLY = {"langchain", "langgraph", "sqlalchemy", "alembic", "fastapi", "psycopg"}
FRONTEND_ONLY = {"streamlit", "requests"}


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _names(requirements: list[str]) -> set[str]:
    """Bare package names, dropping extras and version specifiers."""
    names = set()
    for requirement in requirements:
        name = requirement.split(";")[0].strip()
        for separator in ("[", ">", "<", "=", "!", "~"):
            name = name.split(separator)[0]
        names.add(name.strip().lower())
    return names


def test_shared_dependencies_stay_minimal():
    """A bare `uv add` lands here; use `uv add --group backend|frontend` instead."""
    assert _names(_pyproject()["project"]["dependencies"]) == EXPECTED_SHARED


def test_backend_packages_are_not_shared_or_in_frontend():
    groups = _pyproject()["dependency-groups"]
    backend = _names(groups["backend"])
    frontend = _names(groups["frontend"])
    assert BACKEND_ONLY <= backend
    assert not BACKEND_ONLY & frontend
    assert not BACKEND_ONLY & EXPECTED_SHARED


def test_frontend_packages_are_not_shared_or_in_backend():
    groups = _pyproject()["dependency-groups"]
    backend = _names(groups["backend"])
    frontend = _names(groups["frontend"])
    assert FRONTEND_ONLY <= frontend
    assert not FRONTEND_ONLY & backend
    assert not FRONTEND_ONLY & EXPECTED_SHARED


def test_local_and_ci_still_install_everything():
    """`default-groups` is why plain `uv sync`/`uv run` and CI are unaffected by the split."""
    default_groups = _pyproject()["tool"]["uv"]["default-groups"]
    assert set(default_groups) == {"dev", "backend", "frontend"}


def test_dockerfiles_install_only_their_own_group():
    root = PYPROJECT.parent
    for side in ("backend", "frontend"):
        dockerfile = (root / "docker" / f"{side}.Dockerfile").read_text(encoding="utf-8")
        assert f"--no-default-groups --group {side}" in dockerfile
        other = "frontend" if side == "backend" else "backend"
        assert f"--group {other}" not in dockerfile
