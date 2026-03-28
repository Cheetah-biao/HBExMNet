from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    # The Python release is packaged as a self-contained project rooted at
    # the HBExMNet folder. Experiments, data, and generated files should all
    # resolve relative to this directory.
    return repo_root()


def find_external_data_dir(name: str) -> Path:
    repo = repo_root()
    candidates = [
        repo.parent / name,
        repo.parent.parent / name if repo.parent.parent != repo.parent else repo / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def repo_path(*parts: str) -> Path:
    return repo_root().joinpath(*parts)


def workspace_path(*parts: str) -> Path:
    return workspace_root().joinpath(*parts)
