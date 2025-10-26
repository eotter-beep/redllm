"""Compatibility wrapper for the legacy start_llm entry point."""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Allow running the CLI directly from a source checkout."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(1, str(repo_root))


_ensure_src_on_path()

from redllm.cli import main


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
