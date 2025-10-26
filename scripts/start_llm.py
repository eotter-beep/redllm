"""Compatibility wrapper for the legacy start_llm entry point."""
from __future__ import annotations

from redllm.cli import main


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
