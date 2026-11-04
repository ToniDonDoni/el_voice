"""Resolve which commit the running build came from.

The repository root is not mounted into the voice container, so the app cannot
always read Git metadata directly. Resolution order:
1. EL_VOICE_COMMIT environment variable;
2. /data/COMMIT (written by tools/write_commit.sh before Compose start);
3. workspace/COMMIT;
4. `git rev-parse` when the repository is available (local development runs).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent
_CACHE: str | None = None


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def current_commit() -> str:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    value = os.environ.get("EL_VOICE_COMMIT", "").strip()
    if not value:
        for candidate in (Path("/data/COMMIT"), SOURCE_DIR / "COMMIT"):
            value = _read_file(candidate)
            if value:
                break
    if not value:
        try:
            result = subprocess.run(
                ["git", "-C", str(SOURCE_DIR.parent), "rev-parse", "--short=8", "HEAD"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                value = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            value = ""
    _CACHE = value or "unknown"
    return _CACHE
