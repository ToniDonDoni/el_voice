#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
uv pip compile requirements.in \
  --python-version 3.12 \
  --python-platform aarch64-manylinux_2_28 \
  --generate-hashes \
  --no-header \
  --no-annotate \
  --output-file requirements.lock

# uv resolves the wheel's package metadata via PyPI and otherwise writes the PyPI sdist
# hash for this direct GitHub wheel. Keep the approved artifact hash from requirements.in.
python3 - <<'PY'
import re
from pathlib import Path

lock = Path("requirements.lock")
lines = lock.read_text().splitlines()
for index, line in enumerate(lines):
    if line.startswith("llama-cpp-python @ "):
        match = re.search(r"#sha256=([0-9a-f]{64})", line)
        if not match or index + 1 >= len(lines) or "--hash=" not in lines[index + 1]:
            raise SystemExit("Could not find the pinned llama-cpp-python wheel hash")
        lines[index + 1] = f"    --hash=sha256:{match.group(1)}"
        break
else:
    raise SystemExit("Could not find llama-cpp-python in requirements.lock")
lock.write_text("\n".join(lines) + "\n")
PY
