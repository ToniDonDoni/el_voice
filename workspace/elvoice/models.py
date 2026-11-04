"""Verify downloaded publisher assets against their local download receipts.

Model files live on a read-only bind mount. The download service records the
publisher URL, size and SHA-256 of every file in ``.download-receipts.json``;
this module refuses to hand a path to an inference engine unless the receipt
matches the configured publisher asset and the bytes still match the receipt.
No asset is ever unpickled here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

VAD_MODEL = "vad/silero_vad.onnx"
TTS_MODEL = "tts/v4_cyrillic.pt"
LLM_MODEL = "llm/gemma-3-1b-it-Q4_K_M.gguf"
STT_FILES = (
    "stt/small/config.json",
    "stt/small/tokenizer.json",
    "stt/small/vocabulary.txt",
    "stt/small/model.bin",
)

# Publisher URLs pinned in models.json. The same values must appear in the local
# download receipt, otherwise a replacement file from another source is refused.
ASSET_URLS = {
    VAD_MODEL: "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx",
    TTS_MODEL: "https://models.silero.ai/models/tts/cyr/v4_cyrillic.pt",
    LLM_MODEL: (
        "https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/"
        "f9c28bcd85737ffc5aef028638d3341d49869c27/gemma-3-1b-it-Q4_K_M.gguf"
    ),
    STT_FILES[0]: (
        "https://huggingface.co/Systran/faster-whisper-small/resolve/"
        "536b0662742c02347bc0e980a01041f333bce120/config.json"
    ),
    STT_FILES[1]: (
        "https://huggingface.co/Systran/faster-whisper-small/resolve/"
        "536b0662742c02347bc0e980a01041f333bce120/tokenizer.json"
    ),
    STT_FILES[2]: (
        "https://huggingface.co/Systran/faster-whisper-small/resolve/"
        "536b0662742c02347bc0e980a01041f333bce120/vocabulary.txt"
    ),
    STT_FILES[3]: (
        "https://huggingface.co/Systran/faster-whisper-small/resolve/"
        "536b0662742c02347bc0e980a01041f333bce120/model.bin"
    ),
}

RECEIPT_FILE = ".download-receipts.json"


class AssetError(ValueError):
    """A model file is missing, untrusted or corrupt."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_receipts(root: Path) -> dict:
    receipt_path = root / RECEIPT_FILE
    if not receipt_path.is_file():
        raise AssetError(
            f"Download receipt missing at {RECEIPT_FILE}; run the download service first"
        )
    try:
        receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetError(f"Cannot read download receipt: {exc}") from exc
    if not isinstance(receipts, dict):
        raise AssetError("Download receipt is not a JSON object")
    return receipts


def verify_asset(root: Path, relative_path: str) -> Path:
    """Return a verified path inside ``root`` or raise AssetError."""
    root = Path(root).resolve()
    expected_url = ASSET_URLS.get(relative_path)
    if expected_url is None:
        raise AssetError(f"Asset is not pinned for the voice pipeline: {relative_path}")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AssetError(f"Asset path escapes the model directory: {relative_path}") from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise AssetError(f"Missing model file: {relative_path}")
    receipt = load_receipts(root).get(relative_path)
    if not isinstance(receipt, dict):
        raise AssetError(f"No download receipt for {relative_path}")
    if receipt.get("url") != expected_url:
        raise AssetError(f"Receipt URL does not match the pinned publisher for {relative_path}")
    if receipt.get("size") != path.stat().st_size:
        raise AssetError(f"Size differs from the receipt for {relative_path}")
    if sha256_file(path) != receipt.get("sha256"):
        raise AssetError(f"SHA-256 differs from the receipt for {relative_path}")
    return path


def verify_whisper_files(root: Path) -> list[Path]:
    """Verify every file of the CTranslate2 Whisper model directory."""
    return [verify_asset(root, relative) for relative in STT_FILES]
