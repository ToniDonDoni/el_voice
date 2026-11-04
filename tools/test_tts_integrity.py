"""Offline checks for the guard that runs before executing a model package."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tts_smoke", ROOT / "workspace/tts_smoke.py")
TTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TTS)


class ModelReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / TTS.MODEL_NAME
        self.model.parent.mkdir()
        self.model.write_bytes(b"test fixture; never executable model code")
        self.receipts = self.root / ".download-receipts.json"
        self.receipts.write_text(json.dumps({TTS.MODEL_NAME: {
            "url": TTS.MODEL_URL,
            "sha256": hashlib.sha256(self.model.read_bytes()).hexdigest(),
        }}))

    def test_matching_receipt(self):
        self.assertEqual(TTS.checked_model_path(self.root, TTS.MODEL_NAME), self.model.resolve())

    def test_modified_model_is_rejected(self):
        self.model.write_bytes(b"modified file")
        with self.assertRaisesRegex(ValueError, "checksum"):
            TTS.checked_model_path(self.root, TTS.MODEL_NAME)

    def test_missing_receipt_is_rejected(self):
        self.receipts.unlink()
        with self.assertRaisesRegex(ValueError, "receipt missing"):
            TTS.checked_model_path(self.root, TTS.MODEL_NAME)

    def test_different_source_is_rejected(self):
        data = json.loads(self.receipts.read_text())
        data[TTS.MODEL_NAME]["url"] = "https://example.invalid/replacement.pt"
        self.receipts.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "publisher URL"):
            TTS.checked_model_path(self.root, TTS.MODEL_NAME)

    def test_other_path_is_rejected(self):
        with self.assertRaises(ValueError):
            TTS.checked_model_path(self.root, "../replacement.pt")


if __name__ == "__main__":
    unittest.main()
