"""Deterministic persistence tests for local biometric profiles."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from speaker_profiles import ProfileStore


class SpeakerProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "speaker-profiles.json"
        self.store = ProfileStore(self.path)
        self.embedding = np.zeros(192, dtype=np.float32)
        self.embedding[0] = 1

    def tearDown(self):
        self.directory.cleanup()

    def test_atomic_save_round_trips_only_metadata_and_embedding(self):
        profile_id = self.store.upsert("owner", self.embedding)
        self.assertEqual(profile_id, "owner")
        self.assertEqual(self.store.load()["owner"]["embedding"][0], 1.0)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)
        self.assertFalse(list(self.path.parent.glob(".speaker-profiles-*")))

    def test_invalid_replacement_does_not_overwrite_existing_owner(self):
        self.store.upsert("owner", self.embedding)
        with self.assertRaises(ValueError):
            self.store.upsert("owner", np.ones(192, dtype=np.float32))
        self.assertEqual(self.store.load()["owner"]["embedding"][0], 1.0)

    def test_profile_permissions_are_checked_before_read(self):
        self.store.upsert("owner", self.embedding)
        os.chmod(self.path, 0o644)
        with self.assertRaisesRegex(ValueError, "permissions"):
            self.store.load()

    def test_profile_embedding_shape_finiteness_and_normalization_are_enforced(self):
        for invalid in (np.zeros(191), np.full(192, np.nan), np.ones(192)):
            with self.subTest(shape=np.asarray(invalid).shape):
                with self.assertRaises(ValueError):
                    self.store.upsert("owner", invalid)

    def test_guest_limit_is_bounded_and_disable_delete_are_persisted(self):
        for index in range(4):
            self.store.upsert("guest", self.embedding, f"guest-{index}")
        with self.assertRaisesRegex(ValueError, "limit"):
            self.store.upsert("guest", self.embedding)
        profiles = self.store.load()
        profiles["guest-0"]["enabled"] = False
        self.store.save(profiles)
        self.assertFalse(self.store.load()["guest-0"]["enabled"])


if __name__ == "__main__":
    unittest.main()
