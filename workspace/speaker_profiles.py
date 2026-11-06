"""Private local persistence for target-speaker embeddings."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np


class ProfileStore:
    """Persist only validated 192-dimensional embeddings under the ignored data tree."""

    def __init__(self, path: Path | None = None):
        self.path = path or Path(os.environ.get("EL_VOICE_DATA", "/data")) / "speaker-profiles.json"
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)

    @staticmethod
    def validate_embedding(value: Any) -> list[float]:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape not in {(192,), (1, 192)} or not np.isfinite(vector).all():
            raise ValueError("Speaker embedding must contain 192 finite values")
        vector = vector.reshape(192)
        norm = float(np.linalg.norm(vector))
        if not 0.9 <= norm <= 1.1:
            raise ValueError("Speaker embedding must be L2 normalized")
        return [float(item) for item in vector]

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        if self.path.is_symlink() or self.path.stat().st_mode & 0o077 or self.path.stat().st_size > 65536:
            raise ValueError("Speaker profile permissions are unsafe")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("profiles"), dict):
            raise ValueError("Speaker profile data is malformed")
        profiles: dict[str, dict[str, Any]] = {}
        owners = 0
        for profile_id, profile in raw["profiles"].items():
            if (not isinstance(profile_id, str) or not isinstance(profile, dict)
                    or type(profile.get("enabled")) is not bool
                    or profile.get("kind") not in {"owner", "guest"}):
                raise ValueError("Speaker profile record is malformed")
            profiles[profile_id] = {
                "kind": profile["kind"],
                "enabled": profile["enabled"],
                "embedding": self.validate_embedding(profile.get("embedding")),
            }
            owners += profile["kind"] == "owner"
            if ((profile["kind"] == "owner" and profile_id != "owner")
                    or (profile["kind"] == "guest" and not profile_id.startswith("guest-"))):
                raise ValueError("Speaker profile identifier does not match its type")
        if len(profiles) > 5 or owners > 1 or len(profiles) - owners > 4:
            raise ValueError("Speaker profile count exceeds its configured limit")
        return profiles

    def save(self, profiles: dict[str, dict[str, Any]]) -> None:
        if len(profiles) > 5:
            raise ValueError("At most one owner and four guest profiles are allowed")
        clean = {}
        for profile_id, profile in profiles.items():
            if (profile_id != "owner" and not profile_id.startswith("guest-")) or profile["kind"] not in {"owner", "guest"}:
                raise ValueError("Invalid speaker profile identifier")
            clean[profile_id] = {
                "kind": profile["kind"],
                "enabled": bool(profile["enabled"]),
                "embedding": self.validate_embedding(profile["embedding"]),
            }
        payload = json.dumps({"version": 1, "profiles": clean}, separators=(",", ":"))
        fd, temporary = tempfile.mkstemp(prefix=".speaker-profiles-", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def upsert(self, kind: str, embedding: Any, profile_id: str | None = None) -> str:
        profiles = self.load()
        profile_id = "owner" if kind == "owner" else profile_id or f"guest-{uuid.uuid4().hex[:12]}"
        if kind not in {"owner", "guest"}:
            raise ValueError("Profile kind must be owner or guest")
        if kind == "owner":
            profiles.pop("owner", None)
        elif len([item for item in profiles.values() if item["kind"] == "guest"]) >= 4 and profile_id not in profiles:
            raise ValueError("The guest profile limit has been reached")
        profiles[profile_id] = {"kind": kind, "enabled": True,
                                "embedding": self.validate_embedding(embedding)}
        self.save(profiles)
        return profile_id
