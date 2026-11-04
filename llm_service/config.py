"""Service configuration for the internal EL Voice LLM service.

All values come from the environment so the same image can run Local Voice or
Remote Voice. Prompt and context limits come from the mounted workspace
``settings.json`` through the existing ``elvoice.settings`` module; this keeps
one source of truth for the system prompt and the conversation budget.

Secrets are read here and never logged: only the model name and the base URL
are reported by the public ``/status`` endpoint.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8100
DEFAULT_MODE = "local"
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_MAX_ITERATIONS = 18
DEFAULT_MAX_SEARCH_RESULTS = 5
DEFAULT_SEARCH_SNIPPET_CHARS = 9999
DEFAULT_SEARCH_TIMEOUT = 21.0
DEFAULT_REQUEST_TIMEOUT = 75.0

MODE_ENV = "EL_VOICE_LLM_MODE"
HOST_ENV = "EL_VOICE_LLM_HOST"
PORT_ENV = "EL_VOICE_LLM_PORT"
WORKSPACE_ENV = "EL_VOICE_WORKSPACE"
SETTINGS_ENV = "EL_VOICE_SETTINGS"
MODELS_ENV = "EL_VOICE_MODELS"
ITERATIONS_ENV = "EL_VOICE_LLM_MAX_ITERATIONS"
SEARCH_RESULTS_ENV = "EL_VOICE_SEARCH_MAX_RESULTS"
SEARCH_SNIPPET_ENV = "EL_VOICE_SEARCH_SNIPPET_CHARS"
SEARCH_TIMEOUT_ENV = "EL_VOICE_SEARCH_TIMEOUT"
SEARCH_PROVIDER_ENV = "EL_VOICE_SEARCH_PROVIDER"
REQUEST_TIMEOUT_ENV = "EL_VOICE_LLM_REQUEST_TIMEOUT"

KEY_ENV = "OPENCODE_GO_API_KEY"
BASE_URL_ENV = "OPENCODE_GO_BASE_URL"
MODEL_ENV = "OPENCODE_GO_MODEL"
SESSION_ENV = "OPENCODE_GO_SESSION_ID"


def workspace_dir() -> Path:
    """Locate the mounted voice application source with ``settings.json``."""
    configured = os.environ.get(WORKSPACE_ENV)
    if configured and (Path(configured) / "settings.json").is_file():
        return Path(configured).resolve()
    container = Path("/workspace")
    if (container / "settings.json").is_file():
        return container.resolve()
    return Path(__file__).resolve().parents[1] / "workspace"


def import_voice_settings():
    """Import ``elvoice.settings`` from the mounted workspace."""
    workspace = workspace_dir()
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from elvoice import settings as voice_settings  # noqa: PLC0415 - path first

    return voice_settings


def _validated_base_url(value: str) -> str:
    """Reject non-HTTPS provider endpoints so credentials never travel in clear text."""
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ConfigError(
            "OPENCODE_GO_BASE_URL must be an HTTPS URL; refusing a non-HTTPS provider "
            "endpoint so credentials are never sent in clear text."
        )
    return value


def _text(value: str | None, default: str) -> str:
    if value is None:
        return default
    value = value.strip()
    return value or default


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _number(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class RemoteProvider:
    """OpenCode Go provider settings; the key is never exposed by ``repr``."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = field(default="", repr=False)
    session_id: str = ""
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT


@dataclass(frozen=True)
class ServiceConfig:
    mode: str
    host: str
    port: int
    workspace: Path
    models_root: Path
    settings: dict
    history_turns: int
    max_prompt_chars: int
    max_iterations: int
    max_search_results: int
    search_snippet_chars: int
    search_timeout: float
    search_provider: str
    remote: RemoteProvider

    @property
    def system_prompt(self) -> str:
        llm = self.settings["llm"]
        if self.mode == "remote":
            # Remote mode has web_search; local mode must not be told about it.
            return str(llm.get("remote_system_prompt") or llm["system_prompt"])
        return str(llm["system_prompt"])

    @property
    def llm_settings(self) -> dict:
        return dict(self.settings["llm"])

    def public_status(self) -> dict:
        """Content-free configuration safe to return over the network."""
        return {
            "mode": self.mode,
            "context": {
                "history_turns": self.history_turns,
                "max_prompt_chars": self.max_prompt_chars,
            },
            "limits": {
                "max_iterations": self.max_iterations,
                "max_search_results": self.max_search_results,
                "search_snippet_chars": self.search_snippet_chars,
                "search_timeout_s": self.search_timeout,
            },
            "search": {
                "provider": self.search_provider,
                "enabled": self.mode == "remote",
            },
            "remote": {
                "base_url": self.remote.base_url,
                "model": self.remote.model,
            }
            if self.mode == "remote"
            else None,
        }


def load_config() -> ServiceConfig:
    """Read and validate the service configuration from the environment."""
    mode = _text(os.environ.get(MODE_ENV), DEFAULT_MODE).lower()
    if mode not in ("local", "remote"):
        raise ValueError(f"{MODE_ENV} must be 'local' or 'remote'")

    voice_settings = import_voice_settings()
    settings_path = os.environ.get(SETTINGS_ENV)
    settings = voice_settings.load_settings(settings_path)
    conversation = settings["conversation"]

    host = _text(os.environ.get(HOST_ENV), DEFAULT_HOST)
    port = _integer(PORT_ENV, DEFAULT_PORT, 1, 65535)
    models_root = Path(os.environ.get(MODELS_ENV, "/models"))

    remote = RemoteProvider(
        base_url=_validated_base_url(
            _text(os.environ.get(BASE_URL_ENV), DEFAULT_BASE_URL).rstrip("/")
        ),
        model=_text(os.environ.get(MODEL_ENV), DEFAULT_MODEL),
        api_key=(os.environ.get(KEY_ENV) or "").strip(),
        session_id=_text(os.environ.get(SESSION_ENV), f"el-voice-{uuid.uuid4().hex}"),
        request_timeout=_number(
            REQUEST_TIMEOUT_ENV, DEFAULT_REQUEST_TIMEOUT, 5.0, 300.0
        ),
    )

    return ServiceConfig(
        mode=mode,
        host=host,
        port=port,
        workspace=workspace_dir(),
        models_root=models_root,
        settings=settings,
        history_turns=int(conversation["history_turns"]),
        max_prompt_chars=int(conversation["max_prompt_chars"]),
        max_iterations=_integer(ITERATIONS_ENV, DEFAULT_MAX_ITERATIONS, 1, 64),
        max_search_results=_integer(
            SEARCH_RESULTS_ENV, DEFAULT_MAX_SEARCH_RESULTS, 1, 10
        ),
        search_snippet_chars=_integer(
            SEARCH_SNIPPET_ENV, DEFAULT_SEARCH_SNIPPET_CHARS, 50, 20000
        ),
        search_timeout=_number(SEARCH_TIMEOUT_ENV, DEFAULT_SEARCH_TIMEOUT, 1.0, 60.0),
        search_provider=_text(os.environ.get(SEARCH_PROVIDER_ENV), "ddgs").lower(),
        remote=remote,
    )
