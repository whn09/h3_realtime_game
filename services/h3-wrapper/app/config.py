"""Configuration, read once from the environment.

Deliberately dependency-free (no pydantic-settings) so this can be dropped onto a
GPU box with a minimal pip install.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- SGLang upstream -----------------------------------------------------
    # One wrapper instance fronts exactly one SGLang instance. Run N wrappers
    # for N GPUs; the orchestrator does the scheduling across them.
    sglang_base_url: str = field(default_factory=lambda: _env("SGLANG_BASE_URL", "http://127.0.0.1:30000"))
    sglang_model: str = field(default_factory=lambda: _env("SGLANG_MODEL", "MiniMaxAI/MiniMax-H3"))
    # Which --model-variant this upstream was launched with. Used to reject
    # requests the instance structurally cannot serve, instead of failing deep
    # inside the model 9 seconds later.
    sglang_variant: str = field(default_factory=lambda: _env("SGLANG_VARIANT", "fl2va"))
    request_timeout_s: float = field(default_factory=lambda: float(_env("REQUEST_TIMEOUT_S", "600")))
    poll_interval_s: float = field(default_factory=lambda: float(_env("POLL_INTERVAL_S", "0.25")))

    # --- Concurrency ---------------------------------------------------------
    # A GPU slot is a serial resource. Keep this at 1 unless benchmarking
    # proves the instance benefits from in-flight batching (checklist item 5).
    max_concurrent: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT", "1"))

    # --- Local storage / serving --------------------------------------------
    # nginx serves CLIPS_DIR at PUBLIC_BASE_URL. Keeping the bytes on the GPU
    # box and letting a CDN pull from nginx avoids a ~1s S3 upload on the
    # critical path; S3 archival happens asynchronously, off the hot path.
    clips_dir: Path = field(default_factory=lambda: Path(_env("CLIPS_DIR", "/var/www/clips")))
    work_dir: Path = field(default_factory=lambda: Path(_env("WORK_DIR", "/tmp/h3-wrapper")))
    public_base_url: str = field(default_factory=lambda: _env("PUBLIC_BASE_URL", "http://127.0.0.1:8080/clips").rstrip("/"))

    # --- Async archival ------------------------------------------------------
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", ""))
    s3_prefix: str = field(default_factory=lambda: _env("S3_PREFIX", "h3game/clips").strip("/"))

    # --- Binaries ------------------------------------------------------------
    ffmpeg: str = field(default_factory=lambda: _env("FFMPEG_BIN", "ffmpeg"))
    ffprobe: str = field(default_factory=lambda: _env("FFPROBE_BIN", "ffprobe"))

    # --- Debugging -----------------------------------------------------------
    # Echo the untouched upstream JSON on every response. Invaluable while the
    # exact SGLang response shape is still unconfirmed; turn off in production.
    echo_raw: bool = field(default_factory=lambda: _env_bool("ECHO_RAW", True))


settings = Settings()
