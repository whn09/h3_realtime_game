"""SGLang `/v1/videos` client.

The cookbook documents the *request* schema thoroughly but not the *response*,
and does not say whether the endpoint is synchronous or job-based. Rather than
guess one shape and break on contact, this module:

  1. builds the request from our own typed contract,
  2. accepts sync-or-async transparently (polls if handed a job id),
  3. recursively locates the video payload whatever key it hides under,
  4. always retains the untouched JSON so `/probe` can answer checklist item 1.

Once the real shape is confirmed on hardware, `_find_outputs` can be collapsed
to the single true path -- but keep the raw echo.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from .config import settings
from .models import GenerateRequest

# Keys that plausibly carry a video location or payload, most-specific first.
_URLISH_KEYS = (
    "url", "uri", "video_url", "video_uri", "file_url",
    "path", "file_path", "filepath", "file", "video_path", "video", "output_path",
)
_B64_KEYS = ("b64_json", "b64", "base64", "video_base64", "data_base64", "content")

# Terminal + non-terminal status vocabularies seen across SGLang/OpenAI-ish APIs.
_PENDING_STATUS = {"queued", "pending", "running", "in_progress", "processing", "starting"}
_FAILED_STATUS = {"failed", "error", "cancelled", "canceled", "expired"}

_VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv")
_B64_RE = re.compile(r"^[A-Za-z0-9+/\r\n]{512,}={0,2}$")


class SGLangError(RuntimeError):
    pass


@dataclass
class VideoOutput:
    """One located video, in whichever form the upstream chose to return it."""

    kind: str  # "url" | "path" | "b64"
    value: str
    json_path: str  # where we found it, for debugging the unknown schema


def build_payload(req: GenerateRequest, condition_uris: dict[str, str]) -> dict[str, Any]:
    """Translate our request into SGLang's documented schema.

    `condition_uris` maps our logical slots ("first_frame", "last_frame",
    "ref:0"...) to `file://` URIs already materialised on local disk.
    """
    conditions: list[dict[str, Any]] = []

    if uri := condition_uris.get("first_frame"):
        conditions.append(
            {"type": "image", "uri": uri, "role": "keyframe", "frame_index": 0}
        )
    if uri := condition_uris.get("last_frame"):
        conditions.append(
            {"type": "image", "uri": uri, "role": "keyframe", "frame_index": -1}
        )
    for i, ref in enumerate(req.references):
        cond: dict[str, Any] = {
            "type": ref.type,
            "uri": condition_uris[f"ref:{i}"],
            "role": "reference",
        }
        if ref.start_time_seconds is not None:
            cond["start_time_seconds"] = ref.start_time_seconds
        conditions.append(cond)

    payload: dict[str, Any] = {
        "model": settings.sglang_model,
        "task": req.task,
        "prompt": req.resolved_prompt(),
        "seconds": req.seconds,
        "target": {
            "short_edge": req.short_edge,
            "aspect_ratio": req.aspect_ratio,
            "duration_seconds": req.seconds,
        },
        "quality": req.quality,
        "num_inference_steps": req.num_inference_steps,
        "num_outputs_per_prompt": req.num_outputs_per_prompt,
    }
    if conditions:
        payload["conditions"] = conditions
    if req.flow_shift is not None:
        payload["flow_shift"] = req.flow_shift
    if req.audio_flow_shift is not None:
        payload["audio_flow_shift"] = req.audio_flow_shift
    if req.seed is not None:
        payload["seed"] = req.seed

    payload.update(req.extra)
    return payload


async def submit(client: httpx.AsyncClient, payload: dict[str, Any]) -> Any:
    """POST the generation request and resolve it to a terminal response."""
    url = f"{settings.sglang_base_url.rstrip('/')}/v1/videos"
    resp = await client.post(url, json=payload)
    if resp.status_code >= 400:
        raise SGLangError(f"SGLang {resp.status_code}: {resp.text[:2000]}")
    body = resp.json()

    job_id = _pending_job_id(body)
    if job_id is None:
        return body
    return await _poll(client, job_id)


async def _poll(client: httpx.AsyncClient, job_id: str) -> Any:
    """Poll a job-style response until terminal. No-op for sync upstreams."""
    import asyncio

    url = f"{settings.sglang_base_url.rstrip('/')}/v1/videos/{job_id}"
    deadline = settings.request_timeout_s
    waited = 0.0
    while waited < deadline:
        await asyncio.sleep(settings.poll_interval_s)
        waited += settings.poll_interval_s
        resp = await client.get(url)
        if resp.status_code >= 400:
            raise SGLangError(f"SGLang poll {resp.status_code}: {resp.text[:1000]}")
        body = resp.json()
        status = _status_of(body)
        if status in _FAILED_STATUS:
            raise SGLangError(f"job {job_id} {status}: {str(body)[:1000]}")
        if _pending_job_id(body) is None:
            return body
    raise SGLangError(f"job {job_id} still pending after {deadline}s")


def _status_of(body: Any) -> str | None:
    if isinstance(body, dict):
        for key in ("status", "state"):
            val = body.get(key)
            if isinstance(val, str):
                return val.lower()
    return None


def _pending_job_id(body: Any) -> str | None:
    """Return a job id iff the response is non-terminal (async) and needs polling."""
    if not isinstance(body, dict):
        return None
    status = _status_of(body)
    if status in _FAILED_STATUS:
        raise SGLangError(f"SGLang returned status={status}: {str(body)[:1000]}")
    has_video = bool(_find_outputs(body))
    if has_video:
        return None
    if status in _PENDING_STATUS or (status is None and not has_video):
        for key in ("id", "job_id", "task_id", "request_id"):
            val = body.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def _looks_like_video_ref(value: str) -> bool:
    lowered = value.split("?", 1)[0].lower()
    return lowered.endswith(_VIDEO_SUFFIXES)


def _looks_like_b64_video(value: str) -> bool:
    # Only treat long, well-formed base64 as a payload; short strings are ids.
    if len(value) < 512 or not _B64_RE.match(value):
        return False
    try:
        head = base64.b64decode(value[:64] + "==", validate=False)
    except (binascii.Error, ValueError):
        return False
    # mp4/mov begin with a size box then 'ftyp'; webm/mkv with EBML magic.
    return b"ftyp" in head[:16] or head.startswith(b"\x1a\x45\xdf\xa3")


def _find_outputs(body: Any, _path: str = "$") -> list[VideoOutput]:
    """Recursively locate video references, preferring documented-looking keys.

    Breadth-ordered so `data[0].url` wins over a nested debug field.
    """
    found: list[VideoOutput] = []
    queue: list[tuple[Any, str]] = [(body, _path)]

    while queue:
        node, path = queue.pop(0)

        if isinstance(node, dict):
            for key in _URLISH_KEYS:
                val = node.get(key)
                if isinstance(val, str) and (_looks_like_video_ref(val) or val.startswith("file://")):
                    kind = "url" if val.startswith(("http://", "https://")) else "path"
                    found.append(VideoOutput(kind, val, f"{path}.{key}"))
            for key in _B64_KEYS:
                val = node.get(key)
                if isinstance(val, str) and _looks_like_b64_video(val):
                    found.append(VideoOutput("b64", val, f"{path}.{key}"))
            for key, val in node.items():
                if isinstance(val, (dict, list)):
                    queue.append((val, f"{path}.{key}"))

        elif isinstance(node, list):
            for i, val in enumerate(node):
                if isinstance(val, (dict, list)):
                    queue.append((val, f"{path}[{i}]"))
                elif isinstance(val, str) and _looks_like_video_ref(val):
                    kind = "url" if val.startswith(("http://", "https://")) else "path"
                    found.append(VideoOutput(kind, val, f"{path}[{i}]"))

        elif isinstance(node, str) and _looks_like_video_ref(node):
            kind = "url" if node.startswith(("http://", "https://")) else "path"
            found.append(VideoOutput(kind, node, path))

    return _dedupe(found)


def _dedupe(items: Iterable[VideoOutput]) -> list[VideoOutput]:
    seen: set[str] = set()
    out: list[VideoOutput] = []
    for item in items:
        if item.value in seen:
            continue
        seen.add(item.value)
        out.append(item)
    return out


def locate_outputs(body: Any) -> list[VideoOutput]:
    outputs = _find_outputs(body)
    if not outputs:
        raise SGLangError(
            "could not locate a video in the SGLang response; "
            f"keys seen: {_shape(body)}"
        )
    return outputs


def _shape(node: Any, depth: int = 0) -> Any:
    """Compact structural summary, for error messages about unknown schemas."""
    if depth > 3:
        return "..."
    if isinstance(node, dict):
        return {k: _shape(v, depth + 1) for k, v in list(node.items())[:20]}
    if isinstance(node, list):
        return [_shape(node[0], depth + 1), f"...x{len(node)}"] if node else []
    if isinstance(node, str):
        return f"str[{len(node)}]"
    return type(node).__name__


async def materialise(
    client: httpx.AsyncClient, output: VideoOutput, dest: Path
) -> None:
    """Get the produced video onto local disk at `dest`, whatever form it took."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if output.kind == "b64":
        dest.write_bytes(base64.b64decode(output.value))
        return

    if output.kind == "url":
        async with client.stream("GET", output.value) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                async for chunk in resp.aiter_bytes(1 << 20):
                    fh.write(chunk)
        return

    # kind == "path": SGLang runs on this box, so it wrote a file we can read.
    src = Path(output.value.removeprefix("file://"))
    if not src.exists():
        raise SGLangError(
            f"SGLang reported {src} but it is not readable from the wrapper; "
            "run the wrapper on the same host (or share the volume)"
        )
    if src.resolve() == dest.resolve():
        return
    # Hardlink when possible (same filesystem, zero copy), else copy.
    try:
        dest.hardlink_to(src)
    except (OSError, AttributeError):
        import shutil

        shutil.copyfile(src, dest)
