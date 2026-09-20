"""Local publishing (nginx) and off-critical-path S3 archival.

Publishing is a rename inside CLIPS_DIR, so a clip becomes playable the instant
it is complete -- no upload on the hot path. S3 archival is fire-and-forget and
exists for replay/export/audit, not for playback.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from pathlib import Path

import httpx

from .config import settings

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class InputError(RuntimeError):
    pass


async def materialise_input(
    client: httpx.AsyncClient, source: str, dest_dir: Path, name: str
) -> str:
    """Get a conditioning asset onto local disk and return a `file://` URI.

    Accepts a local path, an http(s) URL, or a `data:` URI. SGLang reads the
    file directly from disk, so it must be local to the GPU box.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    if source.startswith("data:"):
        header, _, payload = source.partition(",")
        if not payload:
            raise InputError("malformed data: URI")
        ext = mimetypes.guess_extension(header.split(";")[0].removeprefix("data:")) or ".png"
        dest = dest_dir / f"{name}{ext}"
        dest.write_bytes(base64.b64decode(payload))
        return f"file://{dest}"

    if source.startswith(("http://", "https://")):
        suffix = Path(source.split("?", 1)[0]).suffix
        dest = dest_dir / f"{name}{suffix if suffix in _IMAGE_SUFFIXES else '.png'}"
        async with client.stream("GET", source) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                async for chunk in resp.aiter_bytes(1 << 20):
                    fh.write(chunk)
        return f"file://{dest}"

    path = Path(source.removeprefix("file://"))
    if not path.exists():
        raise InputError(f"conditioning asset not found on this host: {path}")
    # Already local -- hand SGLang the original, no copy.
    return f"file://{path.resolve()}"


def job_dir(job_id: str) -> Path:
    return settings.clips_dir / job_id


def public_url(job_id: str, filename: str) -> str:
    return f"{settings.public_base_url}/{job_id}/{filename}"


def publish(src: Path, job_id: str, filename: str) -> tuple[Path, str]:
    """Move a finished artefact into the nginx-served directory."""
    dest_dir = job_dir(job_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if src.resolve() != dest.resolve():
        src.replace(dest)
    return dest, public_url(job_id, filename)


def archive_async(job_id: str, paths: list[Path]) -> None:
    """Schedule S3 upload without joining it to the request.

    Deliberately fire-and-forget: a failed archive must never stall or fail a
    beat that is already playable.
    """
    if not settings.s3_bucket:
        return
    task = asyncio.create_task(_archive(job_id, paths))
    # Hold a reference so the task is not garbage collected mid-flight.
    _pending.add(task)
    task.add_done_callback(_pending.discard)


_pending: set[asyncio.Task] = set()


async def _archive(job_id: str, paths: list[Path]) -> None:
    try:
        await asyncio.to_thread(_upload_sync, job_id, paths)
    except Exception:  # noqa: BLE001 - archival must never surface to the caller
        log.exception("s3 archive failed for job %s", job_id)


def _upload_sync(job_id: str, paths: list[Path]) -> None:
    import boto3

    s3 = boto3.client("s3")
    for path in paths:
        if not path.exists():
            continue
        key = f"{settings.s3_prefix}/{job_id}/{path.name}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        s3.upload_file(
            str(path),
            settings.s3_bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
