"""h3-wrapper -- the only thing that talks to SGLang.

Responsibilities (all local to the GPU box, see DESIGN.md section 5.2):
  1. translate our typed contract into SGLang's `/v1/videos` schema
  2. resolve conditioning assets to local `file://` URIs
  3. call SGLang, tolerating sync or async response styles
  4. extract the last frame (next beat's keyframe + this beat's freeze frame)
  5. remux faststart, probe, fingerprint for drift
  6. publish into the nginx-served directory and return URLs
  7. report per-stage timings, because the latency budget is the whole project

One wrapper fronts one SGLang instance. Run one per GPU slot; the orchestrator
schedules across them.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import media, sglang, storage
from .config import settings
from .models import GenerateRequest, GenerateResponse, Timings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("h3-wrapper")


class _Clock:
    """Accumulates per-stage timings into a Timings model."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.timings = Timings()

    @asynccontextmanager
    async def stage(self, field: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            setattr(self.timings, field, getattr(self.timings, field) + elapsed)

    def finish(self) -> Timings:
        self.timings.total_ms = (time.perf_counter() - self.started) * 1000.0
        return self.timings


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    settings.work_dir.mkdir(parents=True, exist_ok=True)
    settings.clips_dir.mkdir(parents=True, exist_ok=True)

    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
    )
    # A GPU slot is serial. This semaphore is the last line of defence against
    # the orchestrator over-subscribing an instance and doubling every latency.
    app.state.slot = asyncio.Semaphore(settings.max_concurrent)
    log.info(
        "h3-wrapper up: upstream=%s variant=%s slots=%d clips=%s",
        settings.sglang_base_url, settings.sglang_variant,
        settings.max_concurrent, settings.clips_dir,
    )
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="h3-wrapper", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Liveness plus upstream reachability, so the scheduler can drop a dead GPU."""
    client: httpx.AsyncClient = app.state.client
    upstream_ok = False
    detail = None
    try:
        resp = await client.get(
            f"{settings.sglang_base_url.rstrip('/')}/health", timeout=3.0
        )
        upstream_ok = resp.status_code < 500
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)

    return {
        "ok": True,
        "upstream_ok": upstream_ok,
        "upstream_detail": detail,
        "variant": settings.sglang_variant,
        "slots_total": settings.max_concurrent,
        "slots_free": app.state.slot._value,  # noqa: SLF001 - cheap, no public API
    }


@app.post("/probe")
async def probe(payload: dict[str, Any] = Body(...)) -> JSONResponse:
    """Forward a raw SGLang payload and return the untouched response.

    This is the tool for P0 checklist items 1, 2 and 9: it answers "is the
    endpoint sync or async", "what shape is the response", and "does this
    variant accept a bare t2va request" without any of our normalisation in
    the way.
    """
    client: httpx.AsyncClient = app.state.client
    url = f"{settings.sglang_base_url.rstrip('/')}/v1/videos"

    start = time.perf_counter()
    try:
        resp = await client.post(url, json=payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"upstream unreachable: {exc}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    try:
        body: Any = resp.json()
    except ValueError:
        body = {"_non_json_body": resp.text[:20000]}

    located = []
    try:
        located = [vars(o) for o in sglang.locate_outputs(body)]
    except sglang.SGLangError as exc:
        located = [{"error": str(exc)}]

    return JSONResponse(
        {
            "status_code": resp.status_code,
            "elapsed_ms": round(elapsed_ms, 1),
            "shape": sglang._shape(body),  # noqa: SLF001 - debug endpoint
            "located_outputs": located,
            "raw": body,
        }
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    client: httpx.AsyncClient = app.state.client
    clock = _Clock()
    job_id = req.job_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"

    _reject_impossible_task(req)

    scratch = settings.work_dir / job_id
    scratch.mkdir(parents=True, exist_ok=True)

    # --- conditioning assets ------------------------------------------------
    async with clock.stage("input_fetch_ms"):
        try:
            condition_uris = await _resolve_conditions(client, req, scratch)
        except storage.InputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = sglang.build_payload(req, condition_uris)

    # --- GPU ----------------------------------------------------------------
    async with clock.stage("slot_wait_ms"):
        await app.state.slot.acquire()
    try:
        async with clock.stage("sglang_ms"):
            try:
                raw = await sglang.submit(client, payload)
            except sglang.SGLangError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        outputs = sglang.locate_outputs(raw)
        raw_video = scratch / "raw.mp4"
        async with clock.stage("download_ms"):
            await sglang.materialise(client, outputs[0], raw_video)
    finally:
        app.state.slot.release()

    # --- post-processing (GPU slot already released) ------------------------
    async with clock.stage("probe_ms"):
        info = await media.probe(raw_video)

    final_video = scratch / "beat.mp4"
    if req.faststart:
        async with clock.stage("faststart_ms"):
            try:
                await media.remux_faststart(raw_video, final_video)
            except media.MediaError:
                # Playable without faststart, just slower to start. Never fail
                # a beat over an optimisation.
                log.warning("faststart remux failed for %s, serving as-is", job_id)
                final_video = raw_video
    else:
        final_video = raw_video

    last_frame_path: Path | None = None
    if req.extract_last_frame:
        last_frame_path = scratch / "last.png"
        async with clock.stage("last_frame_ms"):
            try:
                await media.extract_last_frame(final_video, last_frame_path, info.duration_ms)
            except media.MediaError as exc:
                # Without a last frame the next beat cannot chain, so surface
                # this loudly rather than silently producing a discontinuity.
                raise HTTPException(
                    status_code=500, detail=f"last-frame extraction failed: {exc}"
                ) from exc

    poster_path: Path | None = None
    if req.extract_poster:
        poster_path = scratch / "poster.jpg"
        async with clock.stage("poster_ms"):
            try:
                await media.extract_poster(final_video, poster_path)
            except media.MediaError:
                poster_path = None

    stats = None
    if req.analyze and last_frame_path:
        async with clock.stage("analyze_ms"):
            stats = await media.frame_stats(last_frame_path)

    # --- publish ------------------------------------------------------------
    async with clock.stage("publish_ms"):
        video_dest, video_url = storage.publish(final_video, job_id, "beat.mp4")
        last_dest = last_url = None
        if last_frame_path:
            last_dest, last_url = storage.publish(last_frame_path, job_id, "last.png")
        poster_url = None
        if poster_path:
            _, poster_url = storage.publish(poster_path, job_id, "poster.jpg")

    if req.archive:
        storage.archive_async(job_id, [p for p in (video_dest, last_dest) if p])

    timings = clock.finish()
    log.info(
        "job=%s task=%s steps=%d %dx%d %.1fs audio=%s | total=%.0fms sglang=%.0fms "
        "lastframe=%.0fms faststart=%.0fms",
        job_id, req.task, req.num_inference_steps, info.width or 0, info.height or 0,
        (info.duration_ms or 0) / 1000.0, info.has_audio,
        timings.total_ms, timings.sglang_ms, timings.last_frame_ms, timings.faststart_ms,
    )

    return GenerateResponse(
        job_id=job_id,
        video_url=video_url,
        video_path=str(video_dest),
        last_frame_url=last_url,
        last_frame_path=str(last_dest) if last_dest else None,
        poster_url=poster_url,
        media=info,
        frame_stats=stats,
        timings=timings,
        sglang_request=payload if settings.echo_raw else None,
        sglang_raw=raw if settings.echo_raw else None,
    )


def _reject_impossible_task(req: GenerateRequest) -> None:
    """Fail in microseconds instead of nine seconds.

    `--model-variant` is a launch flag, so an instance serves exactly one
    conditioning family. Checking here means the scheduler learns immediately
    that it routed to the wrong pool.
    """
    variant = settings.sglang_variant
    if req.task == "ref2va" and variant != "ref2va":
        raise HTTPException(
            status_code=409,
            detail=f"this instance was launched as --model-variant {variant}; "
                   "ref2va requests need a ref2va instance",
        )
    if req.task == "fl2va" and variant == "ref2va":
        raise HTTPException(
            status_code=409,
            detail="this instance is ref2va; fl2va keyframe conditioning is unavailable",
        )
    if req.task == "fl2va" and not (req.first_frame or req.last_frame):
        raise HTTPException(
            status_code=400,
            detail="task=fl2va requires first_frame and/or last_frame",
        )


async def _resolve_conditions(
    client: httpx.AsyncClient, req: GenerateRequest, scratch: Path
) -> dict[str, str]:
    uris: dict[str, str] = {}
    if req.first_frame:
        uris["first_frame"] = await storage.materialise_input(
            client, req.first_frame, scratch, "first"
        )
    if req.last_frame:
        uris["last_frame"] = await storage.materialise_input(
            client, req.last_frame, scratch, "last_cond"
        )
    for i, ref in enumerate(req.references):
        uris[f"ref:{i}"] = await storage.materialise_input(
            client, ref.source, scratch, f"ref{i}"
        )
    return uris
