"""HTTP and SSE surface.

The API is deliberately thin: four verbs the player can perform (create, choose,
seek, watch) and nothing that returns a long-running result synchronously.
Everything slow -- the Worldsmith, the Director, PromptIR, generation -- happens
behind the event stream, so a request never blocks on a GPU and a lost connection
never loses work (DESIGN.md section 5.1).

`GET /sessions/{sid}/events?after=N` is the load-bearing endpoint. The client
fetches the session document once, then follows the stream; on reconnect it passes
the last sequence number it saw and gets the gap replayed. There is no polling
path, because a UI that polls a 15-second pipeline either wastes requests or shows
stale statuses.

In development this process also serves `/assets`, which is where the fake GPU
backend and the keyframe generator write. In production those are static files
behind nginx and a CDN pull (section 5.1), and `PUBLIC_BASE_URL` points there
instead.

Under `H3_TRANSPORT=http` it serves them a second time, on its own port and its
own socket (`_assets_app`). See that function for why the same files are served
twice rather than once on 0.0.0.0.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.routing import Mount

from .config import settings
from .engine import Engine, EngineError, session_view
from .events import EventHub
from .store import Store
from .worldsmith import PRESETS

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("h3game.main")

_HEARTBEAT_S = 15.0


class CreateRequest(BaseModel):
    premise: str = Field("", max_length=4000)
    genre: str = Field("", max_length=120)
    pov: str = Field("third", pattern="^(first|third)$")
    preset_id: str = Field("", max_length=64)


class ChooseRequest(BaseModel):
    option_index: int | None = None
    custom_action: str | None = Field(None, max_length=400)


class SeekRequest(BaseModel):
    beat_id: str


def _assets_app() -> Starlette:
    """A read-only file server, and deliberately nothing else.

    H3 fetches the conditioning frame itself now, so something on this box has to
    listen on the VPC interface. It is not this API. The control surface has no
    auth of its own -- anyone who can reach it can create sessions and spend GPU
    slots -- so it stays on 127.0.0.1 behind the ssh tunnel, and what gets exposed
    to the other instances is a separate app whose entire routing table is one
    `StaticFiles` mount.

    That makes the exposure a property of the object rather than of a middleware
    someone has to remember: there is no POST anywhere in this app to forget to
    protect, and `StaticFiles` answers GET and HEAD only. The files themselves are
    immutable PNGs and mp4s already being served to the player.
    """
    return Starlette(
        routes=[Mount("/", app=StaticFiles(directory=str(settings.assets_dir)), name="assets")]
    )


async def _serve_assets() -> None:
    """Run that app on its own socket until cancelled."""
    config = uvicorn.Config(
        _assets_app(),
        host=settings.assets_host,
        port=settings.assets_port,
        log_level="warning",   # one line per keyframe fetch is noise, not signal
        access_log=False,
    )
    await uvicorn.Server(config).serve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.assets_dir.mkdir(parents=True, exist_ok=True)

    store = Store()
    store.start()
    hub = EventHub(settings.data_dir)
    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.gpu_timeout_s))
    engine = Engine(store, hub, client)

    app.state.store = store
    app.state.hub = hub
    app.state.engine = engine
    # `midstory_kf` is in here because it is the one setting that changes what the
    # player sees rather than how fast they see it, and "是不是又在画图" should be
    # answerable from the first line of the log rather than by reading a beat's
    # timings after the fact.
    log.info(
        "orchestrator up: backend=%s slots=%d beat=%.3fs pregen_depth=%d midstory_kf=%s",
        engine.gpu.kind, engine.gpu.slots, settings.beat_seconds, settings.pregen_depth,
        "on" if settings.midstory_keyframes else "off",
    )

    assets: asyncio.Task[None] | None = None
    # Only the h3 backend has anything that needs to reach in; the fake backend
    # animates frames locally. Which also keeps the throwaway test server
    # (bench/fake_gpu_server.sh) from fighting the live one for the port.
    if engine.gpu.kind == "h3" and settings.h3_transport != "ssh":
        assets = asyncio.create_task(_serve_assets())
        log.info(
            "assets for the GPU boxes on %s:%d -> %s",
            settings.assets_host, settings.assets_port, settings.assets_dir,
        )

    warm: asyncio.Task[None] | None = None
    if settings.h3_warm:
        # Not awaited: it costs ~8s per replica and the server is useful before it
        # finishes. The pool holds every slot while warming, so the first real beat
        # queues behind it rather than racing it.
        warm = asyncio.create_task(engine.gpu.warm())
    try:
        yield
    finally:
        if warm and not warm.done():
            warm.cancel()
        if assets:
            assets.cancel()
        await engine.shutdown()
        await client.aclose()
        await store.stop()


app = FastAPI(title="h3game-orchestrator", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.assets_dir.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(settings.assets_dir)), name="assets")


def _engine(request: Request) -> Engine:
    return request.app.state.engine


def _store(request: Request) -> Store:
    return request.app.state.store


# --------------------------------------------------------------------------- #
# Meta                                                                         #
# --------------------------------------------------------------------------- #


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    return await _engine(request).health()


@app.get("/presets")
async def presets() -> dict[str, Any]:
    return {
        "presets": PRESETS,
        "beat_seconds": settings.beat_seconds,
        "branch_count": settings.branch_count,
    }


# --------------------------------------------------------------------------- #
# Sessions                                                                     #
# --------------------------------------------------------------------------- #


@app.post("/sessions")
async def create_session(body: CreateRequest, request: Request) -> dict[str, Any]:
    premise = body.premise.strip()
    genre = body.genre.strip()
    if body.preset_id:
        preset = next((p for p in PRESETS if p["id"] == body.preset_id), None)
        if not preset:
            raise HTTPException(404, f"unknown preset {body.preset_id}")
        # A preset is a starting point, not a lock: a premise typed alongside it
        # wins, which is how "pick 末日 then describe your own corner of it" works.
        premise = premise or str(preset["premise"])
        genre = genre or str(preset["genre"])
    if not premise:
        raise HTTPException(400, "premise is required (or pick a preset)")

    session = await _engine(request).create_session(premise, genre, body.pov)
    # Returns immediately with phase=creating. The world bible, the opening
    # keyframe and the first clip all arrive on the event stream.
    return session_view(session)


@app.get("/sessions")
async def list_sessions(request: Request, limit: int = 50) -> dict[str, Any]:
    return {"sessions": _store(request).list_sessions(limit=limit)}


@app.get("/sessions/{sid}")
async def get_session(sid: str, request: Request) -> dict[str, Any]:
    session = _store(request).get(sid)
    if not session:
        raise HTTPException(404, f"session {sid} not found")
    return session_view(session)


@app.delete("/sessions/{sid}")
async def delete_session(sid: str, request: Request) -> dict[str, Any]:
    """Erase a session and everything it rendered.

    Idempotent: deleting a session that is already gone answers 200, not 404. The
    frontend removes the row optimistically, so a retry after a dropped response
    is deleting something it can no longer see, and a 404 there would surface as
    an error for an operation that in fact succeeded.
    """
    existed = await _engine(request).delete(sid)
    return {"deleted": sid, "existed": existed}


@app.post("/sessions/{sid}/choose")
async def choose(sid: str, body: ChooseRequest, request: Request) -> dict[str, Any]:
    try:
        beat = await _engine(request).choose(
            sid, option_index=body.option_index, custom_action=body.custom_action
        )
    except EngineError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"beat_id": beat.id, "status": beat.status.value}


@app.post("/sessions/{sid}/seek")
async def seek(sid: str, body: SeekRequest, request: Request) -> dict[str, Any]:
    try:
        beat = await _engine(request).seek(sid, body.beat_id)
    except EngineError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"beat_id": beat.id, "status": beat.status.value}


@app.get("/sessions/{sid}/beats/{beat_id}/ir")
async def get_ir(sid: str, beat_id: str, request: Request) -> JSONResponse:
    """The exact prompt H3 saw, for debugging a bad-looking beat."""
    path = _store(request).session_dir(sid) / "ir" / f"{beat_id}.txt"
    meta = path.with_suffix(".meta.json")
    if not path.exists():
        raise HTTPException(404, "no IR recorded for that beat yet")
    return JSONResponse(
        {
            "prompt": path.read_text(encoding="utf-8"),
            "meta": json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {},
        }
    )


# --------------------------------------------------------------------------- #
# Event stream                                                                 #
# --------------------------------------------------------------------------- #


@app.get("/sessions/{sid}/events")
async def events(sid: str, request: Request, after: int = 0) -> StreamingResponse:
    store = _store(request)
    if not store.get(sid):
        raise HTTPException(404, f"session {sid} not found")
    bus = request.app.state.hub.bus(sid)

    async def stream():
        queue = bus.subscribe()
        try:
            # Replay first, then live. Subscribing before replaying means an event
            # emitted during the replay is queued rather than lost; the client
            # de-duplicates on `seq`.
            for event in bus.replay(after):
                yield _sse(event)
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_S)
                except asyncio.TimeoutError:
                    # Comment frame: keeps intermediaries from reaping an idle
                    # connection during a long generation.
                    yield ": ping\n\n"
                    continue
                yield _sse(event)
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx must not buffer this
        },
    )


def _sse(event: dict[str, Any]) -> str:
    return (
        f"id: {event.get('seq', 0)}\n"
        f"event: {event.get('type', 'message')}\n"
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    )
