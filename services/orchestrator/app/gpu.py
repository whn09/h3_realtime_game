"""GPU slot scheduling and the two generation backends.

Two SGLang instances are two concurrent generation slots, and slots are the
project's scarcest resource (DESIGN.md section 5.3), so the queue in front of
them is priority-ordered rather than FIFO. The priority that matters is *is a
player currently staring at a freeze frame waiting for this clip*: a beat the
player has already chosen must overtake the speculative sibling that was queued
before it. `bump()` exists for exactly that moment.

`FakeBackend` synthesises clips locally with ffmpeg. It is not a stub -- it goes
through the same slot accounting, produces real mp4s with real audio, and extracts
real last frames, so the whole story engine including last-frame chaining is
exercisable without a GPU. That makes the engine and the frontend developable now,
and it gives the failure paths somewhere to be tested that is not production.
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import itertools
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx

from .config import settings
from .schema import IRSections

log = logging.getLogger("h3game.gpu")

PRIORITY_BLOCKING = 0     # the player is waiting on this clip right now
PRIORITY_PREGEN = 10      # speculative sibling
PRIORITY_SPECULATIVE = 20  # depth-2


class GpuError(RuntimeError):
    pass


class GpuTransportError(GpuError):
    """The request never got a verdict: tunnel down, timeout, scp failed.

    Separate from `GpuError` because the two deserve opposite treatment on the
    real deployment. A 400 or an OOM is a configuration error and retrying it
    only makes the configuration error slower (GAME.md); a dropped tunnel is
    worth exactly one more attempt at identical parameters.
    """


# --------------------------------------------------------------------------- #
# The frame lattice                                                            #
# --------------------------------------------------------------------------- #
#
# `align_num_frames(n, 17, 5)` snaps to the VAE's 17-frames-to-5-latents
# chunking, so reachable lengths are `5 + 17k` and nothing else. The server does
# not reject an off-lattice request -- it rounds silently, and no log line says
# so -- which is the whole reason this is computed here rather than trusted to
# the other end. 345f = 14.375s is the deployed shape.

_LATTICE_STEP = 17
_LATTICE_BASE = 5
_MIN_FRAMES = 22


def legal_frames(n: int) -> bool:
    return n >= _MIN_FRAMES and (n - _LATTICE_BASE) % _LATTICE_STEP == 0


def frames_for_seconds(seconds: float, fps: int | None = None) -> int:
    """Nearest legal frame count at or above `_MIN_FRAMES`.

    Rounds to the *nearest* rung rather than down: asking for 15s and getting
    14.375s is as wrong as getting 15.083s, and the caller asked for a duration,
    not a bound.
    """
    fps = fps or settings.h3_fps
    want = max(_MIN_FRAMES, round(seconds * fps))
    k = round((want - _LATTICE_BASE) / _LATTICE_STEP)
    n = _LATTICE_BASE + max(1, k) * _LATTICE_STEP
    return n if legal_frames(n) else _LATTICE_BASE + _LATTICE_STEP


@dataclass
class GpuRequest:
    job_id: str
    ir: IRSections
    seconds: float = field(default_factory=lambda: settings.beat_seconds)
    first_frame: str | None = None
    # alias -> a path on *that* replica that already holds `first_frame`'s image,
    # so the backend can skip the upload entirely when this beat lands there.
    # Populated from the parent's `GpuResult.remote_last_frame`; see
    # `H3Backend._extract_last_frame`. Advisory only -- an entry for the wrong
    # replica, or none at all, just means the local file gets uploaded as before.
    first_frame_remote: dict[str, str] = field(default_factory=dict)
    task: str = "fl2va"
    short_edge: int = field(default_factory=lambda: settings.short_edge)
    quality: str = field(default_factory=lambda: settings.quality)
    num_inference_steps: int = field(default_factory=lambda: settings.num_inference_steps)
    seed: int | None = None

    def degraded(self) -> "GpuRequest":
        """Cheaper parameters for the retry (DESIGN.md section 7).

        Deliberately drops steps *and* duration: a shorter, coarser beat still
        advances the story, while a second full-cost attempt is likely to fail the
        same way and costs another 9 seconds to find out.
        """
        return GpuRequest(
            job_id=self.job_id + "-r",
            ir=self.ir,
            seconds=min(self.seconds, settings.retry_seconds),
            first_frame=self.first_frame,
            first_frame_remote=dict(self.first_frame_remote),
            task=self.task,
            short_edge=self.short_edge,
            quality="high",
            num_inference_steps=settings.retry_steps,
            seed=self.seed,
        )


@dataclass
class GpuResult:
    video_url: str
    video_path: str
    last_frame_url: str | None = None
    last_frame_path: str | None = None
    # alias -> where this clip's last frame already sits on that replica, so a
    # continuous child landing there needs no upload. Only ever one entry (the
    # replica that generated the clip); the two boxes are separate EC2 hosts and
    # do not share a filesystem.
    remote_last_frame: dict[str, str] = field(default_factory=dict)
    poster_url: str | None = None
    duration_ms: float | None = None
    has_audio: bool = False
    frame_stats: dict[str, float] | None = None
    timings: dict[str, float] = field(default_factory=dict)
    endpoint: str = ""
    degraded: bool = False


# --------------------------------------------------------------------------- #
# Priority slot pool                                                           #
# --------------------------------------------------------------------------- #


class SlotPool:
    """One permit per endpoint, handed out in priority order."""

    def __init__(self, endpoints: list[str]) -> None:
        if not endpoints:
            raise ValueError("SlotPool needs at least one endpoint")
        self._idle: list[str] = list(endpoints)
        self._all: list[str] = list(endpoints)
        self._waiters: list[tuple[int, int, str, asyncio.Future[str]]] = []
        self._counter = itertools.count()
        self._lock = asyncio.Lock()
        self._cooldown: dict[str, float] = {}

    @property
    def size(self) -> int:
        return len(self._all)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "endpoints": self._all,
            "idle": list(self._idle),
            "queued": len(self._waiters),
            "cooling_down": [e for e, until in self._cooldown.items() if until > now],
        }

    def _healthy(self, endpoint: str) -> bool:
        return self._cooldown.get(endpoint, 0.0) <= time.monotonic()

    async def acquire(self, key: str, priority: int) -> str:
        async with self._lock:
            # Prefer an endpoint that is not in cooldown; fall back to any idle
            # one rather than stalling when every endpoint has recently erred.
            for i, ep in enumerate(self._idle):
                if self._healthy(ep):
                    return self._idle.pop(i)
            if self._idle:
                return self._idle.pop(0)
            fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            heapq.heappush(self._waiters, (priority, next(self._counter), key, fut))
        return await fut

    async def release(self, endpoint: str) -> None:
        async with self._lock:
            while self._waiters:
                _, _, _, fut = heapq.heappop(self._waiters)
                if fut.cancelled() or fut.done():
                    continue
                fut.set_result(endpoint)
                return
            self._idle.append(endpoint)

    async def bump(self, key: str, priority: int) -> bool:
        """Raise a queued request's priority. Used when the player's choice lands
        on a clip that is still waiting for a slot."""
        async with self._lock:
            changed = False
            for i, (prio, seq, k, fut) in enumerate(self._waiters):
                if k == key and priority < prio:
                    self._waiters[i] = (priority, seq, k, fut)
                    changed = True
            if changed:
                heapq.heapify(self._waiters)
            return changed

    async def mark_unhealthy(self, endpoint: str) -> None:
        async with self._lock:
            self._cooldown[endpoint] = time.monotonic() + settings.gpu_cooldown_s
        log.warning("endpoint %s benched for %.0fs", endpoint, settings.gpu_cooldown_s)


# --------------------------------------------------------------------------- #
# Backends                                                                     #
# --------------------------------------------------------------------------- #


class WrapperBackend:
    """Talks to h3-wrapper, which owns everything SGLang-shaped."""

    allow_degraded_retry = True

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def health(self, endpoint: str) -> dict[str, Any]:
        resp = await self.client.get(f"{endpoint}/healthz", timeout=3.0)
        return {"ok": resp.status_code == 200, **resp.json()}

    async def generate(self, endpoint: str, req: GpuRequest) -> GpuResult:
        body: dict[str, Any] = {
            "job_id": req.job_id,
            "ir": {
                "description": req.ir.description,
                "soundscape": req.ir.soundscape,
                "music": req.ir.music,
            },
            "task": req.task,
            "seconds": req.seconds,
            "short_edge": req.short_edge,
            "aspect_ratio": settings.aspect_ratio,
            "quality": req.quality,
            "num_inference_steps": req.num_inference_steps,
            "archive": True,
            "analyze": True,
        }
        if req.first_frame:
            body["first_frame"] = req.first_frame
        if req.seed is not None:
            body["seed"] = req.seed

        try:
            resp = await self.client.post(
                f"{endpoint}/generate", json=body, timeout=settings.gpu_timeout_s
            )
        except Exception as exc:  # noqa: BLE001
            raise GpuError(f"{endpoint} unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise GpuError(f"{endpoint} returned {resp.status_code}: {resp.text[:400]}")

        data = resp.json()
        media = data.get("media") or {}
        return GpuResult(
            video_url=data["video_url"],
            video_path=data.get("video_path", ""),
            last_frame_url=data.get("last_frame_url"),
            last_frame_path=data.get("last_frame_path"),
            poster_url=data.get("poster_url"),
            duration_ms=media.get("duration_ms"),
            has_audio=bool(media.get("has_audio")),
            frame_stats=data.get("frame_stats"),
            timings={f"gpu_{k}": v for k, v in (data.get("timings") or {}).items()},
            endpoint=endpoint,
        )


@dataclass(frozen=True)
class Replica:
    """One SGLang instance: an address to call, and an ssh alias.

    The address is where the HTTP API lives -- a private IP now that the servers
    bind 0.0.0.0, a forwarded local port before that. The alias is only used by
    `H3_TRANSPORT=ssh`; under the default `http` transport nothing shells out, and
    it survives as the label in logs, `/healthz`, and `GpuResult.endpoint`.
    """

    alias: str
    endpoint: str          # host:port, no scheme

    @property
    def base(self) -> str:
        return f"http://{self.endpoint}"


def read_replicas() -> list[Replica]:
    """`H3_REPLICAS`, else the file `game_tunnel.sh` writes."""
    spec = settings.h3_replicas.strip()
    if spec:
        out = []
        for part in spec.split(","):
            alias, _, endpoint = part.partition("=")
            out.append(Replica(alias.strip(), (endpoint or alias).strip()))
        return out
    path = settings.h3_state_dir / "replicas"
    if not path.exists():
        raise GpuError(
            f"no replicas: run `bash scripts/game_tunnel.sh <alias> ...` or set H3_REPLICAS "
            f"(looked in {path})"
        )
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            alias, endpoint = line.split()
            out.append(Replica(alias, endpoint))
    if not out:
        raise GpuError(f"{path} is empty")
    return out


def _lan_ip() -> str | None:
    """This host's address on the interface that leaves it, or None.

    Asks the routing table where it would send a packet rather than parsing
    `ip addr`, because a box with docker bridges has several addresses and only
    one of them is the one a peer in the VPC can use. UDP, so nothing is sent.

    The peer is deliberately *not* a replica's address. Replicas are often
    `127.0.0.1:<tunnel port>`, and routing to loopback would answer `127.0.0.1` --
    the one address that is wrong here, since it names the GPU box to the GPU box.
    A public address forces the question "which of my interfaces faces outward",
    which in this VPC is the same interface that faces the other instances.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("1.1.1.1", 9))
        ip = s.getsockname()[0]
    except OSError as exc:
        log.warning("could not work out this host's LAN address: %s", exc)
        return None
    finally:
        s.close()
    if ip.startswith("127."):
        log.warning("routing says this host is %s, which no GPU box can fetch from", ip)
        return None
    return ip


def _internal_base_url() -> str:
    """Where the GPU boxes should fetch conditioning frames from.

    Empty if it cannot be determined, which the caller treats as "http transport
    cannot serve frames" and falls back to uploading them.
    """
    if settings.internal_base_url:
        return settings.internal_base_url
    ip = _lan_ip()
    return f"http://{ip}:{settings.assets_port}" if ip else ""


class H3Backend:
    """Talks to the live SGLang Diffusion deployment described in GAME.md.

    Every generation is three legs -- get the conditioning frame to the box,
    submit and poll, get the mp4 back -- and the only question is whether the
    outer two are HTTP or ssh. `H3_TRANSPORT` picks:

        http (default)  `conditions[].uri` is a URL on this box's asset server, so
                        the worker fetches the frame itself; the clip comes back
                        from `GET /v1/videos/{id}/content`. No subprocesses.
        ssh             `cat` the frame over ssh, `scp` the clip back, and extract
                        the next frame on the box with the container's ffmpeg.

    GAME.md finding 3 says this build has no download endpoint and the API moves
    no bytes in either direction. That was wrong, and both halves were measured
    here: the server fetched a 2,044,883-byte keyframe from an http.server on this
    box, and `/content` returned 830,271 bytes beginning with `ftyp`. `url` in the
    poll response is still null, which is presumably where the belief came from.

    What http buys is not speed -- the bytes are the same bytes on the same VPC --
    but that a URL is not addressed to one machine. Under ssh, a frame had to be
    pushed to every replica ahead of time (`prefetch`) or uploaded on the critical
    path, and the on-box extraction in `_extract_last_frame` only paid off for a
    child that happened to land on its parent's box. A URL is fetchable from
    either replica, so the scheduler stops needing to care, and the last frame is
    just a file the local `_postprocess` already writes.

    The ssh path stays because it is the only one that works if the GPU boxes
    cannot route back to us, and because `prefetch` + on-box extraction are real
    measured wins (0.226s on-box against 3.4-10.4s to move the mp4) that should
    not be deleted merely for being unused.

    Frames rather than seconds is the other deployment fact worth stating: the
    request carries `duration_seconds` as a float and `seconds` is never sent
    (the server types that one as an int, so 14.375 becomes a 400), and the value
    is always `frames_for_seconds(...)/fps` so it lands on the lattice.
    """

    # GAME.md is explicit: a 400 is an off-lattice shape or `ref2va`, an OOM is
    # the memory arithmetic, and all three are configuration errors that a retry
    # converts into a slow configuration error. The degraded-parameter retry
    # belongs to backends whose failures are transient.
    allow_degraded_retry = False

    def __init__(self, client: httpx.AsyncClient, replicas: list[Replica]) -> None:
        self.client = client
        self.replicas = {r.alias: r for r in replicas}
        self.root = settings.assets_dir / "_h3"
        self.remote_kf_dir = f"{settings.h3_remote_out}/keyframes"
        # (alias, local path) -> the single in-flight or finished upload of that
        # file to that box. Keyed rather than re-run because `prefetch` and
        # `generate` race for the same frame by design.
        self._uploads: dict[tuple[str, str], asyncio.Task[str]] = {}
        self.http = settings.h3_transport != "ssh"
        # Resolved once, at construction, so a misconfiguration is a line in the
        # startup log rather than a mystery 400 on the player's first beat. Empty
        # means http transport cannot serve frames, and each `generate` falls back
        # to the ssh upload for that one leg.
        self.internal_base = _internal_base_url() if self.http else ""
        log.info(
            "h3 transport=%s assets=%s",
            "http" if self.http else "ssh", self.internal_base or "(ssh upload)",
        )

    # -- addressing our own assets ------------------------------------------ #

    def _asset_url(self, local: Path) -> str | None:
        """The URL a GPU box can GET this file at, or None if there isn't one.

        None is not an error. Anything the backend is handed from outside
        `ASSETS_DIR` -- a frame someone points at by hand, a future cache
        elsewhere on disk -- is simply not published, and the caller uploads it
        instead. Silently serving a path outside the asset root would turn a
        read-only image server into an arbitrary-file read.
        """
        if not self.internal_base:
            return None
        try:
            rel = local.resolve().relative_to(settings.assets_dir.resolve())
        except ValueError:
            log.debug("%s is outside ASSETS_DIR; uploading it instead", local)
            return None
        return f"{self.internal_base}/{quote(rel.as_posix())}"

    # -- ssh plumbing (H3_TRANSPORT=ssh) ------------------------------------ #

    def _ssh_opts(self, alias: str) -> list[str]:
        opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
        # Ride the ControlMaster socket `game_tunnel.sh` opened: measured 2.65-3.4s
        # against 5.4-6.1s cold for the same 946 KB, almost all of it handshake.
        # A ControlPath that does not exist is not an error -- ssh just dials a
        # fresh connection -- so this stays correct without the tunnel script.
        cm = settings.h3_state_dir / f"cm-{alias}"
        if cm.exists():
            opts += ["-o", f"ControlPath={cm}"]
        return opts

    @staticmethod
    def _remote_name(local: Path) -> str:
        """A name that changes when the bytes change.

        Keyframe paths are unique per beat, but a beat can be regenerated in
        place (the degraded retry, a reused session directory), and a stale
        remote copy of a reused path would silently condition the next clip on
        the previous take's last frame.
        """
        st = local.stat()
        digest = hashlib.sha1(
            f"{local.resolve()}:{st.st_mtime_ns}:{st.st_size}".encode()
        ).hexdigest()[:16]
        return f"{digest}{local.suffix or '.png'}"

    async def _upload(self, rep: Replica, local: Path) -> str:
        remote = f"{self.remote_kf_dir}/{self._remote_name(local)}"
        tmp = f"{remote}.part"
        # `mkdir && cat && mv` in one ssh invocation rather than `ssh mkdir` then
        # `scp`: one round trip instead of two, and the rename is atomic, so a
        # half-copied frame can never be handed to the model as a URI.
        cmd = [
            "ssh", *self._ssh_opts(rep.alias), rep.alias,
            f"mkdir -p {self.remote_kf_dir} && cat > {tmp} && mv -f {tmp} {remote}",
        ]
        started = time.perf_counter()
        with local.open("rb") as fh:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdin=fh,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, err = await proc.communicate()
        if proc.returncode:
            raise GpuTransportError(
                f"uploading {local.name} to {rep.alias} failed: {err.decode('utf-8', 'replace')[-300:]}"
            )
        log.debug(
            "uploaded %s to %s in %.0fms", local.name, rep.alias,
            (time.perf_counter() - started) * 1000.0,
        )
        return remote

    def _ensure_remote(self, rep: Replica, local: Path) -> asyncio.Task[str]:
        key = (rep.alias, str(local))
        task = self._uploads.get(key)
        if task is None or (task.done() and task.exception() is not None):
            task = asyncio.create_task(self._upload(rep, local))
            self._uploads[key] = task
        return task

    def prefetch(self, local: str) -> None:
        """Push a frame to every replica now, so no beat waits for it later.

        Fire and forget on purpose: a failure here costs nothing, because
        `generate` awaits the same task and reports the failure at the point
        where it actually matters.

        Nothing to do under http: the frame is already where it needs to be, and
        `generate` sends a URL instead of moving it.
        """
        path = Path(local)
        if self.http and self._asset_url(path) is not None:
            return
        if not path.exists():
            return
        for rep in self.replicas.values():
            self._ensure_remote(rep, path)

    async def _extract_last_frame(self, rep: Replica, remote_mp4: str, job_id: str) -> str | None:
        """Pull the clip's final frame out on the box, and leave it there.

        Measured on P5-1: 0.226s, against 3.4-10.4s to copy the 2.4 MB mp4 down
        and ~1s to push a frame back up. So for a continuous child that lands on
        the same replica as its parent, this removes both legs -- the frame it
        needs to condition on is already a local path on that box.

        Two deployment facts make it this short. There is no ffmpeg on the host,
        but the `h3-game` container has one at /usr/bin/ffmpeg; and the outputs
        directory is bind-mounted at the *same* path inside the container as
        outside, which is already implied by `file_path` from the API being
        directly scp-able. So the mp4 path needs no translation either way.

        PNG rather than JPEG because this frame is what the next clip is
        conditioned on, and the seam is measured in dB -- re-encoding the one
        pixel-exact input we have would spend picture quality to save bytes that
        never travel anywhere.

        Returns None on any failure: the caller still has the local
        download-and-extract path, so this is an optimisation and never a
        dependency.
        """
        remote = f"{self.remote_kf_dir}/{job_id}-last.png"
        tmp = f"{remote}.part"
        # `-sseof -0.2` seeks from the end rather than decoding the whole clip,
        # and `-update 1` keeps ffmpeg from treating the single output as a
        # numbered sequence. Written to `.part` and renamed so a child can never
        # be handed a half-written frame as a URI.
        cmd = [
            "ssh", *self._ssh_opts(rep.alias), rep.alias,
            f"mkdir -p {self.remote_kf_dir} && "
            f"docker exec {settings.h3_container} ffmpeg -y -loglevel error "
            f"-sseof -0.2 -i {remote_mp4} -vframes 1 -update 1 {tmp} && "
            f"mv -f {tmp} {remote}",
        ]
        started = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _, err = await proc.communicate()
        if proc.returncode:
            log.warning(
                "on-box last-frame extraction failed on %s (%s); falling back to "
                "download-then-upload", rep.alias, err.decode("utf-8", "replace")[-200:],
            )
            return None
        log.debug(
            "extracted last frame of %s on %s in %.0fms", job_id, rep.alias,
            (time.perf_counter() - started) * 1000.0,
        )
        return remote

    async def _download(self, rep: Replica, remote: str, dest: Path) -> None:
        cmd = ["scp", "-q", *self._ssh_opts(rep.alias), f"{rep.alias}:{remote}", str(dest)]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _, err = await proc.communicate()
        if proc.returncode:
            raise GpuTransportError(
                f"fetching {remote} from {rep.alias} failed: "
                f"{err.decode('utf-8', 'replace')[-300:]}"
            )
        if not dest.exists() or dest.stat().st_size == 0:
            raise GpuTransportError(f"{remote} came back empty from {rep.alias}")

    async def _fetch(self, rep: Replica, vid: str, dest: Path) -> None:
        """The clip, over the same connection that asked for it.

        Streamed to a `.part` and renamed, for the same reason the ssh upload is:
        `beat.mp4` under `ASSETS_DIR` is being served to a browser by the asset
        server while this runs, and a partial file at the final name is a video
        the player can start and fail to finish.

        `Content-Type` is absent on this endpoint, so the check is the ISO-BMFF
        signature instead. It earns its place: the route is annotated
        `application/json` in the schema, so a build where it describes the file
        rather than returning it would otherwise be a confusing ffprobe error.
        """
        tmp = dest.with_suffix(dest.suffix + ".part")
        size = 0
        try:
            async with self.client.stream(
                "GET", f"{rep.base}/v1/videos/{vid}/content", timeout=settings.gpu_timeout_s
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise GpuError(
                        f"{rep.alias} would not return {vid} ({resp.status_code}): {body[:300]}"
                    )
                head = b""
                with tmp.open("wb") as fh:
                    async for chunk in resp.aiter_bytes(256 * 1024):
                        if len(head) < 32:
                            head += chunk[: 32 - len(head)]
                        size += len(chunk)
                        fh.write(chunk)
        except (httpx.HTTPError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            raise GpuTransportError(f"fetching {vid} from {rep.alias} failed: {exc}") from exc
        if not size:
            tmp.unlink(missing_ok=True)
            raise GpuTransportError(f"{vid} came back empty from {rep.alias}")
        if b"ftyp" not in head:
            tmp.unlink(missing_ok=True)
            raise GpuError(f"{rep.alias} returned {size}B of non-mp4 for {vid}: {head[:32]!r}")
        tmp.replace(dest)

    # -- the API ------------------------------------------------------------ #

    async def health(self, endpoint: str) -> dict[str, Any]:
        rep = self.replicas[endpoint]
        resp = await self.client.get(f"{rep.base}/health", timeout=3.0)
        return {"ok": resp.status_code == 200, "alias": rep.alias, "port": rep.endpoint}

    def _body(self, req: GpuRequest, frames: int, keyframe: str | None) -> dict[str, Any]:
        conditions = []
        if keyframe:
            # `uri` is either a path on the box (ssh transport) or an `http://` URL
            # the worker fetches itself; it takes both, and the caller decides
            # which. `frame_index` is required for a keyframe role and rejected for
            # a reference role. 0 pins the opening frame, which is the one a game
            # wants. VDN serves t2va and fl2va and refuses ref2va outright -- a
            # training limit, not a flag.
            conditions = [
                {"role": "keyframe", "type": "image", "uri": keyframe, "frame_index": 0}
            ]
        body: dict[str, Any] = {
            "prompt": req.ir.to_prompt(),
            "task": "fl2va" if keyframe else "t2va",
            "conditions": conditions,
            "target": {
                "short_edge": req.short_edge,
                "aspect_ratio": settings.aspect_ratio,
                "duration_seconds": frames / settings.h3_fps,
            },
            "flow_shift": settings.h3_flow_shift,
            "audio_flow_shift": settings.h3_audio_flow_shift,
            # A directory prefix, not a file prefix: the server writes
            # `<output_path>/<uuid>.mp4`. Naming it after the job makes the box's
            # outputs sweepable by session instead of by timestamp.
            "output_path": f"{settings.h3_remote_out}/{req.job_id}",
        }
        if req.seed is not None:
            body["seed"] = req.seed
        if req.num_inference_steps > 0:
            # Only when someone explicitly overrode it; see `num_inference_steps`
            # in config.py for why the default is to stay silent.
            body["num_inference_steps"] = req.num_inference_steps
        return body

    async def _submit(self, rep: Replica, body: dict[str, Any]) -> str:
        try:
            resp = await self.client.post(f"{rep.base}/v1/videos", json=body, timeout=60.0)
        except Exception as exc:  # noqa: BLE001
            raise GpuTransportError(f"{rep.alias} unreachable: {exc}") from exc
        if resp.status_code >= 400:
            raise GpuError(f"{rep.alias} rejected the request ({resp.status_code}): {resp.text[:500]}")
        return resp.json()["id"]

    async def _poll(self, rep: Replica, vid: str) -> dict[str, Any]:
        deadline = time.monotonic() + settings.gpu_timeout_s
        while time.monotonic() < deadline:
            try:
                resp = await self.client.get(f"{rep.base}/v1/videos/{vid}", timeout=30.0)
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise GpuTransportError(f"polling {vid} on {rep.alias} failed: {exc}") from exc
            if data.get("status") == "completed":
                return data
            if data.get("status") == "failed":
                raise GpuError(f"{rep.alias} failed {vid}: {str(data.get('error'))[:400]}")
            await asyncio.sleep(settings.h3_poll_s)
        raise GpuTransportError(f"{rep.alias} did not finish {vid} in {settings.gpu_timeout_s:.0f}s")

    async def warm_one(self, endpoint: str) -> None:
        """One throwaway clip at the exact shape this backend will send."""
        rep = self.replicas[endpoint]
        body = self._body(
            GpuRequest(
                job_id=f"warm-{rep.alias}",
                ir=IRSections(description="a wide shot of a quiet harbour at dawn"),
            ),
            frames_for_seconds(settings.beat_seconds),
            None,
        )
        started = time.perf_counter()
        data = await self._poll(rep, await self._submit(rep, body))
        log.info(
            "warmed %s in %.1fs (server %.2fs)", rep.alias,
            time.perf_counter() - started, data.get("inference_time_s") or 0.0,
        )

    async def generate(self, endpoint: str, req: GpuRequest) -> GpuResult:
        rep = self.replicas[endpoint]
        frames = frames_for_seconds(req.seconds)
        seconds = frames / settings.h3_fps
        out_dir = self.root / req.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        video, last, poster = out_dir / "beat.mp4", out_dir / "last.png", out_dir / "poster.jpg"
        timings: dict[str, float] = {}
        started = time.perf_counter()

        keyframe: str | None = None
        served: str | None = None
        already = req.first_frame_remote.get(rep.alias)
        if req.first_frame and self.http:
            served = self._asset_url(Path(req.first_frame))
        if served:
            # A URL, so there is nothing to move and nothing that had to be moved
            # earlier: `gpu_upload_ms` is 0 because the leg does not exist, not
            # because it was won by a prefetch.
            keyframe = served
            timings["gpu_upload_ms"] = 0.0
        elif req.first_frame and already:
            # The parent generated on this same replica and left its last frame
            # here, so there is nothing to move. The common case at 2 replicas:
            # one of each beat's two children lands on the parent's box.
            keyframe = already
            timings["gpu_upload_ms"] = 0.0
            timings["gpu_frame_onbox"] = 1.0
        elif req.first_frame and Path(req.first_frame).exists():
            t0 = time.perf_counter()
            keyframe = await self._ensure_remote(rep, Path(req.first_frame))
            timings["gpu_upload_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        elif req.first_frame:
            log.warning("conditioning frame %s is gone; falling back to t2va", req.first_frame)

        t0 = time.perf_counter()
        vid = await self._submit(rep, self._body(req, frames, keyframe))
        data = await self._poll(rep, vid)
        timings["gpu_server_ms"] = round((data.get("inference_time_s") or 0.0) * 1000.0, 1)
        timings["gpu_onbox_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

        remote = data.get("file_path")
        if not remote and not self.http:
            raise GpuError(f"{rep.alias} completed {req.job_id} with no file_path: {str(data)[:300]}")
        t0 = time.perf_counter()
        remote_last: str | None = None
        if self.http:
            await self._fetch(rep, vid, video)
            # No on-box extraction to pair with it. `_postprocess` writes
            # `last.png` next to the clip a moment from now, and under http that
            # file is addressable by URL from *either* replica -- which is
            # strictly better than a path that only one of them can read.
        else:
            # Concurrent, not sequential: the extraction is 0.23s of the box's time
            # and the download is seconds of the network's, so serialising them
            # would add the whole extraction to the critical path for no reason.
            # The extraction never raises -- it returns None -- so `gather` here
            # cannot lose the download's exception to a sibling failure.
            _, remote_last = await asyncio.gather(
                self._download(rep, remote, video),
                self._extract_last_frame(rep, remote, req.job_id),
            )
        timings["gpu_download_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

        t0 = time.perf_counter()
        stats = await _postprocess(video, last, poster)
        timings["gpu_postprocess_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        timings["gpu_total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)

        rel = f"_h3/{req.job_id}"
        return GpuResult(
            video_url=f"{settings.public_base_url}/{rel}/beat.mp4",
            video_path=str(video),
            last_frame_url=f"{settings.public_base_url}/{rel}/last.png" if last.exists() else None,
            last_frame_path=str(last) if last.exists() else None,
            remote_last_frame={rep.alias: remote_last} if remote_last else {},
            poster_url=f"{settings.public_base_url}/{rel}/poster.jpg" if poster.exists() else None,
            duration_ms=seconds * 1000.0,
            # H3 emits audio with the video; the mux is the server's.
            has_audio=True,
            frame_stats=stats,
            timings=timings,
            endpoint=rep.alias,
        )


class FakeBackend:
    """Synthesise a clip locally so the engine runs without a GPU.

    Produces a slow push-in on the conditioning frame plus a tone whose pitch
    tracks the beat, then extracts the last frame exactly the way the real wrapper
    does -- which means last-frame chaining is genuinely exercised, not mocked.
    """

    allow_degraded_retry = True

    _FONT_CANDIDATES = (
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )

    def __init__(self) -> None:
        self.root = settings.assets_dir / "_fake"
        self.font = next((f for f in self._FONT_CANDIDATES if Path(f).exists()), None)
        self._counter = itertools.count()

    async def generate(self, endpoint: str, req: GpuRequest) -> GpuResult:
        started = time.perf_counter()
        out_dir = self.root / req.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        video = out_dir / "beat.mp4"
        last = out_dir / "last.png"
        poster = out_dir / "poster.jpg"

        # Stand in for GPU time so the frontend sees realistic pending states.
        await asyncio.sleep(settings.fake_gpu_latency_s)

        source = req.first_frame if req.first_frame and Path(req.first_frame).exists() else None
        if source is None:
            raise GpuError("fake backend needs a conditioning frame to animate")

        n = next(self._counter)
        height = req.short_edge
        width = int(height * 16 / 9) // 2 * 2
        tone = 90 + (n % 7) * 30

        base_vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"zoompan=z='min(1+0.0009*on,1.16)':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps=24"
        )
        label = self._label(req)
        attempts: list[list[str]] = []
        if self.font and label:
            drawtext = (
                f",drawtext=fontfile='{self.font}':text='{label}':"
                f"x=(w-text_w)/2:y=h-60:fontsize=20:fontcolor=white@0.85:"
                f"box=1:boxcolor=black@0.45:boxborderw=10"
            )
            attempts.append(self._cmd(source, video, base_vf + drawtext, req.seconds, tone))
        attempts.append(self._cmd(source, video, base_vf, req.seconds, tone))

        last_err = ""
        for cmd in attempts:
            code, err = await _run(cmd)
            if code == 0:
                break
            last_err = err
        else:
            raise GpuError(f"fake ffmpeg failed: {last_err[-400:]}")

        stats = await _postprocess(video, last, poster)

        rel = f"_fake/{req.job_id}"
        return GpuResult(
            video_url=f"{settings.public_base_url}/{rel}/beat.mp4",
            video_path=str(video),
            last_frame_url=f"{settings.public_base_url}/{rel}/last.png" if last.exists() else None,
            last_frame_path=str(last) if last.exists() else None,
            poster_url=f"{settings.public_base_url}/{rel}/poster.jpg" if poster.exists() else None,
            duration_ms=req.seconds * 1000.0,
            has_audio=True,
            frame_stats=stats,
            timings={"gpu_sglang_ms": round((time.perf_counter() - started) * 1000.0, 1)},
            endpoint=endpoint,
        )

    def _cmd(self, src: str, dest: Path, vf: str, seconds: float, tone: int) -> list[str]:
        return [
            settings.ffmpeg, "-y", "-loglevel", "error",
            "-loop", "1", "-i", src,
            "-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=44100",
            "-t", f"{seconds:.2f}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-shortest",
            "-movflags", "+faststart", str(dest),
        ]

    @staticmethod
    def _label(req: GpuRequest) -> str:
        text = req.ir.description[:34].replace("\n", " ")
        # ffmpeg filtergraph metacharacters, escaped for drawtext.
        for a, b in (("\\", ""), (":", "\\:"), ("'", ""), ("%", ""), (",", " ")):
            text = text.replace(a, b)
        return text


async def _postprocess(video: Path, last: Path, poster: Path) -> dict[str, float] | None:
    """Cut the last frame and the poster out of a finished clip, and fingerprint it.

    Shared by both real-clip paths so that last-frame chaining and drift
    measurement are computed identically no matter which backend produced the mp4
    -- if the fake backend and the GPU disagreed here, every drift number
    measured without a GPU would be unusable.
    """
    # No `-frames:v 1`: with `-sseof` that flag grabs the *first* frame of the
    # tail window, not the last frame of the clip. `-update 1` overwrites instead,
    # so the final decoded frame is the one left in the file.
    await _run([settings.ffmpeg, "-y", "-loglevel", "error", "-sseof", "-0.25",
                "-i", str(video), "-update", "1", "-q:v", "2", str(last)])
    if not last.exists():
        await _run([settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
                    "-update", "1", "-q:v", "2", str(last)])
    await _run([settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
                "-frames:v", "1", "-q:v", "3", str(poster)])
    return await asyncio.to_thread(_frame_stats, last) if last.exists() else None


async def _run(cmd: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, err = await proc.communicate()
    return proc.returncode or 0, err.decode("utf-8", "replace")


def _frame_stats(path: Path) -> dict[str, float] | None:
    """Same fingerprint the wrapper computes, so drift logic is backend-agnostic."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    arr = np.asarray(img, dtype="float32") / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    grey = luma
    lap = (
        -4 * grey[1:-1, 1:-1]
        + grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
    )
    return {
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
        "luma": float(luma.mean()), "saturation": float(sat.mean()),
        "contrast": float(luma.std()), "sharpness": float(lap.var()),
    }


# --------------------------------------------------------------------------- #
# Pool                                                                         #
# --------------------------------------------------------------------------- #


class GpuPool:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.backend: Any
        if settings.fake_gpu:
            # One virtual slot per configured wrapper, so queue behaviour matches
            # the real topology even without GPUs.
            self.kind = "fake"
            endpoints = [f"fake://{i}" for i in range(max(1, len(settings.wrapper_urls)))]
            self.backend = FakeBackend()
        elif settings.gpu_backend == "h3":
            self.kind = "h3"
            replicas = read_replicas()
            # The slot identity is the ssh alias, not a URL: the alias is what
            # both the HTTP port and the file transfers are looked up from, and
            # one slot per replica is the real concurrency -- two requests to one
            # box do not interleave, they queue and both get slower.
            endpoints = [r.alias for r in replicas]
            self.backend = H3Backend(client, replicas)
        else:
            self.kind = "wrapper"
            endpoints = list(settings.wrapper_urls)
            self.backend = WrapperBackend(client)
        self.pool = SlotPool(endpoints)
        self.client = client

    @property
    def slots(self) -> int:
        return self.pool.size

    def snapshot(self) -> dict[str, Any]:
        return {"backend": self.kind, "fake": settings.fake_gpu, **self.pool.snapshot()}

    async def bump(self, key: str) -> bool:
        return await self.pool.bump(key, PRIORITY_BLOCKING)

    def prefetch(self, local_path: str | None) -> None:
        """Hint that `local_path` will be some later beat's conditioning frame."""
        fn = getattr(self.backend, "prefetch", None)
        if fn and local_path:
            fn(local_path)

    async def warm(self) -> None:
        """Burn one clip per slot before the first player arrives.

        Acquires every slot first, which both guarantees one warm-up per distinct
        endpoint and keeps the warm-up out of the way of real work. Failures are
        logged and swallowed: an unwarmed pool is slower, not broken.
        """
        fn = getattr(self.backend, "warm_one", None)
        if fn is None:
            return
        held = [await self.pool.acquire(f"warm:{i}", PRIORITY_SPECULATIVE)
                for i in range(self.pool.size)]
        try:
            results = await asyncio.gather(*(fn(ep) for ep in held), return_exceptions=True)
            for ep, res in zip(held, results):
                if isinstance(res, BaseException):
                    log.warning("warming %s failed: %s", ep, res)
        finally:
            for ep in held:
                await self.pool.release(ep)

    async def generate(
        self,
        req: GpuRequest,
        priority: int = PRIORITY_PREGEN,
        on_slot: Callable[[str], None] | None = None,
    ) -> GpuResult:
        """Acquire a slot, generate, retry once degraded on failure.

        `on_slot` fires the moment a slot is granted, which is the queued ->
        generating transition the UI needs; without it the player cannot tell
        "waiting for a GPU" from "the GPU is working".
        """
        endpoint = await self.pool.acquire(req.job_id, priority)
        try:
            if on_slot:
                on_slot(endpoint)
            try:
                return await self.backend.generate(endpoint, req)
            except GpuError as exc:
                transport = isinstance(exc, GpuTransportError)
                if transport:
                    # The box never gave a verdict, so it is the suspect.
                    await self.pool.mark_unhealthy(endpoint)
                if not self.backend.allow_degraded_retry:
                    # On the real deployment a rejection is a configuration error
                    # -- an off-lattice shape, `ref2va`, an OOM -- and retrying it
                    # with different parameters just produces a different
                    # configuration error 8 seconds later. Only a failure that
                    # never reached the model earns another attempt, and then at
                    # identical parameters.
                    if not transport:
                        raise
                    log.warning("transport failure on %s (%s); one identical retry", endpoint, exc)
                    return await self.backend.generate(endpoint, req)
                log.warning("generation failed on %s (%s); retrying degraded", endpoint, exc)
                if not transport:
                    await self.pool.mark_unhealthy(endpoint)
                result = await self.backend.generate(endpoint, req.degraded())
                result.degraded = True
                return result
        finally:
            await self.pool.release(endpoint)

    async def health(self) -> list[dict[str, Any]]:
        out = []
        for ep in self.pool._all:  # noqa: SLF001
            probe = getattr(self.backend, "health", None)
            if probe is None:
                out.append({"endpoint": ep, "ok": True, "fake": True})
                continue
            try:
                out.append({"endpoint": ep, **await probe(ep)})
            except Exception as exc:  # noqa: BLE001
                out.append({"endpoint": ep, "ok": False, "detail": str(exc)})
        return out


def drift_from(baseline: dict[str, float], stats: dict[str, float]) -> dict[str, float]:
    """Deltas against the session baseline, matching bench/chain_drift.py's metrics."""
    def d(key: str) -> float:
        return round(stats.get(key, 0.0) - baseline.get(key, 0.0), 4)

    base_sharp = baseline.get("sharpness", 0.0)
    sharp_pct = (
        round((stats.get("sharpness", 0.0) - base_sharp) / base_sharp * 100.0, 1)
        if base_sharp
        else 0.0
    )
    return {
        "d_luma": d("luma"),
        "d_sat": d("saturation"),
        "d_contrast": d("contrast"),
        "d_rg": round(
            (stats.get("mean_r", 0) - stats.get("mean_g", 0))
            - (baseline.get("mean_r", 0) - baseline.get("mean_g", 0)), 4
        ),
        "d_rb": round(
            (stats.get("mean_r", 0) - stats.get("mean_b", 0))
            - (baseline.get("mean_r", 0) - baseline.get("mean_b", 0)), 4
        ),
        "sharpness_pct": sharp_pct,
    }


# Same thresholds bench/chain_drift.py uses, so a K measured there transfers here
# without reinterpretation.
DRIFT_LIMITS = {"d_luma": 0.06, "d_sat": 0.08, "d_rg": 0.04, "d_rb": 0.04}


def drift_exceeded(drift: dict[str, float]) -> list[str]:
    out = [k for k, lim in DRIFT_LIMITS.items() if abs(drift.get(k, 0.0)) > lim]
    if drift.get("sharpness_pct", 0.0) < -35.0:
        out.append("sharpness")
    return out
