"""Does one real beat generate with no ssh anywhere in it?

Runs `H3Backend.generate` directly against a replica -- the same method the live
service calls -- with the asset server this change adds standing in for the live
one. What it proves that `probe_uri_fetch.py` did not: the URL is built by
`_asset_url` from a path the engine would actually hand over, the clip arrives
through `_fetch`, and `last.png` is produced locally, so the *next* beat in a
chain has a conditioning frame without anything having been left on the box.

It never enters the engine, so no session is created and the player's history is
untouched; the clip lands in `ASSETS_DIR/_h3/httpcheck-*/` and can be deleted.

    python bench/http_transport_check.py                  # first replica
    python bench/http_transport_check.py P5-2 /path/kf.png
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orchestrator"))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from app.config import settings  # noqa: E402
from app.gpu import GpuRequest, H3Backend, _lan_ip, read_replicas  # noqa: E402
from app.main import _assets_app  # noqa: E402
from app.schema import IRSections  # noqa: E402

WANT_ALIAS = sys.argv[1] if len(sys.argv) > 1 else ""
GIVEN_KF = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def newest_keyframe() -> Path:
    """Any real keyframe under ASSETS_DIR will do -- this is about transport, not
    about the picture. Newest because the oldest sessions may predate a layout
    change."""
    cands = sorted(
        settings.assets_dir.glob("*/kf-*.png"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not cands:
        raise SystemExit(f"no keyframes under {settings.assets_dir}; pass one as argv[2]")
    return cands[0]


async def main() -> int:
    kf = GIVEN_KF or newest_keyframe()
    reps = read_replicas()
    rep = next((r for r in reps if r.alias == WANT_ALIAS), reps[0])
    print(f"replica {rep.alias} at {rep.endpoint}\nkeyframe {kf} ({kf.stat().st_size} bytes)")

    # This process's own copy of the asset server, on a free port, so the check
    # does not need the live service to be running the new code yet.
    server = uvicorn.Server(
        uvicorn.Config(_assets_app(), host="0.0.0.0", port=0, log_level="warning", access_log=False)
    )
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]

    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.gpu_timeout_s))
    backend = H3Backend(client, [rep])
    ip = _lan_ip()
    if not ip:
        raise SystemExit("no routable address for this host, so H3 cannot fetch from it")
    backend.internal_base = f"http://{ip}:{port}"
    url = backend._asset_url(kf)
    print(f"serving it at {url}\n")
    if not url:
        raise SystemExit(f"{kf} is not under ASSETS_DIR ({settings.assets_dir})")

    req = GpuRequest(
        job_id=f"httpcheck-{int(time.time())}",
        ir=IRSections(
            description="A quiet valley at dawn, mist over dark pines, slow push in.",
            soundscape="wind, distant birds",
        ),
        first_frame=str(kf),
        seed=1234,
    )
    started = time.perf_counter()
    try:
        res = await backend.generate(rep.alias, req)
    finally:
        server.should_exit = True
        await serving
        await client.aclose()

    print(f"generated in {time.perf_counter() - started:.1f}s")
    for k in ("gpu_upload_ms", "gpu_server_ms", "gpu_onbox_ms", "gpu_download_ms",
              "gpu_postprocess_ms", "gpu_total_ms"):
        print(f"  {k:22} {res.timings.get(k, float('nan')):>9.0f}")

    bad = 0
    video = Path(res.video_path)
    if not video.exists() or video.stat().st_size < 100_000:
        print(f"** clip is {video.stat().st_size if video.exists() else 0} bytes"); bad += 1
    else:
        print(f"  clip {video.stat().st_size} bytes at {video}")
    # The chaining input. Without it every continuous child silently becomes a
    # fresh keyframe, which is the drift this project spends the most on avoiding.
    if not res.last_frame_path or not Path(res.last_frame_path).exists():
        print("** no last frame, so the next beat has nothing to continue from"); bad += 1
    else:
        nxt = backend._asset_url(Path(res.last_frame_path))
        print(f"  last frame {Path(res.last_frame_path).stat().st_size} bytes")
        if not nxt:
            print("** and it is not reachable by URL, so the child would need an upload"); bad += 1
        else:
            print(f"  reachable by the next beat at {nxt}")
    # Any leftover in `.part` means a rename was skipped, and a browser could be
    # handed a truncated clip.
    strays = list(video.parent.glob("*.part"))
    if strays:
        print(f"** partial files left behind: {strays}"); bad += 1
    if backend._uploads:
        print(f"** {len(backend._uploads)} ssh upload(s) happened: {list(backend._uploads)}"); bad += 1
    else:
        print("  no ssh uploads")

    print("\nclean" if not bad else f"\n{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
