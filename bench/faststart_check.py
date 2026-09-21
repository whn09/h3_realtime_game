"""Does the faststart remux do what it claims, on a real H3 clip?

Copies an existing beat.mp4 aside (never touches the archive), then checks three
things that together are the whole feature:

  * the scanner agrees with `ffprobe`/the atom order about whether work is needed;
  * after the remux `moov` comes first, the duration is unchanged, and the frame
    count is unchanged -- `-c copy` must not have re-encoded or truncated anything;
  * a second pass is a no-op, so a clip already written faststart costs nothing.

    set -a; . /home/ubuntu/kunlun/.env; set +a
    ./.venv/bin/python bench/faststart_check.py [<path to a beat.mp4>]

Exit code is 0 only if all three hold.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orchestrator"))

from app.config import settings          # noqa: E402
from app.gpu import _faststart, _moov_after_mdat   # noqa: E402


def atoms(path: Path, limit: int = 6) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    with path.open("rb") as fh:
        off = 0
        while len(out) < limit:
            head = fh.read(8)
            if len(head) < 8:
                break
            size = int.from_bytes(head[:4], "big")
            out.append((head[4:8].decode("latin1"), off))
            if size == 1:
                size = int.from_bytes(fh.read(8), "big")
            if size < 8:
                break
            off += size
            fh.seek(off)
    return out


def probe(path: Path) -> dict[str, str]:
    cmd = [settings.ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error",
           "-count_frames", "-select_streams", "v:0",
           "-show_entries", "stream=nb_read_frames,codec_name,width,height",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    stream = (data.get("streams") or [{}])[0]
    return {
        "codec": stream.get("codec_name", "?"),
        "size": f"{stream.get('width')}x{stream.get('height')}",
        "frames": str(stream.get("nb_read_frames", "?")),
        "duration": f"{float(data['format']['duration']):.3f}",
    }


def newest_clip() -> Path | None:
    root = settings.assets_dir / "_h3"
    clips = sorted(root.glob("*/beat.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return clips[0] if clips else None


async def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_clip()
    if not src or not src.exists():
        print(f"no clip to test (looked in {settings.assets_dir / '_h3'})")
        return 1

    work = Path("/tmp/faststart_check.mp4")
    shutil.copy2(src, work)
    print(f"source   {src}  ({src.stat().st_size / 1e6:.2f}MB)")
    print(f"atoms    {' '.join(f'{k}@{o}' for k, o in atoms(work))}")

    before = probe(work)
    needed = _moov_after_mdat(work)
    print(f"needs remux? {needed}")

    t0 = time.perf_counter()
    did = await _faststart(work)
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"remuxed? {did} in {ms:.0f}ms -> {work.stat().st_size / 1e6:.2f}MB")
    print(f"atoms    {' '.join(f'{k}@{o}' for k, o in atoms(work))}")

    after = probe(work)
    again = _moov_after_mdat(work)
    twice = await _faststart(work)

    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'ok ' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")

    print("\nchecks")
    check("scanner and remux agree", needed == did, f"needed={needed} did={did}")
    check("moov now leads", not again)
    check("second pass is a no-op", not twice)
    check("stream untouched", before == after, f"{before} -> {after}")
    check("original archive untouched", _moov_after_mdat(src) == needed)

    work.unlink(missing_ok=True)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
