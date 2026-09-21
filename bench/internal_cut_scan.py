"""Does each clip *stay* on the picture it opened on, or does H3 cut inside it?

`bench/frame_continuity_check.py` answers a neighbouring question -- is frame 0 the
image the beat was conditioned on -- and it provably cannot see this artefact:
fl2va *pins* frame 0, so the answer there is always yes, and the guarantee says
nothing whatsoever about frame 30. Meanwhile the player's complaint ("开头几帧跟
后边连不上") is precisely about frame 30. This is the instrument for that.

The trap, found by getting it wrong once and catching it before reporting: it is
tempting to correlate frame 0 against a late frame and call a low score a cut. That
flags 20 of 21 beats, because a 14-second moving camera falls below any fixed
correlation threshold on motion alone. A cut is not a low score -- it is a *step*:
one adjacent frame pair that collapses while both its neighbours sit at ~0.99. So
the discriminator is the adjacent-frame series, and slow drift is whatever is left
over once the steps are accounted for. The two are reported separately because they
have different fixes (a cut is a prompt/conditioning problem, drift is a re-anchor
problem).

One ffmpeg decode per clip -- all 345 frames out at once as 64x36 grey rawvideo,
about 800KB -- and everything after that is numpy. The obvious shape, one decode per
sampled frame, is ~50x slower and answers less.

    python bench/internal_cut_scan.py <session-id>
    python bench/internal_cut_scan.py              # the newest session
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orchestrator"))

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402

SID = sys.argv[1] if len(sys.argv) > 1 else ""
W, H = 64, 36
FPS = 24.0
# An adjacent pair below this is a cut. Two frames 1/24s apart score >0.97 even on a
# fast pan, and a scene change lands near zero, so nothing here is decided by the
# exact value -- anywhere in 0.6..0.9 gives the same verdict on the same clips.
STEP = 0.80
# Only a step in the first N frames is the artefact being hunted. A cut late in a
# clip is the model cutting within its own shot, which is ordinary film grammar; a
# cut 1.4s in means the frame we paid to condition on was thrown away, which is the
# bug. 96 frames = 4s.
EARLY = 96


def load_session() -> dict:
    root = settings.data_dir / "sessions"
    if SID:
        return json.loads((root / SID / "session.json").read_text())
    newest = max(root.glob("*/session.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(newest.read_text())


def frames(mp4: Path) -> np.ndarray | None:
    """Every frame of the clip as contrast-normalised 64x36 rows, ready to dot."""
    p = subprocess.run(
        [settings.ffmpeg, "-loglevel", "error", "-i", str(mp4),
         "-vf", f"scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        capture_output=True,
    )
    if p.returncode or not p.stdout:
        return None
    n = len(p.stdout) // (W * H)
    a = np.frombuffer(p.stdout[: n * W * H], dtype=np.uint8).reshape(n, H * W).astype("float32")
    a /= 255.0
    a -= a.mean(axis=1, keepdims=True)
    a /= a.std(axis=1, keepdims=True) + 1e-6
    # Row-normalised, so a dot product averaged over pixels *is* the correlation.
    return a


def main() -> int:
    session = load_session()
    beats = session["beats"]

    print(f"{'beat':>12} {'转场':<11} {'帧':>4} {'f0~f_end':>9} {'最小逐帧跳变':>13} "
          f"{'在第几帧':>8} {'早段硬切':>9}")
    hard: list[tuple] = []
    drift_only: list[tuple] = []
    for bid, b in sorted(beats.items(), key=lambda kv: kv[1]["index"]):
        if b["status"] != "ready" or not b.get("last_frame_path"):
            continue
        mp4 = Path(b["last_frame_path"]).parent / "beat.mp4"
        if not mp4.exists():
            continue
        a = frames(mp4)
        if a is None or len(a) < 8:
            continue
        trans = (b.get("intent") or {}).get("transition", "?")
        adj = (a[1:] * a[:-1]).mean(axis=1)   # one value per frame boundary
        i = int(adj.argmin())                 # the boundary between frame i and i+1
        lo = float(adj[i])
        end = float((a[0] * a[-1]).mean())
        steps = [int(j) + 1 for j in np.flatnonzero(adj < STEP)]
        early = [j for j in steps if j <= EARLY]
        mark = f"{early[0]}帧/{early[0] / FPS:.2f}s" if early else "—"
        print(f"{bid:>12} {trans:<11} {len(a):>4} {end:>9.3f} {lo:>13.3f} "
              f"{i + 1:>8} {mark:>9}")
        if early:
            hard.append((bid, trans, bool(b.get("keyframe_path")), early[0], len(steps)))
        elif end < 0.5:
            drift_only.append((bid, trans, end))

    n = sum(1 for b in beats.values() if b["status"] == "ready" and b.get("last_frame_path"))
    print()
    print(f"{len(hard)}/{n} 拍在开场 4s 内有真正的硬切（逐帧跳变 < {STEP}）")
    for bid, trans, kf, at, total in hard:
        print(f"  {bid} 转场={trans} 关键帧={'有' if kf else '—'}  "
              f"开场画面活了 {at} 帧 / {at / FPS:.2f}s，全片共 {total} 处跳变")
    print(f"{len(drift_only)}/{n} 拍没有硬切、只是画面一路漂走（f0~f_end < 0.5）")
    for bid, trans, end in drift_only:
        print(f"  {bid} 转场={trans}  f0~f_end={end:.3f}")
    # Not an exit code: a session with a cut in it is a finding to read, not a
    # failing test. `bench/prepare_race_check.py` is the one that returns 1.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
