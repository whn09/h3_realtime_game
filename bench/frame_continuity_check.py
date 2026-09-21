"""Does each clip actually start on the image it was conditioned on?

The player reports that a continuous beat sometimes opens on a *different*
picture for a few frames. Two very different causes fit that description and they
are fixed in different files, so this decides between them before anything is
changed:

  * the pipeline handed H3 the wrong image -- then the clip on disk starts on the
    wrong image too, and frame 0 will match something other than the parent's
    last frame (its own keyframe, a sibling's frame, the parent's *opening*);
  * the pipeline was right and the browser painted a stale decoded frame during
    the double-buffer swap -- then frame 0 on disk matches the parent's last
    frame, and nothing in the orchestrator is broken.

So: for every beat, pull frame 0 out of the mp4 and ask which candidate image it
resembles most. Also pull t=0.25s and t=1.0s, because fl2va pins frame 0 but the
model is free to move afterwards, and "wrong for three frames" and "wrong for the
whole clip" are again different bugs.

    python bench/frame_continuity_check.py <session-id>
    python bench/frame_continuity_check.py              # the newest session
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orchestrator"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import settings  # noqa: E402

SID = sys.argv[1] if len(sys.argv) > 1 else ""
TMP = Path(tempfile.mkdtemp(prefix="fcc-"))


def load_session() -> dict:
    root = settings.data_dir / "sessions"
    if SID:
        return json.loads((root / SID / "session.json").read_text())
    newest = max(root.glob("*/session.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(newest.read_text())


def thumb(path: Path) -> np.ndarray | None:
    """A small greyscale thumbnail. Small on purpose: the question is "is this the
    same shot", not "are these bit-identical" -- the conditioning frame is 1280x720
    and the clip is 864x480, so an exact comparison is impossible even when
    everything is correct."""
    try:
        img = Image.open(path).convert("L").resize((64, 36), Image.BILINEAR)
    except Exception:  # noqa: BLE001
        return None
    a = np.asarray(img, dtype="float32") / 255.0
    return (a - a.mean()) / (a.std() + 1e-6)   # contrast-normalised


def frame_at(clip: Path, t: float, tag: str) -> Path | None:
    out = TMP / f"{tag}.png"
    cmd = [settings.ffmpeg, "-y", "-loglevel", "error", "-ss", f"{t}", "-i", str(clip),
           "-frames:v", "1", str(out)]
    if subprocess.run(cmd, capture_output=True).returncode or not out.exists():
        return None
    return out


def score(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Normalised cross-correlation, 1.0 = the same picture. NaN when either side
    is missing, so a missing candidate never wins by default."""
    if a is None or b is None:
        return float("nan")
    return float((a * b).mean())


sess = load_session()
beats = sess["beats"]
print(f"session {sess['id']}  genre={sess.get('genre')!r}  premise={sess.get('premise','')[:40]!r}")
print(f"{len(beats)} beats, {sum(b['status'] == 'ready' for b in beats.values())} ready\n")

rows = []
for bid, b in sorted(beats.items(), key=lambda kv: kv[1]["index"]):
    if b["status"] != "ready" or not b.get("last_frame_path"):
        continue
    clip = Path(b["last_frame_path"]).parent / "beat.mp4"
    if not clip.exists():
        continue
    parent = beats.get(b.get("parent_id") or "", {})
    gp = beats.get(parent.get("parent_id") or "", {})
    trans = (b.get("intent") or {}).get("transition", "?")

    f0 = thumb(p) if (p := frame_at(clip, 0.0, f"{bid}-0")) else None
    f_quarter = thumb(p) if (p := frame_at(clip, 0.25, f"{bid}-q")) else None
    f_one = thumb(p) if (p := frame_at(clip, 1.0, f"{bid}-1")) else None

    # Every image this beat could plausibly have been started from.
    cands: dict[str, np.ndarray | None] = {
        "父末帧": thumb(Path(parent["last_frame_path"])) if parent.get("last_frame_path") else None,
        "自己关键帧": thumb(Path(b["keyframe_path"])) if b.get("keyframe_path") else None,
        "父首帧": thumb(p) if parent.get("last_frame_path")
        and (p := frame_at(Path(parent["last_frame_path"]).parent / "beat.mp4", 0.0, f"{bid}-pf"))
        else None,
        "祖父末帧": thumb(Path(gp["last_frame_path"])) if gp.get("last_frame_path") else None,
    }
    # Siblings, to catch the two branches of one choice being crossed.
    for i, sib in enumerate(parent.get("children", {}).values()):
        sb = beats.get(sib, {})
        if sib != bid and sb.get("keyframe_path"):
            cands[f"兄弟{i}关键帧"] = thumb(Path(sb["keyframe_path"]))

    scored = {k: score(f0, v) for k, v in cands.items()}
    ranked = sorted((v, k) for k, v in scored.items() if v == v)
    best_v, best_k = (ranked[-1] if ranked else (float("nan"), "?"))
    expect = "父末帧" if trans == "continuous" and parent else "自己关键帧"
    ok = best_k == expect
    rows.append((bid, trans, expect, best_k, best_v, scored, score(f_quarter, cands.get(expect)),
                 score(f_one, cands.get(expect)), ok))

print(f"{'beat':>12} {'transition':<11} {'应该起自':<10} {'实际最像':<12} {'相关':>6} "
      f"{'t=.25':>6} {'t=1.0':>6}")
bad = []
for bid, trans, expect, best_k, best_v, scored, q, one, ok in rows:
    flag = "" if ok else "   <<< 不符"
    print(f"{bid:>12} {trans:<11} {expect:<10} {best_k:<12} {best_v:>6.3f} "
          f"{q:>6.3f} {one:>6.3f}{flag}")
    if not ok:
        bad.append((bid, trans, expect, best_k, scored))

print()
if not bad:
    print("每一拍的第 0 帧都最像它应该起自的那张图 -> 管线给对了，问题不在编排器")
else:
    print(f"{len(bad)} 拍的第 0 帧不是它应该起自的图：")
    for bid, trans, expect, best_k, scored in bad:
        print(f"  {bid} ({trans}) 应该是 {expect}，却最像 {best_k}")
        print("    " + "  ".join(f"{k}={v:.3f}" for k, v in scored.items() if v == v))
print(f"\n（帧图留在 {TMP}）")
