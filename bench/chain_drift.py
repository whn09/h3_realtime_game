#!/usr/bin/env python3
"""P0 checklist items 6 and 7 -- does last-frame chaining actually hold up?

The continuity mechanism in DESIGN.md section 3.1 feeds each beat's final frame
in as the next beat's `frame_index: 0` keyframe. That is a feedback loop, and
feedback loops drift. This script runs the loop for real and measures it:

  #6  Colour cast, contrast, saturation and sharpness across N chained beats,
      relative to the seed frame. The answer sets the re-anchoring interval K
      (section 3.3) -- guessing K=6 without data is exactly the kind of number
      that turns out to be 3 or 20.
  #7  Audio seams at beat boundaries. The clips are concatenated so the joins
      can be listened to; if they are obvious, the persistent music bed in
      section 3.5 is mandatory rather than a nice-to-have.

Deliberately keeps the scene fixed across beats: the only thing varying is the
number of chaining hops, so drift is attributable to the chaining itself and
not to the story moving somewhere new.

    python chain_drift.py --wrapper http://127.0.0.1:8000 --beats 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx

from common import make_test_keyframe, print_table, write_csv, write_json

# One continuous scene, one action per beat, same subject and location
# throughout -- the controlled condition drift has to show up against.
DEFAULT_BEATS = [
    "固定镜头，中景。黄昏的山脊上，旅人站定，风吹动他的外套下摆。\n"
    "开阔地的风声，布料抖动，远处零星鸟鸣。\n"
    "稀疏的低音弦乐，单一持续的大提琴长音，60 BPM，小调。",

    "固定镜头，中景。旅人缓缓转头，望向身后的山谷。\n"
    "风声持续，衣料摩擦，脚下碎石轻响。\n"
    "稀疏的低音弦乐，单一持续的大提琴长音，60 BPM，小调。",

    "固定镜头，中景。旅人抬起右手，遮住斜射的夕阳。\n"
    "风声持续，布料抖动。\n"
    "稀疏的低音弦乐，单一持续的大提琴长音，60 BPM，小调。",

    "固定镜头，中景。旅人向前走出一步，停在山脊边缘。\n"
    "风声加强，脚下碎石滚落。\n"
    "稀疏的低音弦乐，单一持续的大提琴长音，60 BPM，小调。",

    "固定镜头，中景。旅人低头看向脚下的深谷，肩膀微微起伏。\n"
    "风声持续，呼吸声轻微。\n"
    "稀疏的低音弦乐，单一持续的大提琴长音，60 BPM，小调。",

    "固定镜头，中景。旅人重新站直，把外套的领子拉紧。\n"
    "风声持续，布料摩擦。\n"
    "稀疏的低音弦乐，单一持续的大提琴长音，60 BPM，小调。",

    "固定镜头，中景。旅人转身背向镜头，面朝远处的山路。\n"
    "风声持续，脚步声两下。\n"
    "稀疏的低音弦乐，单一持续的大提琴长音，60 BPM，小调。",

    "固定镜头，中景。旅人沿山脊缓步走远，身影在夕阳中变小。\n"
    "风声渐远，脚步声规律。\n"
    "稀疏的低音弦乐，单一持续的大提琴长音，60 BPM，小调。",
]


async def generate_beat(
    client: httpx.AsyncClient, wrapper: str, prompt: str, first_frame: str,
    index: int, args: argparse.Namespace,
) -> dict[str, Any]:
    body = {
        "job_id": f"chain-{args.run_id}-{index:02d}",
        "prompt": prompt,
        "task": "fl2va",
        "first_frame": first_frame,
        "seconds": args.seconds,
        "short_edge": args.short_edge,
        "quality": args.quality,
        "num_inference_steps": args.steps,
        # Varying the seed per beat mirrors production (we do not reuse seeds
        # across beats), so the drift measured here is the drift users get.
        "seed": 1000 + index,
        "archive": False,
    }
    resp = await client.post(f"{wrapper.rstrip('/')}/generate", json=body)
    if resp.status_code != 200:
        raise RuntimeError(f"beat {index} failed: http {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def drift_row(index: int, stats: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def delta(key: str) -> float:
        return round(stats[key] - baseline[key], 4)

    def rel(key: str) -> float:
        base = baseline[key]
        return round((stats[key] - base) / base * 100.0, 1) if base else float("nan")

    return {
        "beat": index,
        "luma": round(stats["luma"], 4),
        "d_luma": delta("luma"),
        "saturation": round(stats["saturation"], 4),
        "d_sat": delta("saturation"),
        "contrast": round(stats["contrast"], 4),
        "d_contrast": delta("contrast"),
        "sharpness": round(stats["sharpness"], 6),
        "sharpness_pct": rel("sharpness"),
        # Colour cast: how far the channel balance has moved from the seed.
        "d_rg": round((stats["mean_r"] - stats["mean_g"]) - (baseline["mean_r"] - baseline["mean_g"]), 4),
        "d_rb": round((stats["mean_r"] - stats["mean_b"]) - (baseline["mean_r"] - baseline["mean_b"]), 4),
    }


def build_artifacts(frames: list[Path], videos: list[Path], out_dir: Path) -> None:
    """Contact sheet for eyeballing drift, concatenated clip for hearing seams."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if frames:
        # `tile` combines successive frames of ONE input, so stage the last
        # frames as a numbered image sequence and let the image2 demuxer feed
        # them in order. Trying to tile across multiple -i inputs does not work.
        staging = out_dir / "_frames"
        staging.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            shutil.copyfile(frame, staging / f"frame_{i:03d}.png")

        cols = min(4, len(frames))
        rows = (len(frames) + cols - 1) // cols
        sheet = out_dir / "drift_contact_sheet.png"
        result = subprocess.run(
            ["ffmpeg", "-y", "-framerate", "1", "-i", str(staging / "frame_%03d.png"),
             "-vf", f"scale=480:270,tile={cols}x{rows}",
             "-frames:v", "1", str(sheet)],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"  contact sheet (beat 0 -> {len(frames) - 1}, left to right): {sheet}")
        else:
            print(f"  contact sheet failed: {result.stderr.decode()[-400:]}")

    if videos:
        listing = out_dir / "concat.txt"
        listing.write_text(
            "".join(f"file '{v.resolve()}'\n" for v in videos), encoding="utf-8"
        )
        joined = out_dir / "drift_chain.mp4"
        # Hard cuts with no crossfade on purpose: this is the worst case, so any
        # audible seam here is the seam the music bed has to cover.
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", str(joined)],
            capture_output=True,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                 "-c:v", "libx264", "-c:a", "aac", str(joined)],
                capture_output=True,
            )
        if result.returncode == 0:
            print(f"  chained video (listen to the joins): {joined}")
        else:
            print(f"  concat failed: {result.stderr.decode()[-400:]}")


async def amain() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", default="http://127.0.0.1:8000")
    ap.add_argument("--beats", type=int, default=8)
    ap.add_argument("--seed-image", default=None,
                    help="opening keyframe; synthesised if omitted")
    ap.add_argument("--prompts-file", default=None, help="JSONL or newline-separated IR blocks")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--short-edge", type=int, default=480)
    ap.add_argument("--quality", default="high")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default="results/chain_drift")
    args = ap.parse_args()

    args.run_id = args.run_id or str(int(__import__("time").time()))
    out_dir = Path(args.out_dir) / args.run_id

    prompts = DEFAULT_BEATS
    if args.prompts_file:
        raw = Path(args.prompts_file).read_text(encoding="utf-8")
        prompts = [json.loads(l)["prompt"] if l.strip().startswith("{") else l
                   for l in raw.split("\n\n") if l.strip()]
    prompts = (prompts * ((args.beats // len(prompts)) + 1))[:args.beats]

    seed = (
        Path(args.seed_image) if args.seed_image
        else make_test_keyframe(Path("/tmp/kunlun-bench/keyframe.png"))
    )
    print(f"run {args.run_id}: {args.beats} beats, seed={seed}")

    rows: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    frames: list[Path] = []
    videos: list[Path] = []
    baseline: dict[str, Any] | None = None
    first_frame = str(seed.resolve())

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for i, prompt in enumerate(prompts):
            print(f"\n--- beat {i} (chaining from {'seed' if i == 0 else f'beat {i - 1}'})")
            try:
                data = await generate_beat(client, args.wrapper, prompt, first_frame, i, args)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED: {exc}")
                rows.append({"beat": i, "error": str(exc)[:300]})
                break

            raw_results.append(data)
            stats = data.get("frame_stats")
            timings = data["timings"]
            media = data["media"]
            print(f"  sglang={timings['sglang_ms']:.0f}ms audio={media.get('has_audio')} "
                  f"{media.get('width')}x{media.get('height')} url={data['video_url']}")

            if not stats:
                print("  no frame_stats returned (analyze disabled?); drift cannot be measured")
                break
            if baseline is None:
                baseline = stats
            rows.append(drift_row(i, stats, baseline))

            if data.get("last_frame_path"):
                frames.append(Path(data["last_frame_path"]))
            videos.append(Path(data["video_path"]))

            nxt = data.get("last_frame_path")
            if not nxt:
                print("  no last frame returned; the chain cannot continue")
                break
            first_frame = nxt

    write_csv(out_dir / "drift.csv", rows)
    write_json(out_dir / "raw.json", raw_results)

    print_table(
        "DRIFT VS SEED FRAME", [r for r in rows if "error" not in r],
        ["beat", "luma", "d_luma", "saturation", "d_sat", "contrast", "d_contrast",
         "sharpness_pct", "d_rg", "d_rb"],
    )

    build_artifacts(frames, videos, out_dir)

    print("\nDECISIONS UNLOCKED")
    good = [r for r in rows if "error" not in r and r["beat"] > 0]
    if not good:
        print("  not enough beats completed to judge drift")
        return 1

    # Thresholds are deliberately conservative: these are the points at which a
    # side-by-side comparison starts to read as "different footage".
    limits = {"d_luma": 0.06, "d_sat": 0.08, "d_rg": 0.04, "d_rb": 0.04}
    first_breach: int | None = None
    for row in good:
        breached = [k for k, lim in limits.items() if abs(row[k]) > lim]
        if row["sharpness_pct"] == row["sharpness_pct"] and row["sharpness_pct"] < -35:
            breached.append("sharpness")
        if breached and first_breach is None:
            first_breach = row["beat"]
            print(f"  #6 drift becomes significant at beat {row['beat']} ({', '.join(breached)})")

    if first_breach is None:
        last = good[-1]
        print(f"  #6 no significant drift through beat {last['beat']} -- the assumed "
              f"re-anchoring interval K=6 is safe, and could likely be relaxed")
    else:
        print(f"  -> set the re-anchoring interval K = {max(2, first_breach - 1)} "
              "(DESIGN.md section 3.3), i.e. force a `cut` with a freshly generated "
              "keyframe at least that often")

    sharp = [r["sharpness_pct"] for r in good if r["sharpness_pct"] == r["sharpness_pct"]]
    if sharp:
        print(f"  sharpness change by the last beat: {sharp[-1]:+.0f}% vs seed")
        if sharp[-1] < -20:
            print("     -> softening accumulates; consider a light unsharp pass on the "
                  "extracted keyframe before feeding it back")

    print("  #7 open drift_chain.mp4 and listen at each 15s boundary. If the music "
          "restart is audible, the persistent music bed in DESIGN.md section 3.5 is "
          "required, not optional.")

    print(f"\nartifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
