#!/usr/bin/env python3
"""P0 checklist items 1, 2, 9 -- discover SGLang's actual contract.

Answers, with evidence rather than assumption:

  #1  Is `/v1/videos` synchronous or job-based? What shape is the response, and
      is the video returned as a URL, a local path, or base64?
  #2  Can an instance launched with `--model-variant fl2va` serve a bare t2va
      request (no conditions)? If not, every opening beat must get its keyframe
      from an external image model -- which is what DESIGN.md already assumes,
      so a "no" here confirms the design rather than breaking it.
  #9  Is `seconds=15` (the documented maximum) actually stable? Upper bounds
      are a classic source of off-by-one frame-count bugs.

Run this ON the GPU box, against h3-wrapper's /probe endpoint, which forwards
payloads untouched and echoes the raw response.

    python probe_sglang.py --wrapper http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

from common import make_test_keyframe, print_table, write_json

MODEL = "MiniMaxAI/MiniMax-H3"


def cases(keyframe_uri: str, short_edge: int) -> list[dict[str, Any]]:
    def base(seconds: float) -> dict[str, Any]:
        return {
            "model": MODEL,
            "seconds": seconds,
            "target": {
                "short_edge": short_edge,
                "aspect_ratio": "16:9",
                "duration_seconds": seconds,
            },
            "quality": "high",
            "num_inference_steps": 50,
            "num_outputs_per_prompt": 1,
            "seed": 42,
        }

    keyframe = [
        {"type": "image", "uri": keyframe_uri, "role": "keyframe", "frame_index": 0}
    ]
    prompt = (
        "A lone figure stands on a ridge at dusk, wind moving their coat, "
        "the camera slowly pushing in. Distant wind and cloth rustle. "
        "Sparse low strings, 60 BPM, minor key."
    )

    return [
        {
            "id": "t2va_bare",
            "asks": "#2 does this variant accept t2va with no conditions?",
            "payload": {**base(8), "task": "t2va", "prompt": prompt},
        },
        {
            "id": "i2va_first_frame",
            "asks": "#1 baseline: the shape every gameplay beat will use",
            "payload": {**base(8), "task": "fl2va", "prompt": prompt, "conditions": keyframe},
        },
        {
            "id": "i2va_seconds_15",
            "asks": "#9 is the documented maximum duration stable?",
            "payload": {**base(15), "task": "fl2va", "prompt": prompt, "conditions": keyframe},
        },
        {
            "id": "i2va_seconds_4",
            "asks": "#9 is the documented minimum duration stable?",
            "payload": {**base(4), "task": "fl2va", "prompt": prompt, "conditions": keyframe},
        },
        {
            "id": "i2va_two_outputs",
            "asks": "does num_outputs_per_prompt=2 beat two separate calls?",
            "payload": {
                **base(8), "task": "fl2va", "prompt": prompt,
                "conditions": keyframe, "num_outputs_per_prompt": 2,
            },
        },
    ]


def classify(result: dict[str, Any]) -> tuple[str, str]:
    """Reduce a raw response to (sync|async|error, url|path|b64|none)."""
    if result.get("status_code", 500) >= 400:
        return "error", "none"

    raw = result.get("raw")
    located = [o for o in result.get("located_outputs", []) if "error" not in o]
    if located:
        # A terminal response that already carries the video is synchronous.
        return "sync", located[0].get("kind", "unknown")

    status = None
    if isinstance(raw, dict):
        status = raw.get("status") or raw.get("state")
    if status:
        return f"async(status={status})", "none"
    return "unknown", "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", default="http://127.0.0.1:8000",
                    help="h3-wrapper base URL (its /probe forwards raw payloads)")
    ap.add_argument("--keyframe", default=None,
                    help="existing keyframe path on this host; synthesised if omitted")
    ap.add_argument("--short-edge", type=int, default=480)
    ap.add_argument("--out", default="results/probe_sglang.json")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--only", default=None, help="run a single case by id")
    args = ap.parse_args()

    keyframe_path = (
        Path(args.keyframe) if args.keyframe
        else make_test_keyframe(Path("/tmp/kunlun-bench/keyframe.png"))
    )
    if not keyframe_path.exists():
        print(f"keyframe not found: {keyframe_path}")
        return 2
    keyframe_uri = f"file://{keyframe_path.resolve()}"
    print(f"keyframe: {keyframe_uri}")

    selected = cases(keyframe_uri, args.short_edge)
    if args.only:
        selected = [c for c in selected if c["id"] == args.only]

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    with httpx.Client(timeout=args.timeout) as client:
        for case in selected:
            print(f"\n=== {case['id']}  ({case['asks']})")
            try:
                resp = client.post(f"{args.wrapper.rstrip('/')}/probe", json=case["payload"])
                result = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {
                    "status_code": resp.status_code, "raw": resp.text[:20000]
                }
            except Exception as exc:  # noqa: BLE001
                result = {"status_code": 0, "error": str(exc), "raw": None}

            mode, payload_kind = classify(result)
            record = {
                "id": case["id"],
                "asks": case["asks"],
                "request": case["payload"],
                "status_code": result.get("status_code"),
                "elapsed_ms": result.get("elapsed_ms"),
                "mode": mode,
                "video_payload_kind": payload_kind,
                "shape": result.get("shape"),
                "located_outputs": result.get("located_outputs"),
                "raw": result.get("raw"),
                "error": result.get("error"),
            }
            results.append(record)

            print(f"    http={record['status_code']} mode={mode} payload={payload_kind} "
                  f"elapsed={record['elapsed_ms']}ms")
            if record["error"]:
                print(f"    error: {record['error']}")
            elif record["status_code"] and record["status_code"] >= 400:
                print(f"    body: {json.dumps(record['raw'])[:600]}")
            else:
                print(f"    shape: {json.dumps(record['shape'])[:600]}")

            summary_rows.append({
                "case": case["id"],
                "http": record["status_code"],
                "mode": mode,
                "payload": payload_kind,
                "ms": record["elapsed_ms"],
            })

    write_json(Path(args.out), results)
    print_table("SUMMARY", summary_rows, ["case", "http", "mode", "payload", "ms"])

    # --- turn observations into the decisions they gate ---------------------
    by_id = {r["id"]: r for r in results}
    print("\nDECISIONS UNLOCKED")

    bare = by_id.get("t2va_bare")
    if bare:
        ok = bare["status_code"] == 200 and bare["video_payload_kind"] != "none"
        print(f"  #2 fl2va instance serves bare t2va: {'YES' if ok else 'NO'}")
        print("     -> " + (
            "opening beat can skip the external image model (simpler than designed)"
            if ok else
            "opening beat needs an externally generated keyframe, as DESIGN.md assumes"
        ))

    baseline = by_id.get("i2va_first_frame")
    if baseline and baseline["status_code"] == 200:
        print(f"  #1 endpoint is {baseline['mode']}, video returned as "
              f"{baseline['video_payload_kind']}")
        if baseline["video_payload_kind"] == "b64":
            print("     -> base64 means an extra decode+write per beat; measure it, and "
                  "prefer a path/URL response if SGLang can be configured for one")

    for cid, label in (("i2va_seconds_15", "max"), ("i2va_seconds_4", "min")):
        case = by_id.get(cid)
        if case:
            ok = case["status_code"] == 200 and case["video_payload_kind"] != "none"
            print(f"  #9 seconds {label} bound: {'OK' if ok else 'FAILS'}")
            if not ok and label == "max":
                print("     -> the 19.3s window loses its 15s clip; fall back to 14s "
                      "and re-check the budget in DESIGN.md section 2")

    print(f"\nfull raw responses: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
