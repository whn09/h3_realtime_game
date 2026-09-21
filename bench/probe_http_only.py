"""Can the whole H3 round trip be HTTP, with no ssh and no scp?

Today it cannot, and the reason is the API style this deployment was built
against: `POST /v1/videos` is sent as JSON with `conditions[].uri` pointing at a
path on the GPU box and `output_path` naming a directory on it, so the
conditioning frame has to be put there first (ssh) and the finished mp4 has to be
fetched from there afterwards (scp). Three legs, two of them ssh.

But the same server also documents a multipart form route on the same path, with
`input_reference` as an uploaded file, plus `GET /v1/videos/{id}/content`. If both
work on this checkpoint, all three legs collapse to HTTP.

Two things could stop it, which is why this is a probe and not a refactor:

  * the multipart route has no `task` and no `conditions`, so what role
    `input_reference` takes is the server's choice, not ours. VDN-H3 serves t2va
    and fl2va and refuses ref2va outright -- if the upload is treated as a
    reference rather than as the frame-0 keyframe, this route cannot drive the
    game's fl2va chaining at all.
  * `/content` is annotated `application/json` in the schema, which is what
    FastAPI says when a handler is typed loosely. Whether it streams mp4 bytes or
    describes them in JSON has to be looked at.

Run it on the box, against a replica's address (its private IP once SGLang binds
0.0.0.0, or a tunnel's local port before that):
    python bench/probe_http_only.py 172.31.45.68:30010 /path/to/a/keyframe.png
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

TARGET = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:30010"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else ""
# A bare port stays meaningful, so the same command works before and after the
# servers moved off loopback.
BASE = f"http://{TARGET if ':' in TARGET else f'127.0.0.1:{TARGET}'}"
PROMPT = (
    "A still mountain valley at dawn, mist over dark pines, slow push in. "
    "Cinematic, 35mm, natural light. Ambient wind, no speech."
)


def multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[bytes, str]:
    b = "----probe" + str(int(time.time()))
    out = bytearray()
    for k, v in fields.items():
        out += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, (name, data) in files.items():
        out += (
            f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        out += data + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={b}"


def get(path: str, raw: bool = False):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
        head = dict(r.headers)
        body = r.read()
    return head, (body if raw else json.loads(body))


print(f"probing {BASE} with {IMAGE or '(no image -- t2va)'}\n")

# Almost everything H3 cares about goes through `extra_body`, not through the form
# fields. The flat fields are an OpenAI-video shape and H3 does not speak it: it
# rejected `seconds` as non-integer, then rejected `num_frames` outright in favour
# of `target.duration_seconds`, which is nested and so unrepresentable as a form
# field at all. So this ends up being the JSON body the live code already builds,
# with the image attached alongside instead of pre-uploaded.
fields = {
    "prompt": PROMPT,
    "seed": "1234",
    "extra_body": json.dumps(
        {
            "task": "fl2va",
            "target": {
                "short_edge": 480,
                "aspect_ratio": "16:9",
                # 345 frames at 24fps: the `5 + 17k` lattice rung this deployment
                # is measured at. Anything off the lattice is silently snapped.
                "duration_seconds": 345 / 24,
            },
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        }
    ),
}
files = {}
if IMAGE:
    with open(IMAGE, "rb") as fh:
        files["input_reference"] = (IMAGE.rsplit("/", 1)[-1], fh.read())
    print(f"  uploading {len(files['input_reference'][1])} bytes as input_reference")

body, ctype = multipart(fields, files)
req = urllib.request.Request(
    f"{BASE}/v1/videos", data=body, method="POST", headers={"Content-Type": ctype}
)
t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        created = json.loads(r.read())
except urllib.error.HTTPError as e:
    detail = e.read().decode("utf-8", "replace")
    print(f"** multipart submit rejected: {e.code}\n{detail[:1200]}")
    raise SystemExit(1)

print(f"  submit accepted in {(time.perf_counter() - t0) * 1000:.0f}ms")
print(f"  {json.dumps(created)[:400]}")
vid = created.get("id") or created.get("video_id")
if not vid:
    raise SystemExit("** no id in the create response")

# Poll. The same endpoint the live code already uses, so nothing new is being
# tested here -- it just has to finish before /content can be looked at.
data: dict = {}
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    _, data = get(f"/v1/videos/{vid}")
    st = data.get("status")
    if st in ("completed", "failed"):
        break
    time.sleep(1.0)
print(f"  status={data.get('status')} after {time.perf_counter() - t0:.1f}s")
if data.get("status") != "completed":
    print(f"** {json.dumps(data)[:1200]}")
    raise SystemExit(1)
print(f"  completion payload: {json.dumps(data)[:600]}")

print("\n--- GET /v1/videos/{id}/content ---")
try:
    head, raw = get(f"/v1/videos/{vid}/content", raw=True)
except urllib.error.HTTPError as e:
    print(f"** {e.code}: {e.read().decode('utf-8', 'replace')[:600]}")
    raise SystemExit(1)
ctype_out = head.get("Content-Type", "?")
print(f"  Content-Type: {ctype_out}")
print(f"  bytes: {len(raw)}")
# `ftyp` in the first 12 bytes is the ISO-BMFF signature: this is a real mp4 and
# not a JSON document describing where one lives.
if b"ftyp" in raw[:16]:
    print("  -> real mp4 bytes. The scp leg can go away.")
else:
    print(f"  -> not mp4. head: {raw[:220]!r}")
