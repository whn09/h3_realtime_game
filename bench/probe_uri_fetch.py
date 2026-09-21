"""Will H3 fetch a conditioning frame over HTTP, instead of reading a local path?

This is the last leg. `POST /v1/videos` already works over HTTP, and once SGLang
binds 0.0.0.0 it works without a tunnel. But `fl2va` needs `conditions[].uri`, and
today that is a path on the GPU box's own filesystem -- which is the only reason
the orchestrator ssh's each keyframe over before submitting. The multipart route's
`input_reference` upload does not satisfy it: the server still answers
"conditions requires at least one entry for task 'fl2va'".

So: serve the image from this box over HTTP and hand H3 a URL. If it fetches it,
the upload leg becomes a URL in a JSON body and ssh leaves the submit path
entirely. If it insists on a local path, the upload has to stay however pretty the
rest gets.

Run on the orchestrator box:
    python bench/probe_uri_fetch.py 172.31.45.68:30010 /path/to/keyframe.png
"""

from __future__ import annotations

import http.server
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

TARGET = sys.argv[1] if len(sys.argv) > 1 else "172.31.45.68:30010"
IMAGE = Path(sys.argv[2])
BASE = f"http://{TARGET}"


def private_ip() -> str:
    """The address the GPU boxes would reach this host on. Found by asking the
    routing table where it would send a packet, rather than by parsing `ip addr` --
    a box with docker bridges has several addresses and only one of them is the
    one a peer in the VPC can use."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((TARGET.split(":")[0], 1))
        return s.getsockname()[0]
    finally:
        s.close()


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        data = IMAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        print(f"  [served {len(data)} bytes to {self.client_address[0]}]", flush=True)

    def log_message(self, *a):  # keep the probe's output readable
        pass


srv = http.server.HTTPServer(("0.0.0.0", 0), Handler)
port = srv.server_port
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://{private_ip()}:{port}/{IMAGE.name}"
print(f"serving the keyframe at {url}\nsubmitting to {BASE}\n")

body = {
    "prompt": (
        "A still mountain valley at dawn, mist over dark pines, slow push in. "
        "Cinematic, 35mm, natural light. Ambient wind, no speech."
    ),
    "task": "fl2va",
    "conditions": [{"role": "keyframe", "type": "image", "uri": url, "frame_index": 0}],
    "target": {"short_edge": 480, "aspect_ratio": "16:9", "duration_seconds": 345 / 24},
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 1234,
}

req = urllib.request.Request(
    f"{BASE}/v1/videos", data=json.dumps(body).encode(), method="POST",
    headers={"Content-Type": "application/json"},
)
t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        created = json.loads(r.read())
except urllib.error.HTTPError as e:
    print(f"** submit rejected {e.code}: {e.read().decode('utf-8', 'replace')[:800]}")
    raise SystemExit(1)

vid = created.get("id")
print(f"  accepted as {vid} in {(time.perf_counter() - t0) * 1000:.0f}ms")

data: dict = {}
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    with urllib.request.urlopen(f"{BASE}/v1/videos/{vid}", timeout=30) as r:
        data = json.loads(r.read())
    if data.get("status") in ("completed", "failed"):
        break
    time.sleep(1.0)

print(f"  status={data.get('status')} after {time.perf_counter() - t0:.1f}s")
if data.get("status") != "completed":
    print(f"** {json.dumps(data)[:900]}")
    raise SystemExit(1)
print(f"  {json.dumps(data)[:500]}")

# And the other end: does the finished clip come back as bytes over HTTP, or only
# as a path on the box that scp has to go and get?
print("\n--- GET /v1/videos/{id}/content ---")
try:
    with urllib.request.urlopen(f"{BASE}/v1/videos/{vid}/content", timeout=120) as r:
        head, raw = dict(r.headers), r.read()
except urllib.error.HTTPError as e:
    print(f"** {e.code}: {e.read().decode('utf-8', 'replace')[:500]}")
    raise SystemExit(1)
print(f"  Content-Type: {head.get('Content-Type')}   bytes: {len(raw)}")
if b"ftyp" in raw[:32]:
    print("  -> real mp4 bytes. The scp leg can go too.")
else:
    print(f"  -> not mp4: {raw[:200]!r}")
