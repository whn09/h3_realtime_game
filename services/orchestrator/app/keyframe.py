"""Keyframe generation, for the two moments H3 cannot chain from a previous frame.

Needed in exactly two places:
  * the opening beat, which has no previous frame at all (DESIGN.md section 2.4)
  * a `cut` / `timeskip` / re-anchoring beat, where chaining is what we want to
    *break* (sections 3.2 and 3.3)

Always falls back to a locally synthesised frame. A missing keyframe would block
the entire session at its most fragile moment -- before the player has seen
anything at all -- and a plain gradient opening is a far better outcome than an
error screen. The fallback is marked in the return value so the UI and the logs
both know the difference.

Two request shapes, dispatched on the model id, because which image model an
account actually has is not something this code gets to choose. Stability's
`{prompt, aspect_ratio}` body is the default; Nova Canvas's nested
`{taskType, textToImageParams, imageGenerationConfig}` is kept because it is the
shape to come back to if that model is ever un-deprecated on the account.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .schema import ShotSpec, WorldBible, WorldState

log = logging.getLogger("h3game.keyframe")

_NEGATIVE = (
    "text, watermark, logo, subtitles, blurry, low quality, distorted anatomy, "
    "extra limbs, collage, split screen, frame border"
)

_CAMERA_EN = {
    "wide": "extreme wide establishing shot",
    "medium": "medium shot",
    "closeup": "close-up",
    "pov": "first person point of view",
    "tracking": "tracking shot, subject centred",
    "aerial": "high aerial view looking down",
}


@dataclass
class Keyframe:
    path: Path
    url: str
    source: str          # stability | nova | synthetic
    elapsed_ms: float


def build_prompt(bible: WorldBible, shot: ShotSpec, state: WorldState) -> str:
    """Compose an English image prompt.

    English on purpose: the image models are measurably more literal with English,
    and this prompt never reaches H3, so there is no consistency cost to switching
    languages here.
    """
    parts = [
        _CAMERA_EN.get(shot.type, "medium shot"),
        shot.subject,
        shot.setting or state.location,
        state.time_of_day,
        shot.mood,
        bible.style_anchor,
        bible.genre,
        "cinematic still frame, 16:9, no text",
    ]
    return ", ".join(p.strip() for p in parts if p and p.strip())[:1000]


class Keyframer:
    def __init__(self, assets_dir: Path | None = None) -> None:
        self.assets_dir = assets_dir or settings.assets_dir
        self._client = None

    @property
    def client(self):  # noqa: ANN201 - boto3 client is untyped
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=settings.keyframe_region)
        return self._client

    async def generate(
        self, session_id: str, name: str, prompt: str, seed: int | None = None
    ) -> Keyframe:
        dest = self.assets_dir / session_id / f"{name}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        model = settings.keyframe_model
        invoke = self._invoke_nova if "nova-canvas" in model else self._invoke_stability
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(invoke, prompt, seed or 0), timeout=60.0
            )
            dest.write_bytes(data)
            source = "nova" if invoke is self._invoke_nova else "stability"
        except Exception as exc:  # noqa: BLE001
            if not settings.keyframe_fallback:
                raise
            log.warning("keyframe model unavailable (%s); synthesising a placeholder", exc)
            await asyncio.to_thread(self._synthesise, dest, prompt)
            source = "synthetic"

        return Keyframe(
            path=dest,
            url=f"{settings.public_base_url}/{session_id}/{name}.png",
            source=source,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )

    # -- backends ----------------------------------------------------------- #

    def _invoke_stability(self, prompt: str, seed: int) -> bytes:
        """Stable Image Core / Ultra / SD3.5, which share one body shape.

        `aspect_ratio` rather than width/height: these models only accept a ratio
        from a fixed list and pick their own pixel dimensions (SD3.5 answers 16:9
        with 1344x768). That is fine -- H3 resizes the conditioning frame anyway,
        and asking for the ratio instead of the size means no arithmetic here has
        to agree with the model's supported resolutions.
        """
        body = {
            "prompt": prompt[:2000],
            "negative_prompt": _NEGATIVE,
            "aspect_ratio": settings.aspect_ratio,
            "output_format": "png",
            "seed": abs(seed) % 4_294_967_294,
        }
        resp = self.client.invoke_model(modelId=settings.keyframe_model, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        # A non-null finish reason means the request succeeded but the image was
        # withheld (content filter), so there is nothing to decode.
        reasons = [r for r in (payload.get("finish_reasons") or []) if r]
        if reasons:
            raise RuntimeError(f"image withheld: {reasons}")
        images = payload.get("images") or []
        if not images:
            raise RuntimeError(f"no image in response: {list(payload)}")
        return base64.b64decode(images[0])

    def _invoke_nova(self, prompt: str, seed: int) -> bytes:
        body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": prompt[:1024], "negativeText": _NEGATIVE},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "width": settings.keyframe_width,
                "height": settings.keyframe_height,
                "cfgScale": 6.5,
                # Nova wants a non-negative 32-bit seed.
                "seed": abs(seed) % 2_147_483_647,
                "quality": "standard",
            },
        }
        resp = self.client.invoke_model(
            modelId=settings.keyframe_model, body=json.dumps(body)
        )
        payload = json.loads(resp["body"].read())
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        images = payload.get("images") or []
        if not images:
            raise RuntimeError(f"no image in response: {list(payload)}")
        return base64.b64decode(images[0])

    def _synthesise(self, dest: Path, prompt: str) -> None:
        """A deterministic gradient frame, derived from the prompt's hash.

        Not trying to look generated -- trying to be a neutral, non-distracting
        establishing frame that the first beat can legitimately continue from, and
        that gives the drift metrics a non-degenerate baseline.
        """
        from PIL import Image, ImageDraw

        w, h = settings.keyframe_width, settings.keyframe_height
        seed = sum(ord(c) for c in prompt) or 1
        hue = (seed % 360) / 360.0

        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        for y in range(h):
            t = y / max(1, h - 1)
            # Dusk-ish vertical gradient: bright near the horizon, dark at both ends.
            glow = math.exp(-((t - 0.62) ** 2) / 0.05)
            r = int(28 + 150 * glow * (0.5 + 0.5 * math.cos(hue * 6.283)))
            g = int(26 + 95 * glow)
            b = int(48 + 110 * glow * (0.5 + 0.5 * math.sin(hue * 6.283)))
            if t > 0.72:  # ground plane
                r, g, b = int(r * 0.22), int(g * 0.22), int(b * 0.26)
            draw.line([(0, y), (w, y)], fill=(min(r, 255), min(g, 255), min(b, 255)))

        # A silhouette, so the frame has real high-frequency edges for the
        # sharpness metric to track.
        cx, base = int(w * 0.42), int(h * 0.80)
        draw.ellipse([cx - 13, base - 108, cx + 13, base - 82], fill=(8, 8, 10))
        draw.polygon(
            [(cx - 21, base), (cx - 15, base - 82), (cx + 15, base - 82), (cx + 21, base)],
            fill=(8, 8, 10),
        )
        img.save(dest)
