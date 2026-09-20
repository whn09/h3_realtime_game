"""ffmpeg/ffprobe post-processing, all local to the GPU box.

The single most important function here is `extract_last_frame`. Its output is
used twice per beat (DESIGN.md section 3.1):

  1. as the `frame_index: 0` keyframe condition for the *next* beat, which is
     what makes the world continuous rather than a slideshow of unrelated clips;
  2. as the freeze-frame the browser shows during the 4s decision moment, which
     is pixel-identical to the video's final frame and therefore invisible as a
     transition.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .config import settings
from .models import FrameStats, MediaInfo


class MediaError(RuntimeError):
    pass


async def _run(*args: str) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode or 0, out, err


async def _run_checked(*args: str) -> bytes:
    code, out, err = await _run(*args)
    if code != 0:
        raise MediaError(f"{args[0]} failed ({code}): {err.decode('utf-8', 'replace')[-2000:]}")
    return out


async def probe(path: Path) -> MediaInfo:
    raw = await _run_checked(
        settings.ffprobe,
        "-v", "error",
        "-show_entries", "format=duration,size",
        "-show_streams",
        "-of", "json",
        str(path),
    )
    data = json.loads(raw)
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    info = MediaInfo(
        size_bytes=int(fmt["size"]) if fmt.get("size") else path.stat().st_size,
        has_audio=audio is not None,
    )

    if fmt.get("duration"):
        info.duration_ms = float(fmt["duration"]) * 1000.0

    if video:
        info.width = video.get("width")
        info.height = video.get("height")
        info.video_codec = video.get("codec_name")
        if nb := video.get("nb_frames"):
            try:
                info.nb_frames = int(nb)
            except (TypeError, ValueError):
                pass
        info.fps = _parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate"))
        # Stream duration is more trustworthy than container duration when the
        # muxer wrote a sloppy header.
        if video.get("duration"):
            info.duration_ms = float(video["duration"]) * 1000.0

    if audio:
        info.audio_codec = audio.get("codec_name")
        if rate := audio.get("sample_rate"):
            try:
                info.audio_sample_rate = int(rate)
            except (TypeError, ValueError):
                pass

    return info


def _parse_fps(value: str | None) -> float | None:
    if not value or "/" not in value:
        return None
    num, den = value.split("/", 1)
    try:
        num_f, den_f = float(num), float(den)
    except ValueError:
        return None
    return num_f / den_f if den_f else None


async def extract_last_frame(src: Path, dest: Path, duration_ms: float | None) -> None:
    """Write the final displayed frame of `src` to `dest` as PNG.

    All three strategies rely on `-update 1` *without* a `-frames:v` cap: every
    decoded frame overwrites the same PNG, so whatever ffmpeg decodes last is
    what survives. (Capping at one frame would instead capture the *first*
    frame of the seek window -- a subtly wrong frame that would show up as a
    visible jump at every beat boundary.)

    Ordered cheapest-first, because `-sseof` needs a seekable index that a
    freshly-muxed file may lack:
      1. `-sseof -0.25` -- seek from EOF, decodes ~6 frames.
      2. absolute seek to duration-250ms -- needs ffprobe's duration, but works
         on files strategy 1 gives up on.
      3. full decode -- ~200ms at 480p/15s, cannot fail on a valid file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    attempts: list[list[str]] = [
        [settings.ffmpeg, "-y", "-sseof", "-0.25", "-i", str(src),
         "-update", "1", "-q:v", "2", str(dest)],
    ]
    if duration_ms and duration_ms > 300:
        seek = max(0.0, (duration_ms - 250.0) / 1000.0)
        attempts.append(
            [settings.ffmpeg, "-y", "-ss", f"{seek:.3f}", "-i", str(src),
             "-update", "1", "-q:v", "2", str(dest)]
        )
    attempts.append(
        [settings.ffmpeg, "-y", "-i", str(src),
         "-update", "1", "-q:v", "2", str(dest)]
    )

    errors: list[str] = []
    for args in attempts:
        code, _, err = await _run(*args)
        if code == 0 and dest.exists() and dest.stat().st_size > 1024:
            return
        errors.append(err.decode("utf-8", "replace")[-400:])
        dest.unlink(missing_ok=True)

    raise MediaError("last-frame extraction failed:\n" + "\n---\n".join(errors))


async def extract_poster(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    await _run_checked(
        settings.ffmpeg, "-y", "-ss", "0", "-i", str(src),
        "-update", "1", "-frames:v", "1", "-q:v", "3", str(dest),
    )


async def remux_faststart(src: Path, dest: Path) -> None:
    """Move the moov atom to the front so the browser can start playing on the
    first bytes instead of waiting for the full download. Stream copy, ~20ms."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    await _run_checked(
        settings.ffmpeg, "-y", "-i", str(src),
        "-c", "copy", "-movflags", "+faststart", str(dest),
    )


async def frame_stats(frame_path: Path) -> FrameStats:
    """Perceptual fingerprint used for drift detection across a chained session.

    Runs in a thread: PIL/numpy are synchronous and this is ~5ms, but blocking
    the event loop while two GPUs are in flight is a habit worth not forming.
    """
    return await asyncio.to_thread(_frame_stats_sync, frame_path)


def _frame_stats_sync(frame_path: Path) -> FrameStats:
    import numpy as np
    from PIL import Image

    with Image.open(frame_path) as img:
        rgb = img.convert("RGB")
        # Downscale: these are global statistics, full resolution buys nothing
        # and costs milliseconds.
        rgb.thumbnail((256, 256))
        arr = np.asarray(rgb, dtype=np.float32) / 255.0

    mean = arr.reshape(-1, 3).mean(axis=0)
    luma = float(0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2])

    channel_max = arr.max(axis=2)
    channel_min = arr.min(axis=2)
    saturation = float(
        np.divide(
            channel_max - channel_min,
            channel_max,
            out=np.zeros_like(channel_max),
            where=channel_max > 1e-6,
        ).mean()
    )

    gray = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    contrast = float(gray.std())

    # Variance of the Laplacian: the standard cheap focus/softness measure.
    # Chained i2va tends to lose high-frequency detail generation over
    # generation, and this is what catches it.
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    sharpness = float(lap.var())

    return FrameStats(
        mean_r=float(mean[0]),
        mean_g=float(mean[1]),
        mean_b=float(mean[2]),
        luma=luma,
        saturation=saturation,
        contrast=contrast,
        sharpness=sharpness,
    )
