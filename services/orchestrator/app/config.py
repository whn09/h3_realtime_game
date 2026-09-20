"""Configuration for the orchestrator, read once from the environment.

Same dependency-free style as h3-wrapper's config: this process lives next to
the GPU box (DESIGN.md section 5.1) and should install with a minimal pip set.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(key: str, default: str) -> list[str]:
    return [p.strip().rstrip("/") for p in _env(key, default).split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    # --- GPU pool ------------------------------------------------------------
    # `fake` | `wrapper` | `h3`. `h3` talks to SGLang Diffusion directly over the
    # ssh tunnels (see gpu.H3Backend); `wrapper` goes through h3-wrapper, which
    # does not exist yet. FAKE_GPU=1 still wins, so old invocations keep working.
    gpu_backend: str = field(default_factory=lambda: _env("GPU_BACKEND", "h3").strip().lower())
    # One URL per h3-wrapper, one wrapper per SGLang instance, one instance per
    # GPU slot. Two entries = the MVP topology in DESIGN.md section 5.3.
    wrapper_urls: list[str] = field(
        default_factory=lambda: _env_list("WRAPPER_URLS", "http://127.0.0.1:8000")
    )
    # Synthesise clips locally with ffmpeg instead of calling the wrappers.
    # Exists so the entire story engine -- including last-frame chaining -- is
    # runnable and testable before the GPU boxes exist, and so frontend work
    # does not need a GPU.
    fake_gpu: bool = field(default_factory=lambda: _env_bool("FAKE_GPU", False))
    fake_gpu_latency_s: float = field(default_factory=lambda: _env_float("FAKE_GPU_LATENCY_S", 2.0))
    gpu_timeout_s: float = field(default_factory=lambda: _env_float("GPU_TIMEOUT_S", 900.0))
    # Endpoint is benched for this long after a failure, so one sick GPU does
    # not eat every retry.
    gpu_cooldown_s: float = field(default_factory=lambda: _env_float("GPU_COOLDOWN_S", 30.0))

    # --- The live SGLang deployment (GAME.md) ---------------------------------
    # `alias=host:port,alias=host:port`. Empty means read the file
    # `game_tunnel.sh` wrote, which is the normal case -- the aliases are ssh
    # aliases and the backend needs them for scp, so a bare URL is not enough.
    h3_replicas: str = field(default_factory=lambda: _env("H3_REPLICAS", ""))
    h3_state_dir: Path = field(
        default_factory=lambda: Path(
            _env("H3_STATE_DIR", str(Path(tempfile.gettempdir()) / "h3game"))
        )
    )
    # Where on the box clips and uploaded keyframes live. Instance store, wiped
    # on stop/start, which is fine: everything here is reproducible.
    h3_remote_out: str = field(
        default_factory=lambda: _env("H3_REMOTE_OUT", "/opt/dlami/nvme/vdn/outputs").rstrip("/")
    )
    # Transcribed from the deployed server flags; changing either changes the
    # picture, so they are settings and not literals.
    h3_flow_shift: float = field(default_factory=lambda: _env_float("H3_FLOW_SHIFT", 12.0))
    h3_audio_flow_shift: float = field(
        default_factory=lambda: _env_float("H3_AUDIO_FLOW_SHIFT", 3.0)
    )
    # The docker container the SGLang worker runs in. Only used to borrow its
    # ffmpeg (`docker exec`) for on-box last-frame extraction -- the host itself
    # has none -- so a wrong value degrades to the old download-then-upload path
    # rather than breaking generation.
    h3_container: str = field(default_factory=lambda: _env("H3_CONTAINER", "h3-game"))
    h3_poll_s: float = field(default_factory=lambda: _env_float("H3_POLL_S", 0.2))
    h3_fps: int = field(default_factory=lambda: _env_int("H3_FPS", 24))
    # One throwaway clip per replica at startup. `--warmup-resolutions` warms
    # 768p regardless of the flag, so without this the first clip the player
    # waits for pays ~0.9s of shape change (GAME.md, finding 1).
    h3_warm: bool = field(default_factory=lambda: _env_bool("H3_WARM", True))
    h3_ssh_timeout_s: float = field(default_factory=lambda: _env_float("H3_SSH_TIMEOUT_S", 120.0))

    # --- Beat shape ----------------------------------------------------------
    # 14.375s = 345 frames at 24fps, the only shape this deployment has been
    # measured at. Anything off the `5 + 17k` lattice is silently rounded by the
    # server, so `gpu.frames_for_seconds` snaps before sending.
    beat_seconds: float = field(default_factory=lambda: _env_float("BEAT_SECONDS", 14.375))
    short_edge: int = field(default_factory=lambda: _env_int("SHORT_EDGE", 480))
    aspect_ratio: str = field(default_factory=lambda: _env("ASPECT_RATIO", "16:9"))
    quality: str = field(default_factory=lambda: _env("QUALITY", "high"))
    # 0 means "do not send it", which is the right default on VDN-H3: the
    # checkpoint is a Stage-DMD distill baked to exactly nine sigma grid points
    # (eight DiT forwards), and the server rejects every other value outright --
    #   `got num_inference_steps=8. Use MiniMaxAI/MiniMax-H3 for other schedules.`
    # So the step count is a property of the weights, not a knob this side owns,
    # and omitting it means a re-distilled checkpoint needs no change here.
    num_inference_steps: int = field(default_factory=lambda: _env_int("NUM_INFERENCE_STEPS", 0))
    # Degraded retry parameters (DESIGN.md section 7). 10.125s = 243 frames, the
    # next lattice rung down, at 0.72x the tokens -- on this deployment clip
    # length is the only honest latency knob, for the reason directly above.
    retry_steps: int = field(default_factory=lambda: _env_int("RETRY_STEPS", 0))
    retry_seconds: float = field(default_factory=lambda: _env_float("RETRY_SECONDS", 10.125))

    # --- Pre-generation ------------------------------------------------------
    # 1 = pre-generate both children of the beat being watched. 2 = also
    # pre-generate the children of the branch the Director predicts, so the
    # *second* choice is covered too.
    #
    # Stays at 1 at two replicas, and that is a measurement, not caution. Depth 2
    # was tried on this deployment the moment eager expansion made grandchildren
    # plannable in time -- DESIGN.md 2.3 lever 4 -- and it made the game slower:
    #
    #                        depth 2          depth 1
    #   gpu_total (real)     med 13.5s        med 13.0s     <- identical
    #   slot wait            med 28.5s        med 11.5s     <- 2.5x worse
    #   gpu_download         max 16.0s        max  4.7s     <- scp contention
    #   stall the player felt  med 10.8s      med  1.8s
    #
    # The arithmetic is the whole story: depth 2 wants 2 children + 4 grandchildren
    # = 6 clips in flight against 2 slots, and a clip is ~13s of GPU that cannot be
    # preempted once it starts. PRIORITY_SPECULATIVE only decides who takes the
    # *next* free slot, so the children the player is about to need still queue
    # three deep behind speculation. The download outliers are the same crowding one
    # layer down -- four concurrent scp's sharing one ControlMaster socket.
    #
    # So the rule this encodes is slots >= branch_count ** depth. At 2 slots and
    # branch 2, depth 1 is exactly saturating and depth 2 is 3x oversubscribed.
    # Raise this when there are more replicas, not before.
    pregen_depth: int = field(default_factory=lambda: _env_int("PREGEN_DEPTH", 1))
    branch_count: int = field(default_factory=lambda: _env_int("BRANCH_COUNT", 2))
    # Force a re-anchoring cut at least this often, independent of measured
    # drift. chain_drift.py replaces this guess with a measurement.
    reanchor_every: int = field(default_factory=lambda: _env_int("REANCHOR_EVERY", 6))
    # After this many consecutive beats in one location, at least one branch has
    # to move the story -- new place or a time jump.
    #
    # 3, because a beat is 14.4s: three is ~43s in one set, which is about as long
    # as a scene in cut cinema runs before it wants a new angle on a new thing,
    # and four starts to feel like the story is stuck. The Director's own bias is
    # the opposite -- `continuous` is its documented default because that is where
    # visual continuity comes from -- so without a counter pushing back, a session
    # will happily spend its whole first act in the room it opened in.
    scene_max_beats: int = field(default_factory=lambda: _env_int("SCENE_MAX_BEATS", 3))
    # Rewrite the compressed backstory every N beats so the Director's context
    # cannot grow without bound.
    resummarise_every: int = field(default_factory=lambda: _env_int("RESUMMARISE_EVERY", 8))

    # --- Bedrock -------------------------------------------------------------
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-west-2"))
    # Bedrock model IDs carry an `anthropic.` prefix. CONFIRM these against the
    # account's actual model access before first run -- an unavailable ID fails
    # at call time, not at startup.
    model_worldsmith: str = field(default_factory=lambda: _env("MODEL_WORLDSMITH", "anthropic.claude-sonnet-5"))
    # Haiku 4.5, not Sonnet 5, and this is the single biggest latency decision in
    # the service. Measured on a real playthrough, `director_total` on Sonnet 5
    # was 27-49s -- and the earlier isolated probe showed that is platform-side
    # latency on this account, not prompt size: input tokens were flat and TTFT
    # varied 13.9-24.5s at concurrency 1 with byte-identical prompts. The same
    # account runs Haiku 4.5 for PromptIR in 3.6-7s. The Worldsmith stays on
    # Sonnet 5: it runs once per session behind the opening ritual, so its
    # latency is free and its output (the frozen World Bible) constrains every
    # later beat, which is exactly where the better model is worth paying for.
    model_director: str = field(default_factory=lambda: _env("MODEL_DIRECTOR", "anthropic.claude-haiku-4-5"))
    model_promptir: str = field(default_factory=lambda: _env("MODEL_PROMPTIR", "anthropic.claude-haiku-4-5"))
    # Extended thinking costs decode time, which the beat budget does not have.
    # Off by default; the Worldsmith is the one call where turning it on is
    # defensible (once per session, hidden behind the 21s opening ritual).
    worldsmith_thinking: bool = field(default_factory=lambda: _env_bool("WORLDSMITH_THINKING", False))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 1))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 120.0))
    # Ceiling on the doubling `LLM.structured` does when a response comes back
    # truncated. A bound rather than unlimited growth because a prompt that wants
    # more than this is not hitting variance, it is mis-specified, and doubling
    # forever would turn that into a very expensive silent loop.
    llm_max_tokens_cap: int = field(
        default_factory=lambda: _env_int("LLM_MAX_TOKENS_CAP", 16000)
    )

    # --- PromptIR ------------------------------------------------------------
    # Generate only `integrated_multimodal_description` and template the
    # soundscape/music sections from the world bible. Wins twice: ~40% less
    # decode (DESIGN.md section 2.3) *and* a byte-identical music description
    # every beat, which is exactly what section 3.5 asks for.
    ir_template_tail: bool = field(default_factory=lambda: _env_bool("IR_TEMPLATE_TAIL", True))
    ir_max_repairs: int = field(default_factory=lambda: _env_int("IR_MAX_REPAIRS", 1))
    ir_max_tokens: int = field(default_factory=lambda: _env_int("IR_MAX_TOKENS", 1600))

    # --- Keyframes -----------------------------------------------------------
    # SD3.5 Large, measured against the alternatives on this account's actual
    # model access: it was both the most faithful to the prompt (the others
    # answered a 修仙 establishing shot with a modern hiker in a field) and the
    # fastest at 7.6s. `amazon.nova-canvas-v1:0` is LEGACY here and answers
    # InvokeModel with `Access denied ... not been actively using the model`.
    keyframe_model: str = field(
        default_factory=lambda: _env("KEYFRAME_MODEL", "stability.sd3-5-large-v1:0")
    )
    # Its own region, because image and text models are not granted together:
    # the only ACTIVE on-demand text-to-image models on this account are in
    # us-west-2, while the text models are being called in us-east-1.
    keyframe_region: str = field(default_factory=lambda: _env("KEYFRAME_REGION", "us-west-2"))
    keyframe_width: int = field(default_factory=lambda: _env_int("KEYFRAME_WIDTH", 1280))
    keyframe_height: int = field(default_factory=lambda: _env_int("KEYFRAME_HEIGHT", 720))
    # If the image model is unavailable, synthesise a gradient placeholder
    # rather than blocking the session. The story must always start.
    keyframe_fallback: bool = field(default_factory=lambda: _env_bool("KEYFRAME_FALLBACK", True))

    # --- Storage / serving ---------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))
    # Keyframes, and every clip the `fake` and `h3` backends produce. The real
    # deployment returns a path on the GPU box and has no download endpoint
    # (GAME.md, finding 3), so the orchestrator copies clips here and serves them
    # itself rather than pointing the browser at the box.
    assets_dir: Path = field(default_factory=lambda: Path(_env("ASSETS_DIR", "./data/assets")))
    public_base_url: str = field(
        default_factory=lambda: _env("PUBLIC_BASE_URL", "http://127.0.0.1:8100/assets").rstrip("/")
    )
    cors_origins: list[str] = field(default_factory=lambda: _env_list("CORS_ORIGINS", "*"))

    ffmpeg: str = field(default_factory=lambda: _env("FFMPEG_BIN", "ffmpeg"))
    ffprobe: str = field(default_factory=lambda: _env("FFPROBE_BIN", "ffprobe"))

    def session_dir(self, session_id: str) -> Path:
        return self.data_dir / "sessions" / session_id


settings = Settings()
