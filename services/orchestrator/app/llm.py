"""Bedrock access for the three LLM roles: Worldsmith, Director, PromptIR.

Three decisions here that are worth stating rather than discovering later:

* **The synchronous Anthropic SDK client, called through `asyncio.to_thread`.**
  The thread hop costs microseconds against calls that take seconds, and it
  avoids depending on the exact spelling of an async Bedrock client class. When
  the async client is confirmed available, `_invoke` is the only function that
  changes.

* **Streaming, always.** Not for incremental display -- for time-to-first-token.
  TTFT is the prefill/decode split, and DESIGN.md section 2.3 turns on knowing
  which of the two is the wall. Measuring it is nearly free once streaming is on,
  and every call records it.

* **A repair loop, not a retry loop.** A model that emitted malformed JSON does
  better when shown its own output and the specific parse or validation error
  than when simply asked again from scratch. So failures feed back as a
  continuation of the same conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .config import settings

log = logging.getLogger("kunlun.llm")

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    ttft_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str | None = None
    attempts: int = 1
    repairs: list[str] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"

    def timings(self, prefix: str) -> dict[str, float]:
        # `attempts` belongs here with the latencies, not just in the log, because
        # a repair round trip is the single most expensive thing that can happen to
        # one of these calls and `total_ms` alone cannot show it. It hid a real bug
        # for a while: the Worldsmith was silently paying two round trips on some
        # openings, and because `total_ms` is the accumulated figure the only
        # visible symptom was "the opening is sometimes 100s and sometimes 50s".
        # One number in the same dict makes that self-evident.
        out = {
            f"{prefix}_ttft_ms": round(self.ttft_ms, 1),
            f"{prefix}_decode_ms": round(self.decode_ms, 1),
            f"{prefix}_total_ms": round(self.total_ms, 1),
            f"{prefix}_attempts": float(self.attempts),
        }
        if self.output_tokens:
            # Against the budget it was given, this is what says whether a repair
            # was bad luck or a ceiling that is simply too low for this prompt.
            out[f"{prefix}_out_tokens"] = float(self.output_tokens)
        return out


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first complete JSON object out of a model response.

    Tolerant on purpose: fenced blocks, a stray preamble, and the leading brace
    being absent (which is what an assistant-prefill response looks like) are all
    normal and none of them should cost a round trip to fix.

    Tolerance has a sharp edge, though, and it cut deep: the scanner returns the
    first *balanced* object it finds, so on a prefill-accepted response -- which
    begins at `"narration": ...` because the opening `{` is ours and never comes
    back -- the first brace in the raw text is the one opening a *nested* object.
    Scanning the raw text therefore yields that nested fragment. Nothing
    downstream can tell: pydantic ignores unknown keys and fills defaults, so a
    perfectly good Director response arrived as
    `DirectorOutput(narration="", options=[])`, `_normalise_options` padded two
    generic branches, and an entire playthrough ran on 继续向前 / 退回原路 with
    no error anywhere. It only started happening when the Director moved to Haiku
    4.5, which *accepts* prefill -- Sonnet 5 rejects it and so had always returned
    the leading brace.

    So the three shapes are told apart up front rather than raced against each
    other, because two of them can both "succeed" on the same input:

    * starts with `{`  -- the object is the text.
    * starts with `"`  -- a prefilled continuation. `"{" + text` is then the *only*
      reading that can be right, so there is deliberately no fallback to scanning
      the interior: if that does not balance the response is truncated, and
      raising sends `structured()` down its truncation path, which retries with a
      bigger budget. Returning an interior fragment instead would hide exactly
      the failure that path exists to fix.
    * anything else    -- a fence or a prose preamble, where the object genuinely
      does sit in the middle and scanning is correct.
    """
    stripped = text.strip()
    candidates: list[str] = []
    if stripped.startswith("{"):
        candidates.append(text)
    elif stripped.startswith('"'):
        candidates.append("{" + text)
    else:
        if stripped.startswith("```"):
            body = stripped.split("```", 2)
            if len(body) >= 2:
                inner = body[1]
                if inner.lower().startswith("json"):
                    inner = inner[4:]
                candidates.append(inner)
        candidates.append(text)
        # Last resort: a preamble that happens to have eaten the brace.
        candidates.append("{" + text)

    for candidate in candidates:
        start = candidate.find("{")
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(candidate[start : i + 1])
                    except ValueError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    raise LLMError(f"no JSON object found in model output: {text[:400]!r}")


# Per-model capability quirks, learned at runtime rather than hardcoded.
#
# Both of these are genuinely mixed across models and not guessable from the name,
# and both fail as a 400 rather than as a soft degradation:
#
#   * `temperature` -- Haiku 4.5 accepts any value; Sonnet 5 answers
#     `400 \`temperature\` is deprecated for this model` for anything but the
#     default.
#   * assistant prefill -- Haiku 4.5 accepts a trailing assistant turn (the `{`
#     trick that stops a model opening with a preamble); Sonnet 5 answers
#     `400 This model does not support assistant message prefill`.
#
# So the first call for a given model pays one corrective retry, and every later
# call in the process omits whatever that model rejected. Callers keep asking for
# what they want and stay unaware of the difference. A hardcoded table would be
# wrong the week a model is added.
#
# (Structured outputs -- `output_config.format` -- would remove the need for the
# prefill trick entirely, but this Bedrock path rejects it with
# `output_config.format: Extra inputs are not permitted`. Worth retrying later:
# it would also delete the repair loop below and the latency it costs.)
_NO_TEMPERATURE: set[str] = set()
_NO_PREFILL: set[str] = set()


def _strip_prefill(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


class LLM:
    def __init__(self, region: str | None = None) -> None:
        self.region = region or settings.aws_region
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from anthropic import AnthropicBedrockMantle

            self._client = AnthropicBedrockMantle(aws_region=self.region)
        return self._client

    # -- raw ---------------------------------------------------------------- #

    def _invoke(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        thinking: bool,
        temperature: float | None,
    ) -> LLMResult:
        """Blocking; always called via `asyncio.to_thread`."""
        # The cache breakpoint goes on the system block only. Everything after it
        # varies per beat, so putting it anywhere later would invalidate the
        # prefix on every call and buy nothing.
        system_param = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_param,
            "messages": _strip_prefill(messages) if model in _NO_PREFILL else messages,
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        elif temperature is not None and model not in _NO_TEMPERATURE:
            # `temperature` is no longer a named parameter on `messages.create` /
            # `.stream` in anthropic 1.x -- passing it directly raises TypeError.
            # The wire API still accepts it where the model supports it, so it goes
            # through `extra_body`. (Verified against anthropic 1.7.0 +
            # AnthropicBedrockMantle.) Temperature and extended thinking do not
            # combine; only send one.
            kwargs["extra_body"] = {"temperature": temperature}

        # At most one corrective retry per quirk, and only for a quirk this call
        # actually triggered.
        for _ in range(3):
            try:
                chunks, ttft, total, final = self._stream(kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                detail = str(exc)
                if "temperature" in detail and "extra_body" in kwargs:
                    log.info("%s rejects temperature; dropping it for this process", model)
                    _NO_TEMPERATURE.add(model)
                    kwargs.pop("extra_body")
                    continue
                if "prefill" in detail and kwargs["messages"][-1].get("role") == "assistant":
                    log.info("%s rejects assistant prefill; dropping it for this process", model)
                    _NO_PREFILL.add(model)
                    kwargs["messages"] = _strip_prefill(kwargs["messages"])
                    continue
                raise
        else:
            raise LLMError(f"{model} kept rejecting the request after capability retries")

        usage = final.usage
        return LLMResult(
            text="".join(chunks),
            ttft_ms=ttft if ttft is not None else total,
            decode_ms=total - (ttft if ttft is not None else total),
            total_ms=total,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            stop_reason=final.stop_reason,
        )

    def _stream(self, kwargs: dict[str, Any]) -> tuple[list[str], float | None, float, Any]:
        start = time.perf_counter()
        ttft: float | None = None
        chunks: list[str] = []
        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_delta" and getattr(
                    event.delta, "type", ""
                ) == "text_delta":
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000.0
                    chunks.append(event.delta.text)
            final = stream.get_final_message()
        return chunks, ttft, (time.perf_counter() - start) * 1000.0, final

    async def text(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1200,
        thinking: bool = False,
        temperature: float | None = 0.7,
        prefill: str | None = None,
    ) -> LLMResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._invoke, model, system, messages, max_tokens, thinking, temperature
                ),
                timeout=settings.llm_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise LLMError(f"{model} timed out after {settings.llm_timeout_s}s") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"{model} call failed: {exc}") from exc

    # -- structured --------------------------------------------------------- #

    async def structured(
        self,
        model: str,
        system: str,
        user: str,
        response_model: type[T],
        max_tokens: int = 2000,
        thinking: bool = False,
        temperature: float | None = 0.7,
        max_repairs: int | None = None,
    ) -> tuple[T, LLMResult]:
        """Call the model and parse into `response_model`, repairing once on failure.

        Prefills the assistant turn with `{` so the model cannot open with a
        preamble. `extract_json_object` handles both the prefilled and the
        non-prefilled shape, so this stays correct whether or not the backend
        honours prefill.
        """
        budget = settings.llm_max_retries if max_repairs is None else max_repairs
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "{"},
        ]
        repairs: list[str] = []
        attempt = 0
        last: LLMResult | None = None
        # Grows only when an attempt is cut off mid-object -- see the truncation
        # branch below.
        budget_tokens = max_tokens

        while True:
            attempt += 1
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._invoke, model, system, messages, budget_tokens, thinking, temperature
                    ),
                    timeout=settings.llm_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                raise LLMError(f"{model} timed out after {settings.llm_timeout_s}s") from exc
            except Exception as exc:  # noqa: BLE001
                raise LLMError(f"{model} call failed: {exc}") from exc

            # Accumulate latency across repairs: the caller cares what the beat
            # cost in total, not what the lucky attempt cost.
            if last is not None:
                result.total_ms += last.total_ms
                result.decode_ms += last.decode_ms
                result.output_tokens += last.output_tokens
            last = result
            result.attempts = attempt
            result.repairs = list(repairs)

            problem: str | None = None
            try:
                obj = extract_json_object(result.text)
                parsed = response_model.model_validate(obj)
            except (LLMError, ValidationError, ValueError) as exc:
                problem = str(exc)[:1200]
            else:
                if result.truncated:
                    # Parsed despite truncation: possible, but the tail is
                    # missing, so flag it rather than trusting it silently.
                    log.warning("%s hit max_tokens but still parsed", model)
                return parsed, result

            repairs.append(problem or "unknown")
            if attempt > budget:
                raise LLMError(
                    f"{model} produced unusable output after {attempt} attempts: {problem}"
                )

            log.warning(
                "%s output rejected (attempt %d, stop_reason=%s, %d out tokens of %d): %s",
                model, attempt, result.stop_reason, result.output_tokens, budget_tokens, problem,
            )

            if result.truncated:
                # The output was not malformed, it was cut off: the model ran out of
                # budget mid-object, so the JSON never closed and the parser was
                # right to reject it. Asking it to "fix" that is the wrong move
                # twice over -- the text it would be repairing is correct as far as
                # it goes, and echoing a near-budget blob back makes the next
                # attempt *more* likely to truncate, not less. So raise the ceiling
                # and ask again cleanly instead.
                #
                # This is what made the Worldsmith intermittently cost two round
                # trips: measured output is 4.6-5.2k tokens against a 6k ceiling, so
                # ordinary variance tips some openings over, and on Sonnet 5 a
                # wasted attempt is 40-70s of TTFT alone.
                budget_tokens = min(budget_tokens * 2, settings.llm_max_tokens_cap)
                log.warning(
                    "%s was truncated; retrying with max_tokens=%d", model, budget_tokens
                )
                messages = [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": "{"},
                ]
                continue

            # Echo back what the model actually produced. The leading brace is only
            # ours when the prefill was accepted, so re-add it conditionally rather
            # than handing the model `{{`.
            echo = result.text if result.text.lstrip().startswith("{") else "{" + result.text
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": echo},
                {
                    "role": "user",
                    "content": (
                        "上面的输出无法使用，错误如下：\n"
                        f"{problem}\n\n"
                        "只输出修正后的完整 JSON 对象，不要解释，不要 markdown 代码块。"
                    ),
                },
                {"role": "assistant", "content": "{"},
            ]
