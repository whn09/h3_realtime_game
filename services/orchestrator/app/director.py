"""The Director: turns world state into two mutually exclusive next shots.

This is where a chain of prompts stops being a random walk and becomes a story.
Three mechanisms do that work, and all three are enforced in code here rather
than left to the prompt:

1. **Consequence binding.** Every branch must name an existing flag or stat in
   `references_state` and move it in `state_delta`. A branch that touches nothing
   is a decorative button, so `_bind_consequences` rewrites it rather than letting
   it through.
2. **Outline pull.** The current act's milestone, beats used and beats budgeted go
   into every call, so the Director is always answering "how do I get there from
   here" instead of "what would be interesting now".
3. **Shot variety.** Recent shot types are injected and enforced, because 15s x N
   from the same camera position is visually hypnotic regardless of the writing.

The Director never raises: a failed call degrades to `generic_options`, which is
always playable. Story never stops (DESIGN.md section 7).
"""

from __future__ import annotations

import logging

from . import ir_validator as V
from . import prompts
from .config import settings
from .llm import LLM, LLMError
from .schema import (
    Beat,
    BranchIntent,
    DirectorOutput,
    ShotSpec,
    StateDelta,
    WorldBible,
    WorldState,
)

log = logging.getLogger("h3game.director")

_SHOT_CYCLE = ["medium", "wide", "closeup", "pov", "tracking", "aerial"]


# --------------------------------------------------------------------------- #
# Context rendering                                                            #
# --------------------------------------------------------------------------- #


def bible_block(bible: WorldBible) -> str:
    chars = "\n".join(
        f"- {c.name}（id={c.id}）：{c.arc or '（弧光未定）'}" for c in bible.characters
    )
    rules = "\n".join(f"- {r}" for r in bible.world_rules)
    return f"""【世界圣经】
题材：{bible.genre}
一句话故事：{bible.logline}
视角：{"第一人称" if bible.pov == "first" else "第三人称"}
主角 id：{bible.protagonist_id}

角色：
{chars or "（无）"}

世界规则：
{rules or "（无）"}

这个世界的数值：{"、".join(bible.stat_names) or "（无）"}

注意：角色外观由圣经冻结，你不需要也不允许描述或改写它。"""


def act_block(bible: WorldBible, state: WorldState) -> str:
    outline = next((a for a in bible.outline if a.act == state.act), None)
    if not outline:
        return f"【结构】当前第 {state.act} 幕（大纲缺失，自行判断节奏）"
    remaining = outline.target_beats - state.beats_in_act
    pressure = (
        "该幕拍数已用过半但里程碑未达成，必须推进主线，两个分支都不许原地踏步。"
        if state.beats_in_act >= outline.target_beats / 2
        else "节奏还有余裕，可以铺垫。"
    )
    if remaining <= 1:
        pressure = "该幕拍数即将用尽，本拍应达成里程碑或直接推入下一幕。"
    return f"""【结构】
当前第 {state.act} 幕，本幕里程碑：{outline.milestone}
本幕已用 {state.beats_in_act} 拍 / 预算 {outline.target_beats} 拍
{pressure}"""


def state_block(state: WorldState) -> str:
    stats = "、".join(f"{k}={v:g}" for k, v in state.stats.items()) or "（无）"
    flags = "\n".join(f"- {k} = {v}" for k, v in state.flags.items()) or "- （无）"
    # Pacing pressure, stated as a fact and then as an instruction. The fact alone
    # is not enough -- the Director has no sense of elapsed screen time and will
    # read "3 beats here" as continuity rather than as a problem.
    scene = f"本场景已连续 {state.beats_in_scene} 拍（{state.beats_in_scene * settings.beat_seconds:.0f} 秒）"
    if state.beats_in_scene >= settings.scene_max_beats:
        scene += (
            f"——已超过 {settings.scene_max_beats} 拍上限，"
            "**本拍至少有一个分支必须换地点（cut）或跳时间（timeskip）**"
        )
    elif state.beats_in_scene == settings.scene_max_beats - 1:
        scene += "——接近上限，考虑让其中一个分支把故事带出这个场景"
    return f"""【当前状态】
地点：{state.location or "未定"}
{scene}
时刻：{state.time_of_day or "未定"}
世界内已流逝：{state.elapsed_in_world or "刚开始"}
在场角色：{"、".join(state.present_characters) or "（无）"}
随身物品：{"、".join(state.inventory) or "（无）"}
数值：{stats}
紧张度：{state.tension:.2f}

已有 flags（**分支必须引用其中至少一个**）：
{flags}"""


def history_block(state: WorldState) -> str:
    recent = "\n".join(f"- {b}" for b in state.recent_beats) or "- （这是开场之后的第一次选择）"
    return f"""【前史】
{state.summary or "（尚无压缩前史）"}

最近几拍：
{recent}"""


# --------------------------------------------------------------------------- #
# Post-validation and repair-in-code                                           #
# --------------------------------------------------------------------------- #


def _pick_unused_shot(recent: list[str], avoid: str | None = None) -> str:
    """Choose a shot type that breaks the recent pattern."""
    tail = recent[-2:]
    for t in _SHOT_CYCLE:
        if t not in tail and t != avoid:
            return t
    return "medium" if avoid != "medium" else "wide"


def _bind_consequences(
    intent: BranchIntent, state: WorldState, index: int
) -> list[V.Violation]:
    """Make sure the branch actually moves something the player can feel.

    Repaired in code rather than by a repair round trip: this is mechanical, and
    an extra Bedrock call would cost seconds the beat budget does not have.
    """
    out: list[V.Violation] = []
    known_flags = set(state.flags)
    known_stats = set(state.stats)
    ref = (intent.references_state or "").strip()

    touches = bool(intent.state_delta.flags_set) or bool(intent.state_delta.stats_delta)
    valid_ref = ref in known_flags or ref in known_stats

    if not valid_ref:
        out.append(V.Violation("references_state", "warning",
                               f"分支 {index} 的 references_state={ref!r} 不在已有 flags/数值里"))
        # Bind to something real so the choice still has weight.
        if known_stats:
            ref = sorted(known_stats)[index % len(known_stats)]
        elif known_flags:
            ref = sorted(known_flags)[index % len(known_flags)]
        intent.references_state = ref

    if not touches:
        out.append(V.Violation("references_state", "warning",
                               f"分支 {index} 没有改动任何状态，已自动绑定到 {ref!r}"))
        if ref in known_stats:
            # Direction is arbitrary but consistent: branch 0 spends, branch 1
            # conserves. Crude, and only reached when the Director skipped its job.
            intent.state_delta.stats_delta[ref] = -5.0 if index == 0 else 2.0
        elif ref:
            intent.state_delta.flags_set[f"{ref}__touched_b{state.beat_index}"] = True
    return out


def _normalise_options(
    out: DirectorOutput, bible: WorldBible, state: WorldState, want: int
) -> list[V.Violation]:
    issues: list[V.Violation] = []

    if len(out.options) > want:
        issues.append(V.Violation("option_count", "warning",
                                  f"导演给了 {len(out.options)} 个分支，截断到 {want}"))
        out.options = out.options[:want]
    if not out.options:
        # Zero options is categorically different from "one short". Padding one
        # missing branch is a repair; padding *all* of them means the response
        # carried no story at all, and what the player then sees is 继续向前 /
        # 退回原路 on every beat -- the game still runs, so nothing else
        # complains. That is exactly how a parser bug went unnoticed for a whole
        # playthrough (see `extract_json_object` on candidate order), so this is
        # loud in the log even though the padding below keeps the beat playable.
        log.error(
            "Director returned zero options at beat %d -- every branch on this "
            "beat will be a generic fallback. Response was parsed and validated, "
            "so suspect the JSON extraction or a prompt/schema mismatch.",
            state.beat_index,
        )
    while len(out.options) < want:
        issues.append(V.Violation("option_count", "warning",
                                  f"导演只给了 {len(out.options)} 个分支，补一个兜底分支"))
        out.options.append(_generic_branch(state, len(out.options)))

    for i, opt in enumerate(out.options):
        opt.label = opt.label.strip()[:14] or f"选项 {i + 1}"
        issues.extend(_bind_consequences(opt, state, i))
        issues.extend(
            V.check_shot_intent(opt.shot, settings.beat_seconds,
                                V.dialogue_budget_for(settings.beat_seconds))
        )
        # A cut has to say where to; otherwise the keyframe generator has nothing
        # to work from and the beat silently becomes a continuation.
        if opt.transition == "cut" and not opt.state_delta.location:
            opt.state_delta.location = opt.shot.setting or state.location
        if opt.transition == "timeskip" and not opt.state_delta.elapsed_in_world:
            opt.state_delta.elapsed_in_world = "一段时间之后"
        if not opt.state_delta.summary_append:
            opt.state_delta.summary_append = f"{opt.shot.subject}{opt.shot.action}"

    # Scene budget spent and nothing moves -- see rule 7 in DIRECTOR_SYSTEM and
    # `config.scene_max_beats`. This can only be repaired mechanically when the
    # Director already described somewhere else and just failed to declare it:
    # `shot.setting` differing from `state.location` is exactly that case, and
    # promoting it to a real `cut` is bookkeeping, not invention. When both
    # branches genuinely stay put there is nothing honest to do here -- inventing
    # a destination would put a place in the film that the story never earned --
    # so it is recorded as a violation, and the prompt is what has to carry it.
    if state.beats_in_scene >= settings.scene_max_beats:
        moved = [
            o for o in out.options
            if o.state_delta.location or o.state_delta.elapsed_in_world
        ]
        if not moved:
            promoted = next(
                (o for o in out.options
                 if o.shot.setting and o.shot.setting != state.location),
                None,
            )
            if promoted is not None:
                promoted.transition = "cut"
                promoted.state_delta.location = promoted.shot.setting
                issues.append(V.Violation(
                    "scene_budget", "warning",
                    f"本场景已连续 {state.beats_in_scene} 拍，分支的 setting 已经换了地方"
                    f"但没写进 state_delta，已提升为 cut：{promoted.shot.setting}"))
            else:
                issues.append(V.Violation(
                    "scene_budget", "warning",
                    f"本场景已连续 {state.beats_in_scene} 拍，两个分支都没有离开"
                    f"{state.location or '原地'}，节奏会显得停滞"))

    # Forced re-anchor wins over whatever the Director asked for: the visual
    # baseline has to be restored on this beat, and only a cut can do it.
    if state.needs_reanchor:
        for opt in out.options:
            if opt.transition == "continuous":
                opt.transition = "cut"
                opt.state_delta.location = opt.state_delta.location or opt.shot.setting or state.location
        issues.append(V.Violation("reanchor", "warning",
                                  "已到重锚定周期或漂移超阈值，本拍强制硬切"))

    recent = state.recent_shot_types
    if len(out.options) >= 2 and out.options[0].shot.type == out.options[1].shot.type:
        out.options[1].shot.type = _pick_unused_shot(recent, avoid=out.options[0].shot.type)
        issues.append(V.Violation("shot_variety", "warning",
                                  "两个分支机位相同，已改写第二个，否则两条预览看起来一样"))
    for i, opt in enumerate(out.options):
        if len(recent) >= 2 and recent[-1] == recent[-2] == opt.shot.type:
            opt.shot.type = _pick_unused_shot(recent)
            issues.append(V.Violation("shot_variety", "warning",
                                      f"分支 {i} 会造成连续三拍同机位，已改写"))

    if not 0 <= out.predicted_choice < len(out.options):
        out.predicted_choice = 0
    return issues


# --------------------------------------------------------------------------- #
# Fallbacks                                                                    #
# --------------------------------------------------------------------------- #


def _generic_branch(state: WorldState, index: int) -> BranchIntent:
    """The last-resort branch. Deliberately physical and always playable."""
    forward = index == 0
    label = "继续向前" if forward else "退回原路"
    action = "向前走出几步，停住并抬头张望" if forward else "转身后退两步，警惕地回头"
    stat = sorted(state.stats)[0] if state.stats else ""
    delta = StateDelta(
        tension=min(1.0, state.tension + (0.1 if forward else -0.05)),
        summary_append=f"{'继续前进' if forward else '退回原路'}，{state.location or '原地'}",
    )
    if stat:
        delta.stats_delta[stat] = -3.0 if forward else 1.0
    return BranchIntent(
        label=label,
        consequence_hint="前路未知" if forward else "时间在流失",
        transition="continuous",
        references_state=stat,
        shot=ShotSpec(
            type=_pick_unused_shot(state.recent_shot_types, avoid=None if forward else "medium"),
            subject="主角",
            action=action,
            setting=state.location or "原地",
            mood="迟疑",
            sfx_focus="脚步声",
        ),
        state_delta=delta,
    )


def generic_options(state: WorldState) -> DirectorOutput:
    return DirectorOutput(
        narration="路在脚下分开。",
        options=[_generic_branch(state, 0), _generic_branch(state, 1)],
        predicted_choice=0,
    )


# --------------------------------------------------------------------------- #
# Director                                                                     #
# --------------------------------------------------------------------------- #


class Director:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def expand(
        self, bible: WorldBible, beat: Beat
    ) -> tuple[DirectorOutput, list[str], dict[str, float]]:
        """Produce the options that follow `beat`. Never raises."""
        state = beat.state_after
        user = prompts.director_user(
            bible_block=bible_block(bible),
            state_block=state_block(state),
            act_block=act_block(bible, state),
            history_block=history_block(state),
            just_happened=beat.intent.state_delta.summary_append
            or f"{beat.intent.shot.subject}{beat.intent.shot.action}",
            shot_history=" → ".join(state.recent_shot_types[-3:]),
        )
        try:
            out, meta = await self.llm.structured(
                settings.model_director,
                prompts.DIRECTOR_SYSTEM,
                user,
                DirectorOutput,
                # Measured: a two-option expansion decodes 1777-2488 tokens, so
                # 3000 sat close enough to the ceiling that it was occasionally
                # breached -- and a truncated object cannot be salvaged by
                # `extract_json_object`, so the whole call is repeated. One such
                # repair cost 51s against a 26s baseline. A cap is not a target,
                # so raising it is free whenever it goes unused.
                max_tokens=4000,
                temperature=0.9,
            )
            timings = meta.timings("director")
        except LLMError as exc:
            log.error("Director failed, falling back to generic options: %s", exc)
            out = generic_options(state)
            return out, [f"director failed: {exc}", "fell back to generic options"], {}

        issues = _normalise_options(out, bible, state, settings.branch_count)
        if out.is_ending and state.act < 3:
            # Guard against an early curtain: an ending in act 1 is almost always
            # the model losing track of the outline, not a deliberate choice.
            issues.append(V.Violation("premature_ending", "warning",
                                      f"第 {state.act} 幕就想收尾，已忽略 is_ending"))
            out.is_ending = None
        return out, [str(i) for i in issues], timings

    async def custom(
        self, bible: WorldBible, beat: Beat, player_action: str
    ) -> tuple[BranchIntent, str, list[str], dict[str, float]]:
        """The free-text path. Returns one branch plus its narration."""
        state = beat.state_after
        user = prompts.director_custom_user(
            bible_block=bible_block(bible),
            state_block=state_block(state),
            act_block=act_block(bible, state),
            just_happened=beat.intent.state_delta.summary_append or "",
            player_action=player_action.strip()[:400],
        )
        try:
            out, meta = await self.llm.structured(
                settings.model_director,
                prompts.DIRECTOR_SYSTEM,
                user,
                DirectorOutput,
                max_tokens=2200,
                temperature=0.9,
            )
            timings = meta.timings("director")
        except LLMError as exc:
            log.error("Director custom path failed: %s", exc)
            fallback = _generic_branch(state, 0)
            fallback.label = player_action.strip()[:14] or "自由行动"
            return fallback, "世界没有回应。", [f"director failed: {exc}"], {}

        issues = _normalise_options(out, bible, state, 1)
        intent = out.options[0]
        intent.label = player_action.strip()[:14] or intent.label
        return intent, out.narration, [str(i) for i in issues], timings

    async def compress_summary(self, bible: WorldBible, state: WorldState) -> str:
        """Fold recent beats into the running summary.

        Run off the critical path -- it exists to stop the Director's context from
        growing without bound, and nothing is waiting on it.
        """
        user = f"""{bible_block(bible)}

已有前史压缩：
{state.summary or "（无）"}

最近几拍：
{chr(10).join("- " + b for b in state.recent_beats)}

当前状态：地点 {state.location}，第 {state.act} 幕，紧张度 {state.tension:.2f}。

把以上内容重写成一段 80-140 字的前史摘要，只保留对后续剧情有约束力的事实（谁做了什么、谁死了、谁记恨谁、哪扇门开了、数值代价），删掉所有修辞。只输出这段文字本身。"""
        try:
            res = await self.llm.text(
                settings.model_director,
                "你是剧本编辑，负责把长前史压缩成紧凑的事实摘要。只输出摘要本身，不要解释。",
                user,
                max_tokens=400,
                temperature=0.3,
            )
            return res.text.strip()
        except LLMError as exc:
            log.warning("summary compression failed: %s", exc)
            return state.summary
