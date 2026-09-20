"""Validation for compiled IR, and for the Director's shot intents.

PromptIR is a *compiler*, so it gets a compiler's error reporting: named rules,
fatal-vs-warning severity, and messages written to be fed straight back to the
model as a repair instruction.

The reference implementation carries 36 rules. The set below is the subset that
can be justified from the documented IR format alone -- when the reference repo
is available its rules replace these, and the `Rule` shape here is what they
should be expressed in. Each rule states *why* it exists, because a rule whose
rationale is lost gets deleted the first time it produces a false positive.

Severity matters for latency. Every fatal violation risks a repair round trip
(seconds), so a rule is only fatal when the output would be structurally broken
or would visibly damage continuity. Everything else is recorded as a warning and
surfaced in the beat's audit record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .schema import Character, IRSections, ShotSpec

_CJK = re.compile(r"[㐀-䶿一-鿿]")
_QUOTED = re.compile(r"[「“\"']([^」”\"']{1,200})[」”\"']")

# Any of these in the description means the compiler started writing prose about
# the video instead of writing the video.
_MARKDOWN = ("```", "###", "## ", "**", "- ", "* ", "1. ", "2. ")
_META = ("请生成", "请注意", "务必", "要求画面", "确保", "尽量做到", "应该呈现", "需要表现出")
# Filler that describes no pixel. It displaces attention inside the context
# window and measurably does nothing for H3 output.
_FILLER = ("电影感", "高质量", "高清", "超清", "4K", "8K", "杰作", "精美", "大师级", "震撼", "完美")
# Belongs in API parameters. In the IR it can only contradict them.
_TECH = ("宽高比", "16:9", "9:16", "分辨率", "1080p", "720p", "帧率", "fps", "比特率")
_CAMERA_TERMS = (
    "大远景", "远景", "全景", "中景", "近景", "特写", "大特写", "主观视角", "第一人称",
    "跟拍", "航拍", "俯视", "仰视", "固定镜头", "手持", "横移", "推进", "拉远", "环绕",
)
_MUSIC_WORDS = ("配乐", "音乐", "弦乐", "钢琴", "鼓点", "BPM", "旋律", "和弦", "小调", "大调", "合成器")
_DIEGETIC_WORDS = ("脚步", "风声", "呼吸", "关门", "碎石", "雨滴", "衣料", "枪声", "引擎", "对白")

Severity = str  # "fatal" | "warning"


@dataclass
class Violation:
    rule: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


@dataclass
class IRContext:
    """Everything a rule needs to know beyond the IR text itself."""

    style_anchor: str = ""
    music_bible: str = ""
    characters: list[Character] = field(default_factory=list)
    dialogue_budget: int = 24
    seconds: float = 15.0
    require_tail: bool = True


def count_cjk(text: str) -> int:
    return len(_CJK.findall(text))


def normalise(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _shingle_coverage(needle: str, haystack: str, window: int = 6) -> float:
    """Fraction of `needle`'s non-overlapping windows that appear in `haystack`.

    Used instead of exact containment because a verbatim-reuse rule that trips on
    a single inserted comma would fire constantly and get switched off. This
    measures *how much* of the frozen text survived, which is the thing that
    actually determines whether a character drifts.
    """
    n, h = normalise(needle), normalise(haystack)
    if len(n) < window:
        return 1.0 if n in h else 0.0
    windows = [n[i : i + window] for i in range(0, len(n) - window + 1, window)]
    if not windows:
        return 1.0
    return sum(1 for w in windows if w in h) / len(windows)


# --------------------------------------------------------------------------- #
# Rules                                                                        #
# --------------------------------------------------------------------------- #

Rule = Callable[[IRSections, IRContext], list[Violation]]


def r_description_present(ir: IRSections, ctx: IRContext) -> list[Violation]:
    if not ir.description or not ir.description.strip():
        return [Violation("description_present", "fatal", "第一段（description）为空")]
    return []


def r_no_internal_newline(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """The three sections are joined by a single newline, so a newline *inside* a
    section silently turns three sections into four or five. H3 then reads the
    soundscape as part of the visual description. Structurally fatal."""
    out = []
    for name, text in (("description", ir.description), ("soundscape", ir.soundscape), ("music", ir.music)):
        if text and "\n" in text:
            out.append(
                Violation(
                    "no_internal_newline",
                    "fatal",
                    f"{name} 内部含换行符；三段之间才用单个 \\n 分隔，段内不能有换行。"
                    "请把它改写成连续的一段文字。",
                )
            )
    return out


def r_description_length(ir: IRSections, ctx: IRContext) -> list[Violation]:
    n = count_cjk(ir.description)
    if n < 120:
        return [Violation("description_length", "fatal",
                          f"第一段只有 {n} 个汉字，太短，15 秒的画面撑不起来（目标 180-320）")]
    if n > 520:
        return [Violation("description_length", "warning",
                          f"第一段有 {n} 个汉字，偏长（目标 180-320），过长会让模型丢掉前半段的约束")]
    return []


def r_no_markdown(ir: IRSections, ctx: IRContext) -> list[Violation]:
    hit = [m for m in _MARKDOWN if m in ir.description]
    if hit:
        return [Violation("no_markdown", "fatal",
                          f"第一段含 markdown 标记 {hit}；IR 必须是纯散文")]
    return []


def r_no_meta_instruction(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """IR states what is on screen. The moment it addresses the model, the model
    has to spend capacity deciding what is description and what is instruction."""
    hit = [m for m in _META if m in ir.description]
    if hit:
        return [Violation("no_meta_instruction", "fatal",
                          f"第一段在对模型说话（{hit}）；IR 是对画面的陈述，不是指令")]
    return []


def r_no_filler(ir: IRSections, ctx: IRContext) -> list[Violation]:
    hit = [m for m in _FILLER if m in ir.description]
    if hit:
        return [Violation("no_filler", "warning",
                          f"第一段含无画面信息的空词 {hit}，应删除或替换为具体描述")]
    return []


def r_no_tech_params(ir: IRSections, ctx: IRContext) -> list[Violation]:
    hit = [m for m in _TECH if m in ir.description]
    if hit:
        return [Violation("no_tech_params", "fatal",
                          f"第一段写了技术参数 {hit}；分辨率/宽高比由 API 参数控制，"
                          "写进 IR 只会与之冲突")]
    return []


def r_camera_term(ir: IRSections, ctx: IRContext) -> list[Violation]:
    if not any(t in ir.description for t in _CAMERA_TERMS):
        return [Violation("camera_term", "warning",
                          "第一段没有任何机位词汇；开头应明确景别与镜头运动")]
    return []


def r_style_anchor_verbatim(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """The frozen visual grammar is the only thing making N independent
    generations look like one film. If it is missing, the beat is off-model."""
    if not ctx.style_anchor.strip():
        return []
    cov = _shingle_coverage(ctx.style_anchor, ir.description)
    if cov < 0.5:
        return [Violation("style_anchor_verbatim", "fatal",
                          f"视觉风格锚点只有 {cov:.0%} 被保留。必须把风格锚点**逐字**"
                          "抄进第一段末尾，不要改写、不要压缩、不要换同义词")]
    if cov < 0.85:
        return [Violation("style_anchor_verbatim", "warning",
                          f"视觉风格锚点保留了 {cov:.0%}，存在改写")]
    return []


def r_character_verbatim(ir: IRSections, ctx: IRContext) -> list[Violation]:
    out = []
    for ch in ctx.characters:
        if not ch.appearance.strip():
            continue
        cov = _shingle_coverage(ch.appearance, ir.description)
        if cov < 0.2:
            out.append(Violation("character_verbatim", "fatal",
                                 f"角色「{ch.name}」的外观描述几乎没有出现（{cov:.0%}）。"
                                 "必须逐字抄入，否则这个角色会在段落之间变脸"))
        elif cov < 0.6:
            out.append(Violation("character_verbatim", "warning",
                                 f"角色「{ch.name}」的外观只保留了 {cov:.0%}，存在改写风险"))
    return out


def r_dialogue_budget(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """Over-long dialogue is the most reliable way to break H3's audio: lip sync
    and voice both degrade when the line cannot fit the shot's duration."""
    spoken = sum(count_cjk(m.group(1)) for m in _QUOTED.finditer(ir.description))
    if spoken > ctx.dialogue_budget * 1.5:
        return [Violation("dialogue_budget", "fatal",
                          f"对白共 {spoken} 个汉字，上限 {ctx.dialogue_budget}。"
                          f"{ctx.seconds:.0f} 秒装不下，请删减或改为沉默")]
    if spoken > ctx.dialogue_budget:
        return [Violation("dialogue_budget", "warning",
                          f"对白 {spoken} 字，略超上限 {ctx.dialogue_budget}")]
    return []


def r_sequence_chain(ir: IRSections, ctx: IRContext) -> list[Violation]:
    """15 seconds is one action. A long chain of sequence connectives is the
    signature of several actions crammed into one beat, which H3 renders as mush."""
    markers = ("然后", "接着", "紧接着", "之后又", "再然后", "最后又")
    hits = sum(ir.description.count(m) for m in markers)
    if hits >= 3:
        return [Violation("sequence_chain", "warning",
                          f"第一段有 {hits} 处顺序连接词，像是把多个动作塞进了一拍；"
                          "15 秒只演一个动作")]
    return []


def r_tail_present(ir: IRSections, ctx: IRContext) -> list[Violation]:
    if not ctx.require_tail:
        return []
    out = []
    if not (ir.soundscape or "").strip():
        out.append(Violation("tail_present", "fatal", "第二段（overall_soundscape）为空"))
    if not (ir.music or "").strip():
        out.append(Violation("tail_present", "fatal", "第三段（non_diegetic_music）为空"))
    return out


def r_soundscape_is_diegetic(ir: IRSections, ctx: IRContext) -> list[Violation]:
    if not ir.soundscape:
        return []
    hit = [m for m in _MUSIC_WORDS if m in ir.soundscape]
    if hit:
        return [Violation("soundscape_is_diegetic", "warning",
                          f"第二段混入了音乐词汇 {hit}；第二段只写画内音，配乐归第三段")]
    return []


def r_music_is_non_diegetic(ir: IRSections, ctx: IRContext) -> list[Violation]:
    if not ir.music:
        return []
    hit = [m for m in _DIEGETIC_WORDS if m in ir.music]
    if hit:
        return [Violation("music_is_non_diegetic", "warning",
                          f"第三段混入了画内音 {hit}；第三段只写配乐")]
    return []


def r_tail_length(ir: IRSections, ctx: IRContext) -> list[Violation]:
    out = []
    for name, text, lo, hi in (
        ("soundscape", ir.soundscape or "", 10, 140),
        ("music", ir.music or "", 10, 140),
    ):
        n = count_cjk(text)
        if text and not (lo <= n <= hi):
            out.append(Violation("tail_length", "warning",
                                 f"{name} 有 {n} 个汉字，建议 {lo}-{hi}"))
    return out


RULES: list[Rule] = [
    r_description_present,
    r_no_internal_newline,
    r_description_length,
    r_no_markdown,
    r_no_meta_instruction,
    r_no_filler,
    r_no_tech_params,
    r_camera_term,
    r_style_anchor_verbatim,
    r_character_verbatim,
    r_dialogue_budget,
    r_sequence_chain,
    r_tail_present,
    r_soundscape_is_diegetic,
    r_music_is_non_diegetic,
    r_tail_length,
]


def validate(ir: IRSections, ctx: IRContext) -> list[Violation]:
    out: list[Violation] = []
    for rule in RULES:
        try:
            out.extend(rule(ir, ctx))
        except Exception as exc:  # noqa: BLE001
            # A buggy rule must never be able to block a beat.
            out.append(Violation(getattr(rule, "__name__", "rule"), "warning",
                                 f"规则本身出错，已跳过：{exc}"))
    return out


def fatals(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.severity == "fatal"]


def repair_instruction(violations: list[Violation]) -> str:
    lines = [f"{i + 1}. {v.message}" for i, v in enumerate(violations)]
    return "编译结果不合规，问题如下：\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# Director-side validation                                                     #
# --------------------------------------------------------------------------- #


def check_shot_intent(shot: ShotSpec, seconds: float, dialogue_budget: int) -> list[Violation]:
    """Catch the Director's two habitual failures before they reach PromptIR.

    Cheap to check here and expensive to notice later: a multi-action shot only
    reveals itself as a smeared, incoherent 15 seconds after a full generation.
    """
    out: list[Violation] = []

    connectives = ("然后", "接着", "紧接着", "，再", "，又", "之后")
    hits = [c for c in connectives if c in shot.action]
    if hits:
        out.append(Violation("single_action", "warning",
                             f"action 含顺序连接词 {hits}，可能塞了多个动作：{shot.action!r}"))
    if count_cjk(shot.action) > 40:
        out.append(Violation("single_action", "warning",
                             f"action 有 {count_cjk(shot.action)} 字，偏长，通常意味着不止一个动作"))

    spoken = sum(count_cjk(d.get("line", "")) for d in shot.dialogue)
    if spoken > dialogue_budget:
        out.append(Violation("dialogue_budget", "warning",
                             f"对白共 {spoken} 字，超出 {seconds:.0f} 秒的预算 {dialogue_budget} 字"))
    if len(shot.dialogue) > 2:
        out.append(Violation("dialogue_budget", "warning",
                             f"{len(shot.dialogue)} 句台词，{seconds:.0f} 秒最多 2 句"))

    # Camera terms inside `action` fight with `type`, which is the field that is
    # supposed to own the framing.
    if any(t in shot.action for t in ("镜头", "特写", "推进", "拉远", "航拍")):
        out.append(Violation("no_camera_in_action", "warning",
                             "action 里出现了镜头术语；机位应由 shot.type 表达"))
    return out


def dialogue_budget_for(seconds: float) -> int:
    """Chinese speech runs ~4 characters/second; budget 45% of the shot so the
    beat has room to breathe and the line is not rushed."""
    return max(8, int(seconds * 4 * 0.45))
