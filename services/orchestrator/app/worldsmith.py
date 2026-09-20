"""The Worldsmith: one premise in, one frozen world bible out.

Runs once per session, and everything downstream depends on it, so this is the
one call where quality matters more than latency -- it is hidden behind the
opening ritual (DESIGN.md section 2.4) rather than inside a beat budget.

It also has to produce a *playable* bible, not just a readable one. A bible with
no stats and no opening flags gives the Director nothing to bind consequences to,
and the whole consequence system quietly degrades into decorative buttons. So
`_ensure_playable` fills those in when the model skips them.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from . import prompts
from .config import settings
from .llm import LLM, LLMError
from .schema import ActOutline, Character, ShotSpec, WorldBible, WorldState

log = logging.getLogger("kunlun.worldsmith")


class InitialState(BaseModel):
    location: str = ""
    time_of_day: str = ""
    present_characters: list[str] = Field(default_factory=list)
    inventory: list[str] = Field(default_factory=list)
    stats: dict[str, float] = Field(default_factory=dict)
    flags: dict[str, object] = Field(default_factory=dict)


class WorldsmithOutput(BaseModel):
    genre: str = ""
    logline: str = ""
    style_anchor: str = ""
    music_bible: str = ""
    ambience: str = ""
    pov: str = "third"
    protagonist_id: str = ""
    characters: list[Character] = Field(default_factory=list)
    world_rules: list[str] = Field(default_factory=list)
    stat_names: list[str] = Field(default_factory=list)
    outline: list[ActOutline] = Field(default_factory=list)
    initial_state: InitialState = Field(default_factory=InitialState)
    opening: ShotSpec | None = None
    opening_keyframe_prompt: str = ""


PRESETS = [
    {
        "id": "apocalypse",
        "title": "末日",
        "premise": "核冬天后的第七年，你在一座被积雪掩埋的城市里独自搜寻幸存者的信号，"
                   "背包里只剩三天的口粮和一台电量将尽的收音机。",
        "genre": "末日废土",
    },
    {
        "id": "city",
        "title": "现代都市",
        "premise": "你是一名深夜出勤的急救调度员，今晚的第一通报警电话来自一个你以为已经拆掉的地址。",
        "genre": "都市悬疑",
    },
    {
        "id": "cultivation",
        "title": "修仙",
        "premise": "师父在你面前坐化，把一枚来历不明的玉符塞进你掌心。山门外，"
                   "三大宗门的人已经在等你走出来。",
        "genre": "东方修仙",
    },
    {
        "id": "deepsea",
        "title": "深海",
        "premise": "你驾驶一艘单人潜水器下潜到四千米，任务是回收一台失联的观测站，"
                   "但声呐上多出了一个不该存在的回波。",
        "genre": "深海科幻",
    },
]

_DEFAULT_STATS = ["体力", "决心"]


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_")
    return s or fallback


def _ensure_playable(out: WorldsmithOutput) -> list[str]:
    """Fill in whatever the bible needs to be playable but the model omitted."""
    notes: list[str] = []

    if not out.characters:
        out.characters = [
            Character(
                id="protagonist",
                name="旅人",
                appearance="三十岁上下，短发被风吹乱，眉骨有一道旧疤，中等身材；"
                           "穿深灰色厚布外套，袖口磨损，脖颈围着一条褪色的靛蓝围巾，"
                           "左肩挎一只帆布背包，背包带上系着一枚黄铜哨子。",
                voice="低沉、气息偏重",
                arc="从独行到不得不依靠别人",
            )
        ]
        notes.append("世界圣经没有角色，已补一个默认主角")

    ids = {c.id for c in out.characters}
    for c in out.characters:
        if not c.id:
            c.id = _slug(c.name, "character")
    if out.protagonist_id not in ids:
        out.protagonist_id = out.characters[0].id
        notes.append("protagonist_id 不在角色表里，已指向第一个角色")

    if out.pov not in ("first", "third"):
        out.pov = "third"

    if not out.stat_names:
        out.stat_names = list(_DEFAULT_STATS)
        notes.append("没有数值体系，已补默认数值")

    if not out.outline:
        out.outline = [
            ActOutline(act=1, milestone="确立主角的处境与第一个具体目标", target_beats=6),
            ActOutline(act=2, milestone="目标受挫，付出一次不可逆的代价", target_beats=7),
            ActOutline(act=3, milestone="与最初的处境正面对决", target_beats=6),
        ]
        notes.append("没有三幕大纲，已补默认大纲")
    for a in out.outline:
        a.target_beats = max(3, min(12, a.target_beats))

    st = out.initial_state
    # Every stat needs a starting value, or the Director cannot bind to it and
    # `_bind_consequences` has nothing real to fall back on.
    for name in out.stat_names:
        st.stats.setdefault(name, 70.0)
    if not st.present_characters:
        st.present_characters = [out.protagonist_id]
    if len(st.flags) < 2:
        # Two opening facts are the minimum for the consequence system to have
        # anything to reference on beat 1.
        st.flags.setdefault("alone", True)
        st.flags.setdefault("supplies_low", True)
        notes.append("开局 flags 不足 2 个，已补默认 flags")

    if out.opening is None:
        out.opening = ShotSpec(
            type="wide",
            subject=out.characters[0].name,
            action="站定，缓慢环视周围的环境",
            setting=st.location or "故事开始的地方",
            mood="孤寂",
            sfx_focus="风声",
        )
        notes.append("没有开场镜头，已补默认开场")
    # The opening must never carry dialogue: there is nothing established yet for
    # a line to land against, and it costs syllable budget on the one beat whose
    # job is purely to establish space.
    out.opening.dialogue = []

    if not st.location:
        st.location = out.opening.setting
    if not st.time_of_day:
        st.time_of_day = "黄昏"
    if not out.ambience:
        out.ambience = "低频的环境底噪，远处偶发的不明声响"
        notes.append("没有环境音底噪，已补默认")
    return notes


class Worldsmith:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def build(
        self, premise: str, genre_hint: str = "", pov_hint: str = ""
    ) -> tuple[WorldBible, WorldState, list[str], dict[str, float]]:
        user = prompts.worldsmith_user(premise, genre_hint, pov_hint)
        try:
            out, meta = await self.llm.structured(
                settings.model_worldsmith,
                prompts.WORLDSMITH_SYSTEM,
                user,
                WorldsmithOutput,
                # 12000, not 6000. Measured output for this prompt is 4.6-5.2k
                # tokens, so 6000 left almost no headroom and ordinary variance tipped
                # some openings into a truncated object -- which cost a whole second
                # round trip, 40-70s of it TTFT, on the one call the player waits for
                # with nothing on screen. Raising the ceiling is free: billing is on
                # tokens actually produced, not on the budget requested.
                max_tokens=12000,
                thinking=settings.worldsmith_thinking,
                temperature=None if settings.worldsmith_thinking else 1.0,
            )
            timings = meta.timings("worldsmith")
        except LLMError as exc:
            raise LLMError(f"世界构建失败：{exc}") from exc

        notes = _ensure_playable(out)
        if notes:
            log.info("worldsmith repairs: %s", notes)

        bible = WorldBible(
            premise=premise,
            genre=out.genre or genre_hint,
            logline=out.logline,
            style_anchor=out.style_anchor,
            music_bible=out.music_bible,
            ambience=out.ambience,
            pov="first" if out.pov == "first" else "third",
            protagonist_id=out.protagonist_id,
            characters=out.characters,
            world_rules=out.world_rules,
            outline=sorted(out.outline, key=lambda a: a.act),
            stat_names=out.stat_names,
            opening=out.opening,
            opening_keyframe_prompt=out.opening_keyframe_prompt,
        )
        st = out.initial_state
        state = WorldState(
            beat_index=0,
            act=1,
            beats_in_act=0,
            location=st.location,
            time_of_day=st.time_of_day,
            elapsed_in_world="刚开始",
            present_characters=st.present_characters,
            inventory=st.inventory,
            stats=dict(st.stats),
            flags=dict(st.flags),
            tension=0.3,
            summary="",
            recent_beats=[],
        )
        return bible, state, notes, timings
