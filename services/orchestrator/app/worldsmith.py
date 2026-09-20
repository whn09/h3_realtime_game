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

log = logging.getLogger("h3game.worldsmith")


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
    style_anchor_en: str = ""
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


# The opening premises. Each one is deliberately built the same way, because the
# Worldsmith is only as good as the corner it is pushed into and these four
# ingredients are what stop it from writing a synopsis:
#
#   1. A place with a look. "被积雪掩埋的城市" gives the keyframe model and H3
#      something to agree on; "一个奇怪的世界" gives them nothing and they
#      disagree, which is visible as a cut that does not match.
#   2. A problem already in progress, not one about to start. The first shot is
#      14.4 seconds long -- there is no room for the protagonist to decide to
#      begin.
#   3. Something scarce and countable (三天的口粮, 电量将尽). This is where the
#      Worldsmith's `stat_names` come from; without it every world gets 体力/决心.
#   4. Second person, because the POV default is third and the contrast is what
#      tells the Worldsmith the player is the one in the frame.
#
# Twelve rather than four: a preset is the whole of the setup screen for most
# players -- the free-text box is a second click most never make -- so the grid is
# the actual genre range of the product. They are ordered roughly familiar-first.
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
    {
        "id": "wuxia",
        "title": "武侠",
        "premise": "你在雨夜的渡口等一条不会来的船，怀里揣着一封没有署名的信。"
                   "岸上的客栈里，有七个人今晚都想让你死。",
        "genre": "江湖武侠",
    },
    {
        "id": "cyberpunk",
        "title": "赛博朋克",
        "premise": "你是个卖记忆的二手贩子。今天收来的一段记忆里，"
                   "有人正用你的脸、你的声音，在一间你从没进过的房间里签下一份文件。",
        "genre": "赛博朋克",
    },
    {
        "id": "republic",
        "title": "民国",
        "premise": "一九三四年的深秋，你受雇去一栋停摆的洋楼里取一只箱子。"
                   "看门人说楼里没人，可三楼的留声机整夜都在放同一首曲子。",
        "genre": "民国怪谈",
    },
    {
        "id": "noir",
        "title": "黑色侦探",
        "premise": "雨水顺着你办公室的窗往下淌。一个不肯报姓名的女人放下一叠钞票，"
                   "只要你查一个已经死了三年的人现在住在哪里。",
        "genre": "黑色电影",
    },
    {
        "id": "space",
        "title": "星际",
        "premise": "你在一艘缓慢自转的货运飞船上值最后一班夜岗，全员还有十一个月才醒。"
                   "刚才，有人从内侧敲了三下气闸。",
        "genre": "太空科幻",
    },
    {
        "id": "western",
        "title": "西部",
        "premise": "你骑着一匹跛了的马进镇，水壶是空的，通缉令上是你的脸。"
                   "镇上唯一的水井被一个坐在摇椅上的老人看着。",
        "genre": "西部片",
    },
    {
        "id": "folkhorror",
        "title": "乡野怪谈",
        "premise": "你回到二十年没回过的山村替祖父下葬，村口的红纸写着你的名字。"
                   "全村人都来了，没有一个人看你的眼睛。",
        "genre": "民俗恐怖",
    },
    {
        "id": "steampunk",
        "title": "蒸汽朋克",
        "premise": "你是这座齿轮之城最后一个会修老钟的人。市政厅的大钟昨夜停在三点十七分，"
                   "而全城的人都还记得那一刻自己在做什么——除了你。",
        "genre": "蒸汽朋克",
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
                appearance_en="around thirty, short wind-tossed dark hair, old scar "
                              "through the eyebrow, medium build, worn dark grey heavy "
                              "cloth coat with frayed cuffs, faded indigo scarf, canvas "
                              "satchel on the left shoulder, brass whistle on the strap",
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

    # The three English fields cannot be repaired -- translating them here would
    # need another model call on the one path the player waits through, and a
    # machine translation of a prompt is not a prompt. So a missing one is recorded
    # and the keyframe falls back to what it did before: fewer constraints on the
    # image, and the same face lottery at every cut. Recorded rather than silent
    # because that is a quality regression nobody would otherwise see -- it looks
    # like the image model having a bad day.
    missing_en = [c.name for c in out.characters if not c.appearance_en.strip()]
    if missing_en:
        notes.append(f"角色缺少英文外观（appearance_en）：{'、'.join(missing_en)}，重锚帧会少一层约束")
    if not out.style_anchor_en.strip():
        notes.append("缺少 style_anchor_en，关键帧只能退回用中文风格锚点")

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
            style_anchor_en=out.style_anchor_en,
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
