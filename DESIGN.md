# 实时生成式开放世界互动影像

> 基于 MiniMax-H3 视频生成 + Bedrock LLM 的浏览器端互动叙事引擎。
> 用户设定世界观 → 引擎实时生成连续的影像段落 → 每段结束由用户选择剧情走向 → 无限延伸。

设计文档版本 v0.1 · 2026-09-20

---

## 0. 一句话产品定义

**不是"AI 生成视频工具"，是"一部你边看边写的电影"。**

衡量成败的唯一指标：**用户做出选择后，下一段影像是否立刻开始播放（零等待）**。
一旦出现 loading spinner，"世界"的幻觉就破了，产品退化成一个视频生成 demo。

因此下面所有架构决策都服务于一件事：**把生成延迟藏进播放时间里**。

---

## 1. 核心循环

```
        ┌─────────────────────────────────────────────────────┐
        │                                                     │
   [世界创建]                                                  │
   用户输入世界观 ──→ 生成 世界圣经 + 三幕大纲 + 主视觉关键帧      │
        │                                                     │
        ▼                                                     │
   [第 0 段] i2va 生成 15s 影像（带音频）                        │
        │                                                     │
        ▼                                                     │
   ┌─ 播放第 N 段 (15s) ──→ 决断时刻 (4s, 末帧定格) ─┐          │
   │                                               │          │
   │   ▲ 后台并行：                                 ▼          │
   │   │  Director 产出 N+1 的两个分支              用户选择     │
   │   │  PromptIR × 2 编译 IR                     A / B /    │
   │   │  H3 × 2 并行生成两段影像                    自由输入 ──┘
   │   │  ffmpeg 抽末帧 + 上传 CDN
   │   └─ 前端 preload 两条视频
   └────────────────────────────────────────────────
```

**关键：两个分支的视频都预先生成好。** 用户选哪个都是瞬时切换，另一条进入故事树缓存（可回溯重玩，不浪费）。

---

## 2. 时间预算 —— 本项目最重要的一张表

### 2.1 单拍（beat）串行链路

| 阶段 | 实现 | 实测/估计 | 能否并行 |
|---|---|---|---|
| Director LLM（出 2 个分支 + world state delta） | Bedrock Sonnet/Haiku，结构化 JSON | 2–3s | 两分支一次调用出，无法再拆 |
| PromptIR 编译 IR | Bedrock Haiku 4.5 + validator + repair | **8.1s**（实测均值） | 两分支**必须并行**调用 |
| H3 生成 15s 影像 | SGLang `/v1/videos` | 8–9s | 两分支**必须**打到 2 个实例 |
| ffmpeg 抽末帧 + 转封装 + 落盘 | GPU 机本地 | 0.3–0.6s | 并行 |
| CDN 首字节 + 前端 preload 起播 | CloudFront | 0.3s | — |
| **合计（串行关键路径）** | | **≈ 20.8s** | |

### 2.2 可用窗口

| 组成 | 时长 | 说明 |
|---|---|---|
| 影像段落 | 15s | `seconds` 上限即 15，用满 |
| 决断时刻 | 4s | 末帧定格 + Ken Burns + 选项卡片飞入 + 倒计时；**这是叙事节奏，不是等待** |
| 转场 crossfade | 0.3s | |
| **合计窗口** | **≈ 19.3s** | |

### 2.3 缺口 1.5s 与四个可用杠杆

按优先级实施：

1. **PromptIR 降到 ≤5s**（收益最大）

   先把 8.1s 拆开，否则会优化错地方：

   | 组成 | 估计 | prompt cache 命中后 |
   |---|---|---|
   | Prefill（guide 摘录 + 36 规则 + 4 gold 示例，粗估 8–15k tok） | 1–2s | **→ ~0.1s** |
   | Decode（IR 三段，粗估 600–1200 out tok @ ~100–150 tok/s） | **5–7s** | 不变 |
   | 网络 + 排队 | 0.3s | 不变 |

   **结论：prompt caching 只省 prefill，8.1s → 约 6–7s。该开（成本 $0.013→$0.005，且每拍间隔 19s ≪ 5min TTL，命中率近 100%），但它不解决延迟。真正的墙是 decode，只跟输出长度和吐字速度有关。**

   按性价比排的 decode 杠杆：
   - **砍输出 token（最划算）**：`overall_soundscape` 与 `non_diegetic_music` 基本由世界圣经的 `musicBible` 冻结，每拍差异极小 → 让 LLM **只生成 `integrated_multimodal_description`**，另两段模板填充。输出量 −40%，decode 降到 3–4s
   - **两个分支并发**调用 Bedrock（8.1s 是单次延迟，不是两次之和）
   - ⬜ **待核实：Bedrock 的 latency-optimized inference 能否用在这条路上。** 我之前写过"Haiku 4.5 支持、通常 +30–60% decode 吞吐"，这个说法我核不实，先撤回：(a) 官方的 Fast Mode 是 Claude API 侧的能力，且只覆盖 Opus 5 / Opus 4.8，不含 Bedrock、不含 Haiku；(b) Bedrock 自己的 latency-optimized 开关是另一回事，但我们推荐走的 Anthropic SDK Mantle client 路径并不暴露它。**结论：把它当成一个待验证项而不是既有杠杆**——如果要用，得先确认该 region + Haiku 4.5 是否在支持列表里，以及是否要为此改用 boto3 Converse（那会牺牲 SDK 侧的 prompt caching 写法一致性）。不要把时间预算押在这条上。
   - 限制 `max_tokens` 并在 prompt 里给出长度上界，不让模型自由发挥
   - Director 输出已高度结构化 → PromptIR 输入更短，少走 repair loop；repair 设 1 次上限
   - **终局方案：把 IR compiler 蒸馏到本地小模型。** 参考实现里的 `oracle.py` 正是为此而生——用 MiniMax 官方 H3-Context-IR API 采 ground truth 做蒸馏（README 明确提到）。7B 模型在 H100 上用 SGLang 跑，decode 可达 1000+ tok/s → **PromptIR 压到 1s 以内**，且 36 条规则的 validator 仍在，质量有硬保底。这条做成后时间预算彻底宽裕，且不再依赖 Bedrock 的网络往返
   - 退路：超时则回退到"模板拼装 IR"（把 Director 的结构化字段直接填进 IR 三段模板，跳过 LLM）。**fail-open 到模板，而不是 fail-closed 到没视频** —— 这是游戏，不是离线批处理

   ⚠️ 参考实现的 8.1s 是 **4 个请求**的均值，样本太小。P0 阶段必须跑 30+ 条拿到 p50/p95，并**把 prefill 与 decode 分开埋点**——决定用户会不会看到 spinner 的是 p95，不是均值。

2. **Director 流式输出，分支 A 先发车**
   让 Director 先吐 A 分支的完整 JSON 再吐 B，A 的 PromptIR 可提前约 1–1.5s 启动。两条视频不需要同时就绪，只要都在 19.3s 内。

3. **选项文字与视频解耦**
   选项 UI 只依赖 Director（3s 就有），**永不阻塞**。视频是"选项的预览与兑现"。卡片上用一个极低调的胶片进度指示表示该分支影像的就绪状态。用户在 T+15s 才需要真正点击，此时 UI 已存在 16s。

4. **投机展开（depth-2，GPU 富余时才开）**
   用户看第 N 段时，不只准备 N+1 的 A/B，还为 Director 预测概率更高的那一支准备 N+2。窗口从 19.3s 变成 38.6s，代价是 +1 条视频/拍的算力。**MVP 不做**，作为算力富余后的平滑升级。

### 2.4 首拍（冷启动）单独处理

开场没有"上一段在播"来掩盖延迟，链路是 世界圣经生成(4s) + 关键帧生成(3s) + PromptIR(5s) + H3(9s) ≈ **21s**。

这 21s 不要用 spinner，用**开场仪式**填满：
- 关键帧图像先到（3s），作为背景逐渐显影
- 世界圣经的文本以打字机效果逐行揭示（地点 / 你是谁 / 世界的规则 / 你要什么）
- 环境音床先起，音乐渐入
- 进度以"世界正在成形 …"的诊断式文案呈现

用户在读设定，21s 是**恰当的仪式长度**，不是等待。

---

## 3. 视觉与听觉连续性

这是"开放世界"和"一堆不相干的 AI 视频"之间的分界线。

### 3.1 帧级衔接：末帧 → 首帧（i2va）

所有段落（除开场）统一走 `task: "fl2va"`，只给首帧：

```jsonc
{
  "task": "fl2va",
  "conditions": [
    { "type": "image", "uri": "file:///clips/beat_017.last.png",
      "role": "keyframe", "frame_index": 0 }
  ]
}
```

GPU 机上的 wrapper 在每次生成后立刻 `ffmpeg -sseof -0.1 -i out.mp4 -frames:v 1 last.png`，
末帧图同时用于：① 下一段的首帧条件 ② 前端"决断时刻"的定格底图（与视频末帧像素级一致，零跳变）。**一次抽帧，两个用途。**

### 3.2 何时**不**衔接（硬切）

Director 显式输出 `transition: "continuous" | "cut" | "timeskip"`：
- `continuous` → 用末帧做首帧
- `cut` / `timeskip`（换场景、过了三天、视角切换）→ 不用末帧，改用 Nova Canvas 从世界圣经 + 目标场景描述新生成一张关键帧。**电影本来就有切镜**，硬切是表现力而非缺陷。

### 3.3 抗漂移：定期重锚定

末帧链式传递会累积色偏、锐度衰减、角色特征漂移。对策：
- 世界圣经里冻结 **style anchor**（胶片规格、镜头、光比、调色）和 **character sheet**（每个角色的外观描述，逐字复用），每次 IR 都注入
- 每 K 段（建议 K=6）或检测到质量退化时，插入一次 `cut`，用 Nova Canvas 基于原始 character sheet 重新生成关键帧，把视觉拉回基准
- 记录每段末帧的亮度/饱和度直方图，偏离基准超阈值即触发重锚定

### 3.4 `ref2va` 的取舍（重要架构约束）

`--model-variant` 是**启动参数**，一个 SGLang 实例只能是 `fl2va` 或 `ref2va` 之一。
- **两个实例都起 `fl2va`**（MVP 决策）：拿到帧级连续性，这是游戏性的刚需
- `ref2va` 的角色参考图能力很诱人（角色一致性更强），但会牺牲帧衔接。留作 v2：起第 3 个实例专门做"角色特写/新角色登场"这类镜头

### 3.5 音频连续性

H3 每段自带音频，段落边界会有音乐断点。三层处理：
1. 世界圣经冻结 `musicBible`（BPM、调性、编制、情绪基线），每段 IR 的 `non_diegetic_music` 字段复用同一描述 → 风格连贯
2. 前端叠一条**持续不断**的低音量 world ambience / music bed（按世界观从预制曲库选，或 MiniMax Music 生一条 loop），跨段不停 → 掩盖接缝
3. 段落交界 200ms 音频 crossfade

---

## 4. 世界状态：从"漂流"到"故事"

纯 prompt 链会迅速退化成语义随机游走。必须有结构化状态 + 长期目标。

### 4.1 世界圣经（World Bible，创建时生成一次，全局不变）

```ts
interface WorldBible {
  id: string;
  premise: string;            // 用户原始输入
  genre: string;              // 末日 / 现代都市 / 修仙 / ...
  logline: string;            // 一句话故事
  styleAnchor: string;        // 冻结的视觉语法，逐字注入每个 IR
  musicBible: string;         // 冻结的音乐语法
  protagonist: Character;     // 玩家扮演谁（第一人称 POV 还是第三人称跟随）
  pov: 'first' | 'third';
  characters: Character[];    // 角色外观圣经，防漂移
  worldRules: string[];       // 该世界的硬规则（修仙：境界体系；末日：感染机制）
  outline: ActOutline[];      // 三幕大纲：给 Director 的长期指北
  openingKeyframeUrl: string;
}

interface Character {
  id: string; name: string;
  appearance: string;   // 逐字复用，不允许 Director 改写
  voice: string;        // 音色描述，给 IR 的对白用
  arc: string;
}

interface ActOutline {
  act: 1 | 2 | 3;
  milestone: string;    // 该幕必须达成的叙事里程碑
  targetBeats: number;  // 预算拍数
}
```

### 4.2 可变状态（每拍更新）

```ts
interface WorldState {
  beatIndex: number;
  currentAct: 1 | 2 | 3;
  location: string;
  timeOfDay: string;
  elapsedInWorld: string;     // 世界内流逝时间
  presentCharacters: string[];
  inventory: string[];
  stats: Record<string, number>;  // 由世界观决定：修仙=灵力/境界，末日=体力/物资/感染度
  flags: Record<string, boolean | string>;  // 后果系统：谁死了、谁记恨你、哪扇门开了
  tension: number;            // 0-1，Director 用来控节奏
  summary: string;            // 压缩的前史，每 8 拍重写一次，防止无限膨胀
  recentBeats: string[];      // 最近 3 拍的原文摘要
}
```

**后果系统是"有重量的选择"的全部来源**：Director 的 system prompt 强制要求——每个新分支必须引用至少一个已有 `flag` 或 `stat`，否则选择就是装饰。

### 4.3 Director 输出契约

一次调用同时产出两个分支（省一次往返），严格 JSON schema + 校验：

```ts
interface DirectorOutput {
  narration: string;            // 给 UI 的一句旁白/章节标题
  stateDelta: Partial<WorldState>;
  options: [BranchIntent, BranchIntent];
  predictedChoice: 0 | 1;       // 用于 depth-2 投机展开的优先级
  isEnding?: { type: string; title: string };
}

interface BranchIntent {
  label: string;                // 选项文案，≤14 字，动词开头
  consequenceHint: string;      // 给玩家的模糊代价暗示（不剧透）
  transition: 'continuous' | 'cut' | 'timeskip';
  shot: {
    type: 'wide' | 'medium' | 'closeup' | 'pov' | 'tracking' | 'aerial';
    subject: string;
    action: string;             // 15 秒内能演完的**一个**动作
    setting: string;
    mood: string;
    dialogue?: { speaker: string; line: string }[];  // 交给 IR 的对白，注意音节预算
    sfxFocus: string;
  };
  stateDeltaIfChosen: Partial<WorldState>;
}
```

**镜头多样性约束**：Director 的 prompt 里注入最近 3 拍的 `shot.type`，禁止连续三拍同机位。否则 15s × N 段会视觉催眠。

**动作单一性约束**：15 秒只能演一个动作。Director 最常犯的错是塞进"他推开门、穿过走廊、爬上屋顶、看见飞船"——H3 会做出一团糊。system prompt 里要有反例。

### 4.4 大纲牵引

Director 每拍都收到：`当前幕 / 该幕里程碑 / 已用拍数 / 预算拍数`。
- 拍数过半仍未接近里程碑 → 提升 `tension`，分支必须推进主线
- 里程碑达成 → 进入下一幕，`summary` 重写
- 第三幕里程碑达成 → 输出 `isEnding`，进入结局段（多结局，由 flags 决定哪一个）

这是"无限漂流"和"一部有结构的电影"的区别。

---

## 5. 系统架构

### 5.1 部署拓扑（推荐方案）

```
┌────────────────────┐        ┌──────────────────────────────────────┐
│  浏览器             │        │  GPU 主机 / 同机房 VM                  │
│  Next.js (Vercel)  │        │                                      │
│  ┌──────────────┐  │  WS    │  ┌──────────────────────────────┐    │
│  │ 双 video A/B │◄─┼────────┼─►│ Orchestrator (长驻 Node)      │    │
│  │ 决断时刻 UI  │  │  SSE   │  │  · 会话状态机 / 拍流水线       │    │
│  │ 故事树面板   │  │        │  │  · GPU 槽位调度器             │    │
│  └──────────────┘  │        │  │  · Redis (会话/故事树/job)    │    │
└────────┬───────────┘        │  └────┬──────────────┬──────────┘    │
         │ 视频/图片           │       │              │               │
         ▼                    │       ▼              ▼               │
┌────────────────────┐        │  ┌─────────┐   ┌──────────────┐     │
│ CloudFront / CDN   │◄───────┼──│ nginx   │   │ h3-wrapper   │     │
└────────────────────┘        │  │ /clips  │   │ (FastAPI)    │     │
         ▲                    │  └─────────┘   │ ·调 SGLang   │     │
         │ 异步备份             │                │ ·ffmpeg抽帧  │     │
    ┌────┴─────┐               │                └──┬────────┬──┘     │
    │ S3 归档   │               │                   ▼        ▼        │
    └──────────┘               │            SGLang#1    SGLang#2     │
                               │            (fl2va)     (fl2va)      │
                               └──────────────────────────────────────┘
                                          │
                                          ▼  Director / PromptIR
                                   AWS Bedrock (Haiku 4.5 / Sonnet)
```

**为什么编排器放在 GPU 机旁而不是 Vercel Functions：**
- 每拍的流水线是**跨请求的有状态长流程**，serverless 里要拆成 queue + 外部状态存储，多出好几跳网络
- 编排器与 SGLang / nginx 同机房 → 调 H3 和拿视频 URL 都是内网，省掉 1–2s（在 1.5s 缺口面前很关键）
- 视频文件不用跨网上传即可播放（nginx 直出 + CDN 回源），省掉 1s 的 S3 上传
- Vercel 端只跑 UI 和 BFF（会话创建、鉴权、静态资源），这是它最擅长的部分

若坚持全 Vercel：编排器改为 Vercel Queues + 一个外部 Redis（Upstash），视频走 Blob。可行，但每拍要多付 ~1.5–2.5s 的网络往返，得靠 depth-2 投机展开来补。

### 5.2 GPU wrapper（必须自己写一层，不要让前端直连 SGLang）

`h3-wrapper` 的职责，全部在 GPU 机本地完成：

```
POST /generate
  body: { ir: {desc, soundscape, music}, seconds, firstFrame?: path,
          shortEdge, steps, quality, seed, jobId }
  → 1. 组装 SGLang /v1/videos 请求（t2va 或 fl2va）
  → 2. 调用 SGLang，落盘 mp4
  → 3. ffmpeg: 抽末帧 png / 探测真实时长 / faststart 重封装
  → 4. 计算末帧直方图（漂移检测）
  → 5. 移动到 /var/www/clips/<jobId>/
  → 6. 异步 fire-and-forget 上传 S3
  ← { videoUrl, lastFrameUrl, posterUrl, durationMs, histogram,
      timings: { sglangMs, ffmpegMs } }
```

`timings` 必须逐项返回 —— 时间预算是这个项目的生死线，没有细粒度埋点就没法调优。

### 5.3 GPU 槽位调度器

2 个 SGLang 实例 = **2 个并发生成槽**。

| 模式 | 每会话占用 | 支持并发会话 | 代价 |
|---|---|---|---|
| **双分支预生成**（MVP） | 2 槽 | **1** | 零等待 |
| 单分支预生成（按 `predictedChoice`） | 1 槽 | 2 | 约 40% 概率触发 ~15s 等待 |
| depth-2 投机 | 3 槽 | 0（需扩容） | 窗口翻倍 |

MVP = 单会话零等待 + 一个"候场室"（排队位次 + 可以先写世界观设定，排到了直接开场）。这比让 4 个人同时卡在 spinner 里体验好得多。

调度器实现：优先级队列，`priority = (是否阻塞播放) × (会话活跃度) × (predictedChoice 权重)`。用户已做出的选择所在分支永远最高优先级——它可能还没生成完。

---

## 6. 前端：无缝播放的实现细节

### 6.1 双 video 元素交换

```
videoA (playing beat N)   videoB (preload="auto", src = 选中分支)
        └── ended ──→ 交换 z-index / 播放 videoB / videoA 载入下一段
```
两个 `<video>` 常驻，永不 unmount，避免重建 decoder 导致的黑屏。两条分支各自 preload：实际用 3 个 element（当前 + 分支A + 分支B），选中后另一条释放。

### 6.2 决断时刻（4s）

视频 `ended` → 立刻显示 `lastFrameUrl` 的 `<img>`（与末帧像素一致 → 无感切换）→ CSS 极缓慢 `scale(1 → 1.04)` Ken Burns → 选项卡片飞入 → 4s 环形倒计时。

- 音乐床不中断
- 倒计时结束未选择 → 按 `predictedChoice` 自动选（**永不停顿**，节奏是体验的一部分）
- 用户提前选择 → 立即转场，不等满 4s（奖励果断）

### 6.3 第三条路：自由输入

选项区常驻一个"或者…"输入框。用户输入任意行动 → 无法预生成 → 显示 8–20s 的过场（末帧定格 + "世界在回应你…" + 打字机呈现 Director 的旁白）。

**这个等待是可接受的**，因为用户主动偏离了预设路径，心理上预期"我要求了特别的东西"。这恰恰是"开放世界"感的来源，不要因为延迟而砍掉。

### 6.4 故事树面板

DAG 可视化。未选择的分支视频已经生成好且缓存 → 点击即可回到岔路口重走，**几乎零边际成本**。这是双分支预生成的免费红利，也是留存钩子。

### 6.5 导出与分享

- `POST /export` → ffmpeg concat 用户路径上所有 clip + 交界 crossfade + 片头片尾字卡 → 一部 2–5 分钟的"你的电影"，可下载/分享
- 分享链接编码选择路径 → 别人可以看你的版本，并**从任意一拍分叉出自己的故事**

---

## 7. 降级与兜底（必须有，GPU 会抖）

| 故障 | 降级策略 |
|---|---|
| 分支视频未就绪且用户已选 | 末帧定格 + 旁白打字机 + 音乐床，绝不黑屏/spinner |
| H3 超时 | 重试一次并降参：`steps 50→32`、`quality high`、`seconds 15→10` |
| H3 二次失败 | Nova Canvas 生成 2–3 张关键帧做 Ken Burns 幻灯 + 旁白，剧情继续（"静帧章节"，包装成风格化手法） |
| PromptIR validator 失败/超时 | 回退模板拼装 IR（Director 字段直填三段模板），**fail-open** |
| Director JSON 不合 schema | 重试一次 → 仍失败则用"通用二选一"兜底（前进/后退、战/逃） |
| 视觉漂移超阈值 | 强制 `cut` + Nova Canvas 重锚定关键帧 |
| 单个 SGLang 实例掉线 | 调度器降级到单分支模式，按 `predictedChoice` 生成 |

原则：**故事永不中断**。任何一层失败，都要有一个仍然能推进叙事的形态。

---

## 8. 上线前必测清单（等你把 API 搭好我来跑）

文档没写清、必须实测确认的事项，按阻塞程度排序：

1. **`/v1/videos` 是同步还是异步？返回体是文件路径 / base64 / URL？** —— 决定 wrapper 的整个形态
2. **`--model-variant fl2va` 的实例能否跑纯 t2va（不传 conditions）？** —— 若不能，开场必须依赖外部图像模型出关键帧（当前设计已按此假设，属安全侧）
3. **i2va（只给首帧）在 15s 上的实际延迟**，相对 t2va 慢多少 —— 直接影响 1.5s 缺口
4. **`num_inference_steps` 30/40/50 的质量-延迟曲线**，以及 `quality` 三档的实际差异 —— 这是最主要的降级旋钮
5. **同实例并发 2 请求 vs 2 实例各 1 请求**的吞吐对比 —— 决定 2 个实例是不是真的等于 2 个槽
6. **末帧链式续接 6 段后的漂移程度**（色偏/锐度/角色特征）—— 决定重锚定周期 K
7. **段落边界的音频可听接缝**有多明显 —— 决定音乐床的必要音量
8. **PromptIR 的 prefill / decode 分项延迟 p50 与 p95**（≥30 条样本，开与不开 prompt caching 各一组）—— 缺口能否闭合全看这里。同时记录 IR 平均输出 token 数，用来估"砍两段模板化"的收益上限
9. `seconds=15` 是否稳定（上限值常有边界 bug）
10. `short_edge` 480 vs 768 的延迟差 —— 决定是否有余量上 720p

---

## 9. 成本量级（每拍）

| 项 | 估计 |
|---|---|
| Director（Sonnet，~1.5k in / 700 out） | ~$0.012 |
| PromptIR × 2（Haiku 4.5，system prompt 缓存） | ~$0.010 |
| Nova Canvas 关键帧（仅 cut 时，约 1/6 拍） | ~$0.007 |
| H3 生成 × 2 | 自建 GPU，约 2 × 9 GPU·s |
| 存储/CDN | 约 2 × 4MB |
| **LLM 侧合计** | **~$0.025 / 拍** |

一局 20 拍 ≈ $0.5 的 LLM 成本 + 约 6 GPU·分钟。GPU 是绝对瓶颈，不是 LLM。

---

## 10. 实施路线

| 阶段 | 目标 | 产出 |
|---|---|---|
| **P0 摸底** | 跑通第 8 节测试清单 1–5 | 一份实测数据表，确认时间预算能否闭合 |
| **P1 骨架** | 单拍能跑：写死世界观 → PromptIR → H3 → 浏览器播放 | h3-wrapper + 最小 Next.js 播放器 |
| **P2 循环** | Director + 双分支预生成 + 决断时刻 + 无缝交换 | 可玩的核心循环，10 拍不断 |
| **P3 世界** | 世界圣经 + 三幕大纲 + flags/stats + 末帧衔接 + 重锚定 | "有故事"而非"有视频" |
| **P4 产品** | 故事树回溯 + 导出电影 + 分享 + 候场室 + 全套降级 | 可以给人玩 |
| **P5 规模** | 槽位调度器 + depth-2 投机 + 多会话 | 多人同时在线 |

**P0 是硬门槛**：如果实测发现 PromptIR 压不到 5s 且 i2va 明显比 t2va 慢，就必须提前启用 depth-2 投机展开（需要第 3、4 个 GPU 实例）。这个结论要在写任何 UI 代码之前拿到。

---

## 11. 内容安全

- 世界观输入过 Bedrock Guardrails（拒绝真实人物、未成年人相关、极端暴力具体化）
- Director 输出过一遍同一 Guardrail，防止在开放叙事中逐步漂移到红线
- 每个会话的世界观 + 所有 IR 落盘存档，可审计
- 前端明确标注 AI 生成，导出视频加水印

---

## 附：SGLang 请求模板

开场（无末帧可用，靠外部关键帧）：
```jsonc
{
  "model": "MiniMaxAI/MiniMax-H3",
  "task": "fl2va",
  "prompt": "<PromptIR 输出的三段 IR，单 \\n 分隔>",
  "conditions": [
    { "type": "image", "uri": "file:///keyframes/<sid>/opening.png",
      "role": "keyframe", "frame_index": 0 }
  ],
  "target": { "short_edge": 480, "aspect_ratio": "16:9", "duration_seconds": 15 },
  "seconds": 15,
  "quality": "high",
  "num_inference_steps": 50,
  "num_outputs_per_prompt": 1,
  "seed": 0
}
```

续接段落：同上，`uri` 换成 `file:///clips/<prevJobId>/last.png`。
硬切段落：同上，`uri` 换成新生成的 Nova Canvas 关键帧。
