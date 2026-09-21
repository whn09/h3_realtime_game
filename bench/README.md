# P0 摸底套件

对应 `DESIGN.md` 第 8 节的必测清单。**P0 是硬门槛**：这些数据决定 19.3s 窗口能否闭合，
从而决定是否需要提前上 depth-2 投机展开（要多两个 GPU 实例）。在写任何 UI 代码之前先跑完。

四个脚本都会在输出末尾打印一段 `DECISIONS UNLOCKED`，直接把测到的数字翻译成它所决定的架构选择。

```bash
pip install -r requirements.txt     # 在 GPU 机上跑
```

## 建议执行顺序

### 1. `probe_sglang.py` — 清单 #1 #2 #9

先跑这个。它决定后面所有脚本能不能用。

```bash
python probe_sglang.py --wrapper http://127.0.0.1:8000
```

回答：`/v1/videos` 是同步还是异步、视频以 URL / 本地路径 / base64 哪种形式返回、
`--model-variant fl2va` 的实例能否吃纯 t2va 请求、`seconds` 的上下边界是否稳定。

走 h3-wrapper 的 `/probe`，payload 原样转发、响应原样回显，不经任何归一化。

### 2. `bench_h3.py` — 清单 #3 #4 #5 #10

```bash
# 延迟矩阵
python bench_h3.py --wrapper http://127.0.0.1:8000 \
    --tasks t2va,i2va --steps 32,40,50 --short-edge 480,768 --reps 5

# 并发形态（模式 A 需要该 wrapper 以 MAX_CONCURRENT=2 启动）
python bench_h3.py --wrapper http://gpu1:8000 --wrapper http://gpu2:8000 --concurrency-test
```

i2va 比 t2va 慢多少、steps/quality/分辨率的延迟曲线、一个实例扛 2 个并发是否等于两个实例。
**延迟是量出来的，画质得自己看** —— 脚本会把每个格子的视频 URL 分组打印出来供并排对比。

最后会打印 BUDGET CHECK：用实测的 i2va p95 去套 19.3s 窗口，直接给 FITS / OVER。

### 3. `bench_promptir.py` — 清单 #8

整条链路里最危险的一段（参考实现实测 8.1s，但只有 4 个样本）。

```bash
export AWS_REGION=us-west-2
python bench_promptir.py --system-file ir_system_prompt.txt --reps 30
```

先把参考实现里真实的 system prompt（官方 guide 摘录 + 36 条规则 + 4 个 gold 示例）导出到文件——
它的体积正是 prefill 成本的来源，用简化版测没有意义。

产出：prefill / decode 分项的 p50 与 p95、开与不开 prompt caching 两组对比、
以及 IR 三段各自占多少输出字符（这直接给出"把 soundscape/music 两段模板化"能省多少 decode）。

走 Anthropic SDK 的 Bedrock Mantle client（不是 boto3 Converse），模型 `anthropic.claude-haiku-4-5`。
如果 `cache_read_mean` 是 0，脚本会报警——说明缓存根本没生效，此时任何"已缓存"的数字都不可信。

### 4. `chain_drift.py` — 清单 #6 #7

```bash
python chain_drift.py --wrapper http://127.0.0.1:8000 --beats 8
```

真的把末帧→首帧的反馈回路跑 8 拍，测色偏 / 对比度 / 饱和度 / 锐度相对种子帧的漂移，
给出重锚定周期 K 的实测值（而不是 `DESIGN.md` 里假设的 6）。

刻意保持场景不变，只让串联跳数变化，这样漂移才归因于串联本身而非剧情换了地方。

产出 `drift_contact_sheet.png`（8 个末帧并排，肉眼看漂移）和 `drift_chain.mp4`
（硬切拼接、不加交叉淡化，故意是最坏情况 —— 在每个 15s 接缝处听音乐是否明显重启）。

## 结果

都写到 `results/`：CSV 给做表，JSON 留原始响应。
`probe_sglang.json` 里的 raw 响应在后面调 wrapper 的时候还要回头查。

---

## 事后量具（读已经跑完的存档，不花钱、不占 GPU）

上面四个是上线前摸底；下面这些是每次玩家报"画面不对"的时候拿来用的。都在编排器那台机器上跑，
读 `DATA_DIR/sessions/<sid>/session.json`，不给 session id 就取最新的一局。

```bash
set -a; . /home/ubuntu/kunlun/.env; set +a      # 它们要 settings.data_dir / settings.ffmpeg
./.venv/bin/python bench/internal_cut_scan.py       [<sid>]
./.venv/bin/python bench/frame_continuity_check.py  [<sid>]
```

- **`internal_cut_scan.py`** —— 片子有没有**在自己片内**切镜。判据是**逐帧相邻**相关性里的台阶
  （某一对相邻帧掉到 0.8 以下，两边还在 0.99），不是"第 0 帧和第 240 帧不像"——后者在 14s 运动镜头
  上会把 21 拍报成 20 拍。`DESIGN.md` 头表第 10 条就是这个脚本量出来的。
- **`frame_continuity_check.py`** —— 片子的第 0 帧是不是我们给的那张图。**它看不见上面那条**：
  `fl2va` 钉住第 0 帧，所以它的答案永远是"对"。两个脚本回答的是不同的问题，别拿一个代替另一个。
- **`prepare_race_check.py`** —— 纯 stub、不联网，唯一会返回非零退出码的一个：`PREPARE_AHEAD`
  提前做的活会不会和正式生成打架（重复画图 / 重复编译 IR）。改 `engine.py` 的准备逻辑之后跑它。
