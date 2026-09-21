# h3_realtime_game

一部你边看边写的电影。设定世界观，引擎生成一段 14.4 秒的影像，影像结束时给你两个剧情走向；
选一个，下一段已经生成好了，立刻接着播。循环无限延伸。

视频来自自建的 [MiniMax-H3](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)
SGLang Diffusion 部署（`fl2va`，末帧续接），文本和图像来自 Bedrock。

送给 H3 的提示词严格按
[MiniMax 官方格式](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)
拼装（指南已 vendor 到 `docs/h3official/`）。这不是风格偏好：台词不套官方那层
`<d>[语言] …</d>` 壳，H3 分不清"该念出这几个字"和"画面里有人在说话"，出来的是口型对得上、
但听不懂的人声。见 DESIGN 3.6。

完整设计、每个数字的来历、以及所有还没做的事，见 **[DESIGN.md](DESIGN.md)**。

## 组成

| 目录 | 是什么 |
| --- | --- |
| `services/orchestrator` | 全部的脑子。世界圣经（Worldsmith）、分支（Director）、提示词编译（PromptIR）、GPU 槽位调度、末帧抽取、SSE 事件流。FastAPI。 |
| `web` | 播放器。视频池、决断浮层、剧情树、历史记录。Next.js 16 App Router。 |
| `services/h3-wrapper` | 每个 SGLang 实例前面的一层薄包装。`GPU_BACKEND=h3` 时不走这条路，留着备用。 |
| `docs/h3official` | MiniMax 官方 `h3-prompt-writing` skill 的原样副本。提示词格式的唯一权威，代码里每一条 `§x.y` 引用都指向它。 |
| `bench` | 量具。DESIGN 第 8 节的摸底套件（延迟、帧级衔接、片内硬切、faststart），外加 `test_h3_format.py`：拿官方指南本身当 oracle 验提示词格式。 |

## 跑起来

两个服务各有一份 `.env.example`，键都带注释，照着抄成 `.env`：

```bash
# 编排器
cd services/orchestrator
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # 至少要填 H3_REPLICAS 和 AWS_BEARER_TOKEN_BEDROCK
.venv/bin/uvicorn app.main:app --port 8100

# 前端
cd web
npm install
cp .env.local.example .env.local
npm run dev
```

没有 GPU 也能开发：`FAKE_GPU=1` 用 ffmpeg 本地合成影像，但走的是同一套槽位记账和同一套末帧
抽取，所以整个故事引擎照样被跑到。

H3 后端全程走 HTTP：`H3_REPLICAS` 填两台 GPU 机的内网地址，条件帧由 H3 自己来拉
（`conditions[].uri` 给 URL），成片从 `GET /v1/videos/{id}/content` 取回。为此编排器会
另起一个**只读**的静态文件服务绑在 `0.0.0.0:ASSETS_PORT`（默认 8101，只服务 `ASSETS_DIR`
里的图片和成片）；控制接口仍然只绑 `127.0.0.1:8100`，只能从 ssh 隧道进来。拓扑和验证结果
见 `deploy/README.md`。

`H3_TRANSPORT=ssh` 是退路：条件帧用 ssh 推上去、成片用 scp 拉回来，`H3_REPLICAS` 里的 ssh
**别名**这时才是必需的。GPU 机不能回连我们的时候用它。

## 状态

能从头玩到尾。当前的已知短板都记在 `DESIGN.md` 里，最大的两个是：开场要等约 110 秒
（其中 Worldsmith 一次调用 89.7 秒，首 token 就等 52.4 秒，纯平台延迟），以及**视觉漂移现在
完全无人管**——重锚定会造成片内硬切，所以默认关掉了，而 20 拍的长链一次都没量过
（`bench/chain_drift.py` 就是为这个写的）。
