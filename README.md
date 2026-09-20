# h3_realtime_game

一部你边看边写的电影。设定世界观，引擎生成一段 14.4 秒的影像，影像结束时给你两个剧情走向；
选一个，下一段已经生成好了，立刻接着播。循环无限延伸。

视频来自自建的 [MiniMax-H3](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)
SGLang Diffusion 部署（`fl2va`，末帧续接），文本和图像来自 Bedrock。

完整设计、每个数字的来历、以及所有还没做的事，见 **[DESIGN.md](DESIGN.md)**。

## 组成

| 目录 | 是什么 |
| --- | --- |
| `services/orchestrator` | 全部的脑子。世界圣经（Worldsmith）、分支（Director）、提示词编译（PromptIR）、GPU 槽位调度、末帧抽取、SSE 事件流。FastAPI。 |
| `web` | 播放器。视频池、决断浮层、剧情树、历史记录。Next.js 16 App Router。 |
| `services/h3-wrapper` | 每个 SGLang 实例前面的一层薄包装。`GPU_BACKEND=h3` 时不走这条路，留着备用。 |
| `bench` | DESIGN 第 8 节的 P0 摸底套件。跑完才知道 19.3s 的隐藏窗口关不关得上。 |

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

H3 后端要求两台 GPU 机的 SGLang 通过 ssh 隧道可达——它只绑 `127.0.0.1:30010`，
并且 API 本身不传字节，条件帧上去、成片下来都走 ssh，所以 `H3_REPLICAS` 里的 ssh **别名**
是有用的，光给 URL 不够。

## 状态

能从头玩到尾。当前的已知短板都记在 `DESIGN.md` 里，最大的两个是：开场要等约 90 秒
（Worldsmith 一次调用 73 秒，纯平台延迟），以及 `REANCHOR_EVERY` 还是个猜的数，
等 `bench/chain_drift.py` 测出真值。
