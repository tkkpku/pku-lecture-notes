# pku-lecture-notes

课程录像转 Markdown 讲义。适用于有录播的课程，不限于北大。

音频走 ASR，画面抽关键帧读文字，两者按时间戳对齐后交给 LLM 融合。目标是能复习的讲义，不是逐字稿。

## 依赖

- Python 3.10+
- ffmpeg 与 ffprobe（在 PATH 中，或填 `paths.ffmpeg_dir`）
- 一个 OpenAI 兼容的 LLM 端点。读图阶段需要多模态模型
- faster-whisper（可选，约 2~3 GB）
- pku3b（可选，仅从教学网直接下载时需要）

## 安装

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-asr.txt          # 需要转写时再装

cp config/config.example.yaml config/config.yaml
cp config/ignored_lectures.example.yaml config/ignored_lectures.yaml
cp config/terms.example.txt config/terms.txt

python scripts/doctor.py --net
```

`config.yaml` 必须是 UTF-8 无 BOM，PyYAML 见到 BOM 会报错。

密钥放环境变量，变量名由 `llm.key_env` 指定：

```powershell
$env:LLM_API_KEY = "sk-..."
```

本项目不读 `.env` 文件。

## 用法

先用合成素材验证链路，不需要账号也不用录像：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/make_synth_tts.ps1
python scripts/make_synth_lecture.py --lecture 第01讲_演示
python src/pipeline.py --course 合成课程 --lecture 第01讲_演示 --video data/raw/合成课程/第01讲_演示.mp4
```

第二个脚本只造画面，语音由第一个脚本用系统 SAPI 生成。缺语音时音频预检会判定该素材无人讲话并中止，这是预期行为。

跑真实数据：

```powershell
# 从教学网下载（pku3b 需先自行登录一次）
python src/pipeline.py --course 课程名 --lecture 第03讲_主题

# 已有视频
python src/pipeline.py --course 课程名 --lecture 第03讲_主题 --video D:\videos\lesson03.mp4

# 只跑部分阶段
python src/pipeline.py --course 课程名 --lecture 第03讲 --stages ocr,fuse
```

`--stages` 可取值：`fetch` `extract` `asr` `ocr` `fuse` `publish` `review`。
`--dry-run` 只对 `fetch`、`extract`、`publish` 生效，`asr`、`ocr`、`fuse` 仍会实际执行。

## 阶段

| 代号 | 模块 | 输入与产出 |
| --- | --- | --- |
| fetch | s1_fetch | 教学网 → raw/<课程>/*.mp4 |
| extract | s2_extract | mp4 → audio.wav、frames/、frames_index.json |
| asr | s3_asr | audio.wav → asr.json / asr.srt / asr.txt |
| ocr | s4_ocr | 帧 → frames_ocr.json |
| fuse | s5_fuse | asr + frames_ocr → lecture.md |
| publish | s6_publish | lecture.md → 讲义、索引、清理中间产物 |
| review | s7_review | 整章讲义 → 章节复习文档 |

`s2b_probe.py` 是 `extract` 的子步骤，当 `probe.enabled` 为真时自动执行，不是独立阶段。

每个阶段幂等：产物存在即跳过，`--force` 覆盖。状态写在 `work/<课程>/<讲次>/run_status.json` 的 `results.<stage>.ok`。

## 配置

完整注释见 `config/config.example.yaml`。常改的几项：

| 键 | 说明 |
| --- | --- |
| `paths.data_root` | 中间产物目录。相对路径按项目根解析，默认 `data` |
| `paths.obsidian_vault` | 讲义输出目录，任意 Markdown 库均可。留空则只出草稿，`publish` 会拒绝执行 |
| `llm.base_url` `llm.model` `llm.key_env` | LLM 端点、模型、密钥所在环境变量名 |
| `asr.model` `asr.compute_type` | 默认 large-v3 + float16。显存不足降 medium，或用 `scripts/check_env.py` 查本机支持的精度 |
| `vision.backend` | `vlm`（默认）、`winocr`、`hybrid`。WinOCR 仅 Windows 可用 |
| `publish.auto_cleanup` | 发布后删除视频、音频、帧。有安全阀：讲义不存在则不删 |

## 目录

```
src/                  流水线各阶段，可单独运行
scripts/doctor.py     环境与配置自检
scripts/make_synth_tts.ps1    合成语音（SAPI，Windows）
scripts/make_synth_lecture.py 合成 PPT 视频，用于演练
scripts/check_env.py  打印 CUDA、算力、可用 compute_type
scripts/winocr.ps1    Windows 内置 OCR
config/               配置模板
docs/ARCHITECTURE.md  数据流与设计取舍
docs/PITFALLS.md      踩过的坑与原因
INSTALL-PROMPT.md     给 AI 助手用的部署步骤
```

## 平台

| 能力 | Windows | macOS / Linux |
| --- | --- | --- |
| 抽取、ASR、融合、发布 | 可用 | 可用（需 ffmpeg） |
| WinOCR | 可用 | 跳过，自动改用 VLM |
| pku3b 下载 | 官方提供 Windows 版 | 需自行解决，或改用 `--video` |

非 Windows 分支未实测。

## 边界

- 只处理有权访问的录像。教学网内容受版权与校规约束，不要公开分发视频、音频或完整讲义。
- 不保存任何登录凭据。`pku3b init` 由使用者本人执行。
- 讲义是复习辅助。LLM 补全处可能有误，关键结论回看原视频或教材。

MIT，见 [LICENSE](LICENSE)。
