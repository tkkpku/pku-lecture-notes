# 部署指令（给 AI 助手）

把下面的内容整段交给 Claude Code、Codex、Cursor 之类的编程助手，让它在目标机器上完成安装。它应当每完成一步就汇报并等待确认。

---

## 你的角色

在目标机器上把 pku-lecture-notes 装到能用合成素材端到端跑通，并把配置填好。只做环境、依赖、配置类工作；讲义质量由使用者判断。

## 硬规矩

1. 不替使用者登录任何账号，不读取、复制、上传、回显密码、cookie、pku3b 配置。
2. 不执行 git push。
3. 不使用 `--force` 覆盖已有产物。
4. `publish.auto_cleanup` 为真时会删除原始视频。在得到明确同意前，把 `cleanup.enabled` 设为 false。
5. 任何一步失败，停下并原样粘贴报错，不要自行重试。

## 步骤

### 1 环境盘点

- `python --version`，需要 3.10+
- `ffmpeg -version`、`ffprobe -version`，不在 PATH 就要先解决
- 有 NVIDIA 显卡时跑 `python scripts/check_env.py`，记录支持的 compute_type

产出：一张三行的表，标注可用与不可用。

### 2 安装依赖

```bash
python -m venv .venv
pip install -r requirements.txt
```

验收：`python -c "import yaml,httpx,PIL,numpy;print('ok')"`。

`requirements-asr.txt` 约 2~3 GB，先不装，等第 4 步需要时再问磁盘空间。

### 3 生成配置

```bash
cp config/config.example.yaml config/config.yaml
cp config/ignored_lectures.example.yaml config/ignored_lectures.yaml
cp config/terms.example.txt config/terms.txt
```

逐项询问使用者，其余保持默认：

| 配置项 | 需要确认的内容 |
| --- | --- |
| `paths.data_root` | 中间产物放在哪里。体积大，建议非系统盘 |
| `paths.obsidian_vault` | 讲义输出目录。不确定就留空，只出草稿 |
| `llm.base_url` `llm.model` `llm.key_env` | 使用的 LLM 端点。读图需要多模态模型 |
| `asr.model` `asr.compute_type` | 按第 1 步的结论给建议 |

密钥只设环境变量，不写入配置文件。变量名必须与 `llm.key_env` 一致。

`config.yaml` 必须存成 UTF-8 无 BOM。

### 4 自检

```bash
python scripts/doctor.py --net
```

验收：无 FAIL。有 WARN 要逐条说明后果，重点看 ffmpeg、pku3b、faster-whisper 三项。

### 5 合成素材演练

```bash
pip install -r requirements-asr.txt
powershell -ExecutionPolicy Bypass -File scripts/make_synth_tts.ps1
python scripts/make_synth_lecture.py --lecture 第01讲_演示
python src/pipeline.py --course 合成课程 --lecture 第01讲_演示 --video data/raw/合成课程/第01讲_演示.mp4
```

必须先造语音。只有画面没有声音时，音频预检会判定该素材无讲课语音并中止，这不是安装错误。

产物应依次出现在 `data/work/合成课程/第01讲_演示/`：`audio.wav`、`frames/`、`asr.json`、`frames_ocr.json`、`lecture.md`。

这一步会真实调用 LLM，费用很小。若要完全不花钱，加 `--stages fetch,extract,asr` 只验前半条链。

失败时按顺序检查：ffmpeg 是否可用、模型是否下载成功、LLM 端点与密钥、doctor 的提示。

### 6 真实数据

先问使用者选哪条路径：

1. 教学网直接下载：需要 pku3b，且必须由使用者本人执行 `pku3b init`。你只负责说明命令与解读输出。
2. 使用者已有 mp4：直接 `--video <路径>`，跳过 fetch。

```bash
python src/pipeline.py --course <课程名> --lecture <讲次> --video <mp4 路径>
```

先只跑一讲，等使用者验收讲义质量后再考虑批跑。批跑前必须确认清理开关的设置。

### 7 汇报

| 项 | 内容 |
| --- | --- |
| Python 与关键依赖版本 | |
| ffmpeg、GPU、compute_type | |
| doctor 结果 | 几 PASS 几 WARN 几 FAIL |
| 演练结果 | 五个产物是否齐全、lecture.md 字数 |
| 真实数据试跑 | 哪一讲、耗时、字数、费用估计 |
| 建议下一步 | |

结束时提醒三点：清理开关会删除原始视频；讲义补全处需回看原视频；录像与讲义受版权约束，不要公开分发。

---

## 说明

需要使用者拍板的只有三处：数据盘位置、使用哪个 LLM 端点（涉及费用）、是否删除原始视频。其余可由助手完成。

常见卡点：

| 现象 | 原因 |
| --- | --- |
| 模型下载失败 | 用环境变量 `HF_ENDPOINT` 换镜像 |
| `CUBLAS_STATUS_NOT_SUPPORTED` | compute_type 不匹配，跑 `check_env.py` 换一个 |
| `[WinError 206]` | 命令行过长，调小 `fuse.segment_max_chars` |
| OCR 结果大量乱码 | 用 winocr 后端处理了真实录播画面，改用 `vision.backend: vlm` |
| 公式不渲染 | 检查是否有环节做了 JSON 转义，用 `audit_fidelity.py` 抽查 |
