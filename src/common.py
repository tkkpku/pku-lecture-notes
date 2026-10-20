"""公共模块：配置加载、路径解析、日志、CUDA DLL 引导、外部工具定位、子进程封装。

[warn] 本机硬约束（2026-09-12 实测，见 docs/RESEARCH.md §4 与 docs/HANDOFF.md）：

1. **CUDA DLL 引导必须在 import ctranslate2 / faster_whisper 之前调用**
   `ctranslate2` 的 Windows wheel 自带 `cudnn64_9.dll`，但**不带** `cublas64_12.dll`；
   缺了会在**首次前向**（不是加载时）报
   `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`。
   解法：pip 装 `nvidia-cublas-cu12`，再用 `setup_cuda_dlls()` 把 `nvidia/*/bin` 挂进 DLL 搜索路径。

2. **hf-mirror 不支持 Xet 协议**：`huggingface_hub>=1.0` 默认走 `cas-server.xethub.hf.co`，
   走镜像会 401 Unauthorized → 必须 `HF_HUB_DISABLE_XET=1`（`apply_hf_env()` 已代劳）。

3. `compute_type` 在生产路径固定 `float16`（Blackwell sm_120 上 int8 系声明支持但风险已知，
   实测见 docs；保守起见不写 `auto`）。

4. **调用 `dsh --profile headless` 必须走 `find_dsh_cli()`，不要直接 exec `dsh.CMD`**：
   多行 prompt 经 `cmd.exe` 会在第一个换行处被静默截断（rc=0，看不出错）。
   详见 `find_dsh_cli()` docstring。
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------- 路径常量 ----------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent              # codex\lecture-pipeline
WORKSPACE_ROOT = PROJECT_ROOT.parent       # codex

_DLL_DIRS_ADDED: list[str] = []
_CONFIG_CACHE: dict | None = None


# ---------- CUDA DLL 引导 ----------
def setup_cuda_dlls(verbose: bool = False) -> list[str]:
    """把 pip 装的 NVIDIA CUDA 运行库目录挂进 DLL 搜索路径。**必须在 import ctranslate2 之前调用。**

    幂等：重复调用不会重复挂载。返回实际挂上的目录列表。
    """
    global _DLL_DIRS_ADDED
    if _DLL_DIRS_ADDED:
        return _DLL_DIRS_ADDED

    added: list[str] = []
    nv_roots: list[Path] = []

    # site-packages/nvidia/*（pip 包 nvidia-cublas-cu12 等的落点）
    try:
        import site

        for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
            nv = Path(sp) / "nvidia"
            if nv.is_dir():
                nv_roots.append(nv)
    except Exception:  # noqa: BLE001
        pass
    nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if nv.is_dir():
        nv_roots.append(nv)

    # 环境变量兜底：允许用户手动指定（如系统 CUDA / Ollama 自带库）
    for extra in os.environ.get("LECTURE_CUDA_DLL_DIRS", "").split(";"):
        if extra.strip() and Path(extra.strip()).is_dir():
            nv_roots.append(Path(extra.strip()))

    for root in dict.fromkeys(nv_roots):
        candidates = [root] if (root / "cublas64_12.dll").exists() else []
        candidates += [d / "bin" for d in sorted(root.iterdir()) if d.is_dir()]
        for d in candidates:
            if not d.is_dir():
                continue
            try:
                os.add_dll_directory(str(d))
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(d))
            if verbose:
                print(f"[cuda-dll] + {d}")

    _DLL_DIRS_ADDED = added
    return added


def apply_hf_env() -> None:
    """设置 HF 镜像、Xet 关闭、**默认强制离线**（必须在 huggingface_hub 导入前调用）。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # hf-mirror 不支持 Xet，否则 401
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # D 盘不支持 symlink
    #  2026-09-19（你批准固化）：**默认强制离线 `HF_HUB_OFFLINE=1`**。
    # ── 由来（真实现行犯，白等 18 分钟）────────────────────────────
    #    `hf-mirror.com` 会解析出 6 个 IP，其中 `160.16.86.14`（SAKURA，tk2-219-19010.vs.sakura.ne.jp）
    #    **TCP 三次握手成功但服务器不回数据**；而 faster-whisper 的 `WhisperModel(...)` 即便模型
    #    已在本地缓存，也会做一次 HF 元数据校验，且该请求**没有超时** → 打到坏节点就**无限挂起**：
    #    实测进程零 CPU、零产物、HF 缓存零增长，白等 18 分钟（预检与 ASR 都会中招）。
    #    本机模型（`models/models--Systran--faster-whisper-large-v3`，2.9 GB）早已下齐，离线即够用。
    # ── 逃生门 ──────────────────────────────────────────────────
    #    需要**首次下载模型 / 更新模型**时，设 `LECTURE_HF_ALLOW_DOWNLOAD=1`（此时不再强制离线）；
    #    外部若已显式设置 `HF_HUB_OFFLINE`，一律尊重（setdefault 语义）。
    allow_dl = str(os.environ.get("LECTURE_HF_ALLOW_DOWNLOAD", "")).strip().lower()
    if allow_dl in ("1", "true", "yes", "on"):
        return
    #  开源版新增：**首次使用（本地还没有模型）时不要强制离线**，
    #    否则新机器一上来就失败（"模型不存在 + 禁止下载"），体验极差。
    #    判据：models/ 下有没有任何一个 models--* 快照目录。
    try:
        cache = Path((load_config().get("paths") or {}).get("data_root") or "data")
        cache = cache if cache.is_absolute() else (PROJECT_ROOT / cache)
        has_model = (cache / "models").is_dir() and any((cache / "models").glob("models--*"))
    except Exception:
        has_model = False
    if has_model:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    else:
        print("[i] 本地还没有 ASR 模型 → 本次允许联网下载（首次约 3 GB）。"
              "下载完成后会自动转为离线模式。")



# ---------- 配置 ----------
def load_config(path: str | Path | None = None, reload: bool = False) -> dict:
    """读 config/config.yaml（带缓存）。"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not reload and path is None:
        return _CONFIG_CACHE
    import yaml

    p = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(p)
    if path is None:
        _CONFIG_CACHE = cfg
    return cfg


def data_root(cfg: dict | None = None) -> Path:
    """数据根目录。**相对路径按项目根解析**（不是按当前工作目录）。

    换机时把 config.yaml 里 data_root 写成 "data" 即可，不必改绝对路径。
    """
    cfg = cfg or load_config()
    p = Path(str((cfg.get("paths") or {}).get("data_root") or "data"))
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def work_dir(cfg: dict | None = None) -> Path:
    return data_root(cfg) / "work"


def raw_dir(cfg: dict | None = None) -> Path:
    return data_root(cfg) / "raw"


def cache_dir(cfg: dict | None = None) -> Path:
    return data_root(cfg) / "cache"


def models_dir(cfg: dict | None = None) -> Path:
    return data_root(cfg) / "models"


def notes_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["paths"]["obsidian_vault"]) / cfg["paths"]["notes_library"]


def lecture_work_dir(course: str, lecture: str, cfg: dict | None = None) -> Path:
    """某讲的中间产物目录：<DATA_ROOT>\\work\\<课程>\\<讲次>"""
    return work_dir(cfg) / _safe(course) / _safe(lecture)


def _safe(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in str(name)).strip()
    return out or "unnamed"


# ---------- 数学分隔符归一（2026-10-12 新增；事故见 HANDOFF §24.7）----------
_MATH_INLINE_RE = re.compile(r"\\\((.*?)\\\)", re.S)
_MATH_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.S)
_FENCE_RE = re.compile(r"```.*?```", re.S)


def _norm_math_seg(seg: str, stats: dict) -> str:
    def _disp(m):
        stats["display"] += 1
        return "\n$$\n" + m.group(1).strip("\n") + "\n$$\n"

    def _inl(m):
        stats["inline"] += 1
        return "$" + m.group(1).strip() + "$"

    return _MATH_INLINE_RE.sub(_inl, _MATH_DISPLAY_RE.sub(_disp, seg))


def normalize_math_delims(text: str) -> tuple[str, dict]:
    """把非 KaTeX 的 `\\(…\\)` / `\\[…\\]` 归一成 `$…$` / `$$…$$`（Obsidian 只认后者）。

    由来（2026-10-12，HANDOFF §24.7）：常微分方程第 08 讲草稿里出现 **42 处 `\\(…\\)`
    与 17 处 `\\[…\\]`** —— Obsidian 默认**不渲染** `\\(`，公式会原样显示成源码；
    而同批理论力学第 08 讲完全干净（模型口味不稳定）。所以必须**确定性归一**，不能赌。

    细则：代码围栏 ``` 内原样不动；已经是 `$…$` 的不重复处理；返回 (新文本, 统计)。
    """
    stats = {"inline": 0, "display": 0}
    if not text or ("\\(" not in text and "\\[" not in text):
        return text, stats
    out, i = [], 0
    for m in _FENCE_RE.finditer(text):
        out.append(_norm_math_seg(text[i:m.start()], stats))
        out.append(m.group(0))          # 代码块原样
        i = m.end()
    out.append(_norm_math_seg(text[i:], stats))
    return "".join(out), stats


# ---------- work 目录身份闸门（2026-10-12 新增；事故见 HANDOFF §24）----------
def work_identity(work: Path) -> str:
    """读 work 目录里 `meta.json` 记录的 video_id（读不到 / 无此文件 → 空串）。"""
    mp = Path(work) / "meta.json"
    if not mp.exists():
        return ""
    import json

    try:
        return str((json.loads(mp.read_text(encoding="utf-8-sig")) or {}).get("video_id") or "")
    except Exception:  # noqa: BLE001
        return ""


def guard_work_identity(course: str, lecture: str, video_id: str,
                        cfg: dict | None = None, log=None) -> dict | None:
    """开工前核对 work 目录属于**这一条回放**；身份不符就把旧目录改名隔离（不删）。

    由来（2026-10-12 事故）：work 目录只按 (课程, 讲次) 命名，但**讲次编号会前移** ——
    教学网只放最近若干条回放，国庆假期那批空录像被挤出列表后，10-08 的真实课继承了
    同一个「第08讲」目录。于是 `extract` 抽了新帧与新音频，`asr` 却看到目录里躺着
    **旧录像的 asr.json**（13 段 / 64 字幻觉）→ 判定「已完成」直接跳过 ⇒ 各阶段全报 OK，
    最后拿**真课的画面 + 空录像的转写**去融合，讲义会无声无息变成废纸。
    （实测：理力第08讲 audio_sec 仍显示旧录像的 7196.2 s，而新录像是 66.8 min。）

    判据：`meta.json` 的 `video_id`（`s2_extract` 写入）≠ 本次的 video_id。
    处置：旧目录改名为 `<讲次>__stale_<旧ID前8位>_<HHMMSS>`，**只改名不删**，
          证据件（如 `_evidence_silent_*.wav`）原样保留；本次照常继续 → 各阶段重建干净目录。
    """
    vid = str(video_id or "").strip()
    if not vid:
        return None
    wd = lecture_work_dir(course, lecture, cfg)
    old = work_identity(wd)
    if not old or old == vid:
        return None
    stamp = datetime.now().strftime("%H%M%S")
    dest = wd.with_name(f"{wd.name}__stale_{old[:8]}_{stamp}")
    n = 1
    while dest.exists():
        dest = wd.with_name(f"{wd.name}__stale_{old[:8]}_{stamp}_{n}")
        n += 1
    wd.rename(dest)
    info = {"course": course, "lecture": lecture, "stale_video_id": old,
            "new_video_id": vid, "quarantined_to": str(dest)}
    if log:
        log.warning(
            "[work 身份闸门] %s / %s 的中间产物属于**另一条回放**（旧 %s，本次 %s）→ "
            "已改名隔离到 %s，将为新回放重建干净目录。[warn] 讲次编号前移所致；旧产物未删除。",
            course, lecture, old, vid, dest.name)
    return info


# ---------- 每次调用钉模型（2026-09-17 新增）----------
def model_patch_args(provider: str = "", model: str = "", cfg: dict | None = None) -> list[str]:
    """为**一次** `dsh` 调用生成 `--patch <overlay>` 参数，把该次调用的模型钉死。

    【为什么需要】`dsh` CLI **没有** `--provider/--model` 参数：headless 调用只能吃
    `settings.yaml` 的全局 `agent-default-model`。于是「讲义融合/读图到底走哪条通道」变成
    **跟着环境漂**的不确定量 —— 实测现行犯：同一天的 16:51 那批调用走
    `provider-a / model-fast`，17:49 那批就走 `provider-b / model-v4`
    （因为中途有人改了全局默认模型）。成本口径与质量口径因此都无法归因。

    【怎么钉】launcher 的 `--patch <yml>`（可重复）是官方支持的叠加层，用它覆盖
    `agent-default-model`：
        - id: agent-default-model
          config:
            provider: provider-b
            model: some-vision-capable-model

    overlay 文件按 (provider, model) 缓存到 `cache/model_patch/<slug>.yml`，内容变了才重写。
    返回 `[]` 表示「不钉」（provider/model 任一为空）→ 调用跟随全局默认。
    """
    provider, model = str(provider or "").strip(), str(model or "").strip()
    if not provider or not model:
        return []
    import re

    slug = re.sub(r"[^0-9A-Za-z._-]", "_", f"{provider}__{model}")
    p = cache_dir(cfg) / "model_patch" / f"{slug}.yml"
    text = (
        "# 由 lecture-pipeline common.model_patch_args() 自动生成：把**这一次** dsh 调用的模型钉死。\n"
        "# 不要手工改这里 —— 要换通道请改 config/config.yaml 的\n"
        "#   fuse.provider / fuse.model（讲义融合）或 vision.vlm.provider / vision.vlm.model（读图）。\n"
        "- id: agent-default-model\n"
        "  config:\n"
        f"    provider: \"{provider}\"\n"
        f"    model: \"{model}\"\n"
    )
    if not p.exists() or p.read_text(encoding="utf-8") != text:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return ["--patch", str(p)]


# ---------- 永久忽略清单（2026-09-17 新增）----------
IGNORE_LIST_PATH = PROJECT_ROOT / "config" / "ignored_lectures.yaml"
_IGNORE_CACHE: dict[str, dict] | None = None


def load_ignored(reload: bool = False) -> dict[str, dict]:
    """读 `config/ignored_lectures.yaml` → `{video_id: 条目}`（带缓存）。

    文件不存在 / 格式为空 → 返回空 dict（**不报错**：忽略清单是可选增强，缺了照常跑）。
    读取用 `utf-8-sig`：有人手改文件时顺手存成带 BOM 的 UTF-8 也能读（PyYAML 见 \\ufeff 会炸）。
    """
    global _IGNORE_CACHE
    if _IGNORE_CACHE is not None and not reload:
        return _IGNORE_CACHE
    out: dict[str, dict] = {}
    if IGNORE_LIST_PATH.exists():
        import yaml

        with open(IGNORE_LIST_PATH, encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
        for it in (data.get("ignored") or []):
            if not isinstance(it, dict):
                continue
            vid = str(it.get("video_id") or "").strip()
            if vid:
                out[vid] = it
    _IGNORE_CACHE = out
    return out


def is_ignored(video_id: str, reload: bool = False) -> dict | None:
    """该回放是否在永久忽略清单里：命中返回条目，否则 None。"""
    if not video_id:
        return None
    return load_ignored(reload=reload).get(str(video_id).strip())


def ignored_match(course: str = "", lecture: str = "", video_id: str = "",
                  reload: bool = False) -> dict | None:
    """忽略清单匹配（pipeline 闸门用）：先按 video_id 精确匹配，再按 (课程, 讲次) 兜底。

    (course, lecture) 兜底的理由：agent 常照着过期清单手工敲课程名/讲次，可能拿不到 video_id；
    此时若放行，就会把一条**已知为空/无声**的回放重新下载一遍（约 900 MB + 数分钟）。

    [!] 兜底**仅在调用方拿不到 video_id 时**生效（2026-10-12 修，事故见 HANDOFF §24）：
      教学网只放最近若干条回放，旧条目会被挤出列表 → **讲次编号整体前移**，于是「国庆假期
      那批空录像」曾经占用的「第08讲」会落到**后来真实上课的 10-08 那讲**头上，与清单里
      的 (课程, 讲次) 撞车 → 真课被拒跑。video_id 才是回放的稳定身份：调用方既然**明确给了**
      video_id 且它不在清单里，就说明与清单条目**不是同一条回放**，不得再兜底。
    """
    hit = is_ignored(video_id, reload=reload)
    if hit:
        return hit
    if str(video_id or "").strip():
        return None
    if not course or not lecture:
        return None
    for it in load_ignored(reload=reload).values():
        if it.get("fallback") is False:
            # 编号前移过的条目：它的 (课程, 讲次) 已经被**后来真课**占用，不得再兜底
            continue
        if (str(it.get("course") or "") == course
                and str(it.get("lecture") or "").strip() == lecture.strip()):
            return it
    return None


# ---------- 纯音频课程 ----------
def is_audio_only(course: str, cfg: dict | None = None) -> bool:
    """该课程是否走「纯音频」通道（**不抽帧、不读图**）。

    由来（2026-10-04 你钦定）：新增课程「工学创新实践」的 PPT 价值不大，
    只需处理录音 → 抽帧与 OCR 两步都可省（每讲省 ~4 分钟抽帧 + 数分钟 VLM 调用）。

    配置（`config.yaml` 的 `vision.audio_only_courses`）影响链路：
      · `s2_extract`：抽完音轨 + 素材预检后**直接返回**，不抽帧、不写 frames_index.json；
      · `s4_ocr`：跳过读图，写一份**空 frames_ocr.json**（保持产物契约，backend="none"）；
      · `s5_fuse`：帧为空 → 提示词里的画面文字段自动退化，纯 ASR 融合；
      · `s6_publish` / 抽检 / 审计：不受影响。
    """
    cfg = cfg or load_config()
    names = (cfg.get("vision") or {}).get("audio_only_courses") or []
    return str(course) in [str(n) for n in names]


# ---------- 日志 ----------
_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "pipeline", cfg: dict | None = None, console: bool = True) -> logging.Logger:
    """带文件+控制台的 logger（文件落 PROJECT_ROOT/logs/）。"""
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{name}_{datetime.now():%Y%m%d}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

    _LOGGERS[name] = logger
    return logger


# ---------- 外部工具 ----------
def tool_paths(cfg: dict | None = None) -> dict[str, str]:
    """返回 {pku3b, ffmpeg, ffprobe(可能空), venv_python}。"""
    cfg = cfg or load_config()
    ff_dir = Path(cfg["paths"]["ffmpeg_dir"])
    ffmpeg = ff_dir / "ffmpeg.exe"
    ffprobe = ff_dir / "ffprobe.exe"
    paths_cfg = cfg.get("paths") or {}
    return {
        "pku3b": paths_cfg.get("pku3b_exe") or "",
        "ffmpeg": str(ffmpeg) if ffmpeg.exists() else "ffmpeg",
        "ffprobe": str(ffprobe) if ffprobe.exists() else "",  # imageio-ffmpeg 不带 ffprobe
        # venv_python 是历史遗留键：开源版里没有调用者真的需要它，缺了就退回当前解释器
        "venv_python": paths_cfg.get("venv_python") or sys.executable,
    }


def ensure_tools_on_path(cfg: dict | None = None) -> list[str]:
    """把 ffmpeg 目录挂进当前进程 PATH（pku3b 下载回放需要 ffmpeg 在 PATH）。"""
    cfg = cfg or load_config()
    added = []
    for d in (Path(cfg["paths"]["ffmpeg_dir"]), Path(cfg["paths"]["pku3b_exe"]).parent):
        if d.is_dir():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(d))
    return added


def find_dsh_cli(cfg: dict | None = None) -> list[str]:
    """定位**可直接 exec** 的 dsh CLI 调用前缀：`[node.exe, <...>/@deepseek-ai/dsh/lib/bin.js]`。

    [warn] 2026-09-12 实测踩坑（本节最凶的静默失败）：
    把**多行 prompt** 作为参数交给 npm 的 `dsh.CMD` 时，`cmd.exe` 把换行当成命令分隔符，
    **argv 在第一个换行处被截断**，后面几行被当作新命令吞掉，而且 **rc=0、stderr 干净**——
    调用方拿到的只是一个「只看到第一行」的回答，完全看不出错。复现：

        ARG1=[line1 HEAD]   ← 传入 "line1 HEAD\\nline2 BODY\\nline3 TAIL" 的实测结果

    解法：绕开 .cmd shim，直接 `node <dsh>/lib/bin.js`（npm shim 内部本来就是这一条命令，
    行为完全等价）；`CreateProcess` 传参不把换行当分隔符，多行 prompt 完整送达。
    另注：Windows 命令行上限 32767 字符，超长 prompt 需靠 `fuse.segment_max_min` 控制块大小。

    解析顺序：环境变量 `LECTURE_DSH_CLI`（用 `|` 分隔的多段命令）→ dsh.CMD 同级的
    `node_modules/@deepseek-ai/dsh/lib/bin.js` → `npm root -g` 兜底。
    """
    env = os.environ.get("LECTURE_DSH_CLI", "")
    if env.strip():
        parts = [p for p in env.split("|") if p.strip()]
        if parts:
            return parts

    node = shutil.which("node") or shutil.which("node.exe")
    cands: list[Path] = []
    shim = shutil.which("dsh") or shutil.which("dsh.cmd") or shutil.which("dsh.CMD") or ""
    if shim:
        cands.append(Path(shim).parent / "node_modules" / "@deepseek-ai" / "dsh" / "lib" / "bin.js")
    if not any(c.exists() for c in cands):
        try:
            cp = run(["npm", "root", "-g"], check=False)
            root = (cp.stdout or "").strip()
            if root:
                cands.append(Path(root) / "@deepseek-ai" / "dsh" / "lib" / "bin.js")
        except Exception:  # noqa: BLE001
            pass
    for c in cands:
        if c.exists():
            return [node or "node", str(c)]
    raise RuntimeError(
        "找不到可直接调用的 dsh CLI（bin.js）。可设 LECTURE_DSH_CLI="
        r'"C:\Program Files\nodejs\node.exe|<...>\@deepseek-ai\dsh\lib\bin.js" 指定。'
        f" 已尝试：{[str(c) for c in cands]}"
    )


def run(
    cmd: list[str],
    check: bool = True,
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env_extra: dict | None = None,
    log: logging.Logger | None = None,
) -> subprocess.CompletedProcess:
    """跑子进程并捕获输出（UTF-8，失败时把 stderr 带进异常）。"""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    if log:
        log.debug("RUN %s", " ".join(str(c) for c in cmd))
    t0 = time.time()
    cp = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    if log:
        log.debug("EXIT %s in %.1fs", cp.returncode, time.time() - t0)
    if check and cp.returncode != 0:
        raise RuntimeError(
            f"命令失败 rc={cp.returncode}: {' '.join(str(c) for c in cmd)}\n"
            f"stderr: {(cp.stderr or '')[-2000:]}"
        )
    return cp


def probe_duration_sec(media: str | Path, cfg: dict | None = None) -> float:
    """读媒体时长（无 ffprobe，用 ffmpeg -i 解析 stderr）。"""
    tp = tool_paths(cfg)
    ffmpeg = tp["ffmpeg"]
    if tp["ffprobe"]:
        cp = run([tp["ffprobe"], "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=nw=1:nk=1", str(media)], check=False)
        try:
            return float((cp.stdout or "").strip())
        except ValueError:
            pass
    cp = run([ffmpeg, "-hide_banner", "-i", str(media)], check=False)
    import re

    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", cp.stderr or "")
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def human_size(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# ---------- 自检入口 ----------
def _selfcheck() -> int:
    print("== 路径 ==")
    print("PROJECT_ROOT =", PROJECT_ROOT)
    cfg = load_config()
    print("config       =", cfg["_config_path"])
    print("data_root    =", data_root(cfg))
    print("notes_dir    =", notes_dir(cfg))
    print("== 外部工具 ==")
    tp = tool_paths(cfg)
    for k, v in tp.items():
        ok = "OK " if (v and (Path(v).exists() or k == "ffmpeg")) else "-- "
        print(f"  [{ok}] {k:12s} {v or '(未找到)'}")
    print("== CUDA DLL ==")
    added = setup_cuda_dlls(verbose=True)
    print(f"  added {len(added)} dir(s)")
    print("== 目录 ==")
    for d in (data_root(cfg), work_dir(cfg), raw_dir(cfg), cache_dir(cfg), models_dir(cfg)):
        print(f"  [{'OK ' if d.exists() else 'MISS'}] {d}")
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
