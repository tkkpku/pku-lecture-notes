"""doctor.py —— 环境自检（装完先跑这个，别直接跑 pipeline）。

用法:
    python scripts/doctor.py            # 基本检查
    python scripts/doctor.py --net      # 额外检查 LLM 端点连通性

退出码：0 = 可以跑；1 = 有 FAIL 项必须修。
"""
from __future__ import annotations

import os
import shutil
import socket
import sys

for _s in (sys.stdout, sys.stderr):          # Windows 控制台默认 GBK
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
rows: list[tuple[str, str, str]] = []


def add(item: str, status: str, note: str) -> None:
    rows.append((item, status, note))


def check_python() -> None:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    (add("Python", PASS, ver) if v >= (3, 9)
     else add("Python", FAIL, f"{ver} 太旧，需要 3.9+"))


def check_deps() -> None:
    for mod, pkg in [("yaml", "PyYAML"), ("httpx", "httpx"),
                     ("PIL", "Pillow"), ("numpy", "numpy")]:
        try:
            __import__(mod)
            add(f"依赖 {pkg}", PASS, "已安装")
        except Exception:
            add(f"依赖 {pkg}", FAIL, f"缺 → pip install -r requirements.txt")


def check_ffmpeg() -> None:
    for exe in ("ffmpeg", "ffprobe"):
        p = shutil.which(exe)
        add(exe, PASS if p else FAIL,
            p or "不在 PATH。装 ffmpeg 并把 bin 目录加进 PATH，或在 config.yaml 填 paths.ffmpeg_dir")


def check_pku3b() -> None:
    cfg = _cfg()
    p = str(((cfg.get("paths") or {}).get("pku3b_exe") or "")).strip()
    found = p if p and Path(p).exists() else shutil.which("pku3b")
    if found:
        add("pku3b", PASS, str(found))
    else:
        add("pku3b", WARN,
            "没找到。只有「从教学网直接抓视频」才需要它；"
            "用 --video 指定现成 mp4 时可以完全没有")


def check_asr() -> None:
    try:
        import faster_whisper  # noqa: F401
        add("faster-whisper", PASS, "已安装")
    except Exception:
        add("faster-whisper", WARN,
            "未安装 → asr 阶段不可用；extract 的音频预检默认开启也会受影响"
            "（临时绕过：config.yaml 里 probe.enabled: false）。装法 pip install -r requirements-asr.txt")


def check_config() -> None:
    p = ROOT / "config" / "config.yaml"
    if not p.exists():
        add("config.yaml", FAIL,
            "不存在。先复制： cp config/config.example.yaml config/config.yaml")
        return
    raw = p.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        add("config.yaml", FAIL, "带 BOM 了！PyYAML 会报错。用「UTF-8 无 BOM」重存")
        return
    try:
        import yaml
        cfg = yaml.safe_load(raw.decode("utf-8")) or {}
        add("config.yaml", PASS, f"可解析，{len(cfg)} 个顶层段")
    except Exception as exc:
        add("config.yaml", FAIL, f"解析失败：{type(exc).__name__}: {exc}")


def check_paths() -> None:
    cfg = _cfg()
    try:
        from common import data_root, notes_dir
        d = data_root(cfg)
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("data_root 可写", PASS, str(d))
    except Exception as exc:
        add("data_root 可写", FAIL, f"{type(exc).__name__}: {exc}")
        return
    try:
        vault = str(((cfg.get("paths") or {}).get("obsidian_vault") or "")).strip()
        if not vault:
            add("讲义输出目录", WARN, "paths.obsidian_vault 为空 → 只出草稿，不发布")
        else:
            add("讲义输出目录", PASS, str(notes_dir(cfg)))
    except Exception as exc:
        add("讲义输出目录", FAIL, f"{type(exc).__name__}: {exc}")


def check_cfg_keys() -> None:
    """检查代码会**无条件读取**的配置键是否齐全。

    踩坑（开源首版）：模板漏了 vision.vlm，s4_ocr 抛 KeyError 被 except 吞掉，
    变成一句"VLM 后端 0 帧成功"的误导性报错 —— 所以这里必须体检键结构。
    """
    cfg = _cfg()
    if not cfg:
        add("配置键结构", FAIL, "读不到 config.yaml")
        return
    missing = []
    if not isinstance(cfg.get("paths"), dict):
        missing.append("paths")
    if not isinstance((cfg.get("vision") or {}).get("vlm"), dict):
        missing.append("vision.vlm")
    if not isinstance(cfg.get("llm"), dict):
        missing.append("llm")
    if not isinstance(((cfg.get("fuse") or {}).get("fidelity")), dict):
        missing.append("fuse.fidelity")
    if missing:
        add("配置键结构", FAIL,
            "缺 " + ", ".join(missing) + " → 运行时会 KeyError。"
            "对照 config/config.example.yaml 补齐（模板已更新到最新）")
    else:
        add("配置键结构", PASS, "paths / vision.vlm / llm / fuse.fidelity 齐全")


def check_terms() -> None:
    p = ROOT / "config" / "terms.txt"
    if p.exists():
        n = len([x for x in p.read_text(encoding="utf-8", errors="replace").splitlines()
                 if x.strip() and not x.strip().startswith("#")])
        add("术语热词 terms.txt", PASS, f"{n} 条")
    else:
        add("术语热词 terms.txt", WARN,
            "没有（cp config/terms.example.txt config/terms.txt 后按你的学科填），"
            "专有名词识别率会低一些")


def check_platform() -> None:
    if sys.platform.startswith("win"):
        add("平台", PASS, "Windows（WinOCR 可用）")
    else:
        add("平台", WARN,
            "非 Windows：WinOCR 会被跳过（自动退化为纯 VLM）")


def check_key() -> None:
    cfg = _cfg()
    key_env = str(((cfg.get("llm") or {}).get("key_env") or "LLM_API_KEY"))
    if (os.environ.get(key_env) or "").strip():
        add(f"密钥 {key_env}", PASS, "环境变量已设置")
    else:
        add(f"密钥 {key_env}", WARN,
            f"未设置 → 融合/读图会失败。PowerShell: $env:{key_env}='sk-...'")


def check_net() -> None:
    cfg = _cfg()
    base = str(((cfg.get("llm") or {}).get("base_url") or "")).strip()
    if not base:
        add("LLM 端点", WARN, "config.yaml 里 llm.base_url 为空")
        return
    u = urlsplit(base)
    host, port = u.hostname, (u.port or (443 if u.scheme == "https" else 80))
    s = socket.socket()
    s.settimeout(4.0)
    try:
        code = s.connect_ex((host, port))
    finally:
        s.close()
    add("LLM 端点", PASS if code == 0 else WARN,
        f"{host}:{port} " + ("可达" if code == 0 else "连不上（网络/代理/DNS）"))


def _cfg() -> dict:
    try:
        from common import load_config
        return load_config()
    except Exception:
        return {}


def main() -> int:
    print("=" * 78)
    print(" pku-lecture-notes · 环境自检")
    print("=" * 78)
    check_python()
    check_deps()
    check_ffmpeg()
    check_config()
    check_cfg_keys()
    check_platform()
    check_paths()
    check_terms()
    check_key()
    check_pku3b()
    check_asr()
    if "--net" in sys.argv:
        check_net()

    w = max(len(r[0]) for r in rows)
    for item, status, note in rows:
        tag = {"PASS": "[ ok ]", "WARN": "[warn]", "FAIL": "[FAIL]"}[status]
        print(f"{tag} {item.ljust(w)}  {note}")
    fails = sum(1 for r in rows if r[1] == FAIL)
    warns = sum(1 for r in rows if r[1] == WARN)
    print("-" * 78)
    print(f" {len(rows) - fails - warns} PASS · {warns} WARN · {fails} FAIL")
    if fails:
        print("\n先修 FAIL 项。")
        return 1
    print("\n建议先拿合成素材演练一遍（不需要任何账号）：")
    print("  python scripts/make_synth_lecture.py --lecture 第01讲_演示")
    print("  python src/pipeline.py --course 合成课程 --lecture 第01讲_演示 "
          "--video data/raw/合成课程/第01讲_演示.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
