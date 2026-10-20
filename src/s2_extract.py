"""阶段 2：从回放视频抽取音轨 + 关键帧（场景检测 + 保底帧 + pHash 去重）。

产出（落 `<DATA_ROOT>\\work\\<课程>\\<讲次>\\`）：

| 文件 | 内容 |
|---|---|
| `audio.wav` | 16 kHz 单声道 pcm_s16le（喂 faster-whisper） |
| `frames/frame_0001.jpg ...` | 去重后的关键帧（宽 resize_width） |
| `frames_index.json` | 每帧 `{idx, file, t_sec, source, phash}` + 视频元数据 |

抽帧策略（见 docs/RESEARCH.md §5）：
  场景检测阈值 0.30（ffmpeg 官方推荐区间 0.3–0.5）→ 每 `interval_sec` 加保底帧
  → 最小间隔去抖（`min_gap_sec`）→ pHash 去重（汉明距 ≤ `phash_threshold`）→ 硬上限 `max_frames`

用法：
  python src/s2_extract.py --video D:\\...\\xxx.mp4 --course 理论力学 --lecture 第01讲 [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ensure_tools_on_path,
    get_logger,
    human_size,
    is_audio_only,
    lecture_work_dir,
    load_config,
    run,
    tool_paths,
)

RE_SHOWINFO = re.compile(r"n:\s*(\d+).*?pts_time:\s*([0-9]+(?:\.[0-9]+)?)")
RE_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
RE_VIDEO = re.compile(r"Video:\s*([a-z0-9_]+).*?(\d{2,5})x(\d{2,5})")
RE_FPS = re.compile(r"(\d+(?:\.\d+)?)\s*fps")


# ---------------- 探测 ----------------
def probe_media(path: str | Path, cfg: dict, log=None) -> dict:
    """读媒体元数据（无 ffprobe，解析 ffmpeg -i 的 stderr）。"""
    tp = tool_paths(cfg)
    cp = run([tp["ffmpeg"], "-hide_banner", "-i", str(path)], check=False)
    err = cp.stderr or ""
    dur = 0.0
    if m := RE_DURATION.search(err):
        h, mi, s = m.groups()
        dur = int(h) * 3600 + int(mi) * 60 + float(s)
    w = h = 0
    if m := RE_VIDEO.search(err):
        vcodec, w, h = m.group(1), int(m.group(2)), int(m.group(3))
    fps = float(m.group(1)) if (m := RE_FPS.search(err)) else 0.0
    info = {
        "path": str(path),
        "size_bytes": Path(path).stat().st_size if Path(path).exists() else 0,
        "duration_sec": round(dur, 2),
        "width": w,
        "height": h,
        "fps": fps,
    }
    if log:
        log.info("探测器: %s | %.1f min | %sx%s @%.1f | %s",
                 Path(path).name, dur / 60, w, h, fps, human_size(info["size_bytes"]))
    return info


# ---------------- 抽音轨 ----------------
def extract_audio(video: str | Path, out_wav: str | Path, cfg: dict, log=None, dry_run=False) -> Path:
    a = cfg["audio"]
    out_wav = Path(out_wav)
    if out_wav.exists() and out_wav.stat().st_size > 1024 and not dry_run:
        if log:
            log.info("音轨已存在，跳过: %s", out_wav.name)
        return out_wav
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        tool_paths(cfg)["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vn", "-ac", str(a["channels"]), "-ar", str(a["sample_rate"]), "-c:a", a["codec"],
        str(out_wav),
    ]
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return out_wav
    t0 = time.time()
    run(cmd, log=log)
    if log:
        log.info("音轨抽取完成: %s (%s, %.1fs)", out_wav.name,
                 human_size(out_wav.stat().st_size), time.time() - t0)
    return out_wav


# ---------------- 抽帧 ----------------
def _run_frame_pass(video, out_pattern, vf, cfg, log=None, dry_run=False) -> list[tuple[float, Path]]:
    """跑一遍 ffmpeg 抽帧（showinfo 输出时间戳），返回 [(t_sec, path)]。"""
    cmd = [
        tool_paths(cfg)["ffmpeg"], "-hide_banner", "-loglevel", "info", "-y",
        "-i", str(video),
        "-vf", vf + f",showinfo,scale={cfg['frames']['resize_width']}:-2",
        "-vsync", "0", "-q:v", "3", out_pattern,
    ]
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return []
    cp = run(cmd, check=False)
    err = cp.stderr or ""
    if cp.returncode != 0 and "Output file is empty" not in err:
        raise RuntimeError(f"抽帧失败 rc={cp.returncode}:\n{err[-2000:]}")
    times: list[tuple[float, Path]] = []
    for line in err.splitlines():
        m = RE_SHOWINFO.search(line)
        if m:
            times.append((float(m.group(2)), Path()))
    files = sorted(Path(out_pattern).parent.glob(Path(out_pattern).name.replace("%04d", "*")))
    # showinfo 的 n 是 select 后的序号，与落盘序列一致
    out = []
    for i, (t, _) in enumerate(times):
        if i < len(files):
            out.append((t, files[i]))
    if log:
        log.info("抽帧 pass: %d 帧 (vf=%s)", len(out), vf)
    return out


def scene_candidates(video, tmp_dir: Path, cfg: dict, log=None, dry_run=False):
    th = cfg["frames"]["scene_threshold"]
    d = tmp_dir / "scene"
    d.mkdir(parents=True, exist_ok=True)
    cands = _run_frame_pass(video, str(d / "s_%04d.jpg"),
                            f"select='gt(scene,{th})'", cfg, log, dry_run)
    return [(t, p, "scene") for t, p in cands]


def interval_candidates(video, tmp_dir: Path, cfg: dict, log=None, dry_run=False):
    iv = cfg["frames"]["interval_sec"]
    d = tmp_dir / "interval"
    d.mkdir(parents=True, exist_ok=True)
    cands = _run_frame_pass(video, str(d / "i_%04d.jpg"), f"fps=1/{iv}", cfg, log, dry_run)
    return [(t, p, "interval") for t, p in cands]


def sample_candidates(video, tmp_dir: Path, cfg: dict, log=None, dry_run=False):
    """定间隔采样候选帧（hash 模式的输入）。时间戳由序号 × sample_sec 推出。

    [warn] 为什么不用 ffmpeg 的 scene 检测：实测同模板 PPT 翻页时 scene 分数仅 0.01~0.05
    （版式/背景/标题栏全同，只有文字变），官方推荐的 0.3~0.5 会漏掉 5 次切换中的 4 次。
    见 docs/RESEARCH.md §5 修正。
    """
    sec = cfg["frames"]["sample_sec"]
    w = cfg["frames"]["resize_width"]
    d = tmp_dir / "sample"
    d.mkdir(parents=True, exist_ok=True)
    cmd = [
        tool_paths(cfg)["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vf", f"fps=1/{sec},scale={w}:-2",
        "-vsync", "0", "-q:v", "3", str(d / "h_%05d.jpg"),
    ]
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return []
    run(cmd, log=log)
    files = sorted(d.glob("h_*.jpg"))
    if log:
        log.info("采样抽帧: %d 帧（每 %.1fs 一帧）", len(files), sec)
    return [(round(i * sec, 2), p, "sample") for i, p in enumerate(files)]


# ---------------- pHash ----------------
def _dct_matrix(n: int):
    import numpy as np

    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    m = np.cos(np.pi * (2 * x + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0, :] = np.sqrt(1.0 / n)
    return m


_DCT_CACHE: dict[int, object] = {}


def phash(img_path: str | Path, hash_size: int = 8, hf_factor: int = 4,
          crop_right: int = 0) -> int:
    """感知哈希（DCT-II 版，自实现免 scipy）。返回 64 bit 整数。

    `crop_right`：算哈希前先裁掉右侧 N 像素。用于北大录播布局——右上角有**教室摄像头小窗**，
    小窗里学生走动/老师移动会让画面「持续变化」，在各处制造大量无价值帧
    （2026-09-13 实测：2 小时课在 t=5944~7130 段密集产出 20+ 个乱码帧）。
    只对 PPT 主区域算哈希即可消除该干扰。

    [warn] **只影响哈希计算**：落盘的帧文件仍是完整画面（VLM 需要完整画面才读得全）。
    """
    import numpy as np
    from PIL import Image

    n = hash_size * hf_factor
    img = Image.open(img_path).convert("L")
    if crop_right > 0:
        w, h = img.size
        img = img.crop((0, 0, max(16, w - int(crop_right)), h))
    img = img.resize((n, n), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float64)
    if n not in _DCT_CACHE:
        _DCT_CACHE[n] = _dct_matrix(n)
    m = _DCT_CACHE[n]
    d = m @ a @ m.T
    low = d[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])
    bits = "".join("1" if v > med else "0" for v in low)
    return int(bits, 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------- 选帧 ----------------
def select_frames(cands: list[tuple[float, Path, str]], cfg: dict, log=None) -> list[dict]:
    """去抖 + pHash 去重 + 硬上限。cands 为 [(t_sec, path, source)]。"""
    f = cfg["frames"]
    cands = sorted([c for c in cands if c[1].exists()], key=lambda c: c[0])
    if log:
        log.info("候选帧 %d（场景 %d / 保底 %d）", len(cands),
                 sum(1 for c in cands if c[2] == "scene"), sum(1 for c in cands if c[2] == "interval"))

    crop = int(f.get("hash_crop_right", 0))
    kept: list[dict] = []
    for t, p, src in cands:
        try:
            h = phash(p, crop_right=crop)
        except Exception as e:  # noqa: BLE001
            if log:
                log.warning("phash 失败，丢弃 %s: %s", p.name, e)
            continue
        dup = any(hamming(h, k["phash"]) <= f["phash_threshold"] for k in kept)
        if dup:
            continue
        if kept and (t - kept[-1]["t_sec"]) < f["min_gap_sec"]:
            # 太密：场景帧可顶掉上一张保底帧，保底帧不顶场景帧
            if src == "scene" and kept[-1]["source"] == "interval":
                kept.pop()
            else:
                continue
        kept.append({"t_sec": round(t, 2), "source": src, "phash": h, "tmp": str(p)})

    if len(kept) > f["max_frames"]:
        idx = [round(i * (len(kept) - 1) / (f["max_frames"] - 1)) for i in range(f["max_frames"])]
        kept = [kept[i] for i in sorted(set(idx))]
        if log:
            log.info("超过上限 %d，均匀抽样保留 %d 帧", f["max_frames"], len(kept))
    return kept


def select_by_change(cands: list[tuple[float, Path, str]], cfg: dict, log=None) -> list[dict]:
    """hash 模式选帧：按时间序比对 pHash，与**上一采样帧**差异超阈值就留一帧。

    为什么不两两全比：翻页检测关心「相对上一页是否变了」。全比会把「回到之前讲过的某页」
    误判成重复而丢掉 —— 而老师回头重讲同一页恰恰是重点信号，必须保留。

    `prev_h` 始终更新为上一采样帧（无论是否保留），这样连续动画不会无限堆积帧，
    且翻页后静止的页面只留一张。
    """
    f = cfg["frames"]
    thr = int(f.get("hash_change_threshold", f["phash_threshold"]))
    crop = int(f.get("hash_crop_right", 0))
    min_gap = f["min_gap_sec"]
    kept: list[dict] = []
    prev_h: int | None = None
    cands = sorted([c for c in cands if c[1].exists()], key=lambda c: c[0])
    if log:
        log.info("变化检测：%d 个采样帧，阈值 hamming>%d，最小间隔 %.0fs，哈希裁剪右侧 %dpx",
                 len(cands), thr, min_gap, crop)
    for t, p, src in cands:
        try:
            h = phash(p, crop_right=crop)
        except Exception as e:  # noqa: BLE001
            if log:
                log.warning("phash 失败，跳过 %s: %s", p.name, e)
            continue
        keep = prev_h is None or (hamming(h, prev_h) > thr and (t - kept[-1]["t_sec"]) >= min_gap)
        if keep:
            kept.append({"t_sec": round(t, 2), "source": src if prev_h is not None else "head",
                         "phash": h, "tmp": str(p)})
        prev_h = h

    if not kept and cands:  # 兜底：全被判为相同也要留第一帧
        t, p, src = cands[0]
        kept.append({"t_sec": round(t, 2), "source": "head", "phash": phash(p, crop_right=crop),
                     "tmp": str(p)})

    if len(kept) > f["max_frames"]:
        idx = [round(i * (len(kept) - 1) / (f["max_frames"] - 1)) for i in range(f["max_frames"])]
        kept = [kept[i] for i in sorted(set(idx))]
        if log:
            log.info("超过上限 %d，均匀抽样保留 %d 帧", f["max_frames"], len(kept))
    if log:
        log.info("变化检测保留 %d 帧：t=%s", len(kept), [k["t_sec"] for k in kept][:20])
    return kept


# ---------------- 主流程 ----------------
def extract_lecture(video: str | Path, course: str, lecture: str, cfg: dict | None = None,
                    force: bool = False, dry_run: bool = False) -> dict:
    cfg = cfg or load_config()
    log = get_logger("s2_extract", cfg)
    ensure_tools_on_path(cfg)
    video = Path(video)
    if not video.exists() and not dry_run:
        raise FileNotFoundError(video)

    wd = lecture_work_dir(course, lecture, cfg)
    wd.mkdir(parents=True, exist_ok=True)
    frames_dir = wd / "frames"
    tmp_dir = wd / "_tmp_frames"
    index_path = wd / "frames_index.json"

    if index_path.exists() and not force:
        log.info("已存在 frames_index.json，跳过（--force 重跑）: %s", index_path)
        return json.loads(index_path.read_text(encoding="utf-8"))

    meta = probe_media(video, cfg, log) if not dry_run else {"path": str(video), "duration_sec": 0}

    wav = extract_audio(video, wd / "audio.wav", cfg, log, dry_run=dry_run)

    probe: dict = {}
    if not dry_run and (cfg.get("probe") or {}).get("enabled", True):
        # [warn] 2026-09-13 新增（bug G）：在抽帧（~4 min）与后续 ASR/VLM 之前，先花 ~15 s 判
        #    「音频里有没有人在说话」。真实教训：无声回放的画面完全正常，会一路白跑到 ASR
        #    才暴露，白烧 40 分钟与全部 LLM 费用（该素材实测 VAD 判 0 段）。
        from s2b_probe import probe_speech

        pc = cfg.get("probe") or {}
        probe = probe_speech(wav, cfg, log, windows=int(pc.get("windows", 3)),
                             sec=int(pc.get("sec", 60)))
        if probe.get("verdict") != "ok":
            raise RuntimeError(
                f"素材音频预检未通过：{probe.get('detail', '')}。该回放音轨疑似无声，已在抽帧前中止"
                f"（防白跑；确要强制执行可把 config 的 probe.enabled 设为 false）。")

    #  2026-10-04：**纯音频课程**——音轨已抽、素材预检已过，这里直接收工（不抽帧）。
    #   见 config.yaml 的 `vision.audio_only_courses` 与 `common.is_audio_only()`。
    if is_audio_only(course, cfg):
        log.info("【纯音频课】跳过抽帧（已完成音轨抽取 + 素材预检）")
        if not dry_run:
            index = {
                "course": course, "lecture": lecture, "video_meta": meta,
                "frames_dir": "", "count": 0, "frames": [], "audio_only": True,
                "note": "纯音频课程：按 config.vision.audio_only_courses 跳过抽帧",
            }
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"count": 0, "frames_dir": "", "index": "", "audio_only": True,
                "audio": str(wav), "probe": probe}

    mode = cfg["frames"].get("mode", "hash")
    if mode == "scene":
        cands = scene_candidates(video, tmp_dir, cfg, log, dry_run) + \
            interval_candidates(video, tmp_dir, cfg, log, dry_run)
        kept = select_frames(cands, cfg, log)
    elif mode == "interval":
        cands = interval_candidates(video, tmp_dir, cfg, log, dry_run)
        kept = select_frames(cands, cfg, log)
    else:  # hash：实测推荐（scene 在同模板 PPT 翻页上漏检严重）
        cands = sample_candidates(video, tmp_dir, cfg, log, dry_run)
        kept = select_by_change(cands, cfg, log)

    if not dry_run:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, k in enumerate(kept, start=1):
            dst = frames_dir / f"frame_{i:04d}.jpg"
            shutil.copyfile(k.pop("tmp"), dst)
            k["idx"] = i
            k["file"] = dst.name
        shutil.rmtree(tmp_dir, ignore_errors=True)
        index = {
            "course": course, "lecture": lecture, "video_meta": meta,
            "frames_dir": str(frames_dir), "count": len(kept),
            "params": {k: cfg["frames"].get(k) for k in
                       ("mode", "sample_sec", "hash_change_threshold", "scene_threshold",
                        "min_gap_sec", "max_frames", "phash_threshold", "resize_width",
                        "interval_sec")},
            "frames": kept,
        }
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("完成：%d 帧 → %s；音轨 %s", len(kept), frames_dir, wav.name)
    return {"count": len(kept), "frames_dir": str(frames_dir), "index": str(index_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 2：抽音轨 + 关键帧")
    ap.add_argument("--video", required=True)
    ap.add_argument("--course", required=True)
    ap.add_argument("--lecture", required=True)
    ap.add_argument("--force", action="store_true", help="忽略已有产物重跑")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    args = ap.parse_args()
    res = extract_lecture(args.video, args.course, args.lecture,
                          force=args.force, dry_run=args.dry_run)
    print("[s2_extract]", json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
