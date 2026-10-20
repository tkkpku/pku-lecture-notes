"""阶段 2a-pre：素材音频预检（2026-09-13 新增）—— 判「这条回放的音轨里到底有没有人在说话」。

[!] 由来（真实教训，一次白跑）：
    理论力学 2026-09-07 第3-4节（回放 6a4fb277349481ac）的画面完全正常，但音轨只有房间底噪。
    faster-whisper 遇到这种音频**不报错**，只静默返回 0 段 —— 于是整条链路白跑到 ASR 才暴露：
    下载 4 min + 抽帧 4 min + 后续本该发生的 VLM/融合费用全部打水漂。
    检测手段其实很便宜（~15 秒、零 LLM token），故前移到**抽帧之前**。

判据（与 s3_asr 同源，避免两套标准）：
    · 取 N 个窗口（默认 3 个，铺在 10% / 50% / 85% 时长处），每窗 W 秒（默认 60 s）
    · 每窗用 silero VAD（`vad_filter=True`）+ `beam_size=1` 转写一次
    · 再用 `config.hallucination` 的同一套阈值/黑名单过一遍，滤掉「静音幻觉」
      （实测无声素材会吐「喵喵 / 響鐘 / 谢谢大家」这类每 30 秒准点一条的假文本）
    · 全部窗口 0 段通过 → **silent**（阻塞该讲）；任一窗口有段 → ok
    · 附带 ffmpeg `volumedetect` 的 mean/max 音量作旁证（只记录，不参与判决）

用法：
    python src/s2b_probe.py --video <DATA_ROOT>\\raw\\<课程>\\x.mp4
    python src/s2b_probe.py --wav  D:\\...\\audio.wav --windows 3 --sec 60
    退出码：0 = ok ｜ 3 = 静默素材（阻塞）｜ 1 = 出错
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
    apply_hf_env,
    cache_dir,
    get_logger,
    load_config,
    models_dir,
    probe_duration_sec,
    run,
    tool_paths,
)

apply_hf_env()  # 必须在 faster_whisper 导入前
from common import setup_cuda_dlls  # noqa: E402

setup_cuda_dlls()


def mean_volume_db(media: str | Path, cfg: dict) -> tuple[float | None, float | None]:
    """返回 (mean_volume_db, max_volume_db)；解析失败返回 (None, None)。"""
    ff = tool_paths(cfg)["ffmpeg"]
    cp = run([ff, "-hide_banner", "-i", str(media), "-af", "volumedetect", "-f", "null", "-"],
             check=False)
    txt = cp.stderr or ""
    out: list[float | None] = []
    for key in ("mean_volume:", "max_volume:"):
        val = None
        m = re.search(rf"{key}\s*(-?[\d.]+)\s*dB", txt)
        if m:
            try:
                val = float(m.group(1))
            except ValueError:
                val = None
        out.append(val)
    return out[0], out[1]


def _cut_window(media: Path, start: float, sec: int, dst: Path, cfg: dict) -> Path:
    ff = tool_paths(cfg)["ffmpeg"]
    run([ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}",
         "-t", str(sec), "-i", str(media), "-ar", "16000", "-ac", "1", str(dst)])
    return dst


def probe_speech(media: str | Path, cfg: dict | None = None, log=None,
                 windows: int = 3, sec: int = 60) -> dict:
    """判「音频里有没有语音」。返回 dict（含 verdict: ok|silent）。"""
    cfg = cfg or load_config()
    log = log or get_logger("s2b_probe", cfg)
    media = Path(media)
    if not media.exists():
        raise FileNotFoundError(f"预检素材不存在：{media}")

    dur = probe_duration_sec(media, cfg) or 0.0
    if dur <= 0:
        raise RuntimeError(f"读不到时长（素材损坏？）：{media}")

    if dur <= sec * windows:
        starts = [0.0]
    else:
        starts = [dur * f for f in (0.10, 0.50, 0.85)][:max(1, windows)]

    tmp_dir = cache_dir(cfg) / "_probe"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    a = cfg["asr"]
    hf = cfg.get("hallucination", {})
    ns_thr = float(hf.get("no_speech_prob", 0.6))
    lp_thr = float(hf.get("min_avg_logprob", -1.0))
    blocklist = [re.compile(p, re.I) for p in (hf.get("blocklist") or [])]

    t0 = time.time()
    from faster_whisper import WhisperModel

    #  2026-10-15 诊断补丁（**零行为变更，只加日志**）：
    #   理力第09讲实测「音轨抽取完成」→「预检窗口 1/3」之间**空白 4 h 14 min**（进程零 CPU = 阻塞等待），
    #   但日志什么都不打，无法区分卡在①模型加载 ②切窗 ffmpeg ③首个窗口转写。故在此埋点。
    #   （已知同类坑：HF 元数据校验打坏节点会无限挂起 → 已由 apply_hf_env() 强制离线兜住；
    #     本机模型快照完整、无残留锁文件 → 本次病因**尚未定论**，故先埋点取证。）
    log.info("预检：加载 Whisper 模型（%s / %s / %s）…", a["model"], a["device"], a["compute_type"])
    model = WhisperModel(a["model"], device=a["device"], compute_type=a["compute_type"],
                         download_root=str(models_dir(cfg)))
    log.info("预检：模型就绪（%.1fs）", time.time() - t0)
    keeps: list[dict] = []
    window_report: list[dict] = []
    try:
        for i, st in enumerate(starts, 1):
            log.info("预检：切割窗口 %d/%d（t=%.0fs）…", i, len(starts), st)
            cut = _cut_window(media, st, sec, tmp_dir / f"probe_{i}.wav", cfg)
            segs, _info = model.transcribe(
                str(cut), language=a.get("language", "zh"), beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=int(a.get("vad_min_silence_ms", 500))),
            )
            n_raw, n_ok = 0, 0
            for s in segs:
                n_raw += 1
                txt = (s.text or "").strip()
                if not txt:
                    continue
                if s.no_speech_prob > ns_thr or s.avg_logprob < lp_thr:
                    continue
                if any(p.search(txt) for p in blocklist):
                    continue
                n_ok += 1
                if len(keeps) < 5:
                    keeps.append({"t_sec": round(st + s.start, 1), "text": txt[:60]})
            window_report.append({"start_sec": round(st, 1), "segments_raw": n_raw,
                                  "segments_ok": n_ok})
            log.info("预检窗口 %d/%d（t=%.0fs）：VAD 后 %d 段，过幻觉闸门 %d 段",
                     i, len(starts), st, n_raw, n_ok)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    mean_v, max_v = mean_volume_db(media, cfg)
    elapsed = round(time.time() - t0, 1)
    ok_segs = sum(w["segments_ok"] for w in window_report)
    verdict = "ok" if ok_segs > 0 else "silent"
    detail = (f"窗口 {len(window_report)} 个 / 共 {ok_segs} 段过闸门；"
              f"mean={mean_v} dB, max={max_v} dB") if verdict == "ok" else (
        f"{len(window_report)} 个窗口全部 0 段（VAD 判无语音）；"
        f"mean={mean_v} dB, max={max_v} dB —— 疑似该回放音轨没有讲课语音")
    res = {
        "media": str(media), "duration_sec": round(dur, 1), "windows": window_report,
        "segments_ok": ok_segs, "mean_volume_db": mean_v, "max_volume_db": max_v,
        "verdict": verdict, "detail": detail, "elapsed_sec": elapsed, "sample": keeps,
    }
    log.info("素材音频预检：%s（%s）", "有语音 [ok]" if verdict == "ok" else "无语音 [x]", detail)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 2a-pre：素材音频预检（有没有人说话）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", default="", help="回放 mp4（会按窗口切片）")
    g.add_argument("--wav", default="", help="已抽好的 audio.wav")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--sec", type=int, default=60)
    ap.add_argument("--json", default="", help="把结果写到指定 json")
    args = ap.parse_args()
    try:
        res = probe_speech(args.video or args.wav, windows=args.windows, sec=args.sec)
    except Exception as e:  # noqa: BLE001
        print(f"[s2b_probe] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[s2b_probe]", json.dumps({k: res[k] for k in
                                     ("verdict", "segments_ok", "mean_volume_db", "detail",
                                      "elapsed_sec")}, ensure_ascii=False))
    return 0 if res["verdict"] == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
