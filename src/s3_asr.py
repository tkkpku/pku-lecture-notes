"""阶段 2b：语音转写（faster-whisper → SRT / JSON），含幻觉过滤与断点续跑。

输入：`work/<课程>/<讲次>/audio.wav`（由 s2_extract 产出）
产出：
  `asr.srt`   字幕（讲义对齐与人工校对都用得上，属清理时保留的轻量产物）
  `asr.json`  段级结构 + 质量诊断（每段 no_speech_prob / avg_logprob / compression_ratio）
  `asr.txt`   纯文本（喂融合阶段）

[warn] 本机硬约束（2026-09-12 实测）：
  * `compute_type` 不要写 `auto`。实测 float16 与 int8_float16 在 sm_120 上**都能跑**
    （int8 更快更省显存，见 docs/RESEARCH.md §4 修正），但 auto 的选择不可控。
  * CUDA 前向需要 cuBLAS：靠 `common.setup_cuda_dlls()` 挂 `nvidia-cublas-cu12` 的 dll 目录，
    **必须在 import faster_whisper 之前**调用。
  * HF 模型下载必须关 Xet（`common.apply_hf_env()`），否则 hf-mirror 报 401。

幻觉过滤（重要，不是可选）：实测合成音频会被脑补出「优优独播剧场——YoYo Television Series
Exclusive」「感谢观看」这类 YouTube 字幕残留。判据三选一即丢弃：
  ① `no_speech_prob > hallucination.no_speech_prob`
  ② `avg_logprob < hallucination.min_avg_logprob`
  ③ 段起始时间 > 音频实际时长（模型凭空延长）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    apply_hf_env,
    get_logger,
    lecture_work_dir,
    load_config,
    models_dir,
    probe_duration_sec,
)

apply_hf_env()  # 必须在 faster_whisper 导入前
from common import setup_cuda_dlls  # noqa: E402

setup_cuda_dlls()


def _fmt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_PUNCT_RE = re.compile(r"[\s，。、,.:：；;！!？?…—\-–()（）\"'“”‘’]+")


def _norm_text(t: str) -> str:
    return _PUNCT_RE.sub("", t or "")


def _apply_filters(records: list[dict], cfg: dict, dur: float,
                   log=None) -> tuple[list[dict], list[dict]]:
    """给段打 `dropped` / `drop_reason`，并做「连续重复幻觉」二次清理。返回 (kept, dropped)。

    [!] 2026-09-13 实测修正（保真度，真实素材现行犯）：
      ① **vad_filter 开启时改用 `no_speech_prob_when_vad`（默认 0.95，≈不再拦）**。
         理由：VAD 已把非语音段滤掉，能进转写的必然有语音；此时 `no_speech_prob` 会在**真实内容**
         上给出 0.65~0.90 的高值。实测本讲 t=445~675s 连续 14 段被它误杀 —— 那块音频 mean −20.8 dB
         （正常段 −20.4 dB）、文本与前后保留段语义无缝衔接、avg_logprob 高到 −0.04。
         「静音素材 → 幻觉」这个原始风险现已由上游**素材音频预检**（`s2b_probe.py`）兜住。
      ② 新增**连续重复幻觉检测**：同一文本（去标点/空白后）连续出现 ≥ `repeat_run` 次即整段丢弃。
         两次实测的幻觉都是这个形态（无声素材的「喵喵/響鐘」链、本讲 t≈3670s 的「有一个记忆」连续 10 次）。
     2026-09-17（你批准放宽）：阈值**按文本长短分档** ——
       · 短文本（≤ `repeat_run_short_maxlen` 字）用 `repeat_run`（默认 3）：经典 whisper 循环；
       · 长文本用 `repeat_run_long`（默认 8）：长句复读更可能是**对同一段音频的重复解码**。
       现行犯：理论力学第03讲 t=1327~1520 s 有 15 组语义通顺、互相衔接的**真实讲课内容**
       被旧的 `repeat_run=3` 全部误杀。残余风险由 blocklist 与上游素材音频预检兜住。
    """
    hf = cfg.get("hallucination", {})
    vad_on = bool(cfg["asr"].get("vad_filter", True))
    ns_thr = float(hf.get("no_speech_prob_when_vad", 0.95) if vad_on
                   else hf.get("no_speech_prob", 0.6))
    lp_thr = float(hf.get("min_avg_logprob", -1.0))
    cr_thr = float(hf.get("max_compression_ratio", 2.4))
    blocklist = [re.compile(p, re.I) for p in (hf.get("blocklist") or [])]
    repeat_run = max(2, int(hf.get("repeat_run", 3)))
    repeat_run_long = max(repeat_run, int(hf.get("repeat_run_long", 8)))
    short_maxlen = max(1, int(hf.get("repeat_run_short_maxlen", 4)))

    for r in records:
        reasons = []
        nsp = float(r.get("no_speech_prob") or 0.0)
        if nsp > ns_thr:
            reasons.append(f"no_speech={nsp:.2f}")
        lp = float(r.get("avg_logprob") or 0.0)
        if lp < lp_thr:
            reasons.append(f"logprob={lp:.2f}")
        if dur and float(r.get("start") or 0) > dur + 1.0:
            reasons.append(f"t_start>{dur:.0f}s")
        cr = float(r.get("compression_ratio") or 0.0)
        if cr > cr_thr:
            reasons.append(f"compression_ratio={cr:.2f}")
        for pat in blocklist:
            if pat.search(r.get("text") or ""):
                reasons.append(f"blocklist:{pat.pattern[:24]}")
                break
        r["dropped"] = bool(reasons)
        r["drop_reason"] = "; ".join(reasons)

    i = 0
    while i < len(records):
        base = _norm_text(records[i].get("text") or "")
        j = i + 1
        while j < len(records) and base and _norm_text(records[j].get("text") or "") == base:
            j += 1
        #  2026-09-17：阈值按文本长短分档（长文本放宽，见 docstring）
        thr = repeat_run if len(base) <= short_maxlen else repeat_run_long
        if base and j - i >= thr:
            for k in range(i, j):
                extra = f"repeat_run={j - i}>=thr{thr}(len={len(base)})"
                records[k]["dropped"] = True
                records[k]["drop_reason"] = (f"{records[k]['drop_reason']}; {extra}"
                                             if records[k].get("drop_reason") else extra)
            if log:
                log.warning("连续重复幻觉 %d 段（t=%.0f~%.0f s，阈值 %d，文本 %d 字）已丢弃：%r",
                            j - i, float(records[i].get("start") or 0),
                            float(records[j - 1].get("end") or 0), thr, len(base),
                            (records[i].get("text") or "")[:30])
        i = j if j > i else i + 1

    return [r for r in records if not r["dropped"]], [r for r in records if r["dropped"]]


def _load_hotwords(cfg: dict, course: str = "") -> list[str]:
    """热词：config 内的 asr.hotwords + 全局 `config/terms.txt` +（若存在）**课程专属** `config/terms.<课程>.txt`。

    [warn] 2026-09-13 新增课程专属热词：terms.txt 文件头原就写着「本课程的专有名词」，
    但实现一直是全局的 —— 结果理论力学的术语（应力 / 面元 / 张量）会被喂给普通物理课。
    域外热词虽不会直接产生错字，但会把解码往那个领域带，故按课程分开。
    命名：`terms.<课程全名>.txt`（如 `terms.理论力学(26-27学年第1学期).txt`），
    也接受去掉括号部分的名字（`terms.普通物理 (II).txt`）。
    """
    words = [str(w) for w in (cfg["asr"].get("hotwords") or [])]
    base = Path(cfg["_config_path"]).parent
    files = [base / "terms.txt"]
    if course:
        stem = course.split("(")[0].strip() or course
        for nm in dict.fromkeys([course, stem]):
            files.append(base / f"terms.{nm}.txt")
    for f in files:
        if f.exists():
            words += [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
                      if ln.strip() and not ln.strip().startswith("#")]
    return list(dict.fromkeys(words))


def transcribe_lecture(course: str, lecture: str, cfg: dict | None = None,
                       force: bool = False, log=None) -> dict:
    cfg = cfg or load_config()
    log = log or get_logger("s3_asr", cfg)
    wd = lecture_work_dir(course, lecture, cfg)
    wav = wd / "audio.wav"
    json_path = wd / "asr.json"
    srt_path = wd / "asr.srt"
    txt_path = wd / "asr.txt"

    if not wav.exists():
        raise FileNotFoundError(f"缺 audio.wav（先跑 s2_extract）：{wav}")
    if json_path.exists() and not force:
        log.info("已存在 asr.json，跳过（--force 重跑）")
        return json.loads(json_path.read_text(encoding="utf-8"))

    a = cfg["asr"]
    dur = probe_duration_sec(wav, cfg)
    hotwords = _load_hotwords(cfg, course)
    log.info("转写开始：%s | %.1f min | model=%s device=%s compute_type=%s",
             wav.name, dur / 60, a["model"], a["device"], a["compute_type"])
    if hotwords:
        log.info("热词 %d 个：%s", len(hotwords), hotwords[:8])

    from faster_whisper import WhisperModel

    t0 = time.time()
    model = WhisperModel(a["model"], device=a["device"], compute_type=a["compute_type"],
                         download_root=str(models_dir(cfg)))
    t_load = time.time() - t0

    kwargs = dict(
        language=a.get("language", "zh"),
        beam_size=int(a.get("beam_size", 5)),
        vad_filter=bool(a.get("vad_filter", True)),
        initial_prompt=a.get("initial_prompt") or None,
        word_timestamps=False,
        # [!] 2026-09-25（真实现行犯修复）：**必须关掉「上文条件」**。
        #    faster-whisper 默认 `condition_on_previous_text=True` —— 把**上一段解码输出**当作下一段的
        #    prompt。若片头有一段静音（本机实测普物第06讲 t=0~397 s 电平 −44.5 dB），whisper 会在那里
        #    吐一句幻觉（「请不吝点赞 订阅 转发支持明镜与点点栏目」），**这句幻觉随即被当成 prompt
        #    传染下去**：整片 119 分钟被判成 183 段、全是同一句话（其中 179 段连续），真实讲课内容
        #    全被 `repeat_run` 闸门连带丢弃 → ASR 产出 0 字（硬闸门拦下，防假绿生效）。
        #    判据组合（三条同时出现即此病）：① RTF 异常低（本讲 0.053，正常 0.13~0.20，因为 VAD 把
        #    大量段落跳过）；② 同一文本连续数百段；③ **分段电平正常**（本讲 −21~−24 dB，与正常课相同）、
        #    且把任一段单独抽出来转写能得到**真实讲课文本**（实测 t=600 s → 「微平均等于 8KBT 除以 π」）。
        #    [warn] 为什么预检（s2b_probe）发现不了：预检是**分窗口独立解码**（每窗 60 s、每次一个新文件），
        #    幻觉在短窗内只有 1~2 条、够不到 `repeat_run` 阈值 → 被误判为真实语音。
        #    关闭后每段独立解码，幻觉不再传染（代价：跨段上下文连贯性略降，后段由 LLM 融合阶段补救）。
        condition_on_previous_text=bool(a.get("condition_on_previous_text", False)),
    )
    if hotwords:
        kwargs["hotwords"] = " ".join(hotwords)
    if a.get("vad_filter"):
        kwargs["vad_parameters"] = dict(min_silence_duration_ms=int(a.get("vad_min_silence_ms", 500)))

    t1 = time.time()
    segments, info = model.transcribe(str(wav), **kwargs)
    segs = list(segments)
    t_asr = time.time() - t1

    corr = cfg["asr"].get("corrections") or {}
    n_corrected = 0
    records: list[dict] = []
    for s in segs:
        text = (s.text or "").strip()
        for bad, good in corr.items():
            if bad in text:
                text = text.replace(bad, good)
                n_corrected += 1
                if log:
                    log.info("同音字纠正：%s → %s", bad, good)
        records.append({
            "start": round(s.start, 2), "end": round(s.end, 2), "text": text,
            "no_speech_prob": round(s.no_speech_prob, 3),
            "avg_logprob": round(s.avg_logprob, 3),
            "compression_ratio": round(getattr(s, "compression_ratio", 0.0), 3),
            "dropped": False, "drop_reason": "",
        })
    kept, dropped = _apply_filters(records, cfg, dur, log)

    log.info("转写完成：加载 %.1fs + 识别 %.1fs | RTF=%.3f | 段 %d（丢弃幻觉 %d）",
             t_load, t_asr, (t_asr / dur) if dur else 0, len(records), len(dropped))
    for d in dropped[:5]:
        log.warning("丢弃幻觉段 [%.1f-%.1f] %s ← %s", d["start"], d["end"], d["text"][:40], d["drop_reason"])

    # [warn] 2026-09-13 新增硬闸门（真实素材现行犯，修 bug B）：
    #    音频里没有语音时，faster-whisper **不报错**，只是返回 0 段 → 旧版照样打印
    #    `[asr] OK (38.6s)`、写出空的 asr.srt/txt/json，pipeline 汇总里「失败=无」。
    #    这是最危险的一类缺陷（假绿）：空转写会一路融合成一「只看画面猜内容」的假讲义，
    #    而成本已经烧掉了。故此处改成硬失败，让 pipeline 记为 failed 并暴露在报告里。
    text_len = sum(len(r["text"]) for r in kept)
    if not kept or (dur > 300 and text_len < 50):
        msg = (f"ASR 产出为空/近空：段 {len(records)}（丢弃 {len(dropped)}）、正文 {text_len} 字 / "
               f"音频 {dur / 60:.1f} min。疑似音频无语音（实测：无声素材会让 whisper 返回 0 段而整条链路照样「成功」）。"
               f"请先 ffmpeg volumedetect 查 audio.wav 电平，并确认该回放音轨本身含讲课语音。")
        log.error(msg)
        raise RuntimeError(msg)

    srt_path.write_text("".join(
        f"{i}\n{_fmt_ts(r['start'])} --> {_fmt_ts(r['end'])}\n{r['text']}\n\n"
        for i, r in enumerate(kept, 1)), encoding="utf-8")
    txt_path.write_text("".join(r["text"] for r in kept), encoding="utf-8")

    payload = {
        "course": course, "lecture": lecture, "audio": str(wav),
        "audio_sec": round(dur, 1), "model": a["model"], "device": a["device"],
        "compute_type": a["compute_type"], "hotwords": hotwords,
        "initial_prompt": a.get("initial_prompt", ""),
        "load_sec": round(t_load, 1), "asr_sec": round(t_asr, 1),
        "rtf": round((t_asr / dur) if dur else 0, 3),
        "language": info.language, "language_prob": round(info.language_probability, 3),
        "segments_total": len(records), "segments_dropped": len(dropped),
        "corrections_applied": n_corrected,
        "text_len": text_len,
        "segments": records,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("产出：%s / %s / %s（正文 %d 字）", srt_path.name, json_path.name, txt_path.name, payload["text_len"])
    return payload


def refilter_lecture(course: str, lecture: str, cfg: dict | None = None, log=None) -> dict:
    """对**已有 asr.json 重新过幻觉闸门**（零 GPU 成本），用于判据修正后回收被误杀的内容。

    与重跑 ASR 等价（whisper 的输出已存盘，只重跑「丢弃判据」这一步），但只需几秒。
    """
    cfg = cfg or load_config()
    log = log or get_logger("s3_asr", cfg)
    wd = lecture_work_dir(course, lecture, cfg)
    json_path = wd / "asr.json"
    if not json_path.exists():
        raise FileNotFoundError(f"缺 asr.json（先跑 s3_asr）：{json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    dur = float(payload.get("audio_sec") or 0)
    before_drop = int(payload.get("segments_dropped") or 0)
    before_len = int(payload.get("text_len") or 0)
    kept, dropped = _apply_filters(payload["segments"], cfg, dur, log)
    payload["segments_dropped"] = len(dropped)
    payload["text_len"] = sum(len(r["text"]) for r in kept)
    payload["refiltered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload["refilter_delta_chars"] = payload["text_len"] - before_len
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (wd / "asr.srt").write_text("".join(
        f"{i}\n{_fmt_ts(r['start'])} --> {_fmt_ts(r['end'])}\n{r['text']}\n\n"
        for i, r in enumerate(kept, 1)), encoding="utf-8")
    (wd / "asr.txt").write_text("".join(r["text"] for r in kept), encoding="utf-8")
    log.info("重筛完成（不重跑解码）：丢弃 %d → %d 段；正文 %d → %d 字（回收 %+d 字）",
             before_drop, len(dropped), before_len, payload["text_len"],
             payload["text_len"] - before_len)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 2b：语音转写")
    ap.add_argument("--course", required=True)
    ap.add_argument("--lecture", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--refilter", action="store_true",
                    help="不重跑解码，只对已有 asr.json 重新过幻觉闸门（判据修正后回收误杀内容）")
    args = ap.parse_args()
    res = (refilter_lecture(args.course, args.lecture) if args.refilter
           else transcribe_lecture(args.course, args.lecture, force=args.force))
    print(json.dumps({k: res[k] for k in
                      ("audio_sec", "rtf", "segments_total", "segments_dropped", "text_len")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
