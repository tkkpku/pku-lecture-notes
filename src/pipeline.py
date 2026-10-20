"""主控 CLI：分段可跑、断点续跑、dry-run。

六个阶段：

| 代号 | 模块 | 输入 → 产出 |
|---|---|---|
| `fetch` | s1_fetch | pku3b → `raw/<课程>/*.mp4`（[!] 需你本人 `pku3b init`） |
| `extract` | s2_extract | mp4 → `work/<讲>/audio.wav` + `frames/` + `frames_index.json` |
| `asr` | s3_asr | audio.wav → `asr.srt` / `asr.json` / `asr.txt`（含幻觉过滤） |
| `ocr` | s4_ocr | frames → `frames_ocr.json`（WinOCR + 可选 VLM 兜底） |
| `fuse` | s5_fuse | asr + frames_ocr → `lecture.md`（讲义草稿） |
| `publish` | s6_publish | lecture.md → Obsidian 讲义 + 索引 + 验收后清理 |

用法：
  # 端到端（真的从 pku3b 抓）
  python src/pipeline.py --course 理论力学 --lecture 第03讲_应力张量

  # 已有视频，跳过抓取（演练/补跑都走这条）
  python src/pipeline.py --course 合成课程 --lecture 第01讲_应力张量 \
      --video <DATA_ROOT>\\raw\\合成课程\\第01讲_应力张量.mp4

  # 只跑某些阶段 / 只看不动
  python src/pipeline.py --course X --lecture Y --stages ocr,fuse --dry-run

断点续跑：每个阶段自身幂等（产物存在即跳过，`--force` 覆盖）。运行状态写
`work/<课程>/<讲次>/run_status.json`。卡点纪律（你钦定）：任一阶段异常**记录后继续**走下一
阶段，不原地死磕；`fetch` 阶段缺凭据时明确记为阻塞项。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    get_logger,
    guard_work_identity,
    ignored_match,
    lecture_work_dir,
    load_config,
    raw_dir,
    run,
    tool_paths,
)

STAGES = ["fetch", "extract", "asr", "ocr", "fuse", "publish", "review"]
_INDEX_RE = re.compile(r"第\s*(\d+)\s*讲")


def _parse_lecture_name(lecture: str) -> tuple[int, str]:
    m = _INDEX_RE.search(lecture)
    idx = int(m.group(1)) if m else 1
    topic = _INDEX_RE.sub("", lecture).strip(" _-")
    return idx, topic or lecture


def _write_meta(course: str, lecture: str, video: Path, cfg: dict, wd: Path, log,
                video_id: str = "") -> Path:
    idx, topic = _parse_lecture_name(lecture)
    dur = 0.0
    try:
        from common import probe_duration_sec

        if video and video.exists():
            dur = probe_duration_sec(video, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("读视频时长失败（忽略）：%s", e)
    meta = {
        "course": course, "lecture": lecture, "index": idx, "topic": topic,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "duration_min": round(dur / 60, 1),
        "source": "北大教学网课程回放", "video": str(video) if video else "",
        # 2026-09-13 新增：记下回放 ID —— 讲次编号以教学网为锚，日后重排顺序要靠它回溯。
        "video_id": video_id,
        "index_anchor": "教学网回放时间升序（同课程内最早 = 第01讲）",
    }
    p = wd / "meta.json"
    if not p.exists():
        wd.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("写入 meta.json：index=%s topic=%s duration=%.1fmin", idx, topic, meta["duration_min"])
    return p


def _find_video(course: str, lecture: str, cfg: dict, explicit: str = "",
                video_id: str = "") -> Path | None:
    """定位本讲要用的 mp4。

    解析顺序（2026-09-13 修订，修掉两处「静默跑错讲次」的坑）：
      1. `--video` 显式指定 → 直接用（不存在则 None）
      2. **本讲 work 目录的 `meta.json`**（权威记录）：里面的 `video` 路径存在、且（若 meta 记了
         `video_id`）与本次要抓的 ID 一致 → 用它。这解决「同一门课多讲共存」的重跑问题。
      3. 文件名含讲次标签的 glob
      4. **兜底**：目录里恰好一个 mp4 **且本次没有指定 `--video-id`** 时才用。
         [warn] 指定了 `--video-id` 却还要靠「目录里唯一那个 mp4」来猜，就是把别的讲次当成这一讲 ——
         实测场景：同一门课的 第01讲 视频还在 raw 目录里时启动 第02讲，兜底会错拿 第01讲。
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    wd = lecture_work_dir(course, lecture, cfg)
    mp = wd / "meta.json"
    if mp.exists():
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            m = {}
        p = Path(str(m.get("video") or ""))
        m_vid = str(m.get("video_id") or "")
        if p.name and p.exists() and (not video_id or not m_vid or m_vid == video_id):
            return p

    d = raw_dir(cfg) / course
    if not d.is_dir():
        return None
    cands = sorted([*d.glob(f"{lecture}*.mp4"), *d.glob(f"*{lecture}*.mp4")],
                   key=lambda p: p.stat().st_size, reverse=True)
    if cands:
        return cands[0]
    any_mp4 = sorted(d.glob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True)
    if len(any_mp4) == 1 and not video_id:
        return any_mp4[0]
    return None


def run_pipeline(course: str, lecture: str, cfg: dict | None = None, video: str = "",
                 video_id: str = "", stages: list[str] | None = None, dry_run: bool = False,
                 force: bool = False, keep: bool = False, log=None,
                 topic: str = "", date: str = "", auto_index: bool = True,
                 force_ignored: bool = False) -> dict:
    cfg = cfg or load_config()
    log = log or get_logger("pipeline", cfg)
    # [warn] 2026-09-13 实测修复（bug I）：原写法 `stages or STAGES` 把**显式空列表**判成 falsy
    #    → 「跑 0 个阶段」被静默升级成「跑全部六阶段」。现行犯：我想做一次零副作用的锚定自测，
    #    `stages=[]` 却启动了整条流水线（开始下载另一个课程的 700 MB 回放）。
    #    改为「None 才回退到全阶段」，显式空 = 什么都不跑。
    stages = [s.strip() for s in (STAGES if stages is None else stages) if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise ValueError(f"未知阶段 {bad}，可选：{STAGES}")

    # ---- 永久忽略清单闸门（2026-09-17 新增）----
    # 清单 = `config/ignored_lectures.yaml`（video_id 为主键，(课程,讲次) 兜底）。
    # 里面的回放**永远出不了讲义**（录像为空 / 音轨无声，都是录播方的问题），放行只会白烧
    # 一次下载（~900 MB）与预检。故在**任何副作用之前**（不建 work 目录、不下载、不锚定）拒绝。
    hit = ignored_match(course=course, lecture=lecture, video_id=video_id, reload=True)
    if hit and not force_ignored:
        log.warning("[skip] 该回放在永久忽略清单里，已拒绝执行：%s / %s / %s（原因：%s）",
                    hit.get("course") or course, hit.get("lecture") or lecture,
                    hit.get("date") or "", hit.get("reason") or "")
        return {"course": course, "lecture": lecture, "stages": stages, "dry_run": dry_run,
                "refused": "ignored", "ignore_entry": hit, "elapsed_sec": 0.0,
                "results": {}, "blocked": [], "failed": []}

    # ---- work 目录身份闸门（2026-10-12 新增，事故见 HANDOFF §24）----
    # work 目录按 (课程, 讲次) 命名，而讲次编号会随教学网列表变动**前移**：新回放会继承
    # 旧回放的目录，里面的 asr.json / frames 被当成「已完成」跳过 → 画面是新、转写是旧。
    # 这里在**任何阶段之前**核对 meta.json 的 video_id，不符即把旧目录改名隔离（不删）。
    # 每次调用都查（run_lecture.ps1 分三次调 pipeline.py），幂等：隔离后 meta.json 就是新的。
    if video_id and not dry_run:
        guard_work_identity(course, lecture, video_id, cfg, log=log)

    # ---- 讲次锚定（以教学网为准，2026-09-13 新增；可用 --no-auto-index 关掉）----
    if auto_index and video_id and not dry_run:
        try:
            from s1_fetch import anchor_index

            idx = anchor_index(course, video_id, cfg, refresh=True, log=log)
        except Exception as e:  # noqa: BLE001
            idx = None
            log.warning("讲次锚定失败（沿用命令行给的编号）：%s: %s", type(e).__name__, e)
        if idx:
            cur = _parse_lecture_name(lecture)[0]
            if idx != cur:
                new_label = (_INDEX_RE.sub(f"第{idx:02d}讲", lecture) if _INDEX_RE.search(lecture)
                             else f"第{idx:02d}讲_{lecture}")
                log.warning("讲次锚定校正：按教学网该回放是第%02d讲 → 标签 %r 改为 %r"
                            "（如需保留原标签用 --no-auto-index）", idx, lecture, new_label)
                lecture = new_label
            else:
                log.info("讲次锚定核对通过：第%02d讲（教学网顺序）", idx)
        else:
            log.warning("讲次锚定查不到该回放（教学网列表里没有或取不到），沿用命令行编号：%s", lecture)

    wd = lecture_work_dir(course, lecture, cfg)
    wd.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("流水线启动：%s / %s | 阶段=%s | dry_run=%s", course, lecture, ",".join(stages), dry_run)

    results: dict[str, dict] = {}
    status_path = wd / "run_status.json"
    vid = _find_video(course, lecture, cfg, video, video_id)
    t_start = time.time()

    for st in stages:
        t0 = time.time()
        try:
            if st == "fetch":
                if vid:
                    results[st] = {"ok": True, "detail": f"已有视频，跳过抓取：{vid}"}
                    log.info("[fetch] 已有视频，跳过：%s", vid)
                    continue
                from s1_fetch import download_video, list_videos

                # [warn] 不以「凭据文件是否存在」为门禁：实测本机凭据在系统钥匙串里，
                #    文件不存在但 pku3b 完全可用。真正判据 = 命令是否成功。
                if video_id:
                    out_course = raw_dir(cfg) / course
                    dl = download_video(video_id, out_course, cfg, log=log, dry_run=dry_run)
                    if not dry_run and dl.get("rc") != 0:
                        results[st] = {"ok": False, "blocked": True,
                                       "detail": f"下载失败 rc={dl.get('rc')}：{dl.get('stderr_tail', '')[:300]}"}
                        log.warning("[fetch] 阻塞：%s", results[st]["detail"])
                    else:
                        results[st] = {"ok": True, "detail": dl}
                        log.info("[fetch] 下载完成：%s", dl.get("newest_mp4", ""))
                        # [warn] 2026-09-13 实测修复（bug A）：把 fetch 刚下好的视频**直接**交给后续阶段，
                        #    不再依赖 _find_video() 的「目录里最大的 mp4」启发式
                        #    （多讲视频共存时那个启发式会挑错讲次，且下载文件名不含讲次标签）。
                        if dl.get("newest_mp4"):
                            vid = Path(dl["newest_mp4"])
                            #  2026-09-19（你批准）：**fetch 成功即写 meta.json**（bug N 根治）。
                            #    原先 `_write_meta()` 只在 extract 阶段调用 → 单独跑 `--stages fetch`
                            #    （预下载；我实战编排正是"趁 GPU 跑 ASR 时先下下一讲的 mp4"）**不留 meta.json**。
                            #    随后在**另一个进程**里跑 extract 时，`_find_video` 三条判据全不命中：
                            #      ② meta.json 不存在；③ 下载文件名是《课名_日期第X-Y节.mp4》，**不含**「第NN讲」；
                            #      ④ 「目录里唯一 mp4」兜底又被 `--video-id` 禁用（J 修复的副作用）
                            #    → 必报「找不到视频（raw/<课程>/*.mp4），跳过」，asr 连带失败。
                            #    实测现行犯：理论力学第04讲、常微分方程第04讲（本轮各中一次，当时靠 `--video` 绕开）。
                            #    [warn] `_write_meta()` 自带「已存在则不覆盖」保护，放这里不会冲掉后续阶段的记录。
                            if not dry_run:
                                _write_meta(course, lecture, vid, cfg, wd, log, video_id=video_id)
                    continue

                lst = list_videos(cfg, log=log, dry_run=dry_run)
                if not dry_run and lst.get("rc") != 0:
                    results[st] = {"ok": False, "blocked": True,
                                   "detail": f"pku3b video list 失败 rc={lst.get('rc')}；"
                                             "若提示未登录，需你本人执行 pku3b init"}
                    log.warning("[fetch] 阻塞：%s", results[st]["detail"])
                    continue
                items = lst.get("items", [])
                courses = sorted({it["course"] for it in items})
                results[st] = {"ok": True, "items": len(items), "courses": len(courses),
                               "detail": f"list 到 {len(items)} 条 / {len(courses)} 门课；"
                                         "用 --video-id <回放ID> 抓指定讲次"}
                log.info("[fetch] %s", results[st]["detail"])
                for it in items:
                    log.info("   • [%s] %s  %s", it["id"], it["course"], it["title"])
                continue

            if st == "extract":
                if not vid or not vid.exists():
                    # [warn] 2026-09-13 实测修复（bug A 的另一半）：vid 原在进入阶段循环前只解析一次，
                    #    而 fetch 阶段本身才生产视频 → `--stages fetch,extract,asr` 同跑时 vid 恒为 None，
                    #    extract 必然「找不到视频」跳过（真实首跑现行犯）。故此处再解析一次兜底。
                    vid = _find_video(course, lecture, cfg, video, video_id)
                if not vid or not vid.exists():
                    results[st] = {"ok": False, "detail": f"找不到视频（raw/{course}/*.mp4），跳过"}
                    log.warning("[extract] %s", results[st]["detail"])
                    continue
                _write_meta(course, lecture, vid, cfg, wd, log, video_id=video_id)
                from s2_extract import extract_lecture

                res = extract_lecture(vid, course, lecture, cfg, force=force, dry_run=dry_run)
                results[st] = {"ok": True, "detail": res}

            elif st == "asr":
                from s3_asr import transcribe_lecture

                res = transcribe_lecture(course, lecture, cfg, force=force, log=log)
                results[st] = {"ok": True, "detail": {k: res[k] for k in
                                                      ("audio_sec", "rtf", "segments_total",
                                                       "segments_dropped", "text_len")}}

            elif st == "ocr":
                from s4_ocr import process_lecture

                res = process_lecture(course, lecture, cfg, force=force, log=log)
                results[st] = {"ok": True, "detail": res.get("stats", {})}

            elif st == "fuse":
                from s5_fuse import fuse_lecture

                res = fuse_lecture(course, lecture, cfg, force=force, dry_run=dry_run, log=log)
                results[st] = {"ok": True, "detail": {k: v for k, v in res.items()
                                                      if k != "prompt_preview"}}

            elif st == "publish":
                if dry_run:
                    results[st] = {"ok": True, "detail": "dry-run：跳过发布"}
                else:
                    from s6_publish import publish_lecture

                    res = publish_lecture(course, lecture, cfg, date=date, topic=topic,
                                          do_cleanup=(False if keep else None),
                                          dry_run=dry_run, log=log)
                    results[st] = {"ok": True, "detail": res}

            elif st == "review":
                # 章节收口判定 +（命中则）自动写复习文档（2026-10-15 你钦定）。
                # [!] 判定用 LLM 而非启发式脚本：模糊语义判断脚本做不好（假警报一多就没人看，
                #    而"没人看"正是这套机制当初被整个漏掉的方式）。见 s7_review 模块头。
                if dry_run:
                    results[st] = {"ok": True, "detail": "dry-run：跳过复习判定"}
                else:
                    from s7_review import review_lecture as _review

                    res = _review(course, lecture, cfg, force=force, log=log)
                    results[st] = {"ok": True, "detail": {
                        "verdict": res.get("verdict"),
                        "generated": (res.get("generated") or {}).get("file"),
                        "chars": (res.get("generated") or {}).get("chars"),
                    }}
        except Exception as e:  # noqa: BLE001
            results[st] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            log.error("[%s] 失败（按卡点纪律继续下一阶段）：%s: %s", st, type(e).__name__, e)

        results[st] = results.get(st, {"ok": False, "detail": "未执行"})
        results[st]["elapsed_sec"] = round(time.time() - t0, 1)
        log.info("[%s] %s (%.1fs)", st, "OK" if results[st].get("ok") else "FAIL", results[st]["elapsed_sec"])

    summary = {
        "course": course, "lecture": lecture, "stages": stages, "dry_run": dry_run,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t_start, 1),
        "results": results,
        "blocked": [k for k, v in results.items() if v.get("blocked")],
        "failed": [k for k, v in results.items() if v.get("ok") is False and not v.get("blocked")],
    }
    status_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("流水线结束：%.1fs | 失败=%s | 阻塞=%s | 状态：%s",
             summary["elapsed_sec"], summary["failed"] or "无", summary["blocked"] or "无", status_path)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="课程录像 → 讲义 流水线主控")
    ap.add_argument("--course", required=True)
    ap.add_argument("--lecture", required=True)
    ap.add_argument("--video", default="", help="指定已有视频（给了就跳过 fetch）")
    ap.add_argument("--video-id", default="", help="pku3b 回放 ID（见 --stages fetch 列出的清单），给了就抓这一讲")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"逗号分隔，可选 {STAGES}")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="忽略已有产物重跑各阶段")
    ap.add_argument("--keep", action="store_true", help="发布后保留视频与中间产物（不清理）")
    ap.add_argument("--topic", default="", help="讲义主题（用于文件名与 frontmatter；缺省从讲义首行推断）")
    ap.add_argument("--date", default="", help="讲义日期 YYYY-MM-DD（缺省用 meta.json 或今天）")
    ap.add_argument("--no-auto-index", action="store_true",
                    help="关闭「以教学网为准」的讲次锚定校正（默认开启，会按教学网顺序改写讲次编号）")
    ap.add_argument("--force-ignored", action="store_true",
                    help="强制执行永久忽略清单（config/ignored_lectures.yaml）里的回放；"
                         "正常情况下不该用它——要改口径就改清单文件")
    args = ap.parse_args()
    s = run_pipeline(args.course, args.lecture, video=args.video, video_id=args.video_id,
                     stages=args.stages.split(","), dry_run=args.dry_run,
                     force=args.force, keep=args.keep, topic=args.topic, date=args.date,
                     auto_index=not args.no_auto_index, force_ignored=args.force_ignored)
    if s.get("refused") == "ignored":
        e = s.get("ignore_entry") or {}
        print(f"\n[refused] 回放 {e.get('video_id', '?')} 在永久忽略清单里：{e.get('reason', '')}")
        print(f"          {e.get('course', '')} / {e.get('lecture', '')} / {e.get('date', '')} "
              f"{e.get('sections', '')}")
        print(f"          证据：{str(e.get('evidence', '')).strip()[:200]}")
        print("          清单文件：config/ignored_lectures.yaml（要改口径就改它，别绕过闸门）")
        print("          如确需强制执行：加 --force-ignored")
        return 2
    print("\n[summary] " + json.dumps({k: s[k] for k in ("elapsed_sec", "failed", "blocked")},
                                      ensure_ascii=False))
    for st, r in s["results"].items():
        print(f"  {st:8s} ok={r.get('ok')} {str(r.get('detail') or r.get('error') or '')[:120]}")
    return 0 if not s["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
