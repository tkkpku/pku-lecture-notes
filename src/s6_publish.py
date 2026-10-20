"""阶段 4：发布到 Obsidian + 索引维护 + 验收后清理。

流程：
  1. 读 `work/<课程>/<讲次>/lecture.md`（s5_fuse 的草稿）
  2. 校验讲义非空且 ≥ `cleanup.min_note_bytes`（安全阀：**防「讲义没生成成功却把视频删了」**）
  3. 补 YAML frontmatter → 落 `<vault>/<paths.notes_library>/<课程>/<date>_第NN讲_<主题>.md`
     （`notes_library` 现为 `02_随堂讲义`——2026-09-21 Vault 重组后由「课程讲义」改名，
      写死在代码里会让讲义分裂到两个目录，故一律读配置）
  4. 重建该课的 `00_索引.md`（扫目录解析 frontmatter，幂等）
  5. 清理该讲的视频与中间产物（保留 `publish.keep_artifacts` 里的轻量文本：asr.srt / meta.json / frames_index.json）

[warn] 清理是**破坏性操作**，规则（你钦定）：
  * 每讲讲义**产出并验收后**才删，绝不提前删
  * 删除前必须过安全阀（讲义存在 + 字节数达标）
  * 本模块默认打印将删除的清单与体积；`--keep` 可跳过清理，`--dry-run` 只看不动
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    human_size,
    lecture_work_dir,
    load_config,
    notes_dir,
    raw_dir,
    work_dir,
)

HERE = Path(__file__).resolve().parent


# ---------------- frontmatter ----------------
#  2026-09-19：s5_fuse 在某块融合失败时会写入「## [第 N 块生成失败]」占位符以便续跑。
#    但旧闸门只在**全部**块失败时才硬失败（bug E），于是「部分块失败」仍会一路走到 publish，
#    把残缺讲义（少了整段内容）当成合格讲义发出去 —— 假绿。这里补上发布前的最后一道闸门：
#    草稿里只要出现占位符，一律拒绝发布（讲义缺失内容比不发更糟）。
_PLACEHOLDER_RE = re.compile(r"\[第\s*\d+\s*块生成失败\]")


# ---------------- 数学分隔符归一化 ----------------
#  2026-10-12：Obsidian 只认 `$...$` / `$$...$$`。s5_fuse 的 prompt 已明确要求这一对，
#    但模型仍会漂移成 LaTeX 原生的 `\[ ... \]`（行间）与 `\( ... \)`（行内）——
#    实测 2026-09-29 常微分方程第07讲混入 37 个 `\[ \]` 块 + 138 个 `\( \)` 行内，
#    在 Obsidian 里**全部退化成字面文本**（块公式尤其惨，整段无法挽救）。
#    故在落盘前加一道**确定性**归一化，不依赖模型自觉；代码围栏内一律不动
#    （那里的 `\(` 可能是正文示例/正则片段）。
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def _normalize_math_line(line: str) -> tuple[str, int]:
    """行内扫描：`\\[`/`\\]` → `$$`，`\\(`/`\\)` → `$`。

    双反斜杠前缀不动（`\\\\[2pt]` 是矩阵行距，不是分隔符）。
    """
    out, i, n, cnt = [], 0, len(line), 0
    while i < n:
        c = line[i]
        if c == "\\" and i + 1 < n and line[i + 1] in "[]()":
            prev = line[i - 1] if i > 0 else ""
            if prev != "\\":
                out.append("$$" if line[i + 1] in "[]" else "$")
                cnt += 1
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out), cnt


def normalize_math_delimiters(body: str) -> tuple[str, int]:
    """把 Obsidian 不认的数学分隔符归一化，返回 `(新正文, 替换处数)`。

    * `\\[ ... \\]` → `$$ ... $$`（原文多为独占行，转换后仍是独占行 → 真块公式）
    * `\\( ... \\)` → `$ ... $`（行内公式）
    * 代码围栏（``` / ~~~）内原样保留
    """
    lines: list[str] = []
    replaced, in_fence, marker = 0, False, ""
    for line in body.split("\n"):
        m = _FENCE_RE.match(line)
        if m:
            ch = m.group(1)[0]
            if not in_fence:
                in_fence, marker = True, ch
            elif ch == marker:
                in_fence, marker = False, ""
            lines.append(line)
            continue
        if in_fence:
            lines.append(line)
            continue
        new, cnt = _normalize_math_line(line)
        replaced += cnt
        lines.append(new)
    return "\n".join(lines), replaced


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    block = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    fm = {}
    for ln in block.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            fm[k.strip()] = v.strip().strip('"')
    return fm, body


def _render_note(fm: dict, body: str) -> str:
    lines = ["---"]
    for k in ("course", "lecture", "index", "date", "topic", "duration_min", "source", "video_id",
              "generated", "pipeline"):
        if fm.get(k) not in (None, ""):
            lines.append(f"{k}: {fm[k]}")
    tags = fm.get("tags") or []
    if tags:
        lines.append("tags:")
        lines += [f"  - {t}" for t in tags]
    lines.append("---")
    return "\n".join(lines) + "\n\n" + body.lstrip("\n")


# ---------------- 索引 ----------------
def rebuild_index(course: str, cfg: dict) -> Path:
    d = notes_dir(cfg) / course
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(d.glob("*.md")):
        if f.name.startswith("00_索引"):
            continue
        fm, body = _parse_frontmatter(f.read_text(encoding="utf-8"))
        rows.append({
            "index": int(fm.get("index") or 0),
            "date": fm.get("date", ""),
            "topic": fm.get("topic", f.stem),
            "link": f.stem,
            "chars": len(body.strip()),
            "file": f.name,
        })
    rows.sort(key=lambda r: (r["index"], r["date"]))
    lines = [
        f"# {course} · 讲义索引", "",
        f"> 由 lecture-pipeline 自动维护，最后更新：{datetime.now():%Y-%m-%d %H:%M}；共 {len(rows)} 讲。",
        "> 讲义按讲课老师原貌整理，未经教科书化改写；`[听不清]` 表示该处录音不可辨。", "",
        "| 讲次 | 日期 | 主题 | 讲义 | 字数 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['index']:02d} | {r['date']} | {r['topic']} | [[{r['link']}]] | {r['chars']} |")
    idx = d / "00_索引.md"
    idx.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return idx


# ---------------- 清理 ----------------
def _video_path(course: str, lecture: str, cfg: dict, wd: Path) -> Path | None:
    meta_p = wd / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        for k in ("video", "video_path", "path"):
            if meta.get(k):
                p = Path(meta[k])
                if p.exists():
                    return p
    idx_p = wd / "frames_index.json"
    if idx_p.exists():
        idx = json.loads(idx_p.read_text(encoding="utf-8"))
        p = Path((idx.get("video_meta") or {}).get("path", ""))
        if p.exists():
            return p
    for cand in (raw_dir(cfg) / course).glob(f"{lecture}*.mp4"):
        return cand
    return None


def cleanup_lecture(course: str, lecture: str, cfg: dict, note: Path, log, dry_run: bool = False) -> dict:
    cl = cfg["cleanup"]
    if not cl.get("enabled"):
        return {"skipped": "cleanup disabled"}
    nbytes = note.stat().st_size
    if cl.get("verify_note_exists") and nbytes < int(cl.get("min_note_bytes", 2000)):
        raise RuntimeError(f"安全阀拦截：讲义 {note} 仅 {nbytes} 字节 < "
                           f"{cl['min_note_bytes']}，拒绝清理（防误删素材）")
    wd = lecture_work_dir(course, lecture, cfg)
    keep = set(cfg["publish"].get("keep_artifacts") or [])
    targets: list[Path] = []
    if cl.get("delete_video"):
        v = _video_path(course, lecture, cfg, wd)
        if v:
            targets.append(v)
    if cl.get("delete_audio"):
        targets.append(wd / "audio.wav")
    if cl.get("delete_frames"):
        targets.append(wd / "frames")
    targets.append(wd / "_tmp_frames")

    plan, freed, removed = [], 0, []
    for t in targets:
        if not t.exists():
            continue
        size = sum(f.stat().st_size for f in t.rglob("*") if f.is_file()) if t.is_dir() else t.stat().st_size
        plan.append((t, size))
        freed += size
    for t, size in plan:
        log.info("%s %s (%s)", "[dry-run] 将删除" if dry_run else "删除", t, human_size(size))
        if not dry_run:
            if t.is_dir():
                shutil.rmtree(t, ignore_errors=True)
            else:
                t.unlink(missing_ok=True)
            removed.append(str(t))
    # 保留清单提示
    leftovers = [p.name for p in sorted(wd.iterdir())] if wd.exists() else []
    log.info("清理完成：释放 %s；保留 %s（keep_artifacts=%s）",
             human_size(freed), leftovers, sorted(keep))
    return {"freed_bytes": freed, "removed": removed, "kept": leftovers, "dry_run": dry_run}


# ---------------- 主流程 ----------------
def _infer_topic(body: str, index: int, lecture: str, meta: dict) -> str:
    """从融合稿推主题（自动化兜底）：H1（去掉「第NN讲」）→ 第一个二级标题 → meta.topic → lecture。

    [warn] 2026-09-13 新增：融合稿的 H1 就是「# 第NN讲」，原来据此推不出主题，会一路 fallback 到
    `lecture` 自身，产出「2026-09-08_第01讲_第01讲.md」这种废名字。现在退到二级标题
    （并去掉「（约 12:34）」这类时间尾巴）。**人工/agent 读稿后给 `--topic` 仍是首选**。
    """
    h1 = next((ln for ln in body.splitlines() if ln.startswith("# ")), "")
    t = re.sub(r"^#+\s*", "", h1).replace(f"第{index:02d}讲", "").strip(" _-·")
    if not t:
        h2 = next((ln for ln in body.splitlines() if ln.startswith("## ")), "")
        t = re.sub(r"^#+\s*", "", h2).strip()
        t = re.sub(r"[（(]约[^）)]*[）)]", "", t).strip()
    mt = str(meta.get("topic") or "").strip()
    if not t and mt and mt != lecture:
        t = mt
    return t or lecture


def publish_lecture(course: str, lecture: str, cfg: dict | None = None, date: str = "",
                    index: int | None = None, topic: str = "", do_cleanup: bool | None = None,
                    dry_run: bool = False, log=None) -> dict:
    cfg = cfg or load_config()
    from common import get_logger

    log = log or get_logger("s6_publish", cfg)
    vault = str(((cfg.get("paths") or {}).get("obsidian_vault") or "")).strip()
    if not vault:
        # 兑现文档承诺：paths.obsidian_vault 留空 = 只到草稿，不发布。
        # （旧版会把它当相对路径写进当前工作目录，留下脏文件。）
        raise RuntimeError(
            "paths.obsidian_vault 为空 → 不执行发布。\n"
            "  讲义草稿仍在 work/<课程>/<讲次>/lecture.md，可直接取用；\n"
            "  想发布就在 config/config.yaml 里填一个 Markdown 库目录。"
        )
    wd = lecture_work_dir(course, lecture, cfg)
    draft = wd / "lecture.md"
    if not draft.exists():
        raise FileNotFoundError(f"缺讲义草稿（先跑 s5_fuse）：{draft}")

    meta = {}
    meta_p = wd / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    date = date or meta.get("date") or datetime.now().strftime("%Y-%m-%d")
    index = int(index if index is not None else (meta.get("index") or 1))
    body = draft.read_text(encoding="utf-8")

    #  2026-09-19：**防假绿闸门** —— 草稿含「块生成失败」占位符时拒绝发布。
    if _PLACEHOLDER_RE.search(body):
        hits = _PLACEHOLDER_RE.findall(body)
        raise RuntimeError(
            f"讲义草稿含 {len(hits)} 处块生成失败占位符 {hits}，拒绝发布残缺讲义（防假绿）。"
            f"请先重跑融合：python src\\pipeline.py --course \"{course}\" --lecture \"{lecture}\" "
            f"--stages fuse --force")

    #  2026-10-12：落盘前归一化数学分隔符（Obsidian 只认 $ / $$，见函数 docstring）
    body, _math_fixed = normalize_math_delimiters(body)
    if _math_fixed:
        log.warning("归一化 %d 处数学分隔符（\\[ \\] \\( \\) → $$ / $；Obsidian 兼容）", _math_fixed)

    if not topic:
        topic = _infer_topic(body, index, lecture, meta)
    # H1 补上主题，读起来更像讲义（「# 第02讲」→「# 第02讲 · 平衡态」）
    body = re.sub(rf"^#\s*第\s*{index:02d}\s*讲\s*$", f"# 第{index:02d}讲 · {topic}",
                  body, count=1, flags=re.M)

    fm = {
        "course": course, "lecture": lecture, "index": index, "date": date, "topic": topic,
        "duration_min": round(float(meta.get("duration_min") or 0), 1) or "",
        "source": meta.get("source") or (load_config().get("publish") or {}).get("source") or "",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "pipeline": "lecture-pipeline",
        "tags": ["课程讲义", course],
    }
    # 2026-09-13 新增：带上回放 ID —— 讲次编号以教学网为锚，日后重排/追溯要靠它。
    if meta.get("video_id"):
        fm["video_id"] = meta["video_id"]
    fname = cfg["publish"]["filename_template"].format(date=date, index=index, topic=topic)
    out_dir = notes_dir(cfg) / course
    out_dir.mkdir(parents=True, exist_ok=True)
    note = out_dir / fname
    note.write_text(_render_note(fm, body), encoding="utf-8")
    log.info("讲义已落盘：%s（%s）", note, human_size(note.stat().st_size))

    idx_p = rebuild_index(course, cfg) if cfg["publish"].get("update_index") else None
    if idx_p:
        log.info("索引已更新：%s", idx_p)

    result = {"note": str(note), "bytes": note.stat().st_size, "index_file": str(idx_p) if idx_p else "",
              "topic": topic, "date": date, "index": index}
    want_cleanup = cfg["publish"].get("auto_cleanup") if do_cleanup is None else do_cleanup
    if want_cleanup:
        result["cleanup"] = cleanup_lecture(course, lecture, cfg, note, log, dry_run=dry_run)
    else:
        log.info("按参数跳过清理（视频与中间产物保留在 %s）", work_dir(cfg))
        result["cleanup"] = {"skipped": "requested"}
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 4：发布讲义 + 索引 + 清理")
    ap.add_argument("--course", required=True)
    ap.add_argument("--lecture", required=True)
    ap.add_argument("--date", default="")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--topic", default="")
    ap.add_argument("--keep", action="store_true", help="跳过清理（保留视频与中间产物）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将删除的清单，不真删")
    args = ap.parse_args()
    res = publish_lecture(args.course, args.lecture, date=args.date, index=args.index,
                          topic=args.topic, do_cleanup=(False if args.keep else None),
                          dry_run=args.dry_run)
    print("[s6_publish]", json.dumps(res, ensure_ascii=False)[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
