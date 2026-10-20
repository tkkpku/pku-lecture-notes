"""阶段 1：抓取（pku3b 封装）—— 列出回放、下载视频、落元数据。

[!] 凭据红线（你钦定，最高优先）：
  本模块**只检查凭据配置文件的「存在性」**，绝不打开、读取、回显其内容，
  也不读取 `.credentials.yaml` / `auth.json`。`pku3b init` 必须由你本人执行。
  `creds_status()` 只返回路径与 exists 布尔值。

pku3b 关键用法（HANDOFF §3.1 已实测 help）：
  pku3b init                           # 交互式初始化账密（你本人做）
  pku3b video list [--all-term]        # 列出回放
  pku3b video download <ID> -o <目录>  # 下载（支持断点续传）
  [warn] 下载需要 ffmpeg 在 PATH —— 由 common.ensure_tools_on_path() 负责

[warn] 待实测：`video list` 的输出格式（本机尚未 init，无法取样）。本模块的策略是
  「尽力解析 + 原文原样落盘」，解析不出结构化条目时**不报错**，把原文交给人工/后续解析，
  以免在格式变化时整条流水线卡死（卡点纪律：>20 分钟无进展就跳过）。
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
    cache_dir,
    ensure_tools_on_path,
    get_logger,
    load_config,
    raw_dir,
    run,
)

# 宽松匹配：形如 "1234567  课程名  2026-09-12  第3讲" 的行
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_COURSE_RE = re.compile(r"^\[(.+?)\]$")
# [!] 2026-09-17 修：原写死 `{16}`，而教学网**会吐 15 位 ID** —— 实测 2026-09-17 的 17 条里有 2 条
#    是 15 位（`87198cece5ffafe` 普物习题课 09-15 第02讲、`233caa428490a21` 马原 09-15），
#    它们被静默丢弃 → 讲次锚定表里根本没有这两条 → `list_pending.py` **永远报不出它们是待处理**
#    （即：这两讲永远出不了讲义，且没有任何报错）。回放 ID 是 hex 串、长度随上游截断而变，
#    故放到 `{8,32}`：既容得下变长，又不会把行尾普通单词误当 ID。
#    回归自测：`python scripts/selftest_parser.py`（含「真实 list 文件里每个 • 行都必须解析成功」）。
_ITEM_RE = re.compile(r"^[•·]\s*(.+?)\s+([0-9a-fA-F]{8,32})$")
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_SECTION_RE = re.compile(r"第([\d\-]+)节")


def parse_video_list(text: str) -> list[dict]:
    """解析 `pku3b video list` 的真实输出（2026-09-13 实测格式，带 ANSI 颜色码）。

    实测格式（剥离 ANSI 后）：
        > 课程回放 <
        [信号与系统(26-27学年第1学期)]
        • 2026-09-08第3-4节 (2026-09-08 10:10:00) b6a77a5c719a5aa4
        [嵌入式系统(26-27学年第1学期)]
        • 2026-09-08第10-11节 (2026-09-08 18:40:00) bdf46a9531050839

    → 课程行用 `[...]`，条目行以 `•` 开头、行尾是 hex 回放 ID（实测长度 15~16 位，见 `_ITEM_RE` 注释）。
    """
    clean = _ANSI_RE.sub("", text or "")
    items: list[dict] = []
    course = ""
    for ln in clean.splitlines():
        s = ln.strip()
        if not s:
            continue
        if m := _COURSE_RE.match(s):
            course = m.group(1).strip()
            continue
        if m := _ITEM_RE.match(s):
            title, vid = m.group(1).strip(), m.group(2)
            dm = _DATE_RE.search(title)
            sm = _SECTION_RE.search(title)
            items.append({
                "id": vid,
                "course": course,
                "title": title,
                "date": dm.group(1) if dm else "",
                "sections": sm.group(1) if sm else "",
            })
    return items


# ---------------- 讲次锚定（2026-09-13 新增：**以教学网为准**）----------------
# 由来：回放标题只带「日期 + 节次」，没有讲次号。曾被按「第几次跑流水线」编成第 01 讲，
#      结果 09-10 那节课拿了 01、比它更早的 09-07 反而要靠后缀区分 —— 顺序是错的。
# 规则（你 2026-09-13 钦定）：**同一门课内，按教学网回放的时间先后升序编号，最早的一节 = 第01讲。**
#      教学网列表本身是日期降序，故这里反转；同一天多节次时用起始节次再排一次。
def _slot_start(sections: str) -> int:
    m = re.match(r"\s*(\d+)", sections or "")
    return int(m.group(1)) if m else 99


def build_order(items: list[dict]) -> list[dict]:
    """按教学网锚定讲次序号，返回 [{course, index, id, date, sections, title}]。"""
    by_course: dict[str, list[dict]] = {}
    for it in items:
        by_course.setdefault(it.get("course", ""), []).append(it)
    out: list[dict] = []
    for course in sorted(by_course):
        rows = sorted(by_course[course],
                      key=lambda x: (x.get("date", ""), _slot_start(x.get("sections", ""))))
        for i, it in enumerate(rows, 1):
            out.append({"course": course, "index": i, "id": it.get("id", ""),
                        "date": it.get("date", ""), "sections": it.get("sections", ""),
                        "title": it.get("title", "")})
    return out


def order_map_path(cfg: dict) -> Path:
    return raw_dir(cfg) / "_order_map.json"


def write_order_map(items: list[dict], cfg: dict, log=None) -> dict:
    """把锚定表落盘到 `raw/_order_map.json`（供 pipeline / 工具查询与审计）。"""
    rows = build_order(items)
    courses: dict[str, list[dict]] = {}
    for r in rows:
        courses.setdefault(r["course"], []).append(
            {k: r[k] for k in ("index", "id", "date", "sections", "title")})
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "anchor": "pku3b video list（教学网回放）；同一课程内按时间升序 → 第01讲起",
        "courses": courses,
    }
    p = order_map_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if log:
        log.info("讲次锚定表已更新：%s（%d 门课 / %d 条回放）", p, len(courses), len(rows))
    return payload


def anchor_index(course: str, video_id: str, cfg: dict | None = None,
                 refresh: bool = False, log=None) -> int | None:
    """查「某条回放按教学网应编第几讲」。refresh=True 时先跑一次 video list 刷新锚定表。

    返回 None 表示查不到（列表里没有这条 / 列表取不到）——调用方应保守沿用调用者给的编号并告警。
    """
    cfg = cfg or load_config()
    if refresh or not order_map_path(cfg).exists():
        try:
            lst = list_videos(cfg, log=log)
            if lst.get("rc") == 0 and lst.get("items"):
                write_order_map(lst["items"], cfg, log)
        except Exception as e:  # noqa: BLE001
            if log:
                log.warning("刷新讲次锚定表失败（沿用旧表或放弃锚定）：%s: %s", type(e).__name__, e)
    p = order_map_path(cfg)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    for row in (data.get("courses") or {}).get(course, []):
        if row.get("id") == video_id:
            return int(row.get("index") or 0) or None
    return None


def creds_status(cfg: dict, probe: bool = False, log=None) -> dict:
    """凭据状态。**只查文件存在性，绝不读取内容**（凭据红线）。

    [warn] 2026-09-13 实测修正：**文件不存在 ≠ 未初始化**。本机 `pku3b init` 已成功
    （`video list` 能正常列出 7 门课），但 pku3b 目录下没有任何配置文件 —— 凭据很可能
    存进了系统钥匙串（Windows Credential Manager）。所以不要再拿文件存在性当门禁，
    真正的可用性判据是 `video list` 是否成功（`probe=True` 时由调用方自行判断）。
    """
    p = Path(cfg["fetch"]["pku3b_config"])
    st = {
        "pku3b_config": str(p),
        "file_exists": p.exists(),
        "note": "凭据可能存于系统钥匙串；文件不存在不代表未初始化，实际可用性以 `pku3b video list` 为准。",
    }
    if probe:
        res = list_videos(cfg, log=log)
        st["list_rc"] = res.get("rc")
        st["list_items"] = len(res.get("items", []))
        st["usable"] = res.get("rc") == 0 and bool(res.get("items"))
    return st


def pku3b_path(cfg: dict) -> str:
    exe = cfg["paths"]["pku3b_exe"]
    if Path(exe).exists():
        return exe
    found = shutil.which("pku3b")
    if not found:
        raise FileNotFoundError(f"找不到 pku3b：{exe}")
    return found


def run_pku3b(args: list[str], cfg: dict, log=None, timeout: float = 3600, dry_run: bool = False):
    """跑 pku3b。

    [warn] 2026-09-13 实测修正：`--config` / `--cache-dir` 是**子命令级** option
    （原文 `Usage: pku3b.exe video download [OPTIONS] <ID>`），必须插在**子命令之后、位置参数之前**，
    不能放在 `pku3b` 与子命令之间。（此前写成 `pku3b --config X video list`，因本机凭据在系统钥匙串、
    该分支从未执行而没暴露。）
    """
    from common import cache_dir

    ensure_tools_on_path(cfg)  # pku3b 下载回放需要 ffmpeg 在 PATH
    exe = pku3b_path(cfg)
    args = list(args)
    extra: list[str] = []
    # 凭据配置（与代码仓隔离；只传路径，不读内容）
    cfgfile = Path(cfg["fetch"]["pku3b_config"])
    if cfgfile.exists():
        extra += ["--config", str(cfgfile)]
    # HLS 分片缓存重定向到 D 盘（C 盘只剩 33GB）；断点续传也依赖这个目录
    cache = cache_dir(cfg) / "pku3b"
    if not dry_run:
        cache.mkdir(parents=True, exist_ok=True)
    extra += ["--cache-dir", str(cache)]
    cmd = [exe] + args[:2] + extra + args[2:]
    if dry_run:
        print("[dry-run] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return None
    if log:
        log.info("pku3b: %s", " ".join(args))
    return run(cmd, check=False, timeout=timeout, log=log)


def list_videos(cfg: dict, all_term: bool = False, log=None, dry_run: bool = False) -> dict:
    """列出课程回放。返回 {raw, items, raw_file}。"""
    cfg = cfg or load_config()
    log = log or get_logger("s1_fetch", cfg)
    args = ["video", "list"] + (["--all-term"] if all_term else [])
    cp = run_pku3b(args, cfg, log, timeout=300, dry_run=dry_run)
    if dry_run:
        return {"raw": "", "items": [], "raw_file": ""}
    text = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
    raw_file = raw_dir(cfg) / "_pku3b_video_list.txt"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text(f"# {datetime.now():%Y-%m-%d %H:%M:%S}\n# args: {args}\n# rc={cp.returncode}\n\n{text}",
                        encoding="utf-8")
    items = parse_video_list(text)
    log.info("video list：rc=%s，解析出 %d 条 / %d 门课（原文落盘 %s）",
             cp.returncode, len(items), len({i["course"] for i in items}), raw_file)
    if cp.returncode != 0:
        log.error("video list 失败 rc=%s：%s", cp.returncode, (cp.stderr or "")[-400:])
    elif not items:
        log.warning("rc=0 但未解析出条目——可能是输出格式变化，原文已保存：%s", raw_file)
    return {"raw": text, "items": items, "raw_file": str(raw_file), "rc": cp.returncode}


def download_video(video_id: str, out_dir: str | Path, cfg: dict | None = None,
                   all_term: bool = False, log=None, dry_run: bool = False) -> dict:
    """下载单个回放。out_dir 默认 raw/<课程>；pku3b 自带断点续传。"""
    cfg = cfg or load_config()
    log = log or get_logger("s1_fetch", cfg)
    out_dir = Path(out_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    #  2026-09-25（bug T 根治）：下载前给目录拍快照，用于**精确识别本次新增的 mp4**。
    before = {p.resolve() for p in out_dir.glob("*.mp4")} if not dry_run else set()
    args = ["video", "download", str(video_id)] + (["--all-term"] if all_term else []) + \
        ["-o", str(out_dir)]
    t_started = datetime.now().timestamp()   # 供 HLS 缓存清理划定「本次下载时间窗」
    cp = run_pku3b(args, cfg, log, timeout=7200, dry_run=dry_run)
    if dry_run:
        return {"video_id": video_id, "out_dir": str(out_dir), "dry_run": True}
    mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    # [!] 原实现只取「mtime 最新的 mp4」，在**同一门课多讲共存**时会挑错 ——
    #    实测现行犯（2026-09-25）：信号与系统目录里同时躺着 09-18 与 09-22 两讲，
    #    并行下载下 09-18 那个的 mtime 反而更晚 → 第 04讲(09-22) 的 `newest_mp4` 记成了
    #    **第 03讲(09-18) 的文件** → meta.json 记错 → 另一个进程跑 extract 时按 meta.json
    #    取视频 → **抽的是讲3 的音频 → 讲5 的讲义内容整篇张冠李戴**（两份 ASR 前 4000 字
    #    相似度 1.000 才被发现），且 publish 按错路径找不到文件 → 漏删真实 mp4 826 MB。
    #    改为：**优先取本次新出现的文件**；若本次没有新增（如重下同名讲次），才回退到 mtime。
    fresh = [p for p in mp4s if p.resolve() not in before]
    newest = fresh[0] if fresh else (mp4s[0] if mp4s else None)
    if not fresh and mp4s and log:
        log.warning("本次未检测到新增 mp4（可能是重下同名文件）；回退按 mtime 取：%s", mp4s[0].name)
    res = {
        "video_id": video_id, "out_dir": str(out_dir), "rc": cp.returncode,
        "newest_mp4": str(newest) if newest else "",
        "stderr_tail": (cp.stderr or "")[-800:],
    }
    if cp.returncode != 0:
        log.error("下载失败 rc=%s：%s", cp.returncode, res["stderr_tail"])
    else:
        log.info("下载完成：%s", res["newest_mp4"])
        # [warn] 2026-09-13 实测新增（bug F）：pku3b 在 --cache-dir 下按回放 ID 存 HLS 分片，
        #    实测每讲残留 1.4~1.9 GB（本讲 1.88 GB / 719 文件）且从不清理。
        #    下载确认成功后清掉本 ID 的缓存（安全阀见 clean_hls_cache）。
        res["cache_freed_bytes"] = clean_hls_cache(video_id, res["newest_mp4"], cfg, log,
                                                   started_at=t_started)
    return res


def clean_hls_cache(video_id: str, mp4: str, cfg: dict, log=None,
                    started_at: float | None = None) -> int:
    """删掉本次下载留下的 HLS 分片缓存，返回释放字节数（0 = 跳过）。

    [warn] 2026-09-13 二次修正（bug K）：pku3b 的缓存有**两处** ——
      ① `cache/pku3b/video_download/<回放ID>/`（按 ID 分目录）
      ② **`cache/pku3b/with_cache_bytes-<hash>` 平铺文件**（按内容哈希，每个 ~1 MB，数量上万）
    初版只清了 ①，结果本项目每讲仍残留 ~1.5 GB 在 ②（实测一天下来堆到 7 GB）。
    现在 ② 按「**本次下载时间窗**」清（mtime ≥ 下载开始前 120 秒），避免误删别人正在下的缓存。

     2026-09-17（你指出"清不干净"）**第三次修正**：pku3b 还有第三族残留 ——
    **`with_cache-<hash>` 文件**（用**连字符**，每个 ~1 KB；实测每次运行残留 21 个）。
    旧代码 glob 的是 `with_cache_*`（**下划线**），**整族被静默跳过**，于是它们只增不减。
    现在改成前缀 `with_cache_` + `with_cache-` 双前缀匹配，**文件 unlink、目录 rmtree** 都处理，
    并统一受「本次下载时间窗」约束（目录取内部最大 mtime）。

    安全阀：只有「mp4 已落盘且 > 50 MB」才动手 —— 断流/失败时缓存正是续传的基础。
    """
    p = Path(mp4) if mp4 else None
    if not p or not p.exists() or p.stat().st_size < 50 * 1024 * 1024:
        if log:
            log.warning("跳过 HLS 缓存清理（mp4 未落盘或体积异常）：%s", mp4)
        return 0
    base = cache_dir(cfg) / "pku3b"
    freed = 0

    d = base / "video_download" / str(video_id)
    if d.is_dir():
        freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d, ignore_errors=True)

    if started_at:
        cutoff = started_at - 120

        def _newest_mtime(path: Path) -> float:
            try:
                if path.is_file():
                    return path.stat().st_mtime
                # 目录：取内部最"新"文件的 mtime（空目录退化为目录自身）
                mtimes = [f.stat().st_mtime for f in path.rglob("*") if f.is_file()]
                return max(mtimes) if mtimes else path.stat().st_mtime
            except OSError:
                return 0.0

        for e in base.glob("with_cache*"):
            try:
                if _newest_mtime(e) < cutoff:
                    continue
                sz = (sum(f.stat().st_size for f in e.rglob("*") if f.is_file())
                      if e.is_dir() else e.stat().st_size)
                if e.is_dir():
                    shutil.rmtree(e, ignore_errors=True)
                else:
                    e.unlink(missing_ok=True)
                freed += sz
            except OSError:
                continue

    if log:
        log.info("清理 HLS 分片缓存：%s（释放 %.1f MB）", base, freed / 1024 / 1024)
    return freed


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 1：pku3b 抓取")
    ap.add_argument("--list", action="store_true", help="列出回放")
    ap.add_argument("--all-term", action="store_true")
    ap.add_argument("--download", default="", help="下载指定 ID")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    do_probe = bool(args.list or args.download)
    st = creds_status(cfg, probe=do_probe)
    print("[creds]", json.dumps(st, ensure_ascii=False))
    if do_probe and st.get("usable") is False:
        print("[!] pku3b 当前不可用（video list 失败或没有回放）——若提示未登录，"
              "请你本人执行：pku3b init（我不代做、不读取凭据）")
        if args.list:
            return 3
    if args.list:
        res = list_videos(cfg, args.all_term, dry_run=args.dry_run)
        print(json.dumps({"items": res["items"][:20], "count": len(res["items"])}, ensure_ascii=False))
        return 0
    if args.download:
        out = Path(args.out_dir) if args.out_dir else raw_dir(cfg)
        print(json.dumps(download_video(args.download, out, cfg, args.all_term, dry_run=args.dry_run),
                         ensure_ascii=False))
        return 0
    print("用法：--list 或 --download <ID> [--out-dir DIR]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
