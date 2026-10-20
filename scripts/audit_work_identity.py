r"""work 目录身份普查（2026-10-12 新增，配合 HANDOFF §24 的身份闸门）。

**为什么需要**：work 目录按 (课程, 讲次) 命名，而教学网只放最近若干条回放 —— 旧条目被挤出
列表后**讲次编号整体前移**，新回放会继承旧回放的目录。若不清查，`asr`/`ocr` 会读到**上一条
回放留下的产物**并判定「已完成」跳过，各阶段全报 OK，实际产出废讲义（详见 HANDOFF §24.3）。

判据：`work\<课程>\<讲次>\meta.json` 的 `video_id` 是否等于当前锚定表同 (课程, 讲次) 的 `id`。

用法：
  python scripts\\audit_work_identity.py            # 人读报告
  python scripts\\audit_work_identity.py --json out.json
退出码：0 无错配；1 有[!]错配（供 CI / 收尾自检用）。

[warn] 名字含 `__stale_` 的目录是**身份闸门隔离出的存档**，一律跳过（它们的 meta 故意保留旧 ID）。
[warn] `⚪ 孤儿引用` 不计入退出码：该讲次已不在锚定表（教学网删了这条回放），属无害残留。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import load_config, work_dir  # noqa: E402

STALE_MARK = "__stale_"


def load_current(cfg: dict) -> dict[tuple[str, str], str]:
    """当前锚定表 → {(课程, 第NN讲): 回放ID}。"""
    from common import data_root

    p = data_root(cfg) / "raw" / "_order_map.json"
    if not p.exists():
        raise SystemExit(f"取不到锚定表：{p}（先跑 scripts\\list_pending.py --refresh）")
    om = json.loads(p.read_text(encoding="utf-8-sig"))
    cur: dict[tuple[str, str], str] = {}
    for course, rows in (om.get("courses") or {}).items():
        for r in rows or []:
            cur[(course, f"第{int(r['index']):02d}讲")] = str(r.get("id") or "")
    return cur


def scan(cfg: dict) -> dict:
    cur = load_current(cfg)
    root = work_dir(cfg)
    consistent, mismatch, orphan, skipped_stale = [], [], [], 0
    if not root.exists():
        return {"anchor": len(cur), "consistent": consistent, "mismatch": mismatch,
                "orphan": orphan, "stale_skipped": 0, "root": str(root)}
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ldir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            if STALE_MARK in ldir.name:
                skipped_stale += 1
                continue
            mp = ldir / "meta.json"
            if not mp.exists():
                continue
            try:
                m = json.loads(mp.read_text(encoding="utf-8-sig"))
            except Exception as e:  # noqa: BLE001
                orphan.append({"course": cdir.name, "lecture": ldir.name,
                               "video_id": "", "date": "", "why": f"meta.json 读不出（{type(e).__name__}）"})
                continue
            key = (m.get("course"), m.get("lecture"))
            got = str(m.get("video_id") or "")
            row = {"course": key[0], "lecture": key[1], "video_id": got, "date": m.get("date")}
            want = cur.get(key)
            if want is None:
                orphan.append({**row, "why": "该讲次已不在锚定表"})
            elif got != want:
                mismatch.append({**row, "anchor_video_id": want})
            else:
                consistent.append(row)
    return {"anchor": len(cur), "root": str(root), "consistent": consistent,
            "mismatch": mismatch, "orphan": orphan, "stale_skipped": skipped_stale}


def main() -> int:
    ap = argparse.ArgumentParser(description="work 目录身份普查（讲次编号前移事故的自检）")
    ap.add_argument("--json", default="", help="结果写到 JSON")
    args = ap.parse_args()

    cfg = load_config()
    r = scan(cfg)
    print(f"锚定表讲次 {r['anchor']} 条 | work 根 {r['root']}")
    print(f"[ok] 身份一致 {len(r['consistent'])} 个 | [!] 错配 {len(r['mismatch'])} 个 | "
          f"⚪ 孤儿 {len(r['orphan'])} 个 | ⏭ 跳过隔离存档 {r['stale_skipped']} 个")
    if r["mismatch"]:
        print("\n[!] 身份错配（该讲会被身份闸门自动隔离；请确认这是你要跑的那一条回放）：")
        for m in r["mismatch"]:
            print(f"   {m['course']} / {m['lecture']}: work 里的 {m['video_id']}（{m['date']}）"
                  f" ≠ 锚定表 {m['anchor_video_id']}")
    if r["orphan"]:
        print("\n⚪ 孤儿引用（讲次已不在锚定表，无害残留）：")
        for o in r["orphan"]:
            print(f"   {o['course']} / {o['lecture']}: {o['video_id'] or '（空）'} —— {o['why']}")
    if not r["mismatch"]:
        print("\n=== 身份普查：[ok] 全绿（无错配）===")
    if args.json:
        Path(args.json).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"（已写 {args.json}）")
    return 1 if r["mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
