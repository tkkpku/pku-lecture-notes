"""保真度来源审计：讲义里出现的词，到底来自「语音(ASR)」「画面(VLM)」，还是两边都没有（疑外部知识）。

判据（你钦定硬约束③：不引入老师没讲的外部知识）：
  · ASR 有 / 画面有  → 合法（老师讲过，被任一通道捕获）
  · 两边都没有       → [warn] 可疑：可能老师讲/投影过但两通道都丢（可接受但要标注），
                       也可能是模型用自身知识补的（违反约束③）

 2026-09-17（会话 F）**变体感知匹配**：原版用严格字符串计数，于是把三类**正常现象**误报成
   「两通道都没有」，让真违规淹没在噪声里（实测普通物理第03讲：4 个可疑词里 3 个是假警报）：
     ① **ASR 同音字误识**：老师说的是「玻尔兹曼」，whisper 转成「沃尔兹曼」；讲义已按上下文改正
        → 严格匹配必然判「ASR 没有」。
     ② **缩写/长短形**：讲义写「等温线」，ASR 只出现「等温」；「范氏方程」vs「范德瓦尔斯」。
     ③ **人名/专名的异写**：「朱棣文」vs ASR 的「朱蒂文」。
   现在的匹配顺序：严格命中 → 别名/同音变体 → 去尾缀缩写（长度 ≥2）。命中的变体会**标出来**
   （如 `ASR 2≈沃尔兹曼`），便于人工复核，而不是悄悄放过。
   别名词表：内置 `_BUILTIN_ALIASES` + 可选外挂 `config/audit_aliases.yaml`（课程专属，格式
   `术语: [变体1, 变体2]`，两侧等价、顺序无关）。

用法：
  python scripts/audit_fidelity.py --course "普通物理 (II)(26-27学年第1学期)" --lecture "第03讲"
  python scripts/audit_fidelity.py --course X --lecture Y --terms 卷积 傅里叶 抽样定理
  python scripts/audit_fidelity.py --course X --lecture Y --strict   # 关掉宽松匹配，看原始严格结果
  （不给 --terms 时用内置通用词表；自测：python scripts/selftest_audit.py）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import PROJECT_ROOT, lecture_work_dir, load_config, notes_dir  # noqa: E402

DEFAULT_TERMS = [
    "孤立", "单相", "平衡态", "弛豫", "准静态", "相变", "局域平衡", "输运", "华氏", "摄氏",
    "温标", "理想气体", "状态参量", "状态方程", "范氏", "范德瓦尔斯", "相图", "临界点",
    "超临界", "三相点", "熵", "卡诺", "绝热", "等容", "玻色", "麦克斯韦",
    "卷积", "傅里叶", "拉普拉斯", "抽样定理", "奈奎斯特", "冲激", "系统函数", "频谱",
    "通解", "特解", "特征方程", "积分因子", "朗斯基", "齐次", "相平面", "稳定性",
    "应力", "张量", "面元", "刚体", "转动惯量", "拉格朗日",
]

# 同音/异写变体表（键 = 讲义里通常写的「正字」，值 = ASR/画面里可能出现的变体）。
# 两侧等价（写哪边都能匹配上），故一般只需登记一次。
_BUILTIN_ALIASES: dict[str, list[str]] = {
    # 物理·热学（2026-09-17 实测现行犯：ASR 把「玻尔兹曼」听成「沃尔兹曼」）
    "玻尔兹曼": ["波尔兹曼", "沃尔兹曼", "波耳兹曼", "玻耳兹曼", "波尔兹曼常数"],
    "范德瓦尔斯": ["范德华", "范德瓦尔", "范德瓦尔斯方程", "范氏方程"],
    "麦克斯韦": ["麦克斯威", "麦克斯维尔"],
    "朱棣文": ["朱蒂文", "朱迪文", "朱地文"],
    "光学黏胶": ["光学联交", "光学粘胶", "光学年交"],
    "弛豫": ["驰豫"],
    "焓": ["含", "焓变"],
    "熵": ["商"],
    # 电子/信号
    "傅里叶": ["傅立叶", "付里叶", "付立叶"],
    "拉普拉斯": ["拉普拉丝", "拉普拉斯变换"],
    "奈奎斯特": ["乃奎斯特", "奈奎斯特定理"],
    "抽样定理": ["采样定理", "抽样定律"],
    "冲激": ["冲击"],                      # 2026-09-17 实测：ASR 把「冲激信号」写成一律「冲击信号」
    "阶跃": ["节约", "阶越", "节约信号"],   # 实测现行犯：ASR 把「阶跃信号」听成「节约信号」
    "频谱": ["拼谱"],
    # 力学/数学
    "朗斯基": ["朗斯机", "朗斯基行列式", "朗斯基行列式"],
    "拉格朗日": ["拉格郎日", "拉格朗日量"],
    "转动惯量": ["转动贯量"],
    "张量": ["张亮"],
    # 2026-09-17 实测新增：理论力学第03讲 ASR 把「拉梅系数」听成「纳门系数」
    "拉梅系数": ["纳门系数", "拉美系数", "拉梅", "纳门"],
}

# 去尾缀后可作等价缩写匹配的后缀（仅当剩余长度 ≥2 才启用，防止「熵」这类单字乱匹配）
SUFFIXES = ("方程", "常数", "定理", "定律", "系数", "函数", "行列式", "过程", "运动",
            "曲线", "线", "图", "数", "量", "原理", "条件")


def load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    """内置别名 + 可选外挂 `config/audit_aliases.yaml`（课程专属扩展，缺省不存在）。"""
    out = {k: list(v) for k, v in _BUILTIN_ALIASES.items()}
    p = path or (PROJECT_ROOT / "config" / "audit_aliases.yaml")
    if p.exists():
        import yaml

        with open(p, encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                out.setdefault(str(k), []).extend(str(x) for x in v)
            elif isinstance(v, str):
                out.setdefault(str(k), []).append(v)
    return out


def variants(term: str, aliases: dict[str, list[str]] | None = None) -> list[str]:
    """术语的等价写法集合：别名组（双向）+ 去尾缀缩写。不含 term 自身。"""
    aliases = aliases or _BUILTIN_ALIASES
    out: set[str] = set()
    for k, vs in aliases.items():
        group = {str(k), *(str(x) for x in vs)}
        if term in group:
            out |= group
    out.discard(term)
    for suf in SUFFIXES:
        if term.endswith(suf) and len(term) - len(suf) >= 2:
            stem = term[: -len(suf)]
            out.add(stem)
            for k, vs in aliases.items():
                if stem == str(k) or stem in {str(x) for x in vs}:
                    out |= {str(k), *(str(x) for x in vs)} - {term}
    return sorted(out)


def count_term(term: str, asr: str, frames_text: str, note: str = "",
               aliases: dict[str, list[str]] | None = None, strict_only: bool = False) -> dict:
    """统计术语在三个文本里的出现情况（宽松模式下带变体回退）。返回 dict（供 CLI 与自测共用）。"""
    def chan(text: str, allow_variants: bool = True) -> tuple[int, str]:
        """统计 term 在 text 里的出现（宽松模式下带变体回退）。

         2026-09-25：**讲义通道必须严格匹配**（`allow_variants=False`）。
          实测假警报：待查词 `特征方程` 在讲义里严格 0 次，却因「去尾缀缩写」得到变体 `特征`，
          而讲义里有 `特征速率`（×3）→ 被算成「讲义出现 3 次」，再因 ASR/画面都查无而报
          「[warn] 两通道都没有」——明明讲义里压根没有这个词。
          语义上变体回退只解决「**讲义里写的词**在 ASR/画面里被念成什么」（如 `范氏方程`↔`范德瓦尔斯`），
          对讲义自身用回退等于凭空制造命中。
        """
        c = text.count(term)
        if c or strict_only or not allow_variants:
            return c, ""
        for v in variants(term, aliases):
            if (cv := text.count(v)):
                return cv, v
        return 0, ""

    n, _ = chan(note, allow_variants=False)
    a, a_via = chan(asr)
    f, f_via = chan(frames_text)
    if n == 0:
        src = "（讲义未出现）"
    elif a and f:
        src = "ASR+画面"
    elif a:
        src = "ASR"
    elif f:
        src = "画面"
    else:
        src = "[warn] 两通道都没有"
    if src.startswith(("ASR", "画面")) or "ASR+" in src:
        marks = [f"≈{v}" for v in (a_via, f_via) if v]
        if marks:
            src += f"（{'、'.join(marks)}）"
    return {"term": term, "note": n, "asr": a, "frames": f,
            "asr_via": a_via, "frames_via": f_via, "source": src}


def _find_note(cfg: dict, course: str, lecture: str) -> Path | None:
    d = notes_dir(cfg) / course
    if not d.is_dir():
        return None
    cands = [p for p in d.glob(f"*{lecture}*.md") if not p.name.startswith("00_")]
    return cands[0] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description="保真度来源审计（讲义术语 vs ASR / 画面）")
    ap.add_argument("--course", required=True)
    ap.add_argument("--lecture", required=True)
    ap.add_argument("--terms", nargs="*", default=None, help="自定义待查词表（缺省用内置通用词表）")
    ap.add_argument("--json", default="")
    ap.add_argument("--strict", action="store_true", help="关闭变体/缩写宽松匹配（看原始严格结果）")
    ap.add_argument("--aliases", default="", help="外挂别名表 YAML（缺省 config/audit_aliases.yaml）")
    args = ap.parse_args()

    cfg = load_config()
    wd = lecture_work_dir(args.course, args.lecture, cfg)
    asr_p, frames_p = wd / "asr.txt", wd / "frames_ocr.json"
    if not asr_p.exists() or not frames_p.exists():
        print(f"[!] 缺 {asr_p.name} 或 {frames_p.name}（该讲中间产物可能已被清理）", file=sys.stderr)
        return 2
    note = _find_note(cfg, args.course, args.lecture)
    if note is None:
        print(f"[!] 讲义库里找不到 {args.lecture} 的讲义", file=sys.stderr)
        return 2

    asr = asr_p.read_text(encoding="utf-8")
    frames = json.loads(frames_p.read_text(encoding="utf-8"))
    fb = "\n".join((r.get("vlm_text") or "") for r in frames.get("frames", []))
    text = note.read_text(encoding="utf-8")
    terms = args.terms or DEFAULT_TERMS
    aliases = load_aliases(Path(args.aliases) if args.aliases else None)

    print(f"讲义：{note.name}")
    print(f"源：ASR {len(asr):,} 字 ｜ 画面 {len(fb):,} 字 ｜ 待查词 {len(terms)} 个"
          f" ｜ 匹配={'严格' if args.strict else '宽松（含同音/缩写变体）'}\n")
    print(f"{'词':<12}{'讲义':>5}{'ASR':>6}{'画面':>6}   来源")
    print("-" * 64)
    suspicious: list[str] = []
    groups: dict[str, list[str]] = {"both": [], "asr_only": [], "frame_only": []}
    rows = []
    for t in terms:
        r = count_term(t, asr, fb, note=text, aliases=aliases, strict_only=args.strict)
        rows.append(r)
        if r["note"] == 0:
            continue
        if r["source"].startswith("ASR+"):
            groups["both"].append(t)
        elif r["source"].startswith("ASR"):
            groups["asr_only"].append(t)
        elif r["source"].startswith("画面"):
            groups["frame_only"].append(t)
        else:
            suspicious.append(t)
        print(f"{t:<12}{r['note']:>5}{r['asr']:>6}{r['frames']:>6}   {r['source']}")

    total = sum(len(v) for v in groups.values()) + len(suspicious)
    print(f"\n可溯源 {total - len(suspicious)}/{total} 词（ASR+画面 {len(groups['both'])}｜"
          f"仅 ASR {len(groups['asr_only'])}｜仅画面 {len(groups['frame_only'])}）")
    print(f"[warn] 无法溯源（两通道都没有）：{len(suspicious)} 个 —— {suspicious}")
    if suspicious and not args.strict:
        print("   （宽松匹配已开启仍判可疑 → 优先人工复核这几个词是否属约束③违规）")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"note": str(note), "suspicious": suspicious, "rows": rows,
             "groups": groups, "strict": bool(args.strict)},
            ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
