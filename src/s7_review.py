"""阶段 7：**章节收口判定 + 阶段性复习文档自动生成**（2026-10-15 你钦定）。

## 为什么是 LLM 而不是脚本

你原话：
> 「判断章节是否结束**不应该用脚本来做**，这种模糊的处理应当调用 llm，在每个文档成文后
>   发一次请求让 llm 判断本节课是否为某一章节做了结束，若成则再去用别的请求写复习文档」

[ok] 我的教训：同日先写了一版**标题启发式脚本**，跑出来把「收尾与下课」「作业安排」
   全判成收口，一屏假警报。**假警报一多就没人看** —— 而这套机制当初正是**因为没人看而被整个漏掉**。
   模糊语义判断交给 LLM，确定性工作（拼装/归一/落盘/自检）留给代码。

## 全自动流程（命中即写，不再询问；2026-10-15 你确认）

```
① judge      1 次调用，输入 = 本讲讲义全文  → JSON {is_chapter_end, chapter_label, confidence, evidence}
        ↓ 命中
② extract    N 次调用（本章每讲一次）      → 逐讲结构化提取
③ outline    1 次调用                     → JSON 大纲（节号/标题/要点/目标字数）
④ sections   M 次调用                     → 正文分段撰写
⑤ appendix   1 次调用                     → 公式速查表 + 强调汇总 + 备考清单
⑥ 拼装 → 数学分隔符归一 → 自检 → 落 Vault `00_第N章复习_主题.md`
```

## 状态与范围

`work/<课程>/_review_state.json` 记账「已出复习到第几讲」。
首次运行没有状态文件时，**从已有 `00_第*章复习_*.md` 里正则扫 `第NN讲` 取最大值**做自举
（如常微分方程第一章复习提到第01~06讲 → 下一章从第07讲起）。

## 纪律（硬性）

- **以老师课堂讲义为核心依据**；不引入讲义之外的知识。
- 公式只用 `$...$` / `$$...$$`（Obsidian 只认这两组；`\\(…\\)` 不渲染 → 写出前必须归一）。
- `[听不清]` 处如需恢复**必须留痕**，不许静默补全。
- 体量：≥1× 单讲讲义，目标 2~3×。
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
    load_config,
    normalize_math_delims,
    notes_dir,
)

STAGE = "review"
LEC_RE = re.compile(r"第(\d+)讲")
CN_NUM = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
          "十一", "十二", "十三", "十四", "十五", "十六"]

# ---------------------------------------------------------------- 提示词

JUDGE_PROMPT = """你在为一门大学课程的随堂讲义做「**章节收口判定**」。

下面是《{course}》{lecture}（{date}）的完整讲义。请判断：**这一讲是否为一个章节 / 专题画上了句号。**

判据（命中任一即为"是"）：
1. 老师在本讲中**明确宣布**本章/本专题结束，或宣布进入下一章；
2. 本讲是**复习课 / 总结课 / 习题课**，且覆盖**整章或整个专题**而非单节；
3. 本讲末尾在做**整章 / 整专题的梳理**，且前面若干讲同属该章/该专题。

[warn] **不算收口**：只是某一小节的"小结"；只是本讲内容的常规收尾（下课、作业安排、通知、考勤）；
只是复盘上一讲内容。

**分级要求**（重要）：本课程老师会在**每 1~2 讲**结束一个知识板块，请如实分级 ——
- `"章"`：课程大纲级别的一整章（较大）；
- `"专题"`：一个自成一体的知识专题（中等，如「分子动理论」「输运过程」）；
- `"小节"`：单一小节 / 单个概念（**不算收口**）；
- `"无"`：本讲没有结束任何板块。

[!] **`chapter_label` 必须是"知识内容的名字"**，例如「热学」「静力学」「初等积分法」「输运过程」「SolidWorks 建模」。
**严禁**使用「第X讲」「第X节」「这节课」「上一讲」这类**指代性**说法 ——
[warn] 本课程讲义内部的讲次编号可能与教学网编号**不一致**（实测差 1 讲），照抄内部编号会得到错误的章名。
若老师口称「第 X 讲」，你要**换成内容名**。

**只输出 JSON**（不要解释文字、不要 markdown 代码块）：
{{"is_chapter_end": true 或 false,
 "granularity": "章" | "专题" | "小节" | "无",
 "chapter_label": "内容名（不含任何『第X讲/第X节』字样）；未命中填空字符串",
 "confidence": 0 到 1 之间的小数,
 "evidence": "讲义里的依据，引用原文片段，不超过 100 字"}}

讲义全文：
---
{note}
---
"""

EXTRACT_PROMPT = """你在为《{course}》的《{chapter_label}》复习文档做**材料准备**。

下面是《{course}》{lecture} 的随堂讲义全文。请做**结构化提取**。

[!] 铁律：
- **只提取讲义里真实写了的内容**。不得补充讲义之外的知识，不得自行推导新结论。
- 公式用 LaTeX 原样保留（行内 `$...$`、行间 `$$...$$`）；**不要**改成 `\\(…\\)`、`\\[…\\]`。
- 讲义里标了 `[听不清]` 的地方，**原样标出**，绝不许猜测填补。
- 老师标了「反复强调」的条目必须保留。

按下列小标题输出 Markdown（不要总标题、不要寒暄、不要代码围栏）：

### A. 本讲知识点清单
### B. 公式与定义（写全）
### C. 例题（题面 + 解法要点；老师课堂原题，条件写全）
### D. 老师强调点 / 易错点
### E. 与其它讲的衔接（讲义里提到的前后关联）

讲义全文：
---
{note}
---
"""

OUTLINE_PROMPT = """你在为《{course}》编写《{chapter_label}》的**阶段性复习文档**（面向复习与备考）。

**本篇覆盖的讲次**：{scope}
（[warn] 上面的「{chapter_label}」只是**收口那一讲**的名字，未必等于整章的名字 —— 见下方命名规则。）

下面是该章各讲的结构化提取材料（按讲次顺序）。

[!] **命名规则（`doc_title`）**：`doc_title` 必须概括**上面列出的全部讲次**，而不是只概括最后一讲。
例：覆盖「第01~08讲」且内容为 状态参量 / 分子动理论 / 统计分布 / 输运过程 时，
应命名「热学篇」或「热学基础」，**不能**只写「输运过程」（那只覆盖最后一讲）。

请先给出**文档大纲**。要求：
- 结构面向"**复习/备考**"：把同类知识**聚合**，**不要**按讲次写流水账。
- 必须覆盖这六块内容（可把①与知识板块合并编排）：
  ① 判型决策表（拿到题目"该怎么办"的分支表）
  ② 逐类知识点 + 公式 + 陷阱
  ③ 例题精讲（老师课堂原题）
  ④ 老师「反复强调」条目汇总
  ⑤ 备考 / 自测清单
  ⑥ 公式速查表
- 全文**目标总字数约 {target_chars} 字符**，请把字数分配到各节。

[!] **JSON 输出纪律**（违反会导致解析失败）：
- `doc_title`、`title`、`brief` 里**一律不要写 LaTeX 公式**——用中文名称代替
  （写「行列式表示」而不是美元符号包起来的 Delta；写「莱布尼茨公式」而不是 frac 分式）。
- 需要强调符号时用文字（如「角速度 ω」），**不要**用美元符号包裹。
- 字符串里**不要出现反斜杠**：JSON 不认识 LaTeX 的反斜杠转义。
- 正文里的公式留到后面撰写阶段再写，这里只出大纲。

> [warn] 本模板会被 `.format()` 处理：**除占位符外，任何字面花括号都必须双写**。
> 2026-10-17 实测现行犯：在提示词里写 LaTeX 例子 `frac{{d}}{{dx}}` 的原始形态（单花括号）
> 被当成占位符 → `KeyError: 'd'`，连炸两次（所以这一行改成了纯文字描述）。

**只输出 JSON**（不要解释、不要代码围栏）：
{{"doc_title": "复习文档标题（不含「复习」二字之外的装饰，如「初等积分法」）",
 "sections": [{{"no": 1, "title": "节标题", "brief": ["该节要写什么，3~5 条"], "target_chars": 6000}}, ...]}}

材料：
---
{extractions}
---
"""

SECTION_PROMPT = """你在撰写《{course}》{chapter_label}复习文档的**第 {i}/{n} 批**内容。

【全章大纲】（把握上下文用，**只写本批指定的节**）
{outline}

【本批要写的节】
{section_spec}

【原始材料】（提取自老师课堂讲义，**这是唯一的知识来源**）
{extractions}

[!] 铁律（违反即失败）：
1. **以老师课堂讲义为核心依据**，不得引入老师没讲过的外部知识、不得自行发挥。
2. 公式只用 LaTeX 的 `$...$`（行内）与 `$$...$$`（行间）。**禁止** `\\(…\\)`、`\\[…\\]`。
3. 讲义标了 `[听不清]` 而此处需要恢复的，**必须留痕**（如「（第 X 讲 t≈MM:SS 转写缺口，此处按上下文恢复）」）。
4. 不得出现「我认为」「这里补充」「额外说明」之类越界措辞。
5. 本批目标字数约 **{target}** 字符。
6. **直接输出正文 Markdown**（从 `## ` 级标题开始），不要寒暄、不要代码围栏、不要重复大纲。
"""

APPENDIX_PROMPT = """你在为《{course}》{chapter_label}复习文档补**最后三块**。

材料：各讲结构化提取
---
{extractions}
---

请依次输出（`## ` 级标题）：

## 公式速查表
（Markdown 表格：`公式 | 含义 | 出现讲次`，覆盖全章主要公式）

## 老师反复强调汇总
（条目列表，每条注明出自哪一讲）

## 备考自测清单
（可勾选列表 `- [ ] …`，覆盖全章，约 25~40 条）

[!] 铁律同上：**不引入讲义之外的知识**；公式只用 `$...$` / `$$...$$`。
直接输出正文 Markdown，不要寒暄、不要代码围栏。
"""


# ---------------------------------------------------------------- 工具

# 指代性说法（讲次/节次）——章名里出现就是脏值。
# 由来（2026-10-15 实测）：讲义内部讲次与教学网锚定编号**差 1 讲**（锚定第09讲、内容自称「第八讲」），
# 判定器照抄内部编号 → label 变成 '第七讲：输运过程'，直接污染文件名。标签必须是**内容名**。
_LEC_REF_RE = re.compile(r"第\s*[0-9一二三四五六七八九十两]+\s*[讲节次课]")
_LEAD_REF_RE = re.compile(r"^(?:第\s*[0-9一二三四五六七八九十两]+\s*[讲节次课])\s*[：:、,，]?\s*")


def _sanitize_label(label: str) -> str:
    """把章名里的「第X讲/第X节」指代洗掉；洗空了就返回空串（由调用方兜底）。"""
    t = (label or "").strip()
    if not t:
        return ""
    t = _LEAD_REF_RE.sub("", t)          # 去掉开头的「第七讲：」
    t = _LEC_REF_RE.sub("", t)           # 去掉句中残留的讲次指代
    t = re.sub(r"^[：:、,，\s]+|[：:、,，\s]+$", "", t)
    return t.strip()


# 非法 JSON 转义：反斜杠后面跟的字符不是 json 合法转义（" \ / b f n r t u）
# [!] 2026-10-17 实测现行犯：大纲步的 JSON 里 LLM 写了 LaTeX（$\Delta(t,s)$、\frac），
#    `json.loads` 直接 `Invalid \escape: line 1 column 599` 炸掉整篇复习文档（理力静力学那篇）。
#    更阴的是 `\f` `\n` `\t` **属于**合法转义 → 不报错但被静默转成控制字符，**悄悄毁掉公式**。
#    双管齐下：① 源头用 OUTLINE_PROMPT 禁止在标题/要点里写 LaTeX；② 这里兜底修复非法转义。
_BAD_ESC_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _repair_json_escapes(s: str) -> str:
    """把非法 JSON 转义（如 `\\frac` 里的 `\\f` 之外的 `\\m`/`\\s`）补成 `\\\\x`，保住字面反斜杠。"""
    return _BAD_ESC_RE.sub(lambda m: "\\\\", s)


def _parse_json_obj(text: str) -> dict:
    """从 LLM 输出里抠出第一个 JSON 对象（容忍代码围栏、前后废话、非法 LaTeX 转义）。"""
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError(f"LLM 输出里找不到 JSON 对象：{t[:200]!r}")
    body = t[i:j + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        if "escape" not in str(e).lower():
            raise
        fixed = _repair_json_escapes(body)
        obj = json.loads(fixed)          # 再失败就让异常冒出去（别吞）
        obj["_json_repaired"] = True
        return obj


def _call(prompt: str, cfg: dict, log=None) -> str:
    """统一走 s5_fuse.call_llm（直连优先、dsh 回退），保持单一口径。"""
    from s5_fuse import call_llm

    return call_llm(prompt, cfg, log)


def _strip_wrapper(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = "\n".join(ln for ln in t.splitlines() if not ln.strip().startswith("```")).strip()
    return t


# ---------------------------------------------------------------- 讲义库

def published_lectures(course: str, cfg: dict) -> list[dict]:
    """该课已发布讲义 → [{idx, lecture, date, path, text}]（按讲次升序）。"""
    d = notes_dir(cfg) / course
    out: list[dict] = []
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.md")):
        # 不锁死年份：原来写死 "2026-*.md"，跨年就永远扫不到讲义。
        # 讲义文件名形如「YYYY-MM-DD_第NN讲_主题.md」，靠 LEC_RE 认讲次即可。
        m = LEC_RE.search(p.name)
        if not m:
            continue
        out.append({
            "idx": int(m.group(1)),
            "lecture": f"第{int(m.group(1)):02d}讲",
            "date": p.name[:10],
            "path": p,
            "text": p.read_text(encoding="utf-8", errors="replace"),
        })
    out.sort(key=lambda x: x["idx"])
    return out


def _state_path(course: str, cfg: dict) -> Path:
    from common import work_dir

    return work_dir(cfg) / course / "_review_state.json"


def load_state(course: str, cfg: dict) -> dict:
    p = _state_path(course, cfg)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:  # noqa: BLE001
            pass
    return {"course": course, "chapters": [], "covered_upto": 0}


def save_state(course: str, cfg: dict, state: dict) -> None:
    p = _state_path(course, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def covered_upto(course: str, cfg: dict, log=None) -> int:
    """已出复习覆盖到第几讲。

    优先读状态文件；没有则**从已有复习文档自举** —— 正则扫 `第NN讲` 引用取最大值
    （常微分方程第一章复习提到第01~06讲 → 6）。
    """
    st = load_state(course, cfg)
    if st.get("covered_upto"):
        return int(st["covered_upto"])
    d = notes_dir(cfg) / course
    mx = 0
    for p in d.glob("00_第*复习*.md") if d.is_dir() else []:
        for n in LEC_RE.findall(p.read_text(encoding="utf-8", errors="replace")):
            mx = max(mx, int(n))
    if mx and log:
        log.info("状态文件缺失 → 从已有复习文档自举：已覆盖到 第%02d讲", mx)
    return mx


# ---------------------------------------------------------------- 判定

def judge_chapter_end(course: str, lecture: str, date: str, note_text: str,
                      cfg: dict, log=None) -> dict:
    """一次 LLM 调用：本讲是否为某章收口。返回 dict（含 is_chapter_end）。"""
    prompt = JUDGE_PROMPT.format(course=course, lecture=lecture, date=date, note=note_text)
    raw = _call(prompt, cfg, log)
    try:
        obj = _parse_json_obj(raw)
    except Exception as e:  # noqa: BLE001
        if log:
            log.warning("判定输出无法解析为 JSON（%s），原文尾部：%s", e, raw[-200:])
        return {"is_chapter_end": False, "chapter_label": "", "confidence": 0.0,
                "evidence": f"解析失败：{e}", "parse_error": True}
    obj.setdefault("is_chapter_end", False)
    obj.setdefault("granularity", "")
    obj.setdefault("chapter_label", "")
    obj.setdefault("confidence", 0.0)
    obj.setdefault("evidence", "")
    # 章名清洗：去掉「第X讲/第X节」指代（讲义内部编号与锚定编号实测差 1，照抄会污染文件名）
    obj["chapter_label_raw"] = obj.get("chapter_label") or ""
    obj["chapter_label"] = _sanitize_label(obj["chapter_label_raw"])
    # §3.7 只认「章 / 专题」两级；LLM 自报「小节」的一律降级为不算收口
    if obj["is_chapter_end"] and str(obj.get("granularity") or "").strip() == "小节":
        obj["is_chapter_end"] = False
        obj["downgraded"] = "granularity=小节（§3.7 只认章/专题）"
    return obj


# ---------------------------------------------------------------- 生成

def _target_chars(lectures: list[dict]) -> int:
    """目标总字数 ≈ 单讲讲义平均字符数 × 2.5（§3.7：≥1×，建议 2~3×）。"""
    if not lectures:
        return 30000
    avg = sum(len(x["text"]) for x in lectures) / len(lectures)
    return max(30000, int(avg * 2.5))


def _gen_extractions(course: str, chapter_label: str, lectures: list[dict],
                     cfg: dict, log=None) -> tuple[str, list[dict]]:
    blocks, recs = [], []
    for i, lec in enumerate(lectures, 1):
        if log:
            log.info("提取材料 %d/%d：%s", i, len(lectures), lec["lecture"])
        txt = _strip_wrapper(_call(EXTRACT_PROMPT.format(
            course=course, chapter_label=chapter_label,
            lecture=lec["lecture"], note=lec["text"]), cfg, log))
        recs.append({"lecture": lec["lecture"], "chars": len(txt)})
        blocks.append(f"\n\n########## {lec['lecture']}（{lec['date']}）##########\n\n{txt}")
    return "".join(blocks).strip(), recs


def _gen_outline(course: str, chapter_label: str, extractions: str,
                 target: int, cfg: dict, log=None, scope: str = "") -> dict:
    raw = _call(OUTLINE_PROMPT.format(course=course, chapter_label=chapter_label,
                                      extractions=extractions, target_chars=target,
                                      scope=scope or chapter_label), cfg, log)
    obj = _parse_json_obj(raw)
    secs = [s for s in (obj.get("sections") or []) if isinstance(s, dict) and s.get("title")]
    if not secs:
        raise RuntimeError("大纲解析出的 sections 为空")
    obj["sections"] = secs
    return obj


def _batch_sections(secs: list[dict], max_chars: int = 11000) -> list[list[dict]]:
    """把各节按目标字数贪心分组，每组一次调用（控输出长度）。"""
    batches, cur, cur_n = [], [], 0
    for s in secs:
        n = int(s.get("target_chars") or 5000)
        if cur and cur_n + n > max_chars:
            batches.append(cur)
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        batches.append(cur)
    return batches


def _gen_sections(course: str, chapter_label: str, outline: dict, batches: list[list[dict]],
                  extractions: str, cfg: dict, log=None) -> tuple[list[str], list[dict]]:
    outline_txt = json.dumps(outline, ensure_ascii=False, indent=2)
    parts, recs = [], []
    for i, batch in enumerate(batches, 1):
        spec = "\n".join(
            f"- 第{s.get('no')}节《{s['title']}》｜目标 {s.get('target_chars')} 字｜要点："
            + "；".join(str(x) for x in (s.get("brief") or []))
            for s in batch)
        tgt = sum(int(s.get("target_chars") or 5000) for s in batch)
        if log:
            log.info("撰写正文 %d/%d（%d 节 · 目标 %d 字）", i, len(batches), len(batch), tgt)
        txt = _strip_wrapper(_call(SECTION_PROMPT.format(
            course=course, chapter_label=chapter_label, i=i, n=len(batches),
            outline=outline_txt, section_spec=spec, extractions=extractions, target=tgt), cfg, log))
        parts.append(txt)
        recs.append({"batch": i, "sections": [s.get("no") for s in batch], "chars": len(txt)})
    return parts, recs


def _gen_appendix(course: str, chapter_label: str, extractions: str,
                  cfg: dict, log=None) -> str:
    if log:
        log.info("撰写附录（公式速查表 / 强调汇总 / 备考清单）")
    return _strip_wrapper(_call(APPENDIX_PROMPT.format(
        course=course, chapter_label=chapter_label, extractions=extractions), cfg, log))


# ---------------------------------------------------------------- 主流程

def review_lecture(course: str, lecture: str, cfg: dict | None = None,
                   force: bool = False, only_judge: bool = False,
                   log=None) -> dict:
    """对一讲做：判定 →（命中则）生成复习文档。"""
    cfg = cfg or load_config()
    log = log or get_logger("s7_review")
    t0 = time.time()

    lecs = published_lectures(course, cfg)
    me = next((x for x in lecs if x["lecture"] == lecture), None)
    if me is None:
        raise FileNotFoundError(f"讲义库里找不到 {course} / {lecture}（先 publish？）")

    log.info("章节收口判定：%s / %s（讲义 %d 字符）…", course, lecture, len(me["text"]))
    verdict = judge_chapter_end(course, lecture, me["date"], me["text"], cfg, log)
    log.info("判定结果：is_chapter_end=%s  confidence=%s  label=%r  依据=%s",
             verdict["is_chapter_end"], verdict.get("confidence"),
             verdict.get("chapter_label"), str(verdict.get("evidence"))[:80])

    out = {"course": course, "lecture": lecture, "verdict": verdict,
           "generated": None, "elapsed_sec": None}
    if not verdict["is_chapter_end"] or only_judge:
        out["elapsed_sec"] = round(time.time() - t0, 1)
        return out

    # ---- 已覆盖则跳过（2026-10-17 修，防状态污染 + 重复生成）----
    #   场景：LLM 会把**章内的专题收口**也判成命中（如常微分方程第 03/04/05 讲，
    #   而「第一章 初等积分法」那篇复习已是按第 01~06 讲出的）。这些讲次 ≤ covered_upto，
    #   若照旧走「上一章之后 → 本讲」，范围会算空、退化成单讲，**并把 covered_upto 往回退**。
    #   故：本讲编号 ≤ 已覆盖终点 → 直接跳过（不清状态、不生成）。
    covered = covered_upto(course, cfg, log)
    if me["idx"] <= covered:
        log.info("已覆盖：%s（第%02d讲）≤ 已出复习覆盖到第%02d讲 → 跳过生成",
                 lecture, me["idx"], covered)
        out["skipped"] = "already_covered"
        out["covered_upto"] = covered
        out["elapsed_sec"] = round(time.time() - t0, 1)
        return out

    # ---- 本章范围：上一章之后 → 本讲 ----
    start = covered + 1
    chapter = [x for x in lecs if start <= x["idx"] <= me["idx"]]
    if not chapter:
        chapter = [me]
    # 章名兜底：LLM 给的被洗空了（例如它只说「这节课」）→ 用本讲首个二级标题的内容名
    label = verdict.get("chapter_label") or ""
    if not label:
        m = re.search(r"^##\s+(.+)$", me["text"], re.M)
        label = _sanitize_label(m.group(1)) if m else ""
        if label and log:
            log.info("章名兜底：LLM 未给出内容名 → 取本讲首节标题 %r", label)
    if not label:
        label = f"第{CN_NUM[len(load_state(course, cfg).get('chapters', []))]}章"
    log.info("本章范围：%s ~ %s（共 %d 讲）｜章名：%s",
             chapter[0]["lecture"], chapter[-1]["lecture"], len(chapter), label)

    target = _target_chars(chapter)
    log.info("目标体量：%d 字符（单讲均 %.1fk × 2.5）", target,
             sum(len(x['text']) for x in chapter) / len(chapter) / 1000)

    scope = f"{chapter[0]['lecture']} ~ {chapter[-1]['lecture']}（共 {len(chapter)} 讲，{len(chapter)} 份材料）"
    extractions, ext_recs = _gen_extractions(course, label, chapter, cfg, log)
    outline = _gen_outline(course, label, extractions, target, cfg, log, scope=scope)
    # [!] 命名修正（2026-10-17 实测踩坑）：judge 给的是**收口那一讲**的名字，不是整章的名字。
    #   普物 第01~08 讲（状态参量/分子动理论/统计分布/输运过程）被 judge 叫作「输运过程」→
    #   整篇热学复习被命名成「输运过程复习」，**误导读者**。
    #   规则：范围 ≥3 讲且 label 不含「第X章/篇/部分」时，改用大纲的 doc_title（它看得到全章材料）。
    _is_chapter_no = bool(re.match(r"^第[^\s]{1,2}[章篇部]", label or ""))
    if len(chapter) >= 3 and not _is_chapter_no:
        dt = (outline.get("doc_title") or "").strip()
        if dt and dt != label:
            log.info("章名修正：judge 给的 %r 只代表末讲 → 改用大纲标题 %r（覆盖 %s）",
                     label, dt, scope)
            label = dt
    batches = _batch_sections(outline["sections"])
    parts, sec_recs = _gen_sections(course, label, outline, batches, extractions, cfg, log)
    appendix = _gen_appendix(course, label, extractions, cfg, log)

    # ---- 拼装 ----
    toc = "\n".join(f"{s.get('no')}. {s['title']}" for s in outline["sections"])
    head = (
        f"# {label} 复习\n\n"
        f"> **覆盖讲次**：{'、'.join(x['lecture'] for x in chapter)}"
        f"（{chapter[0]['date']} ~ {chapter[-1]['date']}）\n"
        f"> **材料依据**：{course} 随堂讲义（老师课堂内容）；未引入讲义之外的知识。\n"
        f"> **生成方式**：由随堂讲义自动汇编（{datetime.now():%Y-%m-%d %H:%M}）"
        f"｜判定依据：{verdict.get('evidence', '')}\n\n"
        f"## 目录\n\n{toc}\n\n---\n"
    )
    body = "\n\n".join(parts)
    doc = f"{head}\n{body}\n\n---\n\n{appendix}\n"

    # ---- 数学分隔符归一（血债：HANDOFF §24.7）----
    doc, mstats = normalize_math_delims(doc)
    if mstats["inline"] or mstats["display"]:
        log.warning("数学分隔符归一：\\(…\\) ×%d → $…$ ；\\[…\\] ×%d → $$…$$",
                    mstats["inline"], mstats["display"])

    # ---- 落盘 ----
    d = notes_dir(cfg) / course
    d.mkdir(parents=True, exist_ok=True)
    chap_no, chap_title = _split_label(label, outline.get("doc_title", ""))
    fname = f"00_{chap_no}复习_{chap_title}.md" if chap_no else f"00_{chap_title}复习.md"
    dest = d / fname
    if dest.exists() and not force:
        dest = d / f"{dest.stem}_v2{dest.suffix}"
    dest.write_text(doc, encoding="utf-8")

    # ---- 状态记账 ----
    st = load_state(course, cfg)
    st.setdefault("chapters", []).append({
        "label": label, "file": dest.name,
        "lectures": [x["lecture"] for x in chapter],
        "chars": len(doc), "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    st["covered_upto"] = me["idx"]
    save_state(course, cfg, st)

    out["generated"] = {
        "file": str(dest), "chars": len(doc), "target_chars": target,
        "sections": len(outline["sections"]), "batches": len(batches),
        "extractions": ext_recs, "section_recs": sec_recs,
        "math_delims_normalized": mstats,
        "toc": [s["title"] for s in outline["sections"]],
    }
    out["elapsed_sec"] = round(time.time() - t0, 1)
    log.info("复习文档已落盘：%s（%d 字符 · 目标 %d · %d 节）",
             dest, len(doc), target, len(outline["sections"]))
    return out


def _split_label(label: str, doc_title: str) -> tuple[str, str]:
    """把章名拆成 (章号, 主题)，用于 §3.7 规定的文件名 `00_第N章复习_主题.md`。

    例：'第一章 初等积分法' → ('第一章', '初等积分法')
        '热学篇'           → ('热学篇', doc_title)
        '第二部分 静力学'   → ('第二部分', '静力学')
        '输运过程'（纯专题名，无章号）→ ('', '输运过程')   ← 调用方据此生成 `00_输运过程复习.md`
    """
    lab = (label or "").strip()
    # [warn] 「第X部分」必须放在 **前面**：否则会被 `第[^\s]{1,2}[章篇部]` 懒惰匹配成「第二部」+「分 …」
    #    （2026-10-15 由 selftest_review_label 的 L5 抓出 —— 自测用例的价值就在这）
    m = re.match(r"^(第[一二三四五六七八九十百0-9]+部分|第[^\s]{1,2}[章篇部])\s*(.*)$", lab)
    if m:
        chap = m.group(1).strip()
        title = (m.group(2) or "").strip() or (doc_title or "").strip()
        return chap, title
    return "", (lab or doc_title or "复习").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="章节收口判定 + 复习文档生成（LLM 判定，非启发式）")
    ap.add_argument("--course", required=True)
    ap.add_argument("--lecture", required=True)
    ap.add_argument("--force", action="store_true", help="同名复习文档已存在时覆盖（默认加 _v2）")
    ap.add_argument("--only-judge", action="store_true", help="只判定，不生成")
    ap.add_argument("--json", default="", help="结果写 JSON")
    args = ap.parse_args()

    log = get_logger("s7_review")
    res = review_lecture(args.course, args.lecture, force=args.force,
                         only_judge=args.only_judge, log=log)
    print(json.dumps(res, ensure_ascii=False, indent=2)[:4000])
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
