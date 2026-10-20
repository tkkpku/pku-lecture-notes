"""阶段 3：时间轴融合 —— ASR 转写 × 关键帧文字 → 忠实还原老师授课原貌的 Markdown 讲义。

核心约束（你 2026-09-12 钦定，写在 config `fuse.fidelity` 里，逐条注入 prompt）：
  * 以**老师本人的授课风格**为主，保护原貌；不润色成教科书腔
  * 保留老师原话、重复强调、口头禅、举例与跑题段
  * 听不清处标 `[听不清]`，**不许臆造**
  * **不引入老师没讲的外部知识**（防幻觉/防补充课本定义）
  * 不对齐既有课本笔记写作法典

分块策略（`segment_mode: anchor`）：按时间把 ASR 段与关键帧切成不超过 `segment_max_min`
分钟的「内容块」，每块一次 LLM 调用；块内关键帧文字作为 anchor（主题与公式来源）。
一节课 ~120 分钟 → ~10 块 → ~10 次调用，成本可忽略。

LLM 后端：`dsh --profile <fuse.profile> "<prompt>"`（复用本机模型通道，见 HANDOFF §4 实测）。

产出：`work/<课程>/<讲次>/lecture.md`（讲义草稿，尚未含 frontmatter/索引，由 s6_publish 收尾）
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
    find_dsh_cli,
    get_logger,
    is_audio_only,
    lecture_work_dir,
    load_config,
    model_patch_args,
    normalize_math_delims,
    run,
)

FIDELITY_RULES = {
    "polish_colloquial": (
        "**调和口语**：把老师的口语整理成通顺的讲义文字 —— 去掉「是吧／对吧／那其实在就比如说／"
        "这个这个」这类纯语气碎片与口头禅，理顺句子结构、补上必要的逻辑连接，让段落读起来连贯。"
        "但必须**保留老师的表达习惯、术语用法与讲课节奏**；不要改写成教科书腔，不要变成另一个人写的文章。"
    ),
    "fill_from_context": (
        "**按上下文补全（你的核心任务）**：遇到语音转写听漏、同音字错识、公式符号缺失、画面文字乱码时，"
        "**依据上下文补成最可能的内容**（公式要还原成正确的 LaTeX）。判据分三档，请严格照做："
        "① **句内缺一两个词或半个短语、而句子意思明确** → **必须补**"
        "（例：「锻炼〔提出〕物理问题」「每章大概〔十道〕题」）—— 这类地方标 `[听不清]` 属于失职；"
        "② **缺的是实体**（人名／书名／机构名／具体数字／公式系数）→ 只有能从上下文可靠推出"
        "（含前文已出现过的同一实体）时才补，否则标 `[听不清]`；"
        "③ **整句缺失且无法推断** → 标 `[听不清]`，绝不硬编。"
        "补出的内容要贴合老师原意与课程语境，不要顺手改成书面语。"
    ),
    "no_external_knowledge": (
        "**严禁引入老师没讲的外部知识**：不补课本定义、不加背景介绍、不写「延伸阅读／补充说明」。"
        "[warn] 这条同时是「补全」的边界：只能补「**老师讲了但被识别丢掉**」的内容，"
        "不能补「课本上有但老师没讲」的内容。"
    ),
    "dedup_repetition": (
        "**重复看情况处理**：同一句话反复念叨 → 合并成一处；"
        "若是老师**反复强调的重点** → 保留一次，并在句末注明「（老师此处反复强调）」。"
    ),
    "on_topic_only": (
        "**跑题只留与学习相关的**：作业、考试、学习方法、课程安排、对学生的叮嘱 → 保留；"
        "纯闲聊、时事、个人经历 → **删除，不要写进讲义**。"
    ),
    "no_textbook_alignment": "不要按课本/笔记的写作规范重组内容，保持讲课的原始顺序与侧重。",
    #  2026-09-17（你批准）：事实性 normalize 允许
    "factual_normalize": (
        "**事实性 normalize（允许）**：若老师的口误或语音识别错误造成了**事实性错误**，"
        "按事实改正后再写进讲义。典型情形：人名被听成同音字（「朱蒂文」→「朱棣文」）、"
        "身份／机构／年份说错或听错（「数学家」→「物理学家」）、专名异写（「沃尔兹曼」→「玻尔兹曼」、"
        "「光学联交」→「光学黏胶」）。"
        "**边界（不可越过）**：① 只改能被上下文与画面**确证**的事实；"
        "② **不改**老师讲的物理内容、公式、论证步骤与结论；"
        "③ 不借「改正」之名把自己的知识或观点写进去；"
        "④ 拿不准就不改（保留原样，必要时标 `[听不清]`）。"
    ),
}

PROMPT_TEMPLATE = """【执行模式】这是**纯文本转换任务**：你不需要、也不得使用任何工具——不要读文件、不要写文件、不要执行命令、不要搜索工作区。
只根据下面给出的材料直接输出结果。**若材料不完整，就在输出里写明「材料不完整，缺少 XX」，不要自己去查找或补救。**

（缘由：2026-09-12 实测——`dsh --profile headless` 是持 danger-full-access 权限的完整 agent；
一次 prompt 被 cmd.exe 截断后，它收到残缺任务，竟自行 grep 工作区并动手改了流水线源码。必须显式约束。）

你是一位**课堂讲义整理者**：把老师的口语讲课 + 幻灯片内容，整理成**通顺、可读、又不失真**的 Markdown 讲义。
下面是这节课第 {idx}/{total} 个时段（{t0} - {t1}）的原始材料。

【本时段的幻灯片画面文字】（按时间顺序；来源：关键帧视觉识别，可能有识别错误）
{slides}

【本时段老师的讲课录音转写】（来源：语音识别，可能有同音字错误与遗漏）
{asr}

===== 你的任务 =====
把这一时段整理成 Markdown 讲义段落，作为整讲讲义的一部分。
**你的核心价值有两条**：
(a) 把口语调和成能读的讲义（不是逐字稿）；
(b) **依据上下文补全识别丢失的内容**（公式、术语、被听漏的句子）—— 这是本案最看重的能力。

===== 必须遵守的规则（不可违背）=====
{rules}
7. 转写里的**同音字／术语误识**要积极纠正（如「硬力」→「应力」、「面圆」→「面元」、「西格玛」→「σ」）；
   依据上下文能判断的就直接改，判断不了才保留原样。
8. 幻灯片文字可整理为要点列表；公式一律用 LaTeX（行内 `$...$`，行间 `$$...$$`）；
   画面文字不要硬塞进正文，要与老师讲的内容对应起来。
9. 语音转写的**遗漏、碎片、乱码**：先用上下文补全（这是你的强项）；确实补不出来的，标 `[听不清]`。
   **宁可标 `[听不清]` 也不要硬编** —— 但「能补而不补」同样是失职。
10. **不要把材料行首的时间戳原样抄进正文**。要定位时，最多在小节标题后写一次 `（约 mm:ss）`。
11. 输出的是**讲义**：句子要通顺、段落要连贯、该有的逻辑连接要补上；
    但用词与术语要跟老师保持一致，**不要改写成教科书腔**。
    画面上的要点用列表呈现即可，**不要**写「（画面要点 2/6）」这类机械标注。
12. 画面材料里可能混有**版权水印**（如「课堂教学视频受知识产权保护，未经北京大学授权，
    不得录屏转载，违者追究法律责任」）和**屏幕时间戳**（如「2026/09/10 星期四 07:57」）——
    这些**不是讲课内容**，请一律忽略，**不要**把它们写进讲义。
    画面里若只有这类噪声、没有别的可读内容，就直接跳过该处，不要凭空补内容。
13. **不引入老师没讲的知识**：不要补课本定义、不要加背景介绍、不要写「延伸阅读／补充说明」。
    补全的边界 = **只能是「老师讲了但被识别丢掉」的内容**。
14. **重复与跑题**：同一句话反复念叨 → 合并；若是反复强调的重点 → 保留一次并注明
    「（老师此处反复强调）」。跑题只留与学习相关的（作业、考试、学习方法、课程安排、叮嘱），
    纯闲聊/时事/个人经历 → 直接删掉，不要写进讲义。

===== 输出格式（严格遵守）=====
只输出这一段的 Markdown 正文，形如：

## <本页/本段主题，用老师自己的叫法>
<老师讲解的连贯段落；公式用 LaTeX>

- <画面要点，如有>

**绝对不要**输出：前言、总结、「我已完成」之类的对话、代码块围栏、对你自己工作的解释。
**不要**写「以上是本段内容」这类收尾语。
"""


def _fmt_mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _chunk_text_chars(c: dict) -> int:
    """块内文本量（ASR 正文 + 画面文字），用于**命令行长度保护**（见 build_chunks 末尾）。"""
    n = sum(len(s.get("text") or "") for s in c.get("segs") or [])
    for f in c.get("frames") or []:
        n += len(f.get("vlm_text") or f.get("ocr_text") or "")
    return n


def _split_chunk(c: dict) -> list[dict]:
    """把一块按 ASR 段对半切成两块（关键帧按时间归属重新分配）。"""
    segs = c["segs"]
    if len(segs) < 2:
        return [c]
    mid = len(segs) // 2
    left = {"t0": c["t0"], "t1": segs[mid - 1]["end"], "segs": segs[:mid], "frames": []}
    right = {"t0": segs[mid]["start"], "t1": c["t1"], "segs": segs[mid:], "frames": []}
    for f in c.get("frames") or []:
        (left if f["t_sec"] < right["t0"] else right)["frames"].append(f)
    return [left, right]


def build_chunks(asr: dict, frames_payload: dict, cfg: dict, log=None) -> list[dict]:
    """把 ASR 段与关键帧按时间切成内容块。"""
    max_sec = float(cfg["fuse"].get("segment_max_min", 12)) * 60
    segs = [s for s in asr.get("segments", []) if not s.get("dropped")]
    frames = frames_payload.get("frames", [])
    dur = float(asr.get("audio_sec") or 0) or (segs[-1]["end"] if segs else 0)

    chunks, cur = [], {"t0": 0.0, "t1": 0.0, "segs": [], "frames": []}
    for s in segs:
        if cur["segs"] and s["start"] - cur["t0"] > max_sec:
            cur["t1"] = cur["segs"][-1]["end"]
            chunks.append(cur)
            cur = {"t0": s["start"], "t1": 0.0, "segs": [], "frames": []}
        cur["segs"].append(s)
    cur["t1"] = cur["segs"][-1]["end"] if cur["segs"] else dur
    chunks.append(cur)

    # 关键帧按时间落入所在块（帧在块起始之后的算本块；跨块边界的帧归后一块）
    for f in frames:
        for c in chunks:
            if f["t_sec"] >= c["t0"] - 1e-6 and (f["t_sec"] < c["t1"] or c is chunks[-1]):
                c["frames"].append(f)
                break
    #  2026-09-19（真实现行犯修复）：**命令行长度保护**。
    #    本函数原先只按「时间上限」切块，而 s5_fuse 把整块 prompt 作为 argv 传给
    #    `node … bin.js [--patch …] --profile headless "<prompt>"`，受 Windows CreateProcess
    #    32767 字符命令行上限约束。实测理论力学第04讲第 3 块：prompt = 32712 字符，
    #    加上 node 路径 / bin.js / --patch / --profile 后 ≈ 33000 → **[WinError 206] 文件名或
    #    扩展名太长**，该块失败并在讲义里留下「[第 N 块生成失败]」占位符（而当时 pipeline 仍报 OK）。
    #    故除时间上限外再设**文本量上限**：超限的块在这里递归二分（正常块不受任何影响）。
    max_chars = int(cfg["fuse"].get("segment_max_chars", 0) or 0)
    n_split = 0
    if max_chars > 0:
        for _ in range(64):  # 防死循环
            over = [c for c in chunks if _chunk_text_chars(c) > max_chars and len(c["segs"]) >= 2]
            if not over:
                break
            over_ids = {id(c) for c in over}
            new: list[dict] = []
            for c in chunks:
                if id(c) in over_ids:
                    new.extend(_split_chunk(c))
                    n_split += 1
                else:
                    new.append(c)
            chunks = new
        if n_split and log:
            log.info("文本量保护：单块上限 %d 字符 → 细分 %d 次，共 %d 块",
                     max_chars, n_split, len(chunks))

    if log:
        log.info("内容分块：%d 块（上限 %.0f 分钟/块）", len(chunks), max_sec / 60)
        for i, c in enumerate(chunks, 1):
            log.debug("  块%d %s-%s 段%d 帧%d 文本%d字", i, _fmt_mmss(c["t0"]), _fmt_mmss(c["t1"]),
                      len(c["segs"]), len(c["frames"]), _chunk_text_chars(c))
    return chunks


# 录播画面的固有噪声（不属于讲课内容，必须剔除）：
#   · 底部半透明版权水印「课堂教学视频受知识产权保护，未经北京大学授权，不得录屏转载…」
#   · 屏幕上的日期/时间戳「2026/09/10 星期四 07:57」
# 2026-09-13 实测：不过滤的话，它们会被逐帧重复写进讲义开头，污染正文。
_NOISE_RE = re.compile(
    r"知识产权|未经北京大学授权|课堂教学视频|不得(录屏|翻录)转载|违者(追究|将追究)?法律责任"
    r"|^\s*\d{4}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{1,2}\s*星期|^\s*\d{1,2}\s*[:：]\s*\d{2}\s*$"
)


def _clean_frame_text(txt: str) -> str:
    """剔除画面固有噪声（版权水印/屏幕时间戳），只留真正的讲课画面文字。"""
    keep = []
    for ln in (txt or "").splitlines():
        s = ln.strip()
        if s and not _NOISE_RE.search(s):
            keep.append(s)
    return "\n".join(keep)


def _slide_block(frames: list[dict]) -> str:
    """把关键帧画面文字压成「时间锚点 + 跨帧逐行去重」的文本块。

     2026-09-19（真实现行犯修复）：**跨帧逐行去重**。
      实测现行犯：理论力学 2026-09-17 第04讲 40~49 分钟是一页**长公式推导 PPT**，
      pHash 变化检测把「公式逐行添加」当成换页 → 该块留下 15 帧几乎同一页；
      每帧 VLM 都读出**同一份残缺公式**（含大量 `[看不清]`）→ 块内画面文字膨胀到
      **13070 字符（占该块 prompt 的 68%）且高度重复**。LLM 拿到这种输入会试图复原整页推导，
      陷入长时间自我推理 → `dsh headless 失败 rc=1` → 该块失败、讲义缺内容（而 pipeline 仍报 OK）。
      口径：**保内容、不丢信息** —— 规范化后相同的行只在首次出现的帧下输出一次；
      后出现的新行照常输出（公式逐行新增的部分不会丢）。
    """
    if not frames:
        return "（本时段没有检测到幻灯片切换）"
    out: list[str] = []
    seen: set[str] = set()
    for f in frames:
        raw = f.get("vlm_text") or f.get("ocr_text") or ""
        txt = _clean_frame_text(raw).strip()
        fresh: list[str] = []
        for ln in txt.splitlines():
            key = re.sub(r"\s+", "", ln)
            if not key or key in seen:
                continue
            seen.add(key)
            fresh.append(ln)
        if fresh:
            out.append(f"[{_fmt_mmss(f['t_sec'])}] " + "\n".join(fresh))
    return "\n".join(out) if out else "（本时段画面无新增文字）"


def _asr_block(segs: list[dict]) -> str:
    if not segs:
        return "（本时段没有识别到语音）"
    return "\n".join(f"[{_fmt_mmss(s['start'])}] {s['text']}" for s in segs)


def build_prompt(chunk: dict, cfg: dict, idx: int, total: int) -> str:
    fid = cfg["fuse"].get("fidelity", {})
    rules = "\n".join(f"{i}. {FIDELITY_RULES[k]}" for i, k in enumerate(
        [k for k in FIDELITY_RULES if fid.get(k)], start=1))
    return PROMPT_TEMPLATE.format(
        idx=idx, total=total,
        t0=_fmt_mmss(chunk["t0"]), t1=_fmt_mmss(chunk["t1"]),
        slides=_slide_block(chunk["frames"]),
        asr=_asr_block(chunk["segs"]),
        rules=rules,
    )


def llm_available() -> str:
    try:
        return find_dsh_cli()[1]
    except RuntimeError:
        return ""


def call_llm(prompt: str, cfg: dict, log=None) -> str:
    """融合调用：默认**直连** OpenAI 兼容端点（不产生 dsh 会话）；可回退 dsh headless。

     2026-09-26（你钦定改造）：`llm.mode: direct`（默认）走 `llm_client.chat_text()` ——
      纯 HTTP 调用，**没有** dsh 的 system prompt / 工具 schema（~25K token 固定开销），
      也**不再往 `{DSH_HOME}\\sessions\\` 落会话目录**（实测一天批跑堆 400+ 个 / 380 MB）。
      `llm.mode: dsh` 时走下方原路径（端点变更或排障时的回退）。

    [warn] 回退路径的前提（历史教训）：必须用 `find_dsh_cli()` 拿 `node + bin.js`，**不能**直接
    exec `dsh.CMD` —— 多行 prompt 经 cmd.exe 会在第一个换行处被静默截断（2026-09-12 实测）。
    """
    import llm_client as LLM

    if LLM.enabled(cfg):
        return LLM.chat_text(prompt, cfg, log)

    cli = find_dsh_cli()
    fcfg = cfg.get("fuse") or {}
    profile = fcfg.get("profile", "headless")
    pin = (model_patch_args(fcfg.get("provider", ""), fcfg.get("model", ""), cfg)
           if fcfg.get("pin_model", True) else [])
    if log:
        log.debug("dsh 调用：%s --profile %s <prompt %d 字符>｜模型=%s", cli[0], profile, len(prompt),
                  (f"{fcfg.get('provider')}/{fcfg.get('model')}" if pin else "跟随全局默认"))
    cp = run(cli + pin + ["--profile", profile, prompt], check=False, timeout=900, log=None)
    if cp.returncode != 0:
        raise RuntimeError(f"dsh headless 失败 rc={cp.returncode}:\n{(cp.stderr or '')[-1500:]}")
    out = (cp.stdout or "").strip()
    if not out:
        raise RuntimeError(
            f"dsh headless 返回空输出（prompt {len(prompt)} 字符）。"
            f"stderr 尾部：{(cp.stderr or '')[-800:]}"
        )
    return out


def _strip_wrapper(text: str) -> str:
    """去掉 dsh headless 可能的包装（代码围栏、自述行）。"""
    t = text.strip()
    if t.startswith("```"):
        lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
        t = "\n".join(lines).strip()
    return t


# ---------- 语气碎片确定性清理（2026-09-17 新增）----------
# 由来：保真规则①要求去掉「是吧／对吧／那其实／就比如说／这个这个／然后呢」这类纯语气碎片，
# 但 LLM 偶尔会漏掉 1 处（实测第 1/2 讲各残留 1 处，`scan_note.py` 判 [x]）。
# LLM 不可靠的地方就用确定性后处理兜住 —— 但**只在标点边界上动手**，绝不碰正文里的同形字串
# （例如老师真的在提问、或「对不对」出现在术语/引文中时，不满足边界条件 → 原样保留）。
_FILLER_TOKENS = ("是吧", "对吧", "那其实", "就比如说", "这个这个", "然后呢")
_FILLER = "(?:" + "|".join(_FILLER_TOKENS) + ")"
# ① 削词缀式改写：句中作连接词时，删整词会伤可读性 → **保留有意义的词干，只削语气性词缀**。
#    实测形态：「…推导出来的规律，那其实就是可以把所有点都测出来」
#              「…课本里头都有。就比如说你要算一个抛物线形状的问题」
_FILLER_REWRITES = [
    (re.compile("就比如说"), "比如"),        # 「就」「说」是语气，例子本身要留
    (re.compile("那其实"), "其实"),
    (re.compile("这个这个"), "这个"),        # 纯重复
    (re.compile("然后呢(?=[，,、；;。])"), "然后"),
]
_SCRUB_RULES = [
    # 「……，是吧。」→「……。」（连前置逗号一起去掉，避免留下「，。」）
    (re.compile(rf"[，,、；;]{_FILLER}(?=[。！？!?]|$)"), ""),
    # 「……，对吧，……」→「……，……」：匹配「，对吧」、**保留后面那个逗号**（替换为空串，
    # 否则会留出「，，」双逗号 —— 首版就踩了这个，被 selftest_filler 的 S1/S6 当场抓住）
    (re.compile(rf"[，,、；;]{_FILLER}(?=[，,、；;])"), ""),
    # 行首「就比如说，……」→「……」
    (re.compile(rf"(?m)^[＊*\-\s]*{_FILLER}[，,、；;]"), ""),
]


def scrub_fillers(text: str) -> tuple[str, int]:
    """清掉语气碎片，返回 (清理后的文本, 处数)。

    两道处理，都不越界：
      ① **削词缀**：`就比如说→比如`、`那其实→其实`、`这个这个→这个` —— 句中作连接词时用，
         保留有意义的词干（删整词会让句子读不通）；
      ② **删整词**：`是吧／对吧／然后呢` 这类纯语气词，**只在标点边界上**动手，
         绝不碰正文里的同形字串（不在边界的原样保留）。
    """
    n = 0
    for pat, rep in _FILLER_REWRITES:
        text, k = pat.subn(rep, text)
        n += k
    for pat, rep in _SCRUB_RULES:
        text, k = pat.subn(rep, text)
        n += k
    return text, n


def fuse_lecture(course: str, lecture: str, cfg: dict | None = None, force: bool = False,
                 dry_run: bool = False, log=None, call=None) -> dict:
    cfg = cfg or load_config()
    log = log or get_logger("s5_fuse", cfg)
    wd = lecture_work_dir(course, lecture, cfg)
    asr_p = wd / "asr.json"
    ocr_p = wd / "frames_ocr.json"
    out_md = wd / "lecture.md"
    #  2026-10-04：**纯音频课程**允许没有 frames_ocr.json（或其为空）——按纯 ASR 融合。
    #   非纯音频课程仍要求该文件存在（防「忘了跑 s4_ocr 就融合」）。
    audio_only = is_audio_only(course, cfg)
    if not asr_p.exists():
        raise FileNotFoundError(f"缺 {asr_p.name}（先跑 s3_asr）")
    if not ocr_p.exists() and not audio_only:
        raise FileNotFoundError(f"缺 {ocr_p.name}（先跑 s4_ocr）")
    if out_md.exists() and not force:
        log.info("已存在 lecture.md，跳过（--force 重跑）")
        return {"md": str(out_md), "chunks": None, "skipped": True}

    asr = json.loads(asr_p.read_text(encoding="utf-8"))
    if ocr_p.exists():
        frames_payload = json.loads(ocr_p.read_text(encoding="utf-8"))
    else:
        frames_payload = {"frames": []}
    n_frames = len(frames_payload.get("frames") or [])
    if audio_only or n_frames == 0:
        log.info("【纯音频课】无画面文字参与融合（frames=%d，audio_only=%s）", n_frames, audio_only)
    chunks = build_chunks(asr, frames_payload, cfg, log)
    prompts = [build_prompt(c, cfg, i, len(chunks)) for i, c in enumerate(chunks, 1)]

    if dry_run:
        log.info("[dry-run] 将发起 %d 次 LLM 调用；首块 prompt 长度 %d 字符",
                 len(prompts), len(prompts[0]) if prompts else 0)
        return {"md": None, "chunks": len(chunks),
                "prompt_preview": prompts[0][:1200] if prompts else "", "dry_run": True}

    caller = call or (lambda p: call_llm(p, cfg, log))
    parts, t0, n_fail = [], time.time(), 0
    for i, p in enumerate(prompts, 1):
        log.info("融合第 %d/%d 块…", i, len(prompts))
        try:
            txt = _strip_wrapper(caller(p))
            txt, n_scrub = scrub_fillers(txt)
            if n_scrub:
                log.info("第 %d 块清理语气碎片 %d 处", i, n_scrub)
            parts.append(txt)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            log.error("第 %d 块失败，写入占位以便续跑：%s", i, e)
            parts.append(f"## [第 {i} 块生成失败]\n\n> 错误：{e}\n")
    # [warn] 2026-09-13 实测修复（bug E）：原先「全块失败」也会写出讲义草稿并返回 OK（草稿里只有占位符），
    #    后续 publish 若草稿恰好够 2000 字节还会被当成合格讲义发布 → 假绿。全块失败即硬失败。
    if prompts and n_fail == len(prompts):
        raise RuntimeError(
            f"融合 {n_fail}/{len(prompts)} 块**全部失败**，拒绝写讲义草稿（防假绿）。"
            f"请查上一条 ERROR 的根因（模型通道 / dsh headless / find_dsh_cli）。")
    md = f"# {lecture}\n\n" + "\n\n".join(parts).strip() + "\n"
    # 数学分隔符归一（2026-10-12 新增，HANDOFF §24.7）：模型偶尔输出 `\(…\)` / `\[…\]`，
    # 而 Obsidian 默认**不渲染**这两组分隔符 → 公式会原样暴露成源码。确定性归一，别赌模型口味。
    md, mstats = normalize_math_delims(md)
    if mstats["inline"] or mstats["display"]:
        log.warning("数学分隔符归一：\\(…\\) ×%d → $…$ ；\\[…\\] ×%d → $$…$$（原为非 KaTeX 写法）",
                    mstats["inline"], mstats["display"])
    out_md.write_text(md, encoding="utf-8")
    log.info("讲义草稿完成：%s（%d 字，%d 块，%.0fs）", out_md, len(md), len(parts), time.time() - t0)
    return {"md": str(out_md), "chunks": len(chunks), "chars": len(md),
            "math_delims_normalized": mstats, "elapsed_sec": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 3：时间轴融合 → 讲义草稿")
    ap.add_argument("--course", required=True)
    ap.add_argument("--lecture", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    res = fuse_lecture(args.course, args.lecture, force=args.force, dry_run=args.dry_run)
    print("[s5_fuse]", json.dumps(res, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
