"""阶段 2d：关键帧文字理解（WinOCR 打底 + VLM 兜底）。

后端（config `vision.backend`）：

| 值 | 行为 | 成本 |
|---|---|---|
| `winocr` | Windows 内置 OCR（Windows.Media.Ocr，走 PowerShell 5.1 + WinRT） | 0 |
| `vlm` | 多模态模型逐帧/批量读图（走 `dsh --profile <profile>`） | ≈$0.02/讲的图像 token |
| `hybrid` | 先 OCR，可疑帧（太短/拉丁主导/可疑串）升级 VLM | 介于两者 |

[warn] 实测（2026-09-12，合成素材）：WinOCR **中文正文可用**，但
  ① 中文被空格拆成单字 → 已由 `normalize_ocr_text()` 合并；
  ② **公式彻底崩坏**：`lim(ΔA→0)` → `Iim(AA—O)`、`ΔF/ΔA` → `AF / AA`；
  ③ **漏字**：深底白字的标题「一、应力矢量」丢了首字「一」。
  故 hybrid/vlm 对本任务是质量刚需，不是省钱问题（见 docs/RESEARCH.md §6：图像 token 便宜到可忽略）。

产出：`work/<课程>/<讲次>/frames_ocr.json`
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
    run,
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_CLASS = r"\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
_LATIN_RE = re.compile(r"[A-Za-z]")
_SUSPICIOUS_RE = re.compile(r"(?:\b[A-Z]{1,2}\b\s*){2,}|Iim|lirn|O\b")
_JSON_BLOCK_RE = re.compile(r"\[.*\]", re.S)

VLM_PROMPT = """【执行模式】这是纯文本转换任务，请严格照做：
* **只允许**调用 `read_image` 工具读取下面列出的图片（每一张都要读）；
* **禁止**写文件、禁止执行命令、禁止搜索工作区、禁止调用其他任何工具；
* 图片路径可能含中文，照原样传入即可（实测中文路径可用）；
* 读不到某张图时，就在该项的 text 里写「未收到图片」，**不要**去磁盘上翻找替代文件。

（缘由：2026-09-12 实测——`dsh --profile headless` 是持 danger-full-access 权限的完整 agent，
prompt 一旦残缺它会自行用工具去改源码；但 `@路径` 本身只是文本提示，**读图必须依赖 read_image 工具**，
所以不能一刀切禁掉全部工具，只能白名单。）

你在为课堂录像做关键帧文字提取（OCR 的替代/增强）。

规则（必须严格遵守）：
1. **只提取画面中真实存在的文字**，逐字照抄，不要改写、不要翻译、不要总结、不要补充任何画面外的知识。
2. 数学公式用 LaTeX 表示（例如 \\sigma_{{ij}} = \\sigma_{{ji}}）。
   [warn] 本提示词走 str.format()，LaTeX 的下标花括号**必须写成双花括号转义**，
      否则会被当成占位符抛 `KeyError: 'ij'`（2026-09-12 实测踩过这个坑）。
3. 字迹不清或被遮挡处写 [看不清]，**绝不猜测**。
4. 只输出 JSON 数组，不要任何解释文字、不要 markdown 代码块。

输出格式：
[{{"file": "<文件名>", "text": "<画面中的全部文字，按阅读顺序>"}}]

待处理图片（共 {n} 张）：
{refs}
"""

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
_WINOCR_PS1 = _SCRIPT_DIR / "winocr.ps1"

#  2026-09-26：**直连模式**的读图 prompt（图片随请求一起 base64 附上，不用任何工具）。
#    与上面的 VLM_PROMPT 相比：去掉了「只允许调 read_image / 禁止其他工具 / @路径」那一整段
#    —— 那些是给 **dsh headless（完整 agent）** 写的防护说明；直连 HTTP 调用没有工具、没有
#    文件系统访问，因此既不需要它们，也不必付那 ~25K token 的固定开销。
VLM_PROMPT_DIRECT = """你在为课堂录像做关键帧文字提取（OCR 的替代/增强）。

本次附上 {n} 张关键帧图片，**按你看到的顺序**与下列文件名一一对应：
{file_list}

规则（必须严格遵守）：
1. **只提取画面中真实存在的文字**，逐字照抄，不要改写、不要翻译、不要总结、不要补充任何画面外的知识。
2. 数学公式用 LaTeX 表示（例如 \\sigma_{{ij}} = \\sigma_{{ji}}）。
3. 字迹不清或被遮挡处写 [看不清]，**绝不猜测**。
4. 只输出 JSON 数组，不要任何解释文字、不要 markdown 代码块。

输出格式：
[{{"file": "<文件名>", "text": "<画面中的全部文字，按阅读顺序>"}}]
"""


# ---------------- WinOCR ----------------
def run_winocr(paths: list[str | Path], cfg: dict, log=None, timeout: float = 900) -> dict[str, dict]:
    """批量调 Windows OCR，返回 {路径: {text, ms, ok}}。"""
    if not paths:
        return {}
    out_json = lecture_work_dir("_tmp", "_ocr") / "winocr_out.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_json.with_name("winocr_list.txt")
    list_file.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    if not _WINOCR_PS1.exists():
        raise FileNotFoundError(f"缺少 {_WINOCR_PS1}")
    t0 = time.time()
    cp = run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(_WINOCR_PS1),
              "-ListFile", str(list_file), "-OutJson", str(out_json)], check=False, timeout=timeout, log=log)
    if cp.returncode != 0 or not out_json.exists():
        raise RuntimeError(f"WinOCR 失败 rc={cp.returncode}\n{(cp.stderr or '')[-1500:]}")
    data = json.loads(out_json.read_text(encoding="utf-8-sig"))
    res = {}
    for r in data.get("results", []):
        res[str(r["path"])] = {"text": r.get("text", ""), "ms": r.get("ms", 0), "ok": r.get("ok", False)}
    if log:
        log.info("WinOCR 完成：%d 张，引擎耗时 %.1fs，墙钟 %.1fs（language=%s）",
                 len(res), data.get("engine_ms_total", 0) / 1000, time.time() - t0, data.get("language"))
    return res


# ---------------- 后处理与质量判据 ----------------
def normalize_ocr_text(t: str) -> str:
    """合并中文之间的空格（WinOCR 中文按字出 box，会把「应力」拆成「应 力」）。"""
    if not t:
        return ""
    t = re.sub(rf"(?<=[{_CJK_CLASS}])\s+(?=[{_CJK_CLASS}])", "", t)
    t = re.sub(r"[ \t\u3000]+", " ", t)
    return t.strip()


# 录播画面固有噪声：版权水印 + 屏幕时间戳（不属于讲课内容）
_WATERMARK_RE = re.compile(
    r"知识产权|未经北京大学授权|课堂教学视频|不得(录屏|翻录)转载|违者(追究|将追究)?法律责任"
    r"|^\s*\d{4}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{1,2}\s*星期|^\s*\d{1,2}\s*[:：]\s*\d{2}\s*$"
)


def strip_watermark(t: str) -> str:
    """剔除版权水印与屏幕时间戳行（2026-09-13 实测：不过滤会逐帧污染讲义开头）。"""
    keep = [ln.strip() for ln in (t or "").splitlines()
            if ln.strip() and not _WATERMARK_RE.search(ln.strip())]
    return "\n".join(keep)


def ocr_quality(text: str) -> dict:
    n = len(text)
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    return {
        "chars": n,
        "cjk": cjk,
        "latin": latin,
        "cjk_ratio": round(cjk / n, 3) if n else 0.0,
        "suspicious": bool(_SUSPICIOUS_RE.search(text)),
    }


def needs_vlm(text: str, q: dict, cfg: dict) -> tuple[bool, str]:
    """判断该帧是否需要升级到 VLM。"""
    if len(text) < cfg["vision"]["ocr_escalate_min_chars"]:
        return True, "too_short"
    if q["cjk_ratio"] < 0.15 and q["latin"] >= 8:
        return True, "latin_dominant"      # 多半是公式页，OCR 必崩
    if q["suspicious"]:
        return True, "suspicious_token"    # 如 lim→Iim、Δ→A
    return False, ""


# ---------------- VLM ----------------
def vlm_available() -> str:
    try:
        return find_dsh_cli()[1]
    except RuntimeError:
        return ""


def _parse_vlm_json(stdout: str) -> list[dict]:
    m = _JSON_BLOCK_RE.search(stdout or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _retry_one_by_one(batch: list[Path], v: dict, cli: list[str], log=None,
                      pin: list[str] | None = None) -> list[dict]:
    """批量 JSON 解析失败时的兜底：**逐张单图重试**。

    2026-09-13 实测：115 帧里 3 批（12 帧）批量返回非 JSON（画面复杂/输出过长被截断），
    降级成 WinOCR 后那些帧几乎全是乱码 —— 恰好落在「二力等效/一般力系的简化」这类
    图多的章节上。单张调用输出短、JSON 结构稳，实测能救回大部分。
    """
    out: list[dict] = []
    for p in batch:
        prompt = VLM_PROMPT.format(n=1, refs=f"@{p.as_posix()}")
        try:
            cp = run(cli + (pin or []) + ["--profile", v["profile"], prompt],
                     check=False, timeout=v.get("timeout_sec", 300), log=None)
        except Exception as e:  # noqa: BLE001
            if log:
                log.warning("单张重试异常 %s: %s", p.name, e)
            continue
        items = _parse_vlm_json(cp.stdout or "")
        if items:
            it = items[0]
            it.setdefault("file", p.name)
            out.append(it)
        elif log:
            log.warning("单张重试仍未解析出 JSON：%s", p.name)
    return out


def _retry_one_by_one_direct(batch: list[Path], cfg: dict, log=None) -> list[dict]:
    """直连模式的逐张重试（批量未解析出 JSON 时）。"""
    import llm_client as LLM

    out: list[dict] = []
    for p in batch:
        prompt = VLM_PROMPT_DIRECT.format(n=1, file_list=f"1. {p.name}")
        try:
            text = LLM.chat_with_images(prompt, [p], cfg, log)
        except Exception as e:  # noqa: BLE001
            if log:
                log.warning("单张重试请求失败 %s：%s", p.name, e)
            text = ""
        items = _parse_vlm_json(text)
        if items:
            out.append(items[0])
        else:
            if log:
                log.warning("单张重试仍未解析出 JSON：%s", p.name)
            out.append({"file": p.name, "text": ""})
    return out


def _run_vlm_direct(paths: list[str | Path], cfg: dict, v: dict, log=None) -> dict[str, str]:
    """ 2026-09-26：**直连多模态读图** —— 图片 base64 直接塞进 user content。

    与 dsh 路径的差别：不依赖 `@路径` + `read_image` 工具，**不产生 dsh 会话**，
    也不付 system prompt / 工具 schema 的固定开销。
    """
    import llm_client as LLM

    if log:
        log.info("读图通道：直连 %s（%s）", (cfg.get("llm") or {}).get("base_url"),
                 (cfg.get("llm") or {}).get("model", ""))
    out: dict[str, str] = {}
    bs = max(1, int(v.get("batch_size", 20)))
    for i in range(0, len(paths), bs):
        batch = [Path(p) for p in paths[i:i + bs]]
        file_list = "\n".join(f"{k + 1}. {p.name}" for k, p in enumerate(batch))
        prompt = VLM_PROMPT_DIRECT.format(n=len(batch), file_list=file_list)
        if log:
            log.info("VLM 批量 %d 张（%d/%d）…", len(batch), i + len(batch), len(paths))
        try:
            text = LLM.chat_with_images(prompt, batch, cfg, log)
        except Exception as e:  # noqa: BLE001
            if log:
                log.warning("VLM 批量请求失败（%d 张）：%s", len(batch), e)
            text = ""
        items = _parse_vlm_json(text)
        if not items:
            if log:
                log.warning("VLM 输出未解析出 JSON，本批 %d 张改为逐张重试", len(batch))
            items = _retry_one_by_one_direct(batch, cfg, log)
        for k, it in enumerate(items):
            fn = str(it.get("file", "")).strip()
            txt = str(it.get("text", "")).strip()
            match = next((p for p in batch if p.name == Path(fn).name), None)
            if match is None and k < len(batch):
                match = batch[k]
            if match is not None:
                out[str(match)] = txt
    return out


def run_vlm(paths: list[str | Path], cfg: dict, log=None) -> dict[str, str]:
    """批量调多模态模型读图。返回 {路径: 提取文本}。

     2026-09-26（你钦定改造）：`llm.mode: direct`（默认）→ 走 `_run_vlm_direct()`
      **直连** OpenAI 兼容端点（图片 base64 入请求）——不产生 dsh 会话、无固定开销。
      `llm.mode: dsh` → 走下方原路径（`--profile headless` + `@路径` + `read_image` 白名单）。

    [warn] 回退路径依赖 `dsh --profile <profile>` 的 `@图片路径` 引用能力（HANDOFF §4 已实测）；
       且必须经 `find_dsh_cli()` 调用（**不要**直接 exec `dsh.CMD`）：prompt 是多行的，
       经 cmd.exe 会在第一个换行处被静默截断（2026-09-12 实测，rc=0 看不出错）。
    """
    import llm_client as LLM

    v = (cfg.get("vision") or {}).get("vlm") or {}
    if LLM.enabled(cfg):
        return _run_vlm_direct(paths, cfg, v, log)

    cli = find_dsh_cli()
    #  2026-09-17：按 vision.vlm.provider/model **显式钉通道**（此前只传 --profile，
    #    这两个 config 字段是装饰品，实际通道跟着全局 agent-default-model 漂）。
    pin = (model_patch_args(v.get("provider", ""), v.get("model", ""), cfg)
           if v.get("pin_model", True) else [])
    if log:
        log.info("读图通道：%s", (f"{v.get('provider')}/{v.get('model')}" if pin
                                 else "跟随全局默认"))
    out: dict[str, str] = {}
    bs = max(1, int(v.get("batch_size", 4)))
    for i in range(0, len(paths), bs):
        batch = [Path(p) for p in paths[i:i + bs]]
        refs = " ".join(f"@{p.as_posix()}" for p in batch)
        prompt = VLM_PROMPT.format(n=len(batch), refs=refs)
        if log:
            log.info("VLM 批量 %d 张（%d/%d）…", len(batch), i + len(batch), len(paths))
        cp = run(cli + pin + ["--profile", v["profile"], prompt],
                 check=False, timeout=v.get("timeout_sec", 300), log=None)
        items = _parse_vlm_json(cp.stdout or "")
        if not items:
            if log:
                log.warning("VLM 输出未解析出 JSON（rc=%s），本批 %d 张改为逐张重试",
                            cp.returncode, len(batch))
            items = _retry_one_by_one(batch, v, cli, log, pin=pin)
        for k, it in enumerate(items):
            fn = str(it.get("file", "")).strip()
            txt = str(it.get("text", "")).strip()
            match = next((p for p in batch if p.name == Path(fn).name), None)
            if match is None and k < len(batch):
                match = batch[k]
            if match is not None:
                out[str(match)] = txt
    return out


# ---------------- 主流程 ----------------
def process_lecture(course: str, lecture: str, cfg: dict | None = None, backend: str = "",
                    force: bool = False, log=None, skip_vlm: bool = False) -> dict:
    cfg = cfg or load_config()
    log = log or get_logger("s4_ocr", cfg)
    wd = lecture_work_dir(course, lecture, cfg)
    idx_path = wd / "frames_index.json"
    out_path = wd / "frames_ocr.json"

    #  2026-10-04：**纯音频课程**——不读图，直接写空产物（保持下游产物契约）。
    #   见 config.yaml 的 `vision.audio_only_courses` 与 `common.is_audio_only()`。
    if is_audio_only(course, cfg):
        if out_path.exists() and not force:
            log.info("【纯音频课】已存在空 frames_ocr.json，跳过")
            return json.loads(out_path.read_text(encoding="utf-8"))
        payload = {
            "course": course, "lecture": lecture, "backend": "none", "audio_only": True,
            "frames": [], "note": "纯音频课程：按 config.vision.audio_only_courses 跳过画面文字识别",
        }
        wd.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("【纯音频课】跳过 OCR（不抽帧、不读图），已写空产物：%s", out_path)
        return payload

    if not idx_path.exists():
        raise FileNotFoundError(f"缺 frames_index.json（先跑 s2_extract）：{idx_path}")
    if out_path.exists() and not force:
        log.info("已存在 frames_ocr.json，跳过（--force 重跑）")
        return json.loads(out_path.read_text(encoding="utf-8"))

    index = json.loads(idx_path.read_text(encoding="utf-8"))
    frames_dir = Path(index["frames_dir"])
    frames = index["frames"]
    backend = backend or (cfg.get("vision") or {}).get("backend") or "vlm"
    log.info("帧文字理解：%d 帧，backend=%s", len(frames), backend)

    ocr_map: dict[str, dict] = {}
    # 即使 backend=vlm 也先跑一遍 WinOCR 作为兜底（6 帧仅 0.6s；VLM 偶尔解析失败，
    # 没有兜底就会在讲义里留下空白页 —— 实测踩过）。
    # [warn] WinOCR 只在 Windows 可用（依赖 PowerShell + WinRT）：其他平台直接跳过。
    if backend in ("winocr", "hybrid", "vlm"):
        if sys.platform.startswith("win"):
            ocr_map = run_winocr([frames_dir / f["file"] for f in frames], cfg, log)
        elif backend in ("winocr", "hybrid"):
            log.warning("WinOCR 仅 Windows 可用，本机已跳过 → 退化为纯 VLM")
            backend = "vlm"

    records = []
    escalate: list[Path] = []
    for f in frames:
        p = frames_dir / f["file"]
        raw = ocr_map.get(str(p), {}).get("text", "")
        text = strip_watermark(normalize_ocr_text(raw))
        q = ocr_quality(text)
        need, why = (False, "")
        if backend == "hybrid":
            need, why = needs_vlm(text, q, cfg)
        elif backend == "vlm":
            need, why = True, "backend_vlm"
        rec = {
            "idx": f["idx"], "file": f["file"], "t_sec": f["t_sec"],
            "ocr_text": text, "ocr_raw": raw, "ocr_ms": ocr_map.get(str(p), {}).get("ms", 0),
            "quality": q, "vlm_text": None, "escalated": need, "escalate_reason": why,
            "source": "winocr" if backend != "vlm" else "vlm",
        }
        records.append(rec)
        if need:
            escalate.append(p)

    if escalate:
        max_calls = int(cfg["vision"].get("max_vlm_calls_per_lecture", 40))
        if len(escalate) > max_calls:
            log.warning("需升级帧数 %d 超过闸门 %d，仅升级前 %d 帧", len(escalate), max_calls, max_calls)
            escalate = escalate[:max_calls]
        try:
            vlm_map = run_vlm(escalate, cfg, log)
            by_file = {Path(k).name: v for k, v in vlm_map.items()}
            for rec in records:
                if rec["file"] in by_file and by_file[rec["file"]]:
                    rec["vlm_text"] = strip_watermark(by_file[rec["file"]])
                    rec["source"] = "vlm"
        except Exception as e:  # noqa: BLE001
            log.error("VLM 后端失败，降级为纯 OCR：%s: %s", type(e).__name__, e)

    stats = {
        "frames": len(records),
        "escalated": sum(1 for r in records if r["escalated"]),
        "vlm_ok": sum(1 for r in records if r["vlm_text"]),
        "ocr_ms_total": sum(r["ocr_ms"] for r in records),
        "empty": sum(1 for r in records if not (r["vlm_text"] or r["ocr_text"])),
    }
    # [warn] 2026-09-13 实测修复（bug D）：VLM 整场失败时旧版照样写出 frames_ocr.json 并返回 OK，
    #    讲义里就只剩 WinOCR 文本 —— 而 WinOCR 对真实录播画面（教室小窗+版权水印+1080p 缩放）
    #    几乎全是乱码（实测「09 ： 34 = 二 000 《应鬲。一川。十 0 心」）→ 又一处假绿。
    if backend == "vlm" and stats["escalated"] > 0:
        if stats["vlm_ok"] == 0:
            msg = (f"VLM 后端 0 帧成功（{stats['escalated']} 帧送检全部失败），"
                   f"画面文字只剩 WinOCR（真实录播画面上实测大面积乱码）。拒绝写 frames_ocr.json。"
                   f"请查 docs/README.md §8「未解析出 JSON」一行与 §9 铁律一（find_dsh_cli）。")
            log.error(msg)
            raise RuntimeError(msg)
        if stats["vlm_ok"] < stats["frames"] * 0.5:
            log.warning("VLM 覆盖率偏低：%d/%d 帧成功（<50%%），未覆盖帧的画面文字可能不可用",
                        stats["vlm_ok"], stats["frames"])
    payload = {"course": course, "lecture": lecture, "backend": backend,
               "stats": stats, "frames": records}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("完成：%s | %s", out_path, json.dumps(stats, ensure_ascii=False))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 2d：关键帧文字理解")
    ap.add_argument("--course", required=True)
    ap.add_argument("--lecture", required=True)
    ap.add_argument("--backend", default="", choices=["", "winocr", "vlm", "hybrid"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    res = process_lecture(args.course, args.lecture, backend=args.backend, force=args.force)
    print("[s4_ocr]", json.dumps(res.get("stats", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
