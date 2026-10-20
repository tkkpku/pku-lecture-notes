"""直连 LLM 客户端（OpenAI 兼容 `/chat/completions`）——**不经过 dsh**。

════════════════════════════════════════════════════════════════════════
由来（2026-09-26 你钦定改造）
════════════════════════════════════════════════════════════════════════
原先 `s4_ocr` / `s5_fuse` 都通过 `dsh --profile headless "<prompt>"` 调模型，
而 **dsh headless 是完整 agent**，于是每次调用都：
  ① 背着一整套 system prompt + 全部工具 schema（README §10 实测 ~25K token 的**固定开销**）；
  ② 在 `{DSH_HOME}\\sessions\\<项目>\\` 落一个**会话目录** —— 实测一天批跑就堆出 400+ 个、
     380 MB，把 Web UI 的会话列表刷爆（你 2026-09-26 反馈「已经堆了 500 多条」）。
改为直连 OpenAI 兼容端点后：固定开销只剩 prompt 本身，**且不产生任何 dsh 会话**。

════════════════════════════════════════════════════════════════════════
配置（`config/config.yaml` 的 `llm:` 段）
════════════════════════════════════════════════════════════════════════
  mode: direct | dsh        # direct=直连（默认）；dsh=回退旧路径（端点变更/调试时用）
  base_url / key_env / model / timeout_sec / max_retries
  max_image_side / jpeg_quality      # 图片缩放与编码（控制请求体大小）

════════════════════════════════════════════════════════════════════════
密钥（**按优先级取，绝不回显**）
════════════════════════════════════════════════════════════════════════
  ① 环境变量 `{key_env}`
  ② `{DSH_HOME}\\.credentials.yaml` 的 `refs.{key_env}`（本机实际走这条：env 未设置）
  代码只把密钥放进 Authorization 头；异常信息里**只报键名，不带值**。
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import httpx
from PIL import Image

DEFAULT_DSH_HOME: Path | None = None   # 只有旧路径 llm.mode: dsh 才用；开源版不留个人默认值


# ---------------------------------------------------------------- 配置解析
def dsh_home() -> Path | None:
    p = (os.environ.get("DSH_HOME") or "").strip()
    return Path(p) if p else DEFAULT_DSH_HOME


def load_api_key(key_env: str) -> str:
    """取密钥：环境变量 → （可选）DSH_HOME/.credentials.yaml 的 refs。

    开源版优先只认环境变量；.credentials.yaml 是给 DeepSeek Harness 用户留的兼容路径。
    **任何日志/异常都不含密钥值**。
    """
    v = (os.environ.get(key_env) or "").strip()
    if v:
        return v
    home = dsh_home()
    cred = (home / ".credentials.yaml") if home else None
    if cred is not None and cred.exists():
        import yaml

        data = yaml.safe_load(cred.read_text(encoding="utf-8-sig")) or {}
        refs = data.get("refs") or {}
        v = str(refs.get(key_env) or "").strip()
        if v:
            return v
    hint = ("（PowerShell: $env:%s='sk-...' ；bash: export %s=sk-...）" % (key_env, key_env))
    raise RuntimeError(
        f"未找到 API 密钥：环境变量 {key_env} 未设置"
        + (f"，且 {cred} 的 refs.{key_env} 也取不到" if cred is not None else hint)
    )


def resolve(cfg: dict) -> dict:
    """从 cfg['llm'] 解析连接参数（缺项用安全默认）。"""
    l = cfg.get("llm") or {}
    key_env = str(l.get("key_env") or "LLM_API_KEY")
    return {
        "base_url": str(l.get("base_url") or "https://api.deepseek.com/v1").rstrip("/"),
        "key_env": key_env,
        "api_key": load_api_key(key_env),
        "model": str(l.get("model") or "deepseek-chat"),
        "timeout": float(l.get("timeout_sec") or 600),
        "retries": int(l.get("max_retries") or 2),
        "max_image_side": int(l.get("max_image_side") or 1280),
        "jpeg_quality": int(l.get("jpeg_quality") or 85),
    }


def enabled(cfg: dict) -> bool:
    """是否走直连（mode: direct）。"""
    return str((cfg.get("llm") or {}).get("mode", "direct")).lower() == "direct"


# ---------------------------------------------------------------- 图片编码
def image_data_url(path: str | Path, max_side: int = 1280, quality: int = 85) -> str:
    """图片 → `data:image/jpeg;base64,...`（按最长边等比缩放，控制请求体大小）。

    本机关键帧宽 1280（config.frames.resize_width），所以 1280 基本是原尺寸；
    缩放只对异常大的图（如手机拍的截图）生效。
    """
    p = Path(path)
    with Image.open(p) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_user_content(prompt: str, images: list[str | Path] | None, params: dict) -> list[dict]:
    """构造 OpenAI 多模态 user content（文本 + 若干 image_url）。"""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in images or []:
        url = image_data_url(img, params["max_image_side"], params["jpeg_quality"])
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


# ---------------------------------------------------------------- 调用
def chat(messages: list[dict], cfg: dict, log=None) -> str:
    """发一次 `/chat/completions`，返回 assistant 文本（含重试）。

    [warn] 与 dsh 路径的关键差别：**没有** system prompt / 工具 schema / 会话落盘。
    """
    params = resolve(cfg)
    url = f"{params['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {params['api_key']}",   # 密钥只在此处出现，绝不入日志
        "Content-Type": "application/json",
    }
    body = {"model": params["model"], "messages": messages}

    last_err: Exception | None = None
    for attempt in range(1, params["retries"] + 2):
        t0 = time.time()
        try:
            with httpx.Client(timeout=params["timeout"]) as cli:
                r = cli.post(url, headers=headers, json=body)
            dt = time.time() - t0
            if r.status_code != 200:
                # 不回显响应体全文（可能含服务端细节）；只留前 300 字符
                snippet = (r.text or "")[:300]
                raise RuntimeError(f"HTTP {r.status_code}：{snippet}")
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"响应无 choices：{str(data)[:200]}")
            msg = choices[0].get("message") or {}
            text = (msg.get("content") or "").strip()
            if not text:
                raise RuntimeError(f"响应 content 为空（finish_reason={choices[0].get('finish_reason')}）")
            if log:
                usage = data.get("usage") or {}
                log.debug("直连 LLM：%.1fs｜in=%s out=%s｜model=%s", dt,
                          usage.get("prompt_tokens"), usage.get("completion_tokens"), params["model"])
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            if log:
                log.warning("直连 LLM 第 %d/%d 次失败：%s", attempt, params["retries"] + 1, e)
            if attempt <= params["retries"]:
                time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"直连 LLM 连续 {params['retries'] + 1} 次失败：{last_err}")


def chat_text(prompt: str, cfg: dict, log=None) -> str:
    """纯文本调用（s5_fuse 用）。"""
    return chat([{"role": "user", "content": prompt}], cfg, log)


def chat_with_images(prompt: str, images: list[str | Path], cfg: dict, log=None) -> str:
    """多图调用（s4_ocr 用）：把图直接 base64 塞进 user content —— **不依赖任何工具**。"""
    params = resolve(cfg)
    content = build_user_content(prompt, images, params)
    return chat([{"role": "user", "content": content}], cfg, log)
