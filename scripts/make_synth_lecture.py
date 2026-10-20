"""生成「合成课程录像」——端到端演练用的假回放素材（不等真实 pku3b 下载）。

产物：`<out_dir>\\<lecture>.mp4`（1280x720 H.264 + AAC），内容 = N 张 PPT 幻灯片
（每张显示 slide_sec 秒）+ 每张对应的中文讲解语音（16kHz mono）。

设计意图：PPT 文本即 **ground truth**，可用于校验 OCR 正确率；语音文本可校验 ASR 正确率
与时间轴对齐（s5_fuse 的 anchor 模式）。

前置：TTS 语音由 PowerShell SAPI 预生成到 `<DATA_ROOT>\\cache\\tts\\seg1..N.wav`
（16kHz/mono/16bit），见 docs/README.md 的合成素材章节。

用法：
  python scripts/make_synth_lecture.py                       # 默认 6 张 × 20s
  python scripts/make_synth_lecture.py --slide-sec 30 --clean
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import (  # noqa: E402
    cache_dir,
    ensure_tools_on_path,
    load_config,
    raw_dir,
    run,
    tool_paths,
)

import numpy as np  # noqa: E402

def _pick_font() -> str:
    """找一个能画中文的字体；找不到就退回 PIL 内置字体（英文可读、中文会变方块）。"""
    cands = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for c in cands:
        if c.exists():
            return str(c)
    print("[warn] 没找到中文 TrueType 字体，退回 PIL 内置字体（中文会显示为方块，但链路仍能跑）")
    return ""


FONT_PATH = _pick_font()


def _font(size: int):
    if FONT_PATH:
        try:
            from PIL import ImageFont
            return ImageFont.truetype(FONT_PATH, size)
        except Exception:
            pass
    from PIL import ImageFont
    return ImageFont.load_default()

# 每张幻灯片 = (标题, [正文行], 是否标题页)
SLIDES: list[tuple[str, list[str], bool]] = [
    ("理论力学 · 第3讲", ["应力张量", "", "北京大学工学院", "2026-09-12"], True),
    ("一、应力矢量", ["T = lim(ΔA→0) ΔF / ΔA", "", "• 面元上的内力与面积之比",
                  "• 极限即该点沿该法向的应力矢量", "• 方向一般不与法线平行"], False),
    ("二、应力张量", ["σij  ( i, j = 1, 2, 3 )", "",
                  "• 九个分量：σ11 σ12 σ13 / σ21 σ22 σ23 / σ31 σ32 σ33",
                  "• i 为作用面法向，j 为应力方向"], False),
    ("三、对称性", ["σij = σji", "", "• 由角动量守恒得出",
                 "• 独立分量：9 → 6", "• 这个结论非常重要"], False),
    ("四、主应力", ["T 与 n 平行", "det(σ - λI) = 0", "",
                 "• 三个实特征值 λ1 ≥ λ2 ≥ λ3", "• 对应的方向即主方向"], False),
    ("五、作业", ["第三题：求给定应力状态的主应力与主方向", "",
                "• 下周一上课前交", "• 交到助教那里"], False),
]


def render_slides(out_dir: Path, size=(1280, 720)) -> list[Path]:
    from PIL import Image, ImageDraw, ImageFont

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (title, lines, is_title) in enumerate(SLIDES, start=1):
        img = Image.new("RGB", size, "white")
        d = ImageDraw.Draw(img)
        if is_title:
            f_big = _font( 72)
            f_mid = _font( 48)
            f_small = _font( 28)
            d.text((90, 180), title, font=f_big, fill=(20, 40, 90))
            d.text((90, 300), lines[0], font=f_mid, fill=(180, 40, 40))
            d.text((90, 480), lines[2], font=f_small, fill=(60, 60, 60))
            d.text((90, 530), lines[3], font=f_small, fill=(60, 60, 60))
        else:
            f_h = _font( 52)
            f_b = _font( 32)
            d.rectangle([0, 0, size[0], 100], fill=(20, 40, 90))
            d.text((60, 22), title, font=f_h, fill="white")
            y = 180
            for ln in lines:
                if ln:
                    d.text((90, y), ln, font=f_b, fill=(20, 20, 20))
                y += 70
            d.text((1100, 650), f"{i} / {len(SLIDES)}", font=_font( 24),
                   fill=(140, 140, 140))
        p = out_dir / f"slide_{i:02d}.jpg"
        img.save(p, quality=92)
        paths.append(p)
    return paths


def _read_wav(p: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(p)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1).astype("<i2")
    return sr, data


def build_audio(tts_dir: Path, slide_sec: float, out_wav: Path, n_slides: int,
                sr: int = 16000) -> Path:
    """把 N 段 TTS 按幻灯片窗口拼成整条音轨（每段起始留 0.5s 静音）。"""
    total = int(sr * slide_sec * n_slides)
    buf = np.zeros(total, dtype="<i2")

    missing = [i + 1 for i in range(n_slides) if not (tts_dir / f"seg{i + 1}.wav").exists()]
    if len(missing) == n_slides:
        # 全缺 → 直接报错，别产出一条全是静音的片子（下游音频预检必然判"无讲课语音"，
        # 用户会以为是链路坏了，其实是素材没造出来）
        raise RuntimeError(
            f"没有任何语音段：{tts_dir}\\seg*.wav 全缺。\n"
            "  合成素材需要先造语音，否则 ASR 阶段拿不到内容。\n"
            "  Windows 上跑： powershell -ExecutionPolicy Bypass -File scripts\\make_synth_tts.ps1\n"
            "  （用系统 SAPI 合成 16 kHz 中文语音；没有中文语音包时会退回英文音色）"
        )
    if missing:
        print(f"[warn] 缺 {len(missing)} 段语音（seg{missing}），这些窗口将留静音")
    for i in range(n_slides):
        p = tts_dir / f"seg{i + 1}.wav"
        if not p.exists():
            print(f"[warn] 缺少语音段 {p}，该窗口留静音")
            continue
        s, d = _read_wav(p)
        if s != sr:
            raise RuntimeError(f"{p.name} 采样率 {s} != {sr}，请用 SAPI 以 16kHz 重新生成")
        start = int(sr * (i * slide_sec + 0.5))
        win_end = int(sr * (i + 1) * slide_sec)
        end = min(start + len(d), win_end, total)
        buf[start:end] = d[: end - start]
        if start + len(d) > win_end:
            print(f"[warn] seg{i+1} 语音长度超出窗口，已截断（可加大 --slide-sec）")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(buf.tobytes())
    return out_wav


def build_video(slides: list[Path], audio: Path, slide_sec: float, out_mp4: Path,
                cfg: dict, work: Path) -> Path:
    ffmpeg = tool_paths(cfg)["ffmpeg"]
    work.mkdir(parents=True, exist_ok=True)
    seg_files = []
    for i, img in enumerate(slides, start=1):
        seg = work / f"seg_{i:02d}.mp4"
        run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-loop", "1", "-framerate", "25", "-t", str(slide_sec), "-i", str(img),
             "-vf", "scale=1280:720,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "23", "-an", str(seg)])
        seg_files.append(seg)
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in seg_files), encoding="utf-8")
    silent = work / "silent.mp4"
    run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", str(silent)])
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out_mp4)])
    for f in seg_files:
        f.unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    silent.unlink(missing_ok=True)
    return out_mp4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide-sec", type=float, default=20.0)
    ap.add_argument("--lecture", default="第01讲_应力张量")
    ap.add_argument("--course", default="合成课程")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--clean", action="store_true", help="清掉中间产物目录")
    args = ap.parse_args()

    cfg = load_config()
    ensure_tools_on_path(cfg)
    out_dir = Path(args.out_dir) if args.out_dir else raw_dir(cfg) / args.course
    work = cache_dir(cfg) / "synth_build"
    tts = cache_dir(cfg) / "tts"

    slides = render_slides(work / "slides")
    print(f"[slides] {len(slides)} 张 → {work / 'slides'}")
    audio = build_audio(tts, args.slide_sec, work / "audio_track.wav", len(slides))
    print(f"[audio] {audio} ({audio.stat().st_size} bytes, {args.slide_sec * len(slides):.0f}s)")
    mp4 = build_video(slides, audio, args.slide_sec, out_dir / f"{args.lecture}.mp4", cfg, work)
    gt = {"slides": [{"idx": i + 1, "title": t, "lines": ls} for i, (t, ls, _) in enumerate(SLIDES)],
          "slide_sec": args.slide_sec, "video": str(mp4), "tts_dir": str(tts)}
    (out_dir / f"{args.lecture}.ground_truth.json").write_text(
        json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[video] {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    print(f"[gt]    {out_dir / (args.lecture + '.ground_truth.json')}")
    if args.clean:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
        print("[clean] 中间产物已清")
    return 0


if __name__ == "__main__":
    sys.exit(main())
