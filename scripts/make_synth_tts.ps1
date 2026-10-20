# ============================================================================
# make_synth_tts.ps1 —— 给"合成素材演练"造语音（Windows 内置 SAPI，无需联网）
# ============================================================================
# 为什么需要这一步：合成素材 = PPT 画面 + **语音**。只有画面没有声音，
# 音频预检（s2b_probe）会判定"该素材没有讲课语音"并直接拒绝，整条链路跑不下去。
#
# 用法（在仓库根目录）:
#     powershell -ExecutionPolicy Bypass -File scripts\make_synth_tts.ps1
#     # 换输出目录：-OutDir "D:\data\cache\tts"
#
# 产物：<data_root>/cache/tts/seg1..6.wav（16 kHz / 单声道 / 16-bit）
# 说明：这是**合成的假课堂语音**，只用于验证链路，不用于生产。
# ============================================================================

param(
    [string]$OutDir = "",
    [int]$Rate = 0,          # SAPI 语速：-10 慢 .. 0 正常 .. 10 快
    [int]$SlideCount = 6
)

$ErrorActionPreference = "Stop"

# 1) 决定输出目录：默认 <项目根>/data/cache/tts（与 config.example.yaml 的 data_root 对齐）
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $root "data\cache\tts"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# 2) 六句话，对应 make_synth_lecture.py 里的六张幻灯片
$lines = @(
    "这一讲我们来讨论应力张量。",
    "首先定义一点处的应力矢量：它是面元上的内力与面积之比，取面元趋于零的极限。",
    "应力有九个分量，分别由作用面法向和应力方向决定。",
    "由角动量守恒可以证明，应力张量是对称的，独立分量从九个减少到六个。",
    "主应力就是让应力矢量与法线平行的三个特征值，按大小排列。",
    "第三题求给定应力状态的主应力与主方向，下周一上课前交给助教。"
)

# 3) 选音色：优先中文女声，没有就随便挑一个（英文音色读中文会很怪，但链路能通）
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices() | Where-Object { $_.Enabled }
$zh = $voices | Where-Object { $_.VoiceInfo.Culture.Name -like "zh*" } | Select-Object -First 1
if ($zh) {
    $synth.SelectVoice($zh.VoiceInfo.Name)
    Write-Host "[i] 音色：$($zh.VoiceInfo.Name)"
} else {
    Write-Host "[warn] 没装中文语音包，用默认音色读中文（会很怪，但能把链路跑通）"
    Write-Host "      想装：设置 → 时间和语言 → 语音 → 添加中文语音包"
}
$synth.Rate = $Rate

# 4) 逐段合成
for ($i = 0; $i -lt $SlideCount; $i++) {
    $text = if ($i -lt $lines.Count) { $lines[$i] } else { "这是第 $($i + 1) 页。" }
    $path = Join-Path $OutDir ("seg{0}.wav" -f ($i + 1))

    # SAPI 直接写 wav，但采样率跟音色走（通常 22.05k/16k）。为对齐 16 kHz，
    # 先存临时文件，再用 ffmpeg 重采样；没有 ffmpeg 就保留原采样率并提示。
    $tmp = Join-Path $OutDir ("_tmp_seg{0}.wav" -f ($i + 1))
    $synth.SetOutputToWaveFile($tmp)
    $synth.Speak($text)
    $synth.SetOutputToNull()

    $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpeg) {
        & ffmpeg -y -loglevel error -i $tmp -ar 16000 -ac 1 -c:a pcm_s16le $path
        Remove-Item $tmp -Force
    } else {
        Move-Item -Force $tmp $path
        Write-Host "[warn] 没找到 ffmpeg，seg$($i+1).wav 保留原始采样率；pipeline 的音频预检可能报采样率不符"
    }
    Write-Host "[ok] seg$($i + 1).wav"
}

$synth.Dispose()
Write-Host ""
Write-Host "[✓] 语音段已生成到：$OutDir"
Write-Host "    下一步：python scripts/make_synth_lecture.py --lecture 第01讲_演示"
