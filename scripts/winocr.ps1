# Windows built-in OCR (Windows.Media.Ocr / WinRT) - batch mode, JSON output.
# Runs under Windows PowerShell 5.1 (pwsh 7 lacks the WinRT projection).
# ASCII-only on purpose: a .ps1 with non-ASCII comments and no BOM breaks PS 5.1 parsing.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File winocr.ps1 -ListFile frames.txt -OutJson ocr.json
#   powershell -NoProfile -ExecutionPolicy Bypass -File winocr.ps1 -ImagePath x.jpg
#
# Output JSON (UTF-8 with BOM, read it with encoding='utf-8-sig'):
#   { "language": "...", "engine_ms_total": N, "results": [ {"path":..., "ok":true, "ms":N,
#       "text":"...", "lines":[{"text":"...", "words":N}] } ] }

param(
  [string]$ListFile = "",
  [string]$OutJson = "",
  [string]$ImagePath = "",
  [string]$LanguageTag = "zh-Hans-CN"
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName System.Runtime.WindowsRuntime | Out-Null
[void][Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
[void][Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
[void][Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
  })[0]

function Await($WinRtTask, $ResultType) {
  $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
  $netTask = $asTask.Invoke($null, @($WinRtTask))
  $netTask.Wait(-1) | Out-Null
  $netTask.Result
}

function Get-Engine([string]$tag) {
  $e = $null
  try {
    $lang = New-Object Windows.Globalization.Language $tag
    $e = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
  } catch { }
  if (-not $e) { $e = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages() }
  return $e
}

$engine = Get-Engine $LanguageTag
if (-not $engine) { Write-Output "FATAL: no OCR engine available"; exit 2 }

function Invoke-Ocr([string]$path, $engine) {
  $t0 = Get-Date
  $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
  $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
  $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
  $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
  $res = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
  $lines = @()
  foreach ($ln in $res.Lines) {
    $lines += [pscustomobject]@{ text = $ln.Text; words = @($ln.Words).Count }
  }
  $stream.Dispose()
  return [pscustomobject]@{
    path  = $path
    ok    = $true
    ms    = [int]((Get-Date) - $t0).TotalMilliseconds
    text  = $res.Text
    lines = $lines
  }
}

# ---- single image mode ----
if ($ImagePath -and (Test-Path -LiteralPath $ImagePath)) {
  Write-Output ("AvailableRecognizerLanguages: " + (([Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages | ForEach-Object { $_.LanguageTag }) -join ", "))
  Write-Output ("engine = " + $engine.RecognizerLanguage.LanguageTag)
  $r = Invoke-Ocr $ImagePath $engine
  Write-Output ("elapsed_ms = " + $r.ms + " | lines = " + @($r.lines).Count)
  Write-Output "--- text ---"
  Write-Output $r.text
  Write-Output "--- end ---"
  exit 0
}

# ---- batch mode ----
if (-not $ListFile -or -not (Test-Path -LiteralPath $ListFile)) {
  Write-Output "no -ListFile/-ImagePath given; language probe only"
  Write-Output ("AvailableRecognizerLanguages: " + (([Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages | ForEach-Object { $_.LanguageTag }) -join ", "))
  exit 0
}

$paths = Get-Content -LiteralPath $ListFile -Encoding UTF8 | Where-Object { $_.Trim() -ne "" }
$results = @()
$total = 0
foreach ($p in $paths) {
  try {
    $r = Invoke-Ocr $p.Trim() $engine
    $total += $r.ms
    $results += $r
  } catch {
    $results += [pscustomobject]@{ path = $p.Trim(); ok = $false; ms = 0; text = ""; error = $_.Exception.Message; lines = @() }
  }
}
$payload = [pscustomobject]@{
  language         = $engine.RecognizerLanguage.LanguageTag
  engine_ms_total  = $total
  count            = @($results).Count
  results          = $results
}
$json = $payload | ConvertTo-Json -Depth 6
if ($OutJson) {
  [System.IO.File]::WriteAllText($OutJson, $json, (New-Object System.Text.UTF8Encoding $true))
  Write-Output ("wrote " + $OutJson + " | images=" + $payload.count + " | total_ms=" + $total)
} else {
  Write-Output $json
}
