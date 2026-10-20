# 数据流与设计取舍

## 数据流

```
教学网回放
  s1_fetch（pku3b）
    → raw/<课程>/第NN讲_*.mp4
  s2b_probe（VAD，3 窗口 × 60 秒，判断有没有人说话）
  s2_extract（ffmpeg）
    → work/<课程>/<讲次>/audio.wav
    → work/<课程>/<讲次>/frames/frame_XXXX.jpg
    → work/<课程>/<讲次>/frames_index.json
  s3_asr（faster-whisper）
    → asr.json / asr.srt / asr.txt
  s4_ocr（WinOCR 或 VLM）
    → frames_ocr.json
  s5_fuse（LLM 分块）
    → lecture.md
  s6_publish
    → <库>/<notes_library>/<课程>/YYYY-MM-DD_第NN讲_主题.md
  s7_review（LLM 判定并生成）
    → <库>/<课程>/00_第N章复习_主题.md
```

`frames_index.json` 结构：`{course, lecture, video_meta, frames: [{idx, file, t_sec, source, phash}]}`。

`run_status.json` 结构：`{course, lecture, results: {<stage>: {ok, detail, error}}}`。看 `results.<stage>.ok`，不要只看退出码。

## 取舍

**关键帧为什么用 pHash 而不是 ffmpeg scene**
课堂 PPT 翻页时版面、背景、标题栏不变，只有文字变，scene 分数只有 0.01~0.05，官方阈值会漏检。改为每 2 秒采样并比较 pHash 汉明距离。同时只对主区域算哈希，避开教室小窗造成的大量废帧，但落盘仍是完整画面，因为 VLM 要读正文。

**ASR 为什么关掉 condition_on_previous_text**
默认开着时，片头一句幻觉会成为整片的提示，导致全片重复同一句话、真实内容被复读过滤连带丢弃。详见 docs/PITFALLS.md §1。

**复读检测为什么按文本长度分档**
短文本重复多为模型循环，长句重复多为同一段音频的重复解码，而那往往是真实讲课内容。

**融合为什么分块且块不能太大**
两个相反的约束：块越大调用次数越少、固定开销摊得越薄；但单次输出过长容易中段遗漏、结构松散，而融合阶段没有兜底重试。20 分钟一块约 6K 字，风险可接受。另外 `segment_max_chars` 防 Windows 32767 字符的命令行上限。

**LLM 为什么直连而不是调 agent**
经 agent 每次调用都要背系统提示词和全部工具定义（实测约 25K token 固定开销），还会落一次会话目录。直连 OpenAI 兼容端点后固定开销只剩提示词本身。

**发布为什么带安全阀**
`cleanup` 会删除原始视频。三层保护：讲义文件必须存在、字节数不小于 `min_note_bytes`、清理按已发布讲次名单执行而不是按通配符猜。

**幂等与身份闸门**
每阶段产物存在即跳过。`common.guard_work_identity()` 在进入阶段前校验 `meta.json` 的来源标识，防止两次不同来源的视频混进同一工作目录，导致 ASR 与关键帧时间轴错配。

## 扩展点

| 想做的事 | 改哪里 |
| --- | --- |
| 换 ASR 后端 | `s3_asr.py`，保持 `asr.json` 契约 |
| 换读图后端 | `s4_ocr.py` 与 `llm_client.py`，只要是 OpenAI 兼容端点 |
| 改讲义格式 | `s5_fuse.py` 的提示词模板、`s6_publish.py` 的 frontmatter |
| 只要草稿不要发布 | `paths.obsidian_vault` 留空，`publish` 会拒绝执行 |
| 不要章节复习 | `--stages` 里不含 `review` |
