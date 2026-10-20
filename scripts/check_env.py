"""环境探针：Windows 侧 venv 的 CUDA / ctranslate2 能力自检（只读，不下载模型）。

用途：在下载 large-v3 之前，先确认 Blackwell sm_120 上 CTranslate2 到底认哪些 compute_type。
跑法：<DATA_ROOT>\\venv\\Scripts\\python.exe scripts\\check_env.py
"""
import sys

print("== python ==", sys.version.replace("\n", " "), "|", sys.executable)

try:
    import ctranslate2

    print("== ctranslate2 ==", ctranslate2.__version__)
    try:
        n = ctranslate2.get_cuda_device_count()
        print("cuda_device_count =", n)
    except Exception as e:  # noqa: BLE001
        print("cuda_device_count FAILED:", type(e).__name__, e)
    for dev in ("cuda", "cpu"):
        try:
            types = sorted(ctranslate2.get_supported_compute_types(dev))
            print(f"supported_compute_types[{dev}] =", types)
        except Exception as e:  # noqa: BLE001
            print(f"supported_compute_types[{dev}] FAILED:", type(e).__name__, e)
except Exception as e:  # noqa: BLE001
    print("ctranslate2 import FAILED:", type(e).__name__, e)

try:
    import faster_whisper

    print("== faster_whisper ==", getattr(faster_whisper, "__version__", "?"))
except Exception as e:  # noqa: BLE001
    print("faster_whisper import FAILED:", type(e).__name__, e)

for mod in ("av", "huggingface_hub", "numpy", "onnxruntime"):
    try:
        m = __import__(mod)
        print(f"== {mod} ==", getattr(m, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print(f"== {mod} == MISSING:", type(e).__name__)
