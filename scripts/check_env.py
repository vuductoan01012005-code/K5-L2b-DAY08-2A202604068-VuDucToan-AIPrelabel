"""Kiểm tra môi trường trước khi bắt đầu lab.   python scripts/check_env.py [--cvat URL]

In PASS/WARN/FAIL cho: Python, thư viện, thiết bị tính toán (CUDA/MPS/CPU), git, dữ liệu, trọng số đã tải,
HuggingFace (cho SAM 3, tuỳ chọn), kết nối CVAT (nếu có --cvat). Không tải gì cả; chỉ kiểm.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent.parent
bad = 0


def line(level, msg, fix=""):
    global bad
    if level == "FAIL":
        bad += 1
    print(f"[{level:4}] {msg}" + (f"\n         -> {fix}" if fix else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvat", help="URL CVAT để thử kết nối, ví dụ http://localhost:8080")
    a = ap.parse_args()

    v = sys.version_info
    line("PASS" if (3, 10) <= v[:2] <= (3, 13) else "WARN", f"Python {v.major}.{v.minor}.{v.micro}",
         "" if (3, 10) <= v[:2] <= (3, 13) else "dùng Python 3.10-3.13 (khuyến nghị 3.12; 3.14+: torch/ultralytics/opencv có thể chưa có bản cài sẵn)")

    need = {"numpy": "numpy", "PIL": "pillow", "cv2": "opencv-python-headless", "matplotlib": "matplotlib",
            "yaml": "pyyaml", "torch": "torch", "ultralytics": "ultralytics", "transformers": "transformers"}
    for mod, pipname in need.items():
        try:
            m = importlib.import_module(mod)
            line("PASS", f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa
            line("FAIL", f"không import được {mod}: {type(e).__name__}", f"pip install -r requirements.txt  (thiếu {pipname})")

    try:
        import torch
        if torch.cuda.is_available():
            line("PASS", f"GPU CUDA: {torch.cuda.get_device_name(0)}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            line("PASS", "GPU Apple (MPS)", "OWLv2 tự chạy CPU trên Mac; nếu lỗi MPS: thêm --device cpu")
        else:
            line("WARN", "Không có GPU: dùng CPU", "vẫn làm được lab: dùng model cỡ n/s, chạy từng tập nhỏ (xem docs/SETUP.md)")
    except Exception:
        pass

    line("PASS" if shutil.which("git") else "FAIL", "git",
         "" if shutil.which("git") else "YOLOE cần git để tự cài CLIP; cài git rồi chạy lại")

    imgs = list((ROOT / "data" / "images").glob("*.jpg"))
    line("PASS" if len(imgs) == 36 else "FAIL", f"data/images: {len(imgs)}/36 ảnh",
         "" if len(imgs) == 36 else "clone lại repo, không sửa/xoá ảnh")
    line("PASS" if (ROOT / "data" / "split.csv").exists() else "FAIL", "data/split.csv")

    models = ROOT / "models"
    have = sorted(p.name for p in models.glob("*.pt")) if models.exists() else []
    line("PASS" if have else "WARN", "trọng số trong models/: " + (", ".join(have) if have else "(chưa có)"),
         "" if have else "python scripts/download_models.py --set core")

    try:
        from huggingface_hub import get_token
        tok = get_token()
        line("PASS" if tok else "WARN", "HuggingFace token: " + ("có" if tok else "chưa đăng nhập (chỉ cần cho SAM 3, tuỳ chọn)"),
             "" if tok else "xem docs/SETUP.md mục SAM 3")
    except Exception:
        line("WARN", "huggingface_hub chưa cài")

    if a.cvat:
        try:
            with urllib.request.urlopen(a.cvat.rstrip("/") + "/api/server/about", timeout=8) as r:
                line("PASS", f"CVAT trả lời ({r.status}): {a.cvat}")
        except Exception as e:  # noqa
            code = getattr(e, "code", None)
            line("PASS" if code in (401, 403) else "FAIL", f"CVAT {a.cvat}: {e}",
                 "" if code in (401, 403) else "kiểm tra URL / docker compose đã chạy chưa (docs/CVAT_AI.md)")
    print("\nSẴN SÀNG." if not bad else f"\n{bad} mục FAIL: sửa rồi chạy lại.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
