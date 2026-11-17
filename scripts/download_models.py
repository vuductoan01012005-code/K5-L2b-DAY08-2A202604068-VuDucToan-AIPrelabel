"""Tải trọng số model vào models/ (một lần).   python scripts/download_models.py --set core

Bộ:
  core   yolo26m, yoloe-26s-seg, sam2.1_b                       (~0.35 GB) bài chính, bắt buộc (làm TRƯỚC buổi học)
  small  yolo26n, yoloe-26s-seg, sam2.1_t                       (~0.15 GB) máy yếu / chỉ CPU
  hf     Grounding DINO tiny + OWLv2 base (HuggingFace)          (~2.0 GB)  NÂNG CAO (docs/ADVANCED.md)
  sam1 / sam1-base / sam2   chỉ tải SAM 1 (mobile_sam, sam_b) hoặc SAM 2.1 (t, s, b)
  sam3   facebook/sam3 (gated)                                    (~3.5 GB)  NÂNG CAO, cần xin quyền
  all    core + hf
YOLOE cần CLIP: lần chạy đầu ultralytics tự `pip install` CLIP từ git (cần git và mạng).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent.parent
SETS = {
    "core": ["yolo26m.pt", "yoloe-26s-seg.pt", "sam2.1_b.pt"],
    "small": ["yolo26n.pt", "yoloe-26s-seg.pt", "sam2.1_t.pt"],
    "sam1": ["mobile_sam.pt"],            # SAM 1 bản nhẹ (~40 MB)
    "sam1-base": ["sam_b.pt"],            # SAM 1 ViT-B (~375 MB)
    "sam2": ["sam2.1_t.pt", "sam2.1_s.pt", "sam2.1_b.pt"],
}
HF = {"hf": ["IDEA-Research/grounding-dino-tiny", "google/owlv2-base-patch16-ensemble"], "sam3": ["facebook/sam3"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="core", choices=["core", "small", "sam1", "sam1-base", "sam2", "hf", "sam3", "all"])
    a = ap.parse_args()
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    fails = []
    if a.set in SETS or a.set == "all":
        from ultralytics.utils.downloads import attempt_download_asset
        for n in SETS["core" if a.set == "all" else a.set]:
            p = models / n
            if p.exists():
                print(f"đã có {n}")
                continue
            try:
                got = attempt_download_asset(str(p))
                ok = Path(got).exists()
            except Exception as e:  # noqa
                ok, got = False, e
            print(("OK   " if ok else "LỖI ") + f"{n}" + ("" if ok else f": {got}"))
            if not ok:
                fails.append(n)
    if a.set in HF or a.set == "all":
        from huggingface_hub import snapshot_download
        for rid in HF["hf" if a.set == "all" else a.set]:
            try:
                snapshot_download(rid)
                print(f"OK   {rid}")
            except Exception as e:  # noqa
                print(f"LỖI {rid}: {str(e).splitlines()[0]}")
                fails.append(rid)
                if rid == "facebook/sam3":
                    print("     SAM 3 là model gated: xin quyền + đăng nhập HF trước (docs/SETUP.md). Bỏ qua được.")
    sys.exit(1 if fails and a.set != "sam3" else 0)


if __name__ == "__main__":
    main()
