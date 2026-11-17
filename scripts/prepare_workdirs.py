"""Copy ảnh vào các thư mục làm việc để upload lên CVAT (mỗi thư mục = một Task).

    python scripts/prepare_workdirs.py              # bài chính: gold + review
    python scripts/prepare_workdirs.py --advanced   # thêm pilot / seg / selfcheck (docs/ADVANCED.md)

Tạo (trong work/tasks/, không commit):
  gold/       4 ảnh gold              -> Task "gold-manual"   (gán tay, KHÔNG AI)
  review/     24 ảnh batch            -> Task "batch-review"  (pre-label + duyệt)
Chỉ với --advanced:
  pilot/      8 ảnh pilot             -> Task "pilot"
  seg/        8 ảnh batch (seg = 1)   -> Task "seg-review"
  selfcheck/  8 ảnh batch ngẫu nhiên (bỏ ảnh quá dày, không thuộc seg) -> Task "selfcheck-blind"
"""
from __future__ import annotations

import argparse
import random
import shutil

from common import IMAGES_DIR, WORK, load_split, rel

DENSE = {"img_005.jpg", "img_018.jpg", "img_021.jpg", "img_024.jpg", "img_028.jpg", "img_029.jpg"}  # > ~50 vật


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--advanced", action="store_true", help="tạo thêm pilot / seg / selfcheck")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    sp = load_split()
    plan = {
        "gold": [k for k, v in sp.items() if v["split"] == "gold"],
        "review": [k for k, v in sp.items() if v["split"] == "batch"],
    }
    if a.advanced:
        plan["pilot"] = [k for k, v in sp.items() if v["split"] == "pilot"]
        plan["seg"] = [k for k, v in sp.items() if v["split"] == "batch" and v["seg"]]
        pool = sorted(k for k, v in sp.items() if v["split"] == "batch" and not v["seg"] and k not in DENSE)
        plan["selfcheck"] = sorted(random.Random(a.seed).sample(pool, 8))
    root = WORK / "tasks"
    for name, files in plan.items():
        d = root / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for f in sorted(files):
            shutil.copy2(IMAGES_DIR / f, d / f)
        print(f"{rel(d)}/  {len(files)} ảnh: " + " ".join(sorted(f[4:7] for f in files)))


if __name__ == "__main__":
    main()
