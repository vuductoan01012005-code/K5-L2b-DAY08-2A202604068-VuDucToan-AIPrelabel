"""Gộp nhãn cuối (release candidate) sau duyệt + sửa QC.   python scripts/build_release.py

Nguồn: submission/labels/batch_reviewed.json (chỉ 24 ảnh batch). Nếu có submission/labels/qc_fixed.json
(COCO export sau khi sửa các ảnh lỗi từ QC), ảnh trong file đó THAY thế bản cũ.
Ghi: submission/release/final_det.coco.json, release_summary.csv (đếm theo lớp), release_manifest.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import SUB, build_coco, die, images_of, load_coco, load_taxonomy, make_ann, rel, save_json, write_csv


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    tax = load_taxonomy()
    src = SUB / "labels" / "batch_reviewed.json"
    if not src.exists():
        die("Chưa có submission/labels/batch_reviewed.json.")
    base = load_coco(src)
    fix_p = SUB / "labels" / "qc_fixed.json"
    fixed = load_coco(fix_p) if fix_p.exists() else None
    batch = images_of(["batch"])
    images, anns, per, aid = [], [], Counter(), 1
    replaced = []
    for iid, n in enumerate(batch, 1):
        use = fixed if (fixed and n in fixed.images) else base
        if use is fixed:
            replaced.append(n)
        if n not in use.images:
            die(f"Thiếu ảnh {n} trong nhãn đã duyệt.")
        w, h = use.images[n]
        images.append({"id": iid, "file_name": n, "width": w, "height": h})
        for o in use.objects[n]:
            conf = o.attrs.get("conf", "")
            anns.append(make_ann(tax, aid, iid, o.cls, o.box, o.polys, o.score,
                                 {"can_review": str(o.attrs.get("can_review", "")).lower() in ("true", "1", "yes"), "conf": conf}))
            per[o.cls] += 1
            aid += 1
    out = SUB / "release"
    save_json(out / "final_det.coco.json", build_coco(tax, images, anns, {"description": "Day08 release candidate", "taxonomy": tax.version}))
    write_csv(out / "release_summary.csv", ["class", "count"], [{"class": c, "count": per.get(c, 0)} for c in tax.classes])
    save_json(out / "release_manifest.json", {"taxonomy_version": tax.version, "images": len(images), "objects": len(anns),
                                              "per_class": dict(per), "images_replaced_by_qc_fix": replaced})
    print(f"Release candidate: {len(images)} ảnh, {len(anns)} đối tượng ({', '.join(f'{k}={v}' for k, v in per.most_common())})")
    if replaced:
        print(f"Đã thay {len(replaced)} ảnh bằng bản sửa QC: {', '.join(replaced)}")
    print(f"-> {rel(out)}/")


if __name__ == "__main__":
    main()
