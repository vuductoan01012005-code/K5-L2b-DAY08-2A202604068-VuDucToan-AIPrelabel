"""Kiểm file COCO export từ CVAT trước khi dùng/commit.

    python scripts/validate_labels.py submission/labels/gold_manual.json --expect gold
    python scripts/validate_labels.py submission/labels/batch_reviewed.json --expect batch
    python scripts/validate_labels.py --all

--expect: các tập (pilot,gold,batch) mà file PHẢI chứa đủ ảnh. --all kiểm mọi file chuẩn trong submission/labels/.
Kiểm: tên lớp có trong taxonomy, box nằm trong ảnh, box quá nhỏ, đủ/thừa ảnh, polygon hợp lệ (--mode seg).
Không kiểm được: box có ôm sát vật không, lớp có đúng thực tế không.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import SUB, images_of, load_coco, load_split, load_taxonomy, rel

STANDARD = {  # file -> (tập ảnh phải có, có polygon?)
    "gold_manual.json": ("gold", False),
    "pilot_reviewed.json": ("pilot", False),
    "batch_reviewed.json": ("batch", False),
    "seg_reviewed.json": ("seg", True),
    "self_check_blind.json": ("", False),
}


def check(path: Path, expect: str, need_poly: bool, tax) -> int:
    errs, warns = [], []
    try:
        cs = load_coco(path)
    except Exception as e:  # noqa
        print(f"[{rel(path)}] LỖI: không đọc được JSON: {e}")
        return 1
    sp = load_split()
    if expect == "seg":
        want = sorted(k for k, v in sp.items() if v["split"] == "batch" and v["seg"])
    elif expect:
        want = images_of([s.strip() for s in expect.split(",") if s.strip()])
    else:
        want = []
    errs += cs.problems
    for n in want:
        if n not in cs.images:
            errs.append(f"thiếu ảnh {n}")
    extra = [n for n in cs.images if n not in sp]
    if extra:
        errs.append(f"ảnh không thuộc bộ dữ liệu: {extra[:3]}")
    if want:
        extra2 = [n for n in cs.images if n in sp and n not in want]
        if extra2:
            warns.append(f"có {len(extra2)} ảnh ngoài phạm vi mong đợi ({', '.join(extra2[:3])}...)")
    nobj = 0
    for n, objs in cs.objects.items():
        w, h = cs.images[n]
        for o in objs:
            nobj += 1
            if o.cls not in tax.classes:
                errs.append(f"{n}: lớp lạ '{o.cls}' (CVAT còn dùng bộ nhãn cũ? tạo lại Project từ cvat/labels_generated.json)")
                continue
            x1, y1, x2, y2 = o.box
            if x2 > w + 1 or y2 > h + 1 or x1 < -1 or y1 < -1:
                errs.append(f"{n}: box #{o.id} {o.cls} nằm ngoài ảnh")
            if o.cls != tax.ignore and (y2 - y1) < tax.rules["min_box_height_px"]:
                warns.append(f"{n}: box #{o.id} {o.cls} cao {y2 - y1:.1f}px < {tax.rules['min_box_height_px']}px (guideline: vật quá nhỏ dùng ignore_region)")
            if need_poly and o.cls != tax.ignore and not o.polys:
                errs.append(f"{n}: #{o.id} {o.cls} không có polygon (task segmentation phải vẽ polygon/mask, không phải rectangle)")
    for m in warns[:12]:
        print(f"[{rel(path)}] cảnh báo: {m}")
    if len(warns) > 12:
        print(f"[{rel(path)}] ... và {len(warns) - 12} cảnh báo nữa")
    for m in errs[:20]:
        print(f"[{rel(path)}] LỖI: {m}")
    if len(errs) > 20:
        print(f"[{rel(path)}] ... và {len(errs) - 20} lỗi nữa")
    print(f"[{rel(path)}] {len(cs.images)} ảnh, {nobj} đối tượng: {'LỖI ' + str(len(errs)) if errs else 'OK'}")
    return len(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?")
    ap.add_argument("--expect", default="")
    ap.add_argument("--mode", choices=["det", "seg"], default="det")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    tax = load_taxonomy()
    bad = 0
    if a.all:
        found = False
        for fn, (exp, poly) in STANDARD.items():
            p = SUB / "labels" / fn
            if p.exists():
                found = True
                bad += check(p, "seg" if fn == "seg_reviewed.json" else exp, poly, tax)
        if not found:
            print("Chưa có file nhãn nào trong submission/labels/.")
            sys.exit(1)
    elif a.file:
        bad = check(Path(a.file), a.expect, a.mode == "seg", tax)
    else:
        ap.error("cần <file> hoặc --all")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
