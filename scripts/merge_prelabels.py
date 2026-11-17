"""Gộp nhiều file pre-label (ví dụ YOLO26 cho 6 lớp COCO + YOLOE cho three_wheeler) thành một file.

Ưu tiên theo THỨ TỰ file trên dòng lệnh (file đầu = ưu tiên cao nhất). Một box của file ưu tiên thấp bị bỏ nếu
chồng lên (IoU >= --iou, không phân biệt lớp) một box đã giữ từ file ưu tiên cao hơn.
Box trong cùng một file không bị loại lẫn nhau.

Ngưỡng được áp TRƯỚC khi loại trùng (--conf mặc định, --thr theo lớp), nên chạy model ở ngưỡng thấp (0.05) một lần rồi
gộp nhiều lần với ngưỡng khác nhau mà không cần chạy lại model. --split giới hạn ảnh (pilot / gold / batch).

Ví dụ (three_wheeler của YOLOE thắng "car/truck" của YOLO26 khi chúng chồng nhau):
  python scripts/merge_prelabels.py work/prelabel/yoloe_batch/pre.coco.json work/prelabel/y26_batch/pre.coco.json \
      --iou 0.6 --conf 0.3 --thr "person=0.4,three_wheeler=0.25" --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import build_coco, die, images_of, load_coco, load_taxonomy, make_ann, box_iou, rel, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="các file pre.coco.json, ưu tiên giảm dần")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng mặc định")
    ap.add_argument("--thr", help="ngưỡng riêng theo lớp, ví dụ person=0.4,three_wheeler=0.25")
    ap.add_argument("--split", help="chỉ giữ ảnh thuộc các tập này (pilot, gold, batch, seg), ví dụ batch,gold")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tax = load_taxonomy()
    sets = [load_coco(Path(f)) for f in a.files]
    thr = {}
    for part in (a.thr or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            die(f"--thr: sai định dạng '{part}': cần dạng lop=nguong (ví dụ person=0.4).")
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in tax.classes:
            die(f"--thr: lớp '{k}' không có trong taxonomy.")
        try:
            thr[k] = float(v.strip())
        except ValueError:
            die(f"--thr: ngưỡng '{v.strip()}' của lớp '{k}' phải là số thực (ví dụ 0.4).")
    names = sorted({n for s in sets for n in s.images})
    if a.split:
        names = [n for n in names if n in set(images_of([s.strip() for s in a.split.split(",") if s.strip()]))]
    images, anns, dropped = [], [], Counter()
    aid = 1
    for iid, name in enumerate(names, 1):
        size = next(s.images[name] for s in sets if name in s.images)
        images.append({"id": iid, "file_name": name, "width": size[0], "height": size[1]})
        kept = []  # (file_idx, Obj)
        for fi, s in enumerate(sets):
            for o in sorted(s.objects.get(name, []), key=lambda o: -(o.score or 0)):
                if (o.score if o.score is not None else 1.0) < thr.get(o.cls, a.conf):
                    dropped[f"{Path(a.files[fi]).parent.name}:{o.cls}(dưới ngưỡng)"] += 1
                    continue
                if any(kf < fi and box_iou(o.box, ko.box) >= a.iou for kf, ko in kept):
                    dropped[f"{Path(a.files[fi]).parent.name}:{o.cls}"] += 1
                    continue
                kept.append((fi, o))
        for _, o in kept:
            anns.append(make_ann(tax, aid, iid, o.cls, o.box, o.polys, o.score, {"can_review": False}))
            aid += 1
    save_json(Path(a.out), build_coco(tax, images, anns, {"description": "merged prelabel", "sources": a.files}))
    per = Counter(tax.classes[x["category_id"] - 1] for x in anns)
    print(f"Gộp {len(a.files)} file -> {len(anns)} box ({', '.join(f'{k}={v}' for k, v in per.most_common())}) "
          f"-> {rel(Path(a.out))}")
    if dropped:
        print("Box bị loại (dưới ngưỡng, hoặc chồng lên box ưu tiên cao hơn): " + ", ".join(f"{k}×{v}" for k, v in dropped.most_common()))


if __name__ == "__main__":
    main()
