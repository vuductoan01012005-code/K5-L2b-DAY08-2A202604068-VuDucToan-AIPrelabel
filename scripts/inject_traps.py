"""Cài "ảnh bẫy": chèn lỗi đã biết vào pre-label để đo bạn có bắt được lỗi của máy không (slide Day 08, phần 04).

    python scripts/inject_traps.py --pre work/prelabel/merged_review/pre.coco.json --student 22001234

Ghi:
  submission/prelabel/pre_traps_import.coco.json  <- file NẠP VÀO CVAT (chỉ ảnh batch; giống hệt pre-label thường, không đánh dấu bẫy)
  submission/prelabel/pre_traps.coco.json         <- đầy đủ (batch + gold), dùng cho review_stats --pre
  submission/prelabel/traps_key.json        <- đáp án: KHÔNG MỞ file này trước khi duyệt xong (honor system)

Bốn loại lỗi cài: missing (xoá một box đúng), extra (thêm box ma), wrong_class (đổi lớp), loose (làm box lệch/lỏng).
Bẫy chỉ cài trên ảnh thuộc tập `batch` (12 bẫy trên 12 trong 24 ảnh), không cài lên ảnh gold. Cùng --student + --seed cho cùng bộ bẫy.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

from common import (SUB, WORK, build_coco, die, images_of, load_coco, load_taxonomy, make_ann, box_iou, rel, save_json,
                    xyxy_to_xywh)

SWAP = {"car": ["truck", "three_wheeler"], "truck": ["car", "bus"], "motorbike": ["bicycle"], "bicycle": ["motorbike"],
        "person": ["bicycle"], "bus": ["truck"], "three_wheeler": ["car", "motorbike"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", required=True, help="pre.coco.json (đã gộp) của tập batch")
    ap.add_argument("--student", required=True, help="MSHV của bạn (làm hạt giống ngẫu nhiên)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=12, help="số bẫy (chia đều 4 loại)")
    ap.add_argument("--out", default=str(SUB / "prelabel" / "pre_traps.coco.json"))
    ap.add_argument("--key", default=str(SUB / "prelabel" / "traps_key.json"))
    a = ap.parse_args()
    tax = load_taxonomy()
    cs = load_coco(Path(a.pre))
    batch = set(images_of(["batch"]))
    rng = random.Random(int(hashlib.sha256(f"{a.student}:{a.seed}".encode()).hexdigest(), 16) % (2 ** 32))
    cand_imgs = [n for n in sorted(cs.images) if n in batch and len(cs.objects[n]) >= 2]
    if not cand_imgs:
        die("Không có ảnh batch nào trong file pre-label. Chạy prelabel.py với --split batch (hoặc all).")
    types = (["missing", "extra", "wrong_class", "loose"] * ((a.n + 3) // 4))[:a.n]
    rng.shuffle(types)
    # mỗi ảnh tối đa 1 bẫy để dễ đo; nếu thiếu ảnh thì cho phép lặp
    order = cand_imgs[:]
    rng.shuffle(order)
    plan = [(types[k], order[k % len(order)]) for k in range(a.n)]

    # sao chép đối tượng (có thể sửa)
    work = {n: [dict(cls=o.cls, box=list(o.box), polys=[list(p) for p in o.polys], score=o.score, id=o.id, orig=o.id)
                for o in cs.objects[n]] for n in cs.images}
    key = []
    nxt = max((o["id"] for v in work.values() for o in v), default=0) + 1

    def pick(n, pred):
        c = [o for o in work[n] if pred(o) and "trap" not in o]
        return rng.choice(c) if c else None

    for typ, n in plan:
        w, h = cs.images[n]
        if typ == "missing":
            o = pick(n, lambda o: (o["score"] or 0) >= 0.5 and o["cls"] != tax.ignore)
            if not o:
                continue
            work[n].remove(o)
            key.append({"image": n, "type": typ, "orig_box": xyxy_to_xywh(tuple(o["box"])), "orig_cls": o["cls"]})
        elif typ == "extra":
            for _ in range(60):
                bw, bh = rng.uniform(0.06, 0.18) * w, rng.uniform(0.10, 0.28) * h
                x, y = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
                box = (x, y, x + bw, y + bh)
                if all(box_iou(box, tuple(o["box"])) < 0.05 for o in work[n]):
                    cls = rng.choice(["car", "motorbike", "person", "truck"])
                    o = dict(cls=cls, box=list(box), polys=[], score=round(rng.uniform(0.45, 0.8), 2), id=nxt, orig=None,
                             trap="extra")
                    nxt += 1
                    work[n].append(o)
                    key.append({"image": n, "type": typ, "fake_box": xyxy_to_xywh(box), "fake_cls": cls, "id": o["id"]})
                    break
        elif typ == "wrong_class":
            o = pick(n, lambda o: o["cls"] in SWAP and (o["score"] or 0) >= 0.5)
            if not o:
                continue
            old = o["cls"]
            o["cls"] = rng.choice([c for c in SWAP[old] if c in tax.classes])
            o["trap"] = typ
            key.append({"image": n, "type": typ, "box": xyxy_to_xywh(tuple(o["box"])), "orig_cls": old,
                        "fake_cls": o["cls"], "id": o["id"]})
        elif typ == "loose":
            o = pick(n, lambda o: (o["box"][2] - o["box"][0]) > 40 and (o["box"][3] - o["box"][1]) > 40 and o["cls"] != tax.ignore)
            if not o:
                continue
            x1, y1, x2, y2 = o["box"]
            bw, bh = x2 - x1, y2 - y1
            orig = tuple(o["box"])
            for _ in range(80):
                s = rng.uniform(0.18, 0.35)
                dx, dy = rng.choice([-1, 1]) * s * bw * rng.uniform(0.3, 1), rng.choice([-1, 1]) * s * bh * rng.uniform(0.3, 1)
                nb = (x1 + dx, y1 + dy, x2 + dx * 0.4 + bw * s * rng.choice([-1, 1]) * 0.3, y2 + dy * 0.4)
                nb = (max(0, nb[0]), max(0, nb[1]), min(w, nb[2]), min(h, nb[3]))
                if 0.45 <= box_iou(orig, nb) <= 0.68:
                    o["box"] = list(nb)
                    o["polys"] = []  # polygon cũ không còn khớp
                    o["trap"] = typ
                    key.append({"image": n, "type": typ, "orig_box": xyxy_to_xywh(orig), "bad_box": xyxy_to_xywh(nb),
                                "cls": o["cls"], "id": o["id"]})
                    break
    images, anns, aid = [], [], 1
    id_map = {}
    for iid, n in enumerate(sorted(cs.images), 1):
        images.append({"id": iid, "file_name": n, "width": cs.images[n][0], "height": cs.images[n][1]})
        for o in work[n]:
            ann = make_ann(tax, aid, iid, o["cls"], tuple(o["box"]), o["polys"], o["score"], {"can_review": False})
            id_map[(n, o["id"])] = aid
            anns.append(ann)
            aid += 1
    for k in key:
        if "id" in k:
            k["ann_id"] = id_map.get((k["image"], k["id"]))
    save_json(Path(a.out), build_coco(tax, images, anns, {"description": "prelabel (đã cài bẫy)"}))
    # file NẠP VÀO CVAT: chỉ ảnh batch (Task review không có ảnh gold)
    ids = {im["id"] for im in images if im["file_name"] in batch}
    imp = Path(a.out).with_name("pre_traps_import.coco.json")
    save_json(imp, build_coco(tax, [im for im in images if im["id"] in ids], [x for x in anns if x["image_id"] in ids],
                              {"description": "prelabel đã cài bẫy, chỉ ảnh batch (nạp vào CVAT)"}))
    save_json(Path(a.key), {"student": a.student, "seed": a.seed, "traps": key})
    c = Counter(k["type"] for k in key)
    print(f"Đã cài {len(key)} bẫy ({', '.join(f'{k}={v}' for k, v in c.items())}) trên {len({k['image'] for k in key})} ảnh.")
    if len(key) < a.n:
        print(
            f"CẢNH BÁO: chỉ cài được {len(key)}/{a.n} bẫy "
            "(pre-label thiếu box đủ điều kiện: cần điểm >= 0.5 cho missing/wrong_class, kích thước > 40x40px cho loose).",
            file=sys.stderr,
        )
    print(f"-> NẠP VÀO CVAT: {rel(imp)}  (chỉ 24 ảnh batch)")
    print(f"-> Đầy đủ (dùng cho review_stats --pre): {rel(Path(a.out))}")
    print(f"-> Đáp án (đừng mở trước khi duyệt xong): {rel(Path(a.key))}")


if __name__ == "__main__":
    main()
