"""Đo chất lượng pre-label và chất lượng duyệt của bạn (slide Day 08, phần 04).

    python scripts/review_stats.py --pre submission/prelabel/pre_traps.coco.json \
        --final submission/labels/batch_reviewed.json --gold submission/labels/gold_manual.json \
        --key submission/prelabel/traps_key.json

Tính:
  1. Bốn trạng thái của mỗi box máy: accepted / edited / deleted, và added (vật bạn vẽ thêm).
       accepted: cùng lớp, IoU(box cuối, box máy) >= accept_iou (mặc định 0.9; mask: 0.85)
       edited  : cùng lớp, 0.5 <= IoU < accept_iou
       deleted : không ghép được (sai lớp hoặc IoU < 0.5 -> tính là deleted + added, như contract của slide)
  2. Nếu có --gold (nhãn tự gán tay từ đầu, tập gold): precision/recall của MÁY so với gold. Nếu --final có cả ảnh gold
     (làm tuỳ chọn, sau >= 24h): thêm nhãn sau duyệt, residual (lỗi máy bạn giữ lại) và new (lỗi bạn tạo ra).
     Lỗi nhân tạo của ảnh bẫy bị loại khỏi bảng 1 (chỉ tính ở mục 3).
  3. Nếu có --key: tỉ lệ bắt lỗi trên các ảnh bẫy.
Ghi submission/qc/review_stats.json và .md.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from common import (SUB, Obj, die, images_of, load_coco, load_taxonomy, match_objects, obj_iou, rel, split_ignore,
                    box_iou, covered_by)


def pc(a, b):
    return "n/a" if not b else f"{a}/{b} = {a / b * 100:.1f}%"


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def ci(k, n):
    w = wilson(k, n)
    return "" if w is None else f" (KTC 95%: {w[0] * 100:.0f}–{w[1] * 100:.0f}%)"


def rate(a, b):
    return None if not b else a / b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", required=True, help="COCO pre-label bạn đã nạp vào CVAT")
    ap.add_argument("--final", required=True, help="COCO export từ CVAT sau khi duyệt")
    ap.add_argument("--gold", help="COCO gold tự gán tay (ảnh gold)")
    ap.add_argument("--key", help="submission/prelabel/traps_key.json")
    ap.add_argument("--mode", choices=["det", "seg"], default="det", help="seg: so IoU mask thay vì IoU box")
    ap.add_argument("--splits", default="batch,gold", help="chỉ tính trên ảnh thuộc các tập này (mặc định batch,gold)")
    ap.add_argument("--out", default=str(SUB / "qc" / "review_stats"), help="tiền tố file đầu ra")
    a = ap.parse_args()

    tax = load_taxonomy()
    r = tax.rules
    use_mask = a.mode == "seg"
    acc_thr = r["accept_mask_iou"] if use_mask else r["accept_iou"]
    pre, fin = load_coco(Path(a.pre)), load_coco(Path(a.final))
    allowed = set(images_of([s.strip() for s in a.splits.split(",") if s.strip()]))
    imgs = sorted(n for n in fin.images if n in allowed)
    if not imgs:
        die("Không có ảnh chung nào giữa file final và các tập đã chọn. Kiểm tra --final và --splits.")
    missing_in_pre = [n for n in imgs if n not in pre.images]
    if missing_in_pre:
        die(f"File pre thiếu ảnh {missing_in_pre[:3]}...: --pre phải là file bạn đã nạp vào CVAT.")

    # vùng quanh các bẫy: lỗi nhân tạo không được tính vào "máy tốt đến đâu" (chỉ tính vào tỉ lệ bắt bẫy)
    trap_boxes: Dict[str, list] = defaultdict(list)
    keyd = None
    if a.key:
        keyd = json.loads(Path(a.key).read_text(encoding="utf-8-sig"))["traps"]
        for t in keyd:
            for fld in ("orig_box", "fake_box", "box", "bad_box"):
                if t.get(fld):
                    b = t[fld]
                    trap_boxes[t["image"]].append((b[0], b[1], b[0] + b[2], b[1] + b[3]))
    n_trap_excl = 0

    def near_trap(o, name):
        return any(box_iou(o.box, tb) >= 0.3 for tb in trap_boxes.get(name, []))

    states: Counter = Counter()
    per_cls: Dict[str, Counter] = defaultdict(Counter)
    n_pre = n_final = n_changed_cls = 0
    pairs_by_img = {}
    for n in imgs:
        size = fin.images[n]
        P, ignP = split_ignore(tax, pre.objects[n])
        F, ignF = split_ignore(tax, fin.objects[n])
        if trap_boxes.get(n):
            n0 = len(P) + len(F)
            P = [o for o in P if not near_trap(o, n)]
            F = [o for o in F if not near_trap(o, n)]
            n_trap_excl += n0 - len(P) - len(F)
        pairs, up, uf = match_objects(P, F, size, r["match_iou"], use_mask, same_class=True)
        pairs_by_img[n] = (pairs, up, uf)
        n_pre += len(P)
        n_final += len(F)
        for pa, fb, v in pairs:
            st = "accepted" if v >= acc_thr else "edited"
            states[st] += 1
            per_cls[pa.cls][st] += 1
        # vùng ignore: box máy nằm trong ignore_region của bạn không tính là deleted
        ign_boxes = [g.box for g in ignF]
        up = [pa for pa in up if not covered_by(pa.box, ign_boxes)]   # box máy nằm trong ignore_region của bạn: không tính là deleted
        uf = [fb for fb in uf if not covered_by(fb.box, ign_boxes)]
        for pa in up:
            states["deleted"] += 1
            per_cls[pa.cls]["deleted"] += 1
        for fb in uf:
            states["added"] += 1
            per_cls[fb.cls]["added"] += 1
        # thông tin phụ: đổi lớp (box máy bị xoá + box cuối thêm, chồng nhau IoU>=0.5)
        for pa in up:
            if any(fb.cls != pa.cls and obj_iou(pa, fb, size, use_mask) >= r["match_iou"] for fb in uf):
                n_changed_cls += 1
    acc, edi, dele, add = (states[k] for k in ("accepted", "edited", "deleted", "added"))

    # ---------------------------------------------------------------- gold
    gold_res = None
    if a.gold:
        gold = load_coco(Path(a.gold))
        gimgs = [n for n in gold.images if n in pre.images]
        if not gimgs:
            print("CẢNH BÁO: không có ảnh gold nào trong file pre (pre-label phải gồm cả ảnh gold: merge với --split \"batch,gold\").")
        else:
            fin_g = [n for n in gimgs if n in fin.images]   # ảnh gold có trong nhãn sau duyệt (không bắt buộc)
            tags = {"pre": gimgs, "fin": fin_g}
            tp = {"pre": 0, "fin": 0}; fp = {"pre": 0, "fin": 0}; fn = {"pre": 0, "fin": 0}
            n_gold = sum(len(split_ignore(tax, gold.objects[n])[0]) for n in gimgs)
            residual = new = 0
            for n in gimgs:
                G, ignG = split_ignore(tax, gold.objects[n])
                size = gold.images[n]
                gb = [g.box for g in ignG]
                keepP = [o for o in split_ignore(tax, pre.objects[n])[0] if not covered_by(o.box, gb)]
                res = {}
                prs = {"pre": keepP}
                if n in fin.images:
                    prs["fin"] = [o for o in split_ignore(tax, fin.objects[n])[0] if not covered_by(o.box, gb)]
                for tag, X in prs.items():
                    pr, ux, ug = match_objects(X, G, size, r["match_iou"], False, same_class=True)
                    tp[tag] += len(pr); fp[tag] += len(ux); fn[tag] += len(ug)
                    res[tag] = (ux, ug)
                if "fin" in res:
                    pre_fp, pre_fn = res["pre"]; fin_fp, fin_fn = res["fin"]
                    pre_fn_ids = {id(g) for g in pre_fn}
                    r_cnt = sum(1 for f in fin_fp if any(box_iou(f.box, p.box) >= 0.7 for p in pre_fp))
                    r_cnt += sum(1 for g in fin_fn if id(g) in pre_fn_ids)
                    residual += r_cnt
                    new += len(fin_fp) + len(fin_fn) - r_cnt
            def prf(t):
                return rate(tp[t], tp[t] + fp[t]), rate(tp[t], tp[t] + fn[t])
            gold_res = {"images": len(gimgs), "gold_objects": n_gold,
                        "pre": {"tp": tp["pre"], "fp": fp["pre"], "fn": fn["pre"], "precision": prf("pre")[0], "recall": prf("pre")[1]},
                        "final": ({"images": len(fin_g), "tp": tp["fin"], "fp": fp["fin"], "fn": fn["fin"], "precision": prf("fin")[0], "recall": prf("fin")[1]}
                                  if fin_g else None),
                        "residual_errors": residual if fin_g else None, "new_errors": new if fin_g else None}

    # ---------------------------------------------------------------- traps
    trap_res = None
    if keyd is not None:
        caught, rows = 0, []
        absent = sorted({t["image"] for t in keyd if t["image"] not in fin.objects})
        if absent:
            die(f"File --final thiếu {len(absent)} ảnh có bẫy ({', '.join(absent[:3])}...): bạn export nhầm Task hoặc thiếu ảnh. "
                "Không tính bẫy trên ảnh vắng mặt (sẽ làm tỉ lệ bắt bẫy sai).")
        for t in keyd:
            n = t["image"]
            F = split_ignore(tax, fin.objects[n])[0]
            def bx(b):
                return (b[0], b[1], b[0] + b[2], b[1] + b[3])
            ok = False
            if t["type"] == "missing":
                ok = any(f.cls == t["orig_cls"] and box_iou(f.box, bx(t["orig_box"])) >= 0.5 for f in F)
            elif t["type"] == "extra":
                ok = not any(box_iou(f.box, bx(t["fake_box"])) >= 0.5 for f in F)
            elif t["type"] == "wrong_class":
                ok = any(f.cls == t["orig_cls"] and box_iou(f.box, bx(t["box"])) >= 0.5 for f in F)
            elif t["type"] == "loose":
                # đúng nếu có box cùng lớp gần box gốc (>= 0.75), hoặc đã rời khỏi box lỗi (< 0.8) mà vẫn trùng vật (>= 0.5)
                ok = any(f.cls == t["cls"] and (box_iou(f.box, bx(t["orig_box"])) >= 0.75 or
                                                (box_iou(f.box, bx(t["bad_box"])) < 0.8 and box_iou(f.box, bx(t["orig_box"])) >= 0.5)) for f in F)
            caught += ok
            rows.append((t["image"], t["type"], ok))
        by_type = defaultdict(lambda: [0, 0])
        for _, ty, ok in rows:
            by_type[ty][1] += 1; by_type[ty][0] += ok
        trap_res = {"n": len(rows), "caught": caught, "rate": rate(caught, len(rows)),
                    "by_type": {k: v for k, v in by_type.items()}, "missed": [(i, t) for i, t, ok in rows if not ok]}

    # ---------------------------------------------------------------- báo cáo
    out = {"mode": a.mode, "images": len(imgs), "pre_boxes": n_pre, "final_objects": n_final, "accepted": acc,
           "edited": edi, "deleted": dele, "added": add, "class_changed_info": n_changed_cls,
           "accepted_rate": rate(acc, n_pre), "edited_rate": rate(edi, n_pre), "deleted_rate": rate(dele, n_pre),
           "added_rate": rate(add, n_final), "per_class": {k: dict(v) for k, v in per_cls.items()},
           "gold": gold_res, "traps": trap_res, "accept_threshold": acc_thr}
    L = [f"# Số đo review ({a.mode})", "",
         f"- Ảnh tính: {len(imgs)} ({a.splits}); box máy: {n_pre}; đối tượng sau duyệt: {n_final}; "
         f"ngưỡng accepted: IoU >= {acc_thr}",
         (f"- Đã **loại {n_trap_excl} đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy." if trap_boxes else ""), "",
         "## 1. Bốn trạng thái (AI error profile)", "| Trạng thái | Số | Tỉ lệ | Mẫu số |", "|---|---|---|---|",
         f"| accepted (giữ nguyên) | {acc} | {pc(acc, n_pre)} | {n_pre} box máy |",
         f"| edited (sửa) | {edi} | {pc(edi, n_pre)} | {n_pre} box máy |",
         f"| deleted (xoá) | {dele} | {pc(dele, n_pre)} | {n_pre} box máy |",
         f"| added (vẽ thêm) | {add} | {pc(add, n_final)} | {n_final} đối tượng sau duyệt |", "",
         f"Trong số deleted + added, **{n_changed_cls} trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. "
         f"Số vật máy **sót thật** ≈ added − đổi lớp = {max(0, add - n_changed_cls)}.", "",
         "### Theo lớp", "| Lớp | accepted | edited | deleted | added |", "|---|---|---|---|---|"]
    for c in tax.object_classes:
        if c in per_cls:
            v = per_cls[c]
            L.append(f"| {c} | {v['accepted']} | {v['edited']} | {v['deleted']} | {v['added']} |")
    if gold_res:
        g = gold_res
        f = lambda x: "n/a" if x is None else f"{x * 100:.1f}%"  # noqa: E731
        L += ["", "## 2. So với gold (nhãn gán tay từ đầu)", f"Ảnh gold: {g['images']}; đối tượng gold: {g['gold_objects']} (mẫu số của recall)",
              "| | TP | FP | FN | Precision | Recall |", "|---|---|---|---|---|---|",
              f"| Máy (pre-label) | {g['pre']['tp']} | {g['pre']['fp']} | {g['pre']['fn']} | {f(g['pre']['precision'])}{ci(g['pre']['tp'], g['pre']['tp'] + g['pre']['fp'])} | {f(g['pre']['recall'])}{ci(g['pre']['tp'], g['pre']['tp'] + g['pre']['fn'])} |",
              "", "*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*", ""]
        if g["final"]:
            L += [f"| Sau duyệt của bạn ({g['final']['images']} ảnh gold có trong nhãn cuối) | {g['final']['tp']} | {g['final']['fp']} | {g['final']['fn']} | {f(g['final']['precision'])} | {f(g['final']['recall'])} |", "",
                  f"- **Residual** (lỗi của máy còn trong nhãn của bạn): {g['residual_errors']} (đếm FP + FN; một lỗi sai lớp hoặc lệch vị trí tính 2)",
                  f"- **New** (lỗi do bạn tạo ra): {g['new_errors']}"]
        else:
            L += ["", "(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)"]
    if trap_res:
        t = trap_res
        L += ["", "## 3. Ảnh bẫy", f"- Bắt được {pc(t['caught'], t['n'])}{ci(t['caught'], t['n'])} (ngưỡng đạt: {r['trap_catch_min'] * 100:.0f}%). Chỉ {t['n']} bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %."]
        for ty, (c, n) in t["by_type"].items():
            L.append(f"  - {ty}: {c}/{n}")
        if t["missed"]:
            L.append("  - lọt: " + ", ".join(f"{i} ({ty})" for i, ty in t["missed"]))
    L += ["", "## Gợi ý đọc số", "- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);",
          "- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;",
          "- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy)."]
    text = "\n".join(L) + "\n"
    base = Path(a.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(str(base) + ".md").write_text(text, encoding="utf8")
    Path(str(base) + ".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    print(text)
    print(f"Đã ghi {rel(Path(str(base) + '.md'))} và .json")


if __name__ == "__main__":
    main()
