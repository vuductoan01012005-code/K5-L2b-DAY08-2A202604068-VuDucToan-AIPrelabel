"""So hai bộ nhãn của CÙNG các ảnh: nhãn cuối của bạn vs bản gán lại mù (hoặc nhãn của bạn học). Dùng cho calibration.

    python scripts/self_consistency.py --a submission/labels/batch_reviewed.json \
        --b submission/labels/self_check_blind.json --images-from b

Ghép box 1-1 theo IoU (không phân biệt lớp, ngưỡng match_iou), rồi đo:
  IoU trung bình của các cặp ghép được; % cùng lớp; số box không ghép được (bỏ sót/thừa).
Đạt khi IoU TB >= self_iou_min và cùng lớp >= self_class_min (xem taxonomy.rules).
Ghi submission/qc/self_check.md/.json và self_check_cases.csv (từng ca lệch -> ứng viên calibration).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import SUB, die, load_coco, load_taxonomy, match_objects, rel, split_ignore, write_csv, covered_by


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="bộ nhãn tham chiếu (ví dụ nhãn cuối)")
    ap.add_argument("--b", required=True, help="bộ nhãn so sánh (ví dụ gán lại mù, hoặc nhãn của bạn học)")
    ap.add_argument("--images-from", choices=["a", "b", "both"], default="b", help="lấy tập ảnh từ file nào (mặc định b)")
    ap.add_argument("--mode", choices=["det", "seg"], default="det")
    ap.add_argument("--out", default=str(SUB / "qc" / "self_check"))
    a = ap.parse_args()
    tax = load_taxonomy()
    r = tax.rules
    if Path(a.a).resolve() == Path(a.b).resolve():
        die("--a và --b là cùng một file: IoU luôn 1. B phải là bộ nhãn gán lại mù (hoặc của bạn học).")
    A, B = load_coco(Path(a.a)), load_coco(Path(a.b))
    if a.images_from == "b":
        imgs = sorted(n for n in B.images if n in A.images)
    elif a.images_from == "a":
        imgs = sorted(n for n in A.images if n in B.images)
    else:
        imgs = sorted(set(A.images) & set(B.images))
    if not imgs:
        die("Hai file không có ảnh chung.")
    ious, same, tot, ua_n, ub_n, cases = [], 0, 0, 0, 0, []
    for n in imgs:
        PA, igA = split_ignore(tax, A.objects[n])
        PB, igB = split_ignore(tax, B.objects[n])
        ign = [o.box for o in igA + igB]
        pairs, ua, ub = match_objects(PA, PB, A.images[n], r["match_iou"], a.mode == "seg", same_class=False)
        ua = [o for o in ua if not covered_by(o.box, ign)]
        ub = [o for o in ub if not covered_by(o.box, ign)]
        for x, y, v in pairs:
            ious.append(v); tot += 1
            if x.cls == y.cls:
                same += 1
                if v < r["self_iou_min"]:
                    cases.append({"image": n, "issue": "iou_thap", "a_cls": x.cls, "b_cls": y.cls, "iou": round(v, 2), "note": ""})
            else:
                cases.append({"image": n, "issue": "sai_lop", "a_cls": x.cls, "b_cls": y.cls, "iou": round(v, 2), "note": ""})
        for o in ua:
            cases.append({"image": n, "issue": "chi_co_o_A", "a_cls": o.cls, "b_cls": "", "iou": "", "note": "B bỏ sót hoặc A thừa"})
        for o in ub:
            cases.append({"image": n, "issue": "chi_co_o_B", "a_cls": "", "b_cls": o.cls, "iou": "", "note": "A bỏ sót hoặc B thừa"})
        ua_n += len(ua); ub_n += len(ub)
    mi = sum(ious) / len(ious) if ious else None
    ca = same / tot if tot else None
    unm = (ua_n + ub_n) / (tot + ua_n + ub_n) if (tot + ua_n + ub_n) else None
    ok = bool(mi is not None and mi >= r["self_iou_min"] and ca is not None and ca >= r["self_class_min"] and unm is not None and unm <= r["self_unmatched_max"])
    res = {"unmatched_rate": unm, "images": len(imgs), "pairs": tot, "mean_iou": mi, "class_agreement": ca, "only_in_a": ua_n, "only_in_b": ub_n, "pass": ok}
    L = ["# Độ nhất quán giữa hai bộ nhãn", "", f"- A: `{rel(Path(a.a))}`; B: `{rel(Path(a.b))}`; {len(imgs)} ảnh chung",
         "| Chỉ số | Giá trị | Mẫu số | Ngưỡng |", "|---|---|---|---|",
         f"| IoU trung bình | {'n/a' if mi is None else '%.3f' % mi} | {tot} cặp box ghép được | ≥ {r['self_iou_min']} |",
         f"| Cùng lớp | {'n/a' if ca is None else '%.1f%%' % (ca * 100)} | {tot} cặp box ghép được | ≥ {r['self_class_min'] * 100:.0f}% |",
         f"| Tỉ lệ box không ghép được (A hoặc B bỏ sót/thừa) | {'n/a' if unm is None else '%.1f%%' % (unm * 100)} | {tot + ua_n + ub_n} box | ≤ {r['self_unmatched_max'] * 100:.0f}% |", f"| Chỉ có ở A (B bỏ sót / A thừa) | {ua_n} | | |", f"| Chỉ có ở B (A bỏ sót / B thừa) | {ub_n} | | |", "",
         f"**{'ĐẠT' if ok else 'CHƯA ĐẠT'}**", "",
         "Mỗi dòng trong `self_check_cases.csv` là một ứng viên calibration: hỏi *guideline mơ hồ* hay *tôi bất cẩn*?"]
    text = "\n".join(L) + "\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".md").write_text(text, encoding="utf8")
    Path(str(out) + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
    write_csv(Path(str(out) + "_cases.csv"), ["image", "issue", "a_cls", "b_cls", "iou", "note"], cases)
    print(text)
    print(f"Đã ghi {rel(Path(str(out) + '.md'))}, .json, _cases.csv")


if __name__ == "__main__":
    main()
