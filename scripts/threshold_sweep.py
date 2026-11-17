"""Quét ngưỡng tin cậy THEO TỪNG LỚP trên nhãn gán tay (gold), để chọn ngưỡng có cơ sở (thay vì đoán 0.5).

    # pre-label đã gộp ở ngưỡng thấp (0.10) cho ảnh gold, và nhãn gán tay của bạn:
    python scripts/threshold_sweep.py --pred work/prelabel/merged_gold/pre.coco.json \
        --gold submission/labels/gold_manual.json

Với mỗi lớp và mỗi ngưỡng t: tính precision / recall / F1 (IoU >= match_iou, cùng lớp).
Gợi ý ngưỡng theo lớp = ngưỡng CAO NHẤT mà recall còn >= --recall-target (ít box thừa nhất trong khi không bỏ sót thêm).
Lớp có < 5 vật gold, hoặc không ngưỡng nào <= 0.5 đạt mục tiêu: KHÔNG chọn (dùng ngưỡng mặc định của merge_prelabels --conf). Ghi submission/qc/threshold_sweep.csv/.png/.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import SUB, die, load_coco, load_taxonomy, match_objects, rel, split_ignore, write_csv, covered_by


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, nargs="+", help="một hoặc nhiều pre.coco.json chạy ở ngưỡng thấp (0.05), vd. YOLO26 + YOLOE")
    ap.add_argument("--gold", required=True, help="nhãn chuẩn: nhãn gán tay (gold_manual.json), hoặc pilot_reviewed.json ở bài nâng cao")
    ap.add_argument("--recall-target", type=float, help="mặc định lấy từ taxonomy.rules.recall_target")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--min-objects", type=int, default=5, help="lớp có ít hơn số vật này trong ảnh gold thì không chọn ngưỡng riêng (mặc định 5)")
    ap.add_argument("--max-thr", type=float, default=0.5, help="trần ngưỡng được chọn (mặc định 0.5): chặn ngưỡng cao do vài vật")
    ap.add_argument("--out", default=str(SUB / "qc" / "threshold_sweep"))
    a = ap.parse_args()
    tax = load_taxonomy()
    tgt = a.recall_target if a.recall_target is not None else tax.rules["recall_target"]
    preds = [load_coco(Path(p)) for p in a.pred]
    gold = load_coco(Path(a.gold))

    class _Merged:  # gom đối tượng của nhiều file dự đoán (mỗi lớp thường chỉ đến từ một model)
        images = {n: s for p in preds for n, s in p.images.items()}
        objects = {n: [o for p in preds for o in p.objects.get(n, [])] for n in images}
    pred = _Merged()
    imgs = sorted(n for n in gold.images if n in pred.images)
    if not imgs:
        die("Hai file không có ảnh chung. --pred phải chạy trên cùng ảnh với --gold.")
    ths = [round(0.10 + i * a.step, 2) for i in range(int(round((0.95 - 0.10) / a.step)) + 1)]  # sàn 0.10: pre-label được gộp ở ngưỡng 0.10, dưới đó không có dữ liệu
    rows, best = [], {}
    for cls in tax.object_classes:
        n_gold = sum(1 for n in imgs for o in gold.objects[n] if o.cls == cls)
        n_pred_low = sum(1 for n in imgs for o in pred.objects[n] if o.cls == cls)
        if n_gold == 0 and n_pred_low == 0:
            continue
        series = []
        for t in ths:
            tp = fp = fn = 0
            for n in imgs:
                G, ig = split_ignore(tax, gold.objects[n])
                G = [g for g in G if g.cls == cls]
                P = [p for p in split_ignore(tax, pred.objects[n])[0] if p.cls == cls and (p.score or 0) >= t
                     and not covered_by(p.box, [x.box for x in ig])]
                pr, up, ug = match_objects(P, G, gold.images[n], tax.rules["match_iou"], False, same_class=True)
                tp += len(pr); fp += len(up); fn += len(ug)
            p = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
            series.append((t, tp, fp, fn, p, rc, f1))
            rows.append({"class": cls, "thr": t, "tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3),
                         "recall": round(rc, 3), "f1": round(f1, 3), "n_gold": n_gold})
        ok = [s for s in series if s[5] >= tgt and s[0] <= a.max_thr]
        if n_gold < a.min_objects:
            best[cls] = (None, f"chỉ {n_gold} vật trong ảnh gold (< {a.min_objects}): mẫu quá ít, KHÔNG chọn ngưỡng theo lớp này, dùng ngưỡng mặc định")
        elif ok:
            s = max(ok, key=lambda s: s[0]); best[cls] = (s[0], f"recall {s[5] * 100:.0f}% (>= {tgt * 100:.0f}%), precision {s[4] * 100:.0f}% ({n_gold} vật gold)")
        else:
            best[cls] = (None, f"không ngưỡng nào <= {a.max_thr} đạt recall {tgt * 100:.0f}% (tối đa {max(x[5] for x in series) * 100:.0f}%, {n_gold} vật): dùng ngưỡng mặc định, nói rõ trong báo cáo")
    out = Path(a.out)
    write_csv(Path(str(out) + ".csv"), ["class", "thr", "tp", "fp", "fn", "precision", "recall", "f1", "n_gold"], rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cls_list = [c for c in tax.object_classes if any(r["class"] == c for r in rows)]
        fig, axs = plt.subplots(1, len(cls_list), figsize=(3.2 * len(cls_list), 3.2), squeeze=False)
        for ax, c in zip(axs[0], cls_list):
            rr = [r for r in rows if r["class"] == c]
            ax.plot([r["thr"] for r in rr], [r["precision"] for r in rr], label="precision")
            ax.plot([r["thr"] for r in rr], [r["recall"] for r in rr], label="recall")
            ax.axhline(tgt, ls=":", c="gray")
            if best[c][0] is not None:
                ax.axvline(best[c][0], ls="--", c="red")
            ax.set_title(f"{c} (gold={rr[0]['n_gold']})", fontsize=9)
            ax.set_ylim(0, 1.02); ax.set_xlabel("ngưỡng")
        axs[0][0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(str(out) + ".png", dpi=110); plt.close(fig)
    except Exception as e:  # noqa
        print(f"(không vẽ được biểu đồ: {e})")
    L = ["# Quét ngưỡng theo lớp", "", f"Mục tiêu recall: {tgt * 100:.0f}%. Ảnh dùng để quét: {len(imgs)}.", "",
         "| Lớp | Ngưỡng gợi ý | Lý do |", "|---|---|---|"]
    for c, (t, why) in best.items():
        L.append(f"| {c} | {'(mặc định)' if t is None else t} | {why} |")
    thr_arg = ",".join(f"{c}={t}" for c, (t, _) in best.items() if t is not None)
    L += ["", "Dán vào lệnh merge_prelabels (lớp không có trong chuỗi dùng `--conf` mặc định 0.3): `--thr \"" + thr_arg + "\"`", "",
          "Lưu ý: bạn chọn ngưỡng trên chính nhãn gold, nên precision / recall của máy trên gold ở ngưỡng đã chọn là con số **lạc quan** (chọn và chấm trên cùng dữ liệu); chỉ có vài ảnh -> ngưỡng là ước lượng thô; lớp có ít vật (xem n_gold) thì kết luận yếu."]
    text = "\n".join(L) + "\n"
    Path(str(out) + ".md").write_text(text, encoding="utf8")
    print(text)
    print(f"Đã ghi {rel(Path(str(out) + '.md'))}, .csv, .png")


if __name__ == "__main__":
    main()
