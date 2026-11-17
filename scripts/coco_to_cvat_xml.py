"""Đổi COCO pre-label sang "CVAT for images 1.1" XML để nạp polygon sạch vào CVAT.

    python scripts/coco_to_cvat_xml.py --coco submission/prelabel/seg_pre.coco.json --shape polygon --out work/cvat_import/seg_pre.xml

Vì sao cần: nếu nạp COCO có `segmentation`, CVAT tạo CẢ polygon LẪN rectangle (cùng nhóm) cho mỗi vật. File XML chỉ
tạo đúng một loại hình. --shape box -> rectangle (dùng được thay cho COCO JSON), --shape polygon -> polygon.
Nạp: Task -> Actions -> Upload annotations -> định dạng "CVAT 1.1" -> chọn file .xml.
Lưu ý: CVAT gán source = file cho mọi nhãn nạp từ file (docs/CVAT_AI.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from common import die, load_coco, load_taxonomy, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--shape", choices=["box", "polygon"], default="polygon")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tax = load_taxonomy()
    cs = load_coco(Path(a.coco))
    out = ["<?xml version=\"1.0\" encoding=\"utf-8\"?>", "<annotations>", "  <version>1.1</version>"]
    n_shape = n_skip = 0
    for fid, name in enumerate(sorted(cs.images)):
        w, h = cs.images[name]
        out.append(f"  <image id=\"{fid}\" name={quoteattr(name)} width=\"{w}\" height=\"{h}\">")
        for o in cs.objects[name]:
            if o.cls not in tax.classes:
                die(f"lớp '{o.cls}' không có trong taxonomy")
            attrs = {"conf": "" if o.score is None else f"{o.score:.2f}", "can_review": "false"}
            if o.cls == tax.ignore:
                attrs = {}
            at = "".join(f"<attribute name={quoteattr(k)}>{escape(v)}</attribute>" for k, v in attrs.items())
            if a.shape == "polygon":
                if not o.polys:
                    n_skip += 1
                    continue
                # CVAT chỉ có polygon một vòng: vật nhiều mảnh (bị che) chỉ giữ MẢNH LỚN NHẤT để mỗi vật = một polygon
                # (mảnh nhỏ bị bỏ; nếu cần, vẽ thêm khi review).
                for p in [max(o.polys, key=len)]:
                    pts = ";".join(f"{p[i]:.2f},{p[i + 1]:.2f}" for i in range(0, len(p), 2))
                    out.append(f"    <polygon label={quoteattr(o.cls)} source=\"file\" occluded=\"0\" points=\"{pts}\" z_order=\"0\">{at}</polygon>")
                    n_shape += 1
            else:
                x1, y1, x2, y2 = o.box
                out.append(f"    <box label={quoteattr(o.cls)} source=\"file\" occluded=\"0\" xtl=\"{x1:.2f}\" ytl=\"{y1:.2f}\" xbr=\"{x2:.2f}\" ybr=\"{y2:.2f}\" z_order=\"0\">{at}</box>")
                n_shape += 1
        out.append("  </image>")
    out.append("</annotations>")
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out) + "\n", encoding="utf8")
    print(f"Ghi {n_shape} {a.shape} cho {len(cs.images)} ảnh -> {rel(p)}" + (f" (bỏ {n_skip} vật không có polygon)" if n_skip else ""))


if __name__ == "__main__":
    main()
