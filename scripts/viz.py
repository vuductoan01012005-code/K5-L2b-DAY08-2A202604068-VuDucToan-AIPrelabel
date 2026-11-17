"""Vẽ nhãn COCO (box + polygon + điểm tin cậy) chồng lên ảnh để xem nhanh.

Ví dụ:
  python scripts/viz.py --coco work/prelabel/yolo26/pre.coco.json --out work/viz/yolo26
  python scripts/viz.py --coco work/prelabel/yolo26/pre.coco.json --images img_012.jpg img_020.jpg --out work/viz/x
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

from common import IMAGES_DIR, ROOT, CocoSet, Obj, die, load_coco, load_taxonomy, rel

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#808000", "#f032e6"]


def color_of(tax, cls: str) -> str:
    try:
        return PALETTE[tax.classes.index(cls) % len(PALETTE)]
    except ValueError:
        return "#ffffff"


def draw_image(img_path: Path, objs: List[Obj], tax, show_ids: bool = False, alpha: int = 70) -> Image.Image:
    im = Image.open(img_path).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    try:
        font = ImageFont.load_default(size=max(11, im.size[0] // 70))
    except TypeError:  # Pillow cũ
        font = ImageFont.load_default()
    for k, o in enumerate(objs):
        c = color_of(tax, o.cls)
        rgb = tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
        for p in o.polys:
            if len(p) >= 6:
                pts = [(p[i], p[i + 1]) for i in range(0, len(p), 2)]
                d.polygon(pts, fill=rgb + (alpha,), outline=rgb + (255,))
        d.rectangle(o.box, outline=rgb + (255,), width=max(2, im.size[0] // 400))
        label = o.cls + (f" {o.score:.2f}" if o.score is not None else "")
        if show_ids:
            label = f"#{o.id} " + label
        tw = d.textlength(label, font=font)
        x, y = o.box[0], max(0, o.box[1] - font.size - 3 if hasattr(font, "size") else o.box[1] - 14)
        d.rectangle([x, y, x + tw + 4, y + (font.size if hasattr(font, "size") else 12) + 3], fill=rgb + (230,))
        d.text((x + 2, y + 1), label, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(im, ov).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, help="file COCO cần vẽ")
    ap.add_argument("--out", required=True, help="thư mục ghi ảnh .jpg")
    ap.add_argument("--images", nargs="*", help="chỉ vẽ các ảnh này (mặc định: tất cả ảnh trong file)")
    ap.add_argument("--img-dir", default=str(IMAGES_DIR))
    ap.add_argument("--ids", action="store_true", help="in id của từng đối tượng")
    a = ap.parse_args()
    tax = load_taxonomy()
    cs: CocoSet = load_coco(Path(a.coco))
    names = a.images or sorted(cs.images)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in names:
        p = Path(a.img_dir) / name
        if not p.exists():
            print(f"bỏ qua {name}: không thấy ảnh trong {a.img_dir}")
            continue
        draw_image(p, cs.objects.get(name, []), tax, a.ids).save(out / (Path(name).stem + ".jpg"), quality=88)
        n += 1
    print(f"Đã vẽ {n} ảnh -> {rel(out)}")


if __name__ == "__main__":
    main()
