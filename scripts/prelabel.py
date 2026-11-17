"""Chạy một model pre-label trên ảnh, ghi kết quả ra COCO 1.0 để nạp vào CVAT (đường "offline").

Mỗi lần chạy = một "run" ghi vào work/runs.csv (model, tham số, số box, thời gian) -> dùng cho model_log.

Các model (từ đơn giản đến phức tạp):
  yolo26   Detector closed-set (COCO 80 lớp).  Thêm --segmenter native để dùng yolo26-seg (mask), hoặc sam2.
  yoloe    Open-vocabulary thời gian thực (gõ chữ -> box + mask).  Cần --prompts.
  gdino    Grounding DINO (HF transformers), gõ chữ -> box.  Nặng hơn, chậm hơn.
  owlv2    OWLv2 (HF), gõ chữ -> box.
  sam3     SAM 3 (HF, gated), gõ chữ -> mọi instance + mask.  Cần quyền truy cập, xem docs/SETUP.md.
Bộ tách mask (--segmenter):
  none     chỉ box (mặc định)
  native   dùng mask của chính model (yolo26-seg, yoloe-seg, sam3)
  sam      mỗi box -> SAM (bất kỳ phiên bản trong --sam: SAM 1 / SAM 2.1) -> mask (kiểu "detector -> segmenter" của Grounded-SAM)
           (`sam2` là tên cũ, vẫn dùng được). SAM 3 chạy riêng bằng --model sam3 (gõ chữ -> mask).

Ví dụ:
  python scripts/prelabel.py --model yolo26 --size m --split pilot --conf 0.25 --out work/prelabel/y26_pilot
  python scripts/prelabel.py --model yoloe --prompts "tuk-tuk=three_wheeler,cyclo=three_wheeler" \
         --classes three_wheeler --split batch --out work/prelabel/yoloe_batch
  python scripts/prelabel.py --model yolo26 --size m --split batch --segmenter sam --sam sam2.1_b.pt --out work/prelabel/y26_sam
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from common import (IMAGES_DIR, ROOT, WORK, Taxonomy, build_coco, die, images_of, load_split, load_taxonomy,
                    make_ann, mask_to_polys, box_iou, pick_device, read_csv, rel, save_json, write_csv, Obj)

MODELS_DIR = ROOT / "models"
Det = Tuple[str, float, Tuple[float, float, float, float], Optional[List[List[float]]]]  # (tên thô, score, box, polys)


# --------------------------------------------------------------------------- tiện ích
def parse_kv(s: Optional[str], cast=str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            die(f"Sai định dạng '{part}': cần dạng khoa=giatri (ví dụ person=0.3).")
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = cast(v.strip())
        except ValueError:
            die(f"Sai giá trị '{v.strip()}' cho '{k.strip()}': không hợp lệ ({cast.__name__}).")
    return out


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def weights_path(name: str) -> str:
    """models/<name> nếu đã có hoặc tải được vào đó; nếu không thì để ultralytics tự xử lý."""
    p = MODELS_DIR / name
    if p.exists():
        return str(p)
    MODELS_DIR.mkdir(exist_ok=True)
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        got = attempt_download_asset(str(p))
        if Path(got).exists():
            return str(got)
    except Exception as e:  # noqa
        print(f"[cảnh báo] không tải được {name} vào models/: {e}", file=sys.stderr)
    return name


# --------------------------------------------------------------------------- backends
class Backend:
    needs_prompts = False
    native_masks = False

    def __init__(self, a, tax: Taxonomy, device: str, min_conf: float, prompts: Dict[str, str]):
        self.a, self.tax, self.device, self.min_conf, self.prompts = a, tax, device, min_conf, prompts

    def detect(self, path: Path) -> List[Det]:
        raise NotImplementedError


class YoloBackend(Backend):
    def __init__(self, *args):
        super().__init__(*args)
        from ultralytics import YOLO
        seg = self.a.segmenter == "native"
        name = self.a.weights or f"yolo26{self.a.size}{'-seg' if seg else ''}.pt"
        self.native_masks = seg
        self.model = YOLO(weights_path(name))

    def detect(self, path):
        r = self.model.predict(str(path), conf=self.min_conf, device=self.device, imgsz=self.a.imgsz,
                               verbose=False)[0]
        out: List[Det] = []
        polys = r.masks.xy if (self.native_masks and r.masks is not None) else None
        for k in range(len(r.boxes)):
            b = r.boxes[k]
            out.append((r.names[int(b.cls)], float(b.conf), tuple(b.xyxy[0].tolist()),
                        [p.flatten().tolist() for p in [polys[k]]] if polys is not None and len(polys[k]) >= 3 else None))
        return out


class YoloeBackend(Backend):
    needs_prompts = True

    def __init__(self, *args):
        super().__init__(*args)
        import os
        from ultralytics import YOLOE
        self.native_masks = self.a.segmenter == "native"
        name = self.a.weights or f"yoloe-26{self.a.size}-seg.pt"
        w = weights_path(name)
        self.phrases = list(self.prompts.keys())
        cwd = os.getcwd()
        MODELS_DIR.mkdir(exist_ok=True)
        os.chdir(MODELS_DIR)  # bộ mã hoá chữ (mobileclip) được tải vào thư mục hiện tại -> giữ gọn trong models/
        try:
            self.model = YOLOE(w)
            self.model.set_classes(self.phrases, self.model.get_text_pe(self.phrases))
        finally:
            os.chdir(cwd)

    def detect(self, path):
        r = self.model.predict(str(path), conf=self.min_conf, device=self.device, imgsz=self.a.imgsz,
                               verbose=False)[0]
        out: List[Det] = []
        has_m = self.native_masks and r.masks is not None
        for k in range(len(r.boxes)):
            b = r.boxes[k]
            poly = None
            if has_m and len(r.masks.xy[k]) >= 3:
                poly = [r.masks.xy[k].flatten().tolist()]
            out.append((self.phrases[int(b.cls)], float(b.conf), tuple(b.xyxy[0].tolist()), poly))
        return out


class GDinoBackend(Backend):
    needs_prompts = True

    def __init__(self, *args):
        super().__init__(*args)
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        mid = self.a.weights or "IDEA-Research/grounding-dino-tiny"
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(self.device).eval()
        self.phrases = list(self.prompts.keys())
        self.text = " ".join(p.lower().rstrip(".") + "." for p in self.phrases)

    def detect(self, path):
        img = Image.open(path).convert("RGB")
        inp = self.proc(images=img, text=self.text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=self.min_conf, text_threshold=self.a.text_thr,
            target_sizes=[img.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        dets: List[Det] = []
        for s, b, lab in zip(res["scores"].tolist(), res["boxes"].tolist(), labels):
            phrase = self._phrase(lab)
            if phrase:
                dets.append((phrase, float(s), tuple(b), None))
        return dets

    def _phrase(self, lab: str) -> Optional[str]:
        # Grounding DINO có thể trả "a tuk - tuk", nhiều cụm dính nhau; so khớp theo chuỗi đã chuẩn hoá.
        n = norm(lab)
        best = None
        for p in self.phrases:
            if norm(p) and norm(p) in n and (best is None or len(norm(p)) > len(norm(best))):
                best = p
        return best


class Owlv2Backend(Backend):
    needs_prompts = True

    def __init__(self, *args):
        super().__init__(*args)
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        mid = self.a.weights or "google/owlv2-base-patch16-ensemble"
        self.torch = torch
        self.proc = Owlv2Processor.from_pretrained(mid)
        self.model = Owlv2ForObjectDetection.from_pretrained(mid).to("cpu" if self.device == "mps" else self.device).eval()
        self.dev = "cpu" if self.device == "mps" else self.device  # OWLv2 trên MPS hay lỗi dtype -> CPU
        self.phrases = list(self.prompts.keys())

    def detect(self, path):
        img = Image.open(path).convert("RGB")
        inp = self.proc(text=[[f"a photo of a {p}" for p in self.phrases]], images=img, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.model(**inp)
        side = max(img.size)  # OWLv2 pad ảnh thành vuông -> target_sizes phải là cạnh dài nhất
        res = self.proc.post_process_grounded_object_detection(out, threshold=self.min_conf,
                                                               target_sizes=[(side, side)])[0]
        dets: List[Det] = []
        for s, b, li in zip(res["scores"].tolist(), res["boxes"].tolist(), res["labels"].tolist()):
            x1, y1, x2, y2 = b
            dets.append((self.phrases[int(li)], float(s),
                         (max(0, x1), max(0, y1), min(img.size[0], x2), min(img.size[1], y2)), None))
        return dets


class Sam3Backend(Backend):
    """SAM 3 (concept prompting) qua HF transformers. Cần: hf auth login + được cấp quyền facebook/sam3."""
    needs_prompts = True
    native_masks = True

    def __init__(self, *args):
        super().__init__(*args)
        import torch
        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError:
            die("transformers của bạn chưa có Sam3Model. Chạy: pip install -U 'transformers>=5.0'")
        mid = self.a.weights or "facebook/sam3"
        self.torch = torch
        try:
            self.proc = Sam3Processor.from_pretrained(mid)
            self.model = Sam3Model.from_pretrained(mid).to(self.device).eval()
        except Exception as e:  # noqa
            die(f"Không tải được {mid}: {e}\nSAM 3 là model gated: xin quyền tại https://huggingface.co/{mid} "
                "rồi chạy `hf auth login` (xem docs/SETUP.md, mục SAM 3).")
        self.phrases = list(self.prompts.keys())

    def detect(self, path):
        img = Image.open(path).convert("RGB")
        dets: List[Det] = []
        for ph in self.phrases:  # một khái niệm một prompt (luật của slide)
            inp = self.proc(images=img, text=ph, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = self.model(**inp)
            res = self.proc.post_process_instance_segmentation(
                out, threshold=self.min_conf, mask_threshold=0.5, target_sizes=[img.size[::-1]])[0]
            masks = res["masks"].cpu().numpy() if len(res["masks"]) else []
            for k, (s, b) in enumerate(zip(res["scores"].tolist(), res["boxes"].tolist())):
                polys = mask_to_polys(masks[k]) if len(masks) else None
                dets.append((ph, float(s), tuple(b), polys or None))
        return dets


BACKENDS = {"yolo26": YoloBackend, "yoloe": YoloeBackend, "gdino": GDinoBackend, "owlv2": Owlv2Backend,
            "sam3": Sam3Backend}


# --------------------------------------------------------------------------- SAM 2: box -> mask
class Sam2Segmenter:
    def __init__(self, name: str, device: str):
        from ultralytics import SAM
        self.model = SAM(weights_path(name))
        self.device = device

    def masks_for(self, path: Path, boxes: List[Tuple[float, float, float, float]]):
        if not boxes:
            return []
        r = self.model(str(path), bboxes=[list(map(float, b)) for b in boxes], device=self.device, verbose=False)[0]
        if r.masks is None:
            return [None] * len(boxes)
        data = r.masks.data.cpu().numpy()
        out = []
        for k in range(len(boxes)):
            out.append(mask_to_polys(data[k]) if k < len(data) else None)
        return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(BACKENDS))
    ap.add_argument("--size", default=None, help="cỡ model YOLO: n s m l x (yolo26) / s m l (yoloe). Mặc định m (yolo26), s (yoloe)")
    ap.add_argument("--weights", help="ghi đè tên file trọng số / id HuggingFace")
    ap.add_argument("--split", help="pilot, gold, batch, seg (cách nhau dấu phẩy) hoặc 'all'; seg = 8 ảnh batch dùng cho segmentation")
    ap.add_argument("--images", nargs="*", help="danh sách tên ảnh cụ thể (thay cho --split)")
    ap.add_argument("--prompts", help="open-vocab: 'cụm chữ=lớp,cụm chữ=lớp' (mặc định lấy từ taxonomy.json cho --classes)")
    ap.add_argument("--classes", help="chỉ GIỮ các lớp này (cách nhau dấu phẩy), ví dụ three_wheeler")
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng tin cậy mặc định")
    ap.add_argument("--thr", help="ngưỡng riêng từng lớp, ví dụ \"person=0.3,car=0.4\" (ghi đè --conf)")
    ap.add_argument("--segmenter", default="none", choices=["none", "native", "sam", "sam2"])
    ap.add_argument("--sam", "--sam2", dest="sam", default="sam2.1_b.pt",
                    help="trọng số SAM khi --segmenter sam: SAM 1 = mobile_sam.pt | sam_b.pt | sam_l.pt ; SAM 2.1 = sam2.1_t.pt | sam2.1_s.pt | sam2.1_b.pt | sam2.1_l.pt")
    ap.add_argument("--map", help="ánh xạ tên thô -> lớp, ví dụ motorcycle=motorbike (bổ sung cho taxonomy.coco_map)")
    ap.add_argument("--nms", type=float, default=0.7, help="open-vocab: IoU để loại box trùng giữa các cụm chữ cùng lớp")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--text-thr", type=float, default=0.25, help="gdino: ngưỡng text")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--out", required=True, help="thư mục đầu ra (ghi pre.coco.json, viz/, run.json)")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--note", default="", help="ghi chú vào runs.csv")
    a = ap.parse_args()

    tax = load_taxonomy()
    if a.size is None:
        a.size = "s" if a.model == "yoloe" else "m"
    device = pick_device(a.device)
    keep = {c.strip() for c in a.classes.split(",") if c.strip()} if a.classes else None
    thr = {k: float(v) for k, v in parse_kv(a.thr, float).items()}
    for k in thr:
        if k not in tax.classes:
            die(f"--thr: lớp '{k}' không có trong taxonomy ({', '.join(tax.object_classes)}).")
    if keep:
        for k in keep:
            if k not in tax.object_classes:
                die(f"--classes: lớp '{k}' không có trong taxonomy.")
    min_conf = min([a.conf] + list(thr.values()))

    # --- danh sách ảnh
    if a.images:
        names = a.images
    elif a.split:
        sp = ["pilot", "gold", "batch"] if a.split == "all" else [s.strip() for s in a.split.split(",")]
        names = images_of(sp)
    else:
        die("Cần --split hoặc --images.")
    for n in names:
        if not (IMAGES_DIR / n).exists():
            die(f"Không thấy ảnh {n} trong data/images/.")

    # --- prompts / ánh xạ nhãn
    prompts: Dict[str, str] = {}
    if BACKENDS[a.model].needs_prompts:
        if a.prompts:
            prompts = {k: str(v) for k, v in parse_kv(a.prompts).items()}
        else:
            for cls in (keep or tax.object_classes):
                for ph in tax.prompts.get(cls, [cls]):
                    prompts[ph] = cls
        for ph, cls in prompts.items():
            if cls not in tax.object_classes:
                die(f"--prompts: '{ph}={cls}' trỏ tới lớp không có trong taxonomy.")
        print(f"Prompts ({len(prompts)}): " + ", ".join(f"'{p}'→{c}" for p, c in prompts.items()))
        mapping = dict(prompts)
    else:
        mapping = dict(tax.coco_map)
        mapping.update({k: str(v) for k, v in parse_kv(a.map).items()})

    print(f"Model: {a.model} | thiết bị: {device} | {len(names)} ảnh | ngưỡng mặc định {a.conf}"
          + (f" | theo lớp {thr}" if thr else ""))
    t0 = time.time()
    be = BACKENDS[a.model](a, tax, device, min_conf, prompts)
    sam2 = Sam2Segmenter(a.sam, device) if a.segmenter in ("sam", "sam2") else None
    if a.segmenter == "native" and not be.native_masks:
        die(f"Model '{a.model}' không có mask gốc. Dùng --segmenter sam hoặc bỏ --segmenter.")
    t_load = time.time() - t0

    images, anns = [], []
    dropped: Counter = Counter()
    below: Counter = Counter()
    ann_id = 1
    t1 = time.time()
    for iid, name in enumerate(names, 1):
        path = IMAGES_DIR / name
        w, h = Image.open(path).size
        images.append({"id": iid, "file_name": name, "width": w, "height": h})
        dets = be.detect(path)
        kept: List[Tuple[str, float, tuple, Optional[list]]] = []
        for raw, s, box, polys in dets:
            cls = mapping.get(raw)
            if cls is None:
                dropped[raw] += 1
                continue
            if keep and cls not in keep:
                continue
            if s < thr.get(cls, a.conf):
                below[cls] += 1
                continue
            kept.append((cls, s, (max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3])), polys))
        if len(prompts) > 1:  # nhiều cụm chữ đồng nghĩa cùng một lớp -> loại box trùng (NMS theo lớp, IoU 0.7)
            kept.sort(key=lambda k: -k[1])
            out_k = []
            for k in kept:
                if not any(o[0] == k[0] and box_iou(o[2], k[2]) >= a.nms for o in out_k):
                    out_k.append(k)
            kept = out_k
        if sam2 and kept:
            ms = sam2.masks_for(path, [k[2] for k in kept])
            kept = [(c, s, b, m or None) for (c, s, b, _), m in zip(kept, ms)]
        for cls, s, box, polys in kept:
            anns.append(make_ann(tax, ann_id, iid, cls, box, polys, s, {"can_review": False}))
            ann_id += 1
        print(f"  {name}: {len(kept)} box", end="\r")
    dt = time.time() - t1
    out = Path(a.out)
    save_json(out / "pre.coco.json", build_coco(
        tax, images, anns, {"description": f"prelabel {a.model}", "model": a.model, "taxonomy": tax.version}))
    print(f"\nĐã ghi {len(anns)} box ({sum(1 for x in anns if x['segmentation'])} có mask) cho {len(images)} ảnh "
          f"-> {rel(out / 'pre.coco.json')}")
    if dropped:
        print("Tên thô KHÔNG có trong ánh xạ (bị bỏ im lặng nếu bạn không đọc dòng này!): "
              + ", ".join(f"{k}×{v}" for k, v in dropped.most_common()))
    if below:
        print("Bị loại do dưới ngưỡng theo lớp: " + ", ".join(f"{k}×{v}" for k, v in below.most_common()))
    per_cls = Counter(tax.classes[x["category_id"] - 1] for x in anns)
    print("Số box theo lớp: " + (", ".join(f"{k}={v}" for k, v in per_cls.most_common()) or "(không có)"))
    print(f"Thời gian: nạp {t_load:.1f}s, chạy {dt:.1f}s ({dt / max(1, len(images)):.2f}s/ảnh)")

    params = " ".join(sys.argv[1:])
    runs = WORK / "runs.csv"
    rows = read_csv(runs) if runs.exists() else []
    run = {"run_id": f"r{len(rows) + 1:02d}", "part": "", "model": a.model + (f"-{a.size}" if "yolo" in a.model else "") + (f"+sam({a.sam})" if a.segmenter in ("sam", "sam2") else ""),
           "params": params, "split": a.split or "custom", "n_boxes": len(anns),
           "sec_per_img": round(dt / max(1, len(images)), 2), "what_went_right": "", "what_went_wrong": "", "decision": "",
           "out": rel(out), "device": device, "n_images": len(images), "n_masks": sum(1 for x in anns if x["segmentation"]),
           "dropped": ";".join(f"{k}x{v}" for k, v in dropped.items()), "note": a.note}
    save_json(out / "run.json", run)
    rows = [r for r in rows if r.get("out") != run["out"]] + [run]
    write_csv(runs, list(run.keys()), rows)

    if not a.no_viz:
        from viz import draw_image
        from common import load_coco
        cs = load_coco(out / "pre.coco.json")
        vd = out / "viz"
        vd.mkdir(exist_ok=True)
        for name in names:
            draw_image(IMAGES_DIR / name, cs.objects[name], tax).save(vd / (Path(name).stem + ".jpg"), quality=85)
        print(f"Ảnh xem nhanh: {rel(vd)}/")


if __name__ == "__main__":
    main()
