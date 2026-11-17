"""Hàm dùng chung cho các script của lab Day 08 (đọc taxonomy, đọc/ghi COCO, IoU box + mask, ghép đối tượng).

Học viên KHÔNG cần sửa file này. Muốn đổi lớp / thuộc tính / ánh xạ nhãn: sửa submission/taxonomy.json.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def configure_utf8_stdio() -> None:
    """Đảm bảo stdout/stderr luôn dùng UTF-8 khi bị pipe/redirect trên Windows."""
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_stdio()

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = ROOT / "submission" / "taxonomy.json"
IMAGES_DIR = ROOT / "data" / "images"
SPLIT_CSV = ROOT / "data" / "split.csv"
WORK = ROOT / "work"
SUB = ROOT / "submission"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def die(msg: str, code: int = 2) -> None:
    print(f"\nLỖI: {msg}", file=sys.stderr)
    sys.exit(code)


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- taxonomy
class Taxonomy:
    def __init__(self, d: dict):
        self.raw = d
        self.version = d.get("version", "?")
        self.classes: List[str] = [c["name"] for c in d["classes"]]
        self.ignore: str = d.get("ignore_class", "ignore_region")
        self.attrs: Dict[str, dict] = {a["name"]: a for a in d.get("attributes", [])}
        self.rules: dict = d["rules"]
        self.coco_map: Dict[str, str] = d.get("coco_map", {})
        self.prompts: Dict[str, List[str]] = d.get("prompts", {})

    @property
    def object_classes(self) -> List[str]:
        return [c for c in self.classes if c != self.ignore]

    def cat_id(self, cls: str) -> int:
        return self.classes.index(cls) + 1


def load_taxonomy() -> Taxonomy:
    if not TAXONOMY_PATH.exists():
        die(f"Không thấy {rel(TAXONOMY_PATH)}. Hãy chạy script từ thư mục gốc của repo.")
    try:
        return Taxonomy(json.loads(TAXONOMY_PATH.read_text(encoding="utf-8-sig")))
    except (json.JSONDecodeError, KeyError) as e:
        die(f"taxonomy.json bị lỗi cú pháp / thiếu trường ({e}).")
        raise


# --------------------------------------------------------------------------- tách tập dữ liệu
def load_split() -> Dict[str, dict]:
    """data/split.csv -> {tên ảnh: {'split': pilot|gold|batch, 'seg': bool}}"""
    if not SPLIT_CSV.exists():
        die("Thiếu data/split.csv.")
    with SPLIT_CSV.open(encoding="utf-8-sig", newline="") as f:
        return {r["image"]: {"split": r["split"], "seg": r["seg"].strip() in ("1", "yes", "true")}
                for r in csv.DictReader(f)}


def images_of(split_names: Sequence[str]) -> List[str]:
    """Tên ảnh thuộc các tập pilot / gold / batch; 'seg' = 8 ảnh batch dùng cho segmentation."""
    sp = load_split()
    names = set(split_names)
    return sorted(k for k, v in sp.items()
                  if v["split"] in names or ("seg" in names and v["split"] == "batch" and v["seg"]))


# --------------------------------------------------------------------------- hình học
def xywh_to_xyxy(b) -> Box:
    return (float(b[0]), float(b[1]), float(b[0]) + float(b[2]), float(b[1]) + float(b[3]))


def xyxy_to_xywh(b: Box) -> List[float]:
    return [round(b[0], 2), round(b[1], 2), round(b[2] - b[0], 2), round(b[3] - b[1], 2)]


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a: Box, b: Box) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


def covered_by(box: Box, regions: List[Box], frac: float = 0.5) -> bool:
    """True nếu >= frac diện tích box nằm trong một vùng ignore."""
    a = area(box)
    if a <= 0:
        return False
    for r in regions:
        iw = max(0.0, min(box[2], r[2]) - max(box[0], r[0]))
        ih = max(0.0, min(box[3], r[3]) - max(box[1], r[1]))
        if iw * ih / a >= frac:
            return True
    return False


# --------------------------------------------------------------------------- mask
def polys_to_mask(polys: List[List[float]], w: int, h: int) -> np.ndarray:
    import cv2
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        if len(p) >= 6:
            pts = np.array(p, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    return m


def mask_to_polys(mask: np.ndarray, eps_frac: float = 0.003, min_area: float = 30.0, keep_frac: float = 0.05) -> List[List[float]]:
    """Mask nhị phân -> danh sách polygon phẳng [x1,y1,x2,y2,...] (đã đơn giản hoá).

    Bỏ các mảnh vụn: chỉ giữ mảnh có diện tích >= keep_frac x mảnh lớn nhất (và >= min_area px).
    """
    import cv2
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    biggest = max(cv2.contourArea(c) for c in cnts)
    out = []
    for c in cnts:
        ar = cv2.contourArea(c)
        if ar < min_area or ar < keep_frac * biggest:
            continue
        eps = eps_frac * cv2.arcLength(c, True)
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(a) >= 3:
            out.append([round(float(v), 1) for v in a.flatten()])
    return out


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    i = np.logical_and(m1, m2).sum()
    u = np.logical_or(m1, m2).sum()
    return float(i) / float(u) if u > 0 else 0.0


# --------------------------------------------------------------------------- đọc COCO
@dataclass
class Obj:
    id: int
    cls: str
    box: Box
    polys: List[List[float]] = field(default_factory=list)
    score: Optional[float] = None
    attrs: dict = field(default_factory=dict)
    _mask: Optional[np.ndarray] = None

    @property
    def has_poly(self) -> bool:
        return bool(self.polys)


@dataclass
class CocoSet:
    path: Optional[Path]
    images: Dict[str, Tuple[int, int]]      # tên ảnh (basename) -> (w, h)
    objects: Dict[str, List[Obj]]           # tên ảnh -> đối tượng (kể cả ignore_region)
    problems: List[str] = field(default_factory=list)


def _score_of(attrs: dict, an: dict) -> Optional[float]:
    for k in ("score", "confidence"):
        if an.get(k) is not None:
            try:
                return float(an[k])
            except (TypeError, ValueError):
                pass
    v = attrs.get("conf") or attrs.get("score")
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def load_coco(path: Path) -> CocoSet:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    cs = CocoSet(Path(path), {}, {})
    id2name: Dict[int, str] = {}
    for im in data.get("images", []):
        base = Path(im["file_name"]).name
        if base in cs.images:
            cs.problems.append(f"ảnh {base} xuất hiện 2 lần")
        id2name[im["id"]] = base
        cs.images[base] = (int(im.get("width", 0)), int(im.get("height", 0)))
        cs.objects.setdefault(base, [])
    for an in data.get("annotations", []):
        base = id2name.get(an.get("image_id"))
        if base is None:
            cs.problems.append(f"annotation {an.get('id')} trỏ tới image_id không có trong 'images'")
            continue
        cls = cats.get(an.get("category_id"), f"?category_id={an.get('category_id')}")
        attrs = an.get("attributes", {}) or {}
        seg = an.get("segmentation") or []
        polys = [list(map(float, p)) for p in seg] if isinstance(seg, list) else []
        if polys:  # box = bao của polygon nếu thiếu bbox
            pass
        box = xywh_to_xyxy(an["bbox"]) if an.get("bbox") else _bbox_of_polys(polys)
        cs.objects[base].append(Obj(an.get("id", 0), cls, box, polys, _score_of(attrs, an), dict(attrs)))
    return cs


def _bbox_of_polys(polys) -> Box:
    xs = [v for p in polys for v in p[0::2]]
    ys = [v for p in polys for v in p[1::2]]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)


def obj_mask(o: Obj, w: int, h: int) -> np.ndarray:
    if o._mask is None:
        if o.polys:
            o._mask = polys_to_mask(o.polys, w, h)
        else:
            m = np.zeros((h, w), np.uint8)
            x1, y1, x2, y2 = [int(round(v)) for v in o.box]
            m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 1
            o._mask = m
    return o._mask


def obj_iou(a: Obj, b: Obj, size: Tuple[int, int], use_mask: bool) -> float:
    """IoU box; hoặc IoU mask nếu use_mask và cả hai có polygon."""
    if use_mask and a.has_poly and b.has_poly:
        return mask_iou(obj_mask(a, *size), obj_mask(b, *size))
    return box_iou(a.box, b.box)


# --------------------------------------------------------------------------- ghi COCO
def build_coco(tax: Taxonomy, images: List[dict], annotations: List[dict], info: Optional[dict] = None) -> dict:
    return {
        "info": info or {},
        "licenses": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(tax.classes)],
        "images": images,
        "annotations": annotations,
    }


def make_ann(tax: Taxonomy, ann_id: int, image_id: int, cls: str, box: Box,
             polys: Optional[List[List[float]]] = None, score: Optional[float] = None,
             attrs: Optional[dict] = None) -> dict:
    xywh = xyxy_to_xywh(box)
    at = {"conf": "" if score is None else f"{score:.2f}", "can_review": "false"}
    at.update(attrs or {})
    # CVAT lưu checkbox dạng chuỗi "true"/"false": đổi bool Python -> chuỗi để nạp vào CVAT không bị lệch
    at = {k: (("true" if v else "false") if isinstance(v, bool) else v) for k, v in at.items()}
    seg = polys or []
    if seg:
        m_area = round(float(polys_area(seg)), 2)
    else:
        m_area = round(xywh[2] * xywh[3], 2)
    ann = {"id": ann_id, "image_id": image_id, "category_id": tax.cat_id(cls), "bbox": xywh,
           "area": m_area, "iscrowd": 0, "segmentation": seg, "attributes": at}
    if score is not None:
        ann["score"] = round(float(score), 4)
    return ann


def polys_area(polys: List[List[float]]) -> float:
    tot = 0.0
    for p in polys:
        xs, ys = p[0::2], p[1::2]
        n = len(xs)
        tot += abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2
    return tot


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf8")


def read_csv(path: Path) -> List[dict]:
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    delim = None
    if lines and lines[0].strip().lower() in ("sep=;", "sep=,"):
        delim = lines[0].strip()[-1]
        text = "\n".join(lines[1:])
        lines = lines[1:]
    if delim is None:
        first = next((ln for ln in lines if ln.strip()), "")
        delim = ";" if (";" in first and "," not in first) else ","
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delim))


def write_csv(path: Path, header: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- ghép đối tượng
def match_objects(pre: List[Obj], fin: List[Obj], size: Tuple[int, int], thr: float, use_mask: bool,
                  same_class: bool = True):
    """Ghép 1-1 tham lam theo IoU giảm dần. Trả về (cặp [(a, b, iou)], a chưa ghép, b chưa ghép).

    same_class=True: chỉ ghép khi cùng lớp (sai lớp -> hai bên đều chưa ghép = xoá + thêm, đúng như slide).
    """
    cands = []
    for i, a in enumerate(pre):
        for j, b in enumerate(fin):
            if same_class and a.cls != b.cls:
                continue
            v = obj_iou(a, b, size, use_mask)
            if v >= thr:
                cands.append((v, i, j))
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))
    ua, ub, pairs = set(), set(), []
    for v, i, j in cands:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        pairs.append((pre[i], fin[j], v))
    return (pairs, [a for i, a in enumerate(pre) if i not in ua], [b for j, b in enumerate(fin) if j not in ub])


def split_ignore(tax: Taxonomy, objs: List[Obj]):
    ign = [o for o in objs if o.cls == tax.ignore]
    real = [o for o in objs if o.cls != tax.ignore]
    return real, ign
