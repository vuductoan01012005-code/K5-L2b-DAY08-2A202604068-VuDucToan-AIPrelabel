"""Hàm auto-annotation cho CVAT (đường "trực tiếp"): chạy YOLO26 trên máy BẠN, kết quả ghi thẳng vào Task CVAT.

Dùng với cvat-cli (cài từ requirements-cvat.txt):

  # chạy một lần trên một task đã tạo
  cvat-cli --server-host http://localhost --server-port 8080 --auth USER task auto-annotate <TASK_ID> \
      --function-file scripts/cvat_function_yolo26.py -p size=str:m -p seg=bool:false --conf-threshold 0.3

  # hoặc đăng ký thành "model" trong CVAT rồi chạy agent (bấm Automatic annotation trên giao diện)
  cvat-cli ... function create-native "YOLO26 lab" --function-file scripts/cvat_function_yolo26.py -p size=str:m
  cvat-cli ... function run-agent <FUNCTION_ID> --function-file scripts/cvat_function_yolo26.py -p size=str:m

Tham số (-p ten=kieu:giatri):  size (n|s|m|l|x), seg (bool: dùng yolo26-seg -> polygon), device (auto|cpu|mps|cuda).
Nhãn của hàm trùng TÊN với taxonomy (person, bicycle, motorbike, car, bus, truck); nhãn nào Project không có sẽ bị bỏ
(Labels matching theo tên: sai tên = mất box im lặng -> luôn kiểm 5 ảnh đầu).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import PIL.Image

import cvat_sdk.auto_annotation as cvataa

_TAX = json.loads((Path(__file__).resolve().parent.parent / "submission" / "taxonomy.json").read_text(encoding="utf-8-sig"))
_COCO_MAP = _TAX["coco_map"]
# chỉ khai báo lớp mà YOLO26 (COCO) thật sự phát hiện được; three_wheeler cần model open-vocab (xem docs/MODELS.md)
_CLASSES = [c["name"] for c in _TAX["classes"] if c["name"] in set(_COCO_MAP.values())]
_MODELS = Path(__file__).resolve().parent.parent / "models"


def _device(pref: str) -> str:
    if pref != "auto":
        return pref
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _weights_path(name: str) -> str:
    p = _MODELS / name
    if p.exists():
        return str(p)
    _MODELS.mkdir(exist_ok=True)
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        got = attempt_download_asset(str(p))
        if Path(got).exists():
            return str(got)
    except Exception:
        pass
    return name


class Yolo26Function:
    def __init__(self, size: str = "m", seg: bool = False, device: str = "auto") -> None:
        from ultralytics import YOLO
        name = f"yolo26{size}{'-seg' if seg else ''}.pt"
        self._model = YOLO(_weights_path(name))
        self._seg = seg
        self._device = _device(device)
        self._ids = {c: i for i, c in enumerate(_CLASSES)}
        self.spec = cvataa.DetectionFunctionSpec(
            labels=[cvataa.label_spec(c, i, type="polygon" if seg else "rectangle") for c, i in self._ids.items()])

    def detect(self, context: cvataa.DetectionFunctionContext, image: PIL.Image.Image):
        conf = context.conf_threshold if context.conf_threshold is not None else 0.25
        r = self._model.predict(image, conf=conf, device=self._device, imgsz=1024, verbose=False)[0]
        out = []
        for k in range(len(r.boxes)):
            cls = _COCO_MAP.get(r.names[int(r.boxes.cls[k])])
            if cls is None:  # ngoài taxonomy (bottle, umbrella...) -> bỏ
                continue
            if self._seg and r.masks is not None and len(r.masks.xy[k]) >= 3:
                out.append(cvataa.polygon(self._ids[cls], r.masks.xy[k].flatten().tolist()))
            else:
                out.append(cvataa.rectangle(self._ids[cls], r.boxes.xyxy[k].tolist()))
        return out


create = Yolo26Function
