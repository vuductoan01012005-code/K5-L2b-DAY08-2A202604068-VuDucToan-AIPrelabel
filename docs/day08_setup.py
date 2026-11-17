"""Day 08: tải model và kiểm tra cài đặt. Chạy trong thư mục day08, venv đã kích hoạt.

    python day08_setup.py download --profile 8gb     # hoặc --profile 16gb
    python day08_setup.py check                      # thư viện + chạy thử 3 model bằng CPU + CVAT
    python day08_setup.py check --light              # máy dùng Colab: chỉ thư viện nhẹ + CVAT
"""
import importlib
import os
import pathlib
import sys
import time
import urllib.request

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

GH = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
PROFILES = {
    "8gb": ["yolo26n.pt", "yoloe-26s-seg.pt", "mobileclip2_b.ts", "mobile_sam.pt", "sam2.1_t.pt"],
    "16gb": ["yolo26n.pt", "yolo26s.pt", "yoloe-26s-seg.pt", "mobileclip2_b.ts", "mobile_sam.pt", "sam2.1_b.pt"],
}
MD = pathlib.Path("models").resolve()


def download(profile):
    MD.mkdir(exist_ok=True)
    for name in PROFILES[profile]:
        dst = MD / name
        if dst.exists() and dst.stat().st_size > 1_000_000:
            print(f"[ có ] {name}")
            continue
        tmp = MD / (name + ".part")
        print(f"[tải ] {name} ...", end="", flush=True)
        try:
            with urllib.request.urlopen(GH + name, timeout=60) as r, open(tmp, "wb") as f:
                total, got = int(r.headers.get("Content-Length") or 0), 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
                    if total:
                        print(f"\r[tải ] {name} {got * 100 // total:3d}% ({total / 1e6:.0f} MB)", end="", flush=True)
            tmp.replace(dst)
            print("  xong")
        except Exception as e:  # noqa
            print(f"\n[LỖI] {name}: {e}\n       -> tải tay: {GH + name} rồi đặt vào thư mục models/")
    print("Xong. Chạy tiếp: python day08_setup.py check")


class Report:
    def __init__(self):
        self.ok = True

    def line(self, ok, msg, fix=""):
        self.ok = self.ok and bool(ok)
        print(("[ OK ] " if ok else "[LỖI] ") + msg + (("\n       -> " + fix) if (fix and not ok) else ""))


def check(light):
    R = Report()
    v = sys.version_info
    R.line((3, 10) <= v[:2] <= (3, 13), f"Python {v.major}.{v.minor}.{v.micro}",
           "cần Python 3.10 - 3.13 (khuyến nghị 3.12): tạo lại venv bằng Python 3.12")
    pkgs = [("numpy", "numpy"), ("cv2", "opencv-python-headless"), ("PIL", "pillow"), ("matplotlib", "matplotlib"), ("yaml", "pyyaml")]
    if not light:
        pkgs += [("torch", "torch"), ("ultralytics", "ultralytics"), ("clip", "CLIP của ultralytics")]
    for mod, name in pkgs:
        try:
            m = importlib.import_module(mod)
            R.line(True, f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa
            R.line(False, f"thiếu {mod}: {type(e).__name__}", f"cài lại theo bước 'Cài thư viện' ({name})")

    if not light:
        have = sorted(p.name for p in MD.glob("*.pt")) if MD.exists() else []
        R.line(bool(have), "models/: " + (", ".join(have) if have else "(trống)"), "tải model: python day08_setup.py download --profile 8gb")
        try:
            from ultralytics import SAM, YOLO, YOLOE
            from ultralytics.utils import ASSETS
            img = str(ASSETS / "bus.jpg")  # ảnh mẫu có sẵn trong ultralytics, không cần mạng
            yolo = next((MD / n for n in ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt") if (MD / n).exists()), None)
            box = None
            if yolo:
                t = time.time()
                r = YOLO(str(yolo))(img, device="cpu", imgsz=640, verbose=False)[0]
                R.line(len(r.boxes) > 0, f"{yolo.name}: {len(r.boxes)} box, {time.time() - t:.1f} giây (CPU)")
                if len(r.boxes):
                    box = r.boxes.xyxy[0].tolist()
            else:
                R.line(False, "chưa có yolo26n.pt trong models/")
            ye = MD / "yoloe-26s-seg.pt"
            if ye.exists():
                cwd = os.getcwd()
                os.chdir(MD)  # bộ mã hoá chữ mobileclip2_b.ts được đọc / tải trong models/
                try:
                    t = time.time()
                    m = YOLOE(str(ye))
                    names = ["bus", "person"]
                    m.set_classes(names, m.get_text_pe(names))
                    r = m(img, device="cpu", imgsz=640, verbose=False)[0]
                finally:
                    os.chdir(cwd)
                R.line(len(r.boxes) > 0, f"YOLOE gõ chữ 'bus', 'person': {len(r.boxes)} box, {time.time() - t:.1f} giây")
            else:
                R.line(False, "chưa có yoloe-26s-seg.pt trong models/")
            sam = next((MD / n for n in ("mobile_sam.pt", "sam2.1_t.pt", "sam2.1_b.pt", "sam_b.pt") if (MD / n).exists()), None)
            if sam and box:
                t = time.time()
                r = SAM(str(sam))(img, bboxes=[box], device="cpu", verbose=False)[0]
                area = int(r.masks.data[0].sum()) if r.masks is not None else 0
                R.line(area > 0, f"{sam.name}: mask {area} px, {time.time() - t:.1f} giây")
            elif not sam:
                R.line(False, "chưa có model SAM (mobile_sam.pt hoặc sam2.1_*.pt) trong models/")
        except Exception as e:  # noqa
            R.line(False, f"chạy thử model lỗi: {type(e).__name__}: {e}", "xem mục 'Gặp lỗi?' của trang hướng dẫn")

    try:
        with urllib.request.urlopen("http://localhost:8080/api/server/about", timeout=8) as r:
            R.line(r.status == 200, "CVAT trả lời ở http://localhost:8080")
    except Exception as e:  # noqa
        R.line(False, f"không kết nối được CVAT ({type(e).__name__})",
               "mở Docker Desktop, vào thư mục cvat-2.81.0 và chạy: docker compose up -d")
    print("\nSẴN SÀNG cho buổi học." if R.ok else "\nCòn mục [LỖI]: sửa rồi chạy lại.")
    sys.exit(0 if R.ok else 1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or "-h" in a or "--help" in a:
        print(__doc__)
        sys.exit(0)
    if a[:1] == ["download"]:
        prof = a[a.index("--profile") + 1] if "--profile" in a and len(a) > a.index("--profile") + 1 else "8gb"
        if prof not in PROFILES:
            sys.exit("--profile phải là 8gb hoặc 16gb")
        download(prof)
    elif a[:1] == ["check"]:
        check("--light" in a)
    else:
        print(__doc__)
