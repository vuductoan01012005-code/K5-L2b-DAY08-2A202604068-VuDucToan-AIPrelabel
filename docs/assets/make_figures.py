"""Sinh các hình minh hoạ (sơ đồ, KHÔNG phải ảnh chụp màn hình) cho README và docs. Giảng viên dùng; học viên không cần chạy.

    python docs/assets/make_figures.py
Cần Pillow + matplotlib (lấy font DejaVu có đủ dấu tiếng Việt). Đầu vào: docs/assets/model_tour.jpg, data/images/*.jpg.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def font(size, bold=False):
    p = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(p, size)


NAVY, TEAL, ORANGE, GREEN, RED, PURPLE, GRAY, LIGHT = "#0b1f3a", "#12b3a8", "#ff8a3d", "#2fbf71", "#e5484d", "#7b61ff", "#6b7686", "#f3f6fa"


def rrect(d, box, r=14, fill=None, outline=None, width=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def text(d, xy, s, size=20, fill="white", bold=False, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def gradient(w, h, c1, c2):
    im = Image.new("RGB", (w, h))
    px = im.load()
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    for x in range(w):
        for y in range(h):
            t = (x / w * 0.7 + y / h * 0.3)
            px[x, y] = tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(3))
    return im


def arrow(d, p1, p2, fill=GRAY, width=4):
    d.line([p1, p2], fill=fill, width=width)
    import math
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    for s in (-0.5, 0.5):
        d.line([p2, (p2[0] - 14 * math.cos(ang + s), p2[1] - 14 * math.sin(ang + s))], fill=fill, width=width)


def fit(path_or_im, w, h):
    im = Image.open(path_or_im).convert("RGB") if not isinstance(path_or_im, Image.Image) else path_or_im
    r = max(w / im.width, h / im.height)
    im = im.resize((int(im.width * r) + 1, int(im.height * r) + 1))
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


# --------------------------------------------------------------------------- hero
def hero():
    W, H = 1600, 520
    im = gradient(W, H, NAVY, "#0f5f66")
    d = ImageDraw.Draw(im)
    text(d, (70, 70), "DAY 08 · LAB", 22, TEAL, True)
    text(d, (70, 110), "AI hỗ trợ gán nhãn", 62, "white", True)
    text(d, (70, 190), "Pre-label · Review · Calibration", 38, "white")
    text(d, (70, 262), "Máy gán sẵn, người sửa.", 26, "#cfe8ee")
    text(d, (70, 300), "Bạn là chốt chặn chất lượng — và phải ĐO được điều đó.", 26, "#cfe8ee")
    x = 70
    for name, c in [("YOLO26", GREEN), ("SAM", PURPLE), ("YOLOE", ORANGE), ("Grounding DINO", TEAL), ("OWLv2", "#4f8cff"), ("CVAT", RED)]:
        w = d.textlength(name, font=font(20, True)) + 36
        rrect(d, (x, 380, x + w, 424), 22, fill=c)
        text(d, (x + 18, 391), name, 20, "white", True)
        x += w + 14
    text(d, (70, 455), "36 ảnh phố châu Á · detection + segmentation · tuk-tuk, xích lô, ba gác", 20, "#9fc7cf")
    tour = Image.open(HERE / "model_tour.jpg").convert("RGB")
    # cắt ba ô: YOLO26, YOLOE, SAM mask
    crops = [tour.crop((0, 30, 640, 455)), tour.crop((650, 30, 1290, 455)), tour.crop((650, 500, 1290, 920))]
    pos = [(930, 40, -4), (1160, 150, 3), (900, 285, 2)]
    for c, (x, y, ang) in zip(crops, pos):
        c = c.resize((450, 300))
        card = Image.new("RGB", (462, 312), "white")
        card.paste(c, (6, 6))
        card = card.convert("RGBA").rotate(ang, expand=True, resample=Image.BICUBIC)
        im.paste(card, (x, y), card)
    im.save(HERE / "hero.png")


# --------------------------------------------------------------------------- roadmap
def roadmap():
    W, H = 1600, 360
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    items = [("0", "Cài đặt", "TRƯỚC lớp", "venv · 3 model\nCVAT · (Colab)", GRAY), ("1", "Gold", "25'", "gán tay 4 ảnh\nKHÔNG AI", GREEN),
             ("2", "Tour 3 model", "30'", "YOLO26 · YOLOE\nSAM", PURPLE), ("3", "Ngưỡng", "10'", "theo lớp\ntrên gold", ORANGE),
             ("4", "Review", "95'", "24 ảnh + 12 bẫy\nchecklist 4 bước", RED), ("5", "Mask SAM", "15'", "3 vật khó\nclick dương/âm", TEAL),
             ("6", "Calibration\n+ QC + báo cáo", "45'", "guideline v1.1\nRelease/Hold/Rework", NAVY)]
    n = len(items)
    step = (W - 280) / (n - 1)
    d.line([(140, 120), (W - 140, 120)], fill="#d5dbe3", width=8)
    for i, (num, title, t, sub, c) in enumerate(items):
        cx = int(140 + i * step)
        d.ellipse((cx - 38, 82, cx + 38, 158), fill=c)
        text(d, (cx, 120), f"{num}", 40, "white", True, "mm")
        text(d, (cx, 180), title.split("\n")[0], 24, NAVY, True, "ma")
        yy = 210
        if "\n" in title:
            text(d, (cx, 208), title.split("\n")[1], 24, NAVY, True, "ma")
            yy = 238
        text(d, (cx, yy), t, 20, c, True, "ma")
        for k, line in enumerate(sub.split("\n")):
            text(d, (cx, yy + 32 + k * 24), line, 18, GRAY, False, "ma")
    text(d, (60, 20), "Lộ trình 7 Part  (4 giờ trên lớp + cài đặt làm trước)", 26, NAVY, True)
    im.save(HERE / "roadmap.png")


# --------------------------------------------------------------------------- bốn họ model
def families():
    W, H = 1600, 430
    im = Image.new("RGB", (W, H), LIGHT)
    d = ImageDraw.Draw(im)
    text(d, (40, 20), "Bốn họ model pre-label: mỗi họ một kiểu đầu vào", 28, NAVY, True)
    cards = [("Click", "SAM 1 · SAM 2.1", "ảnh + 1 click", "mask", PURPLE, "click"),
             ("Gõ chữ", "YOLOE · Grounding DINO\nOWLv2 · SAM 3", 'ảnh + "tuk-tuk"', "box (+ mask)", ORANGE, "text"),
             ("Ảnh mẫu", "T-Rex · OWLv2 (mẫu)", "ảnh + vật mẫu", "box vật giống mẫu", TEAL, "ex"),
             ("Danh mục cố định", "YOLO26", "ảnh", "box + 1 trong 80 lớp", GREEN, "cls")]
    cw = (W - 80 - 3 * 24) // 4
    for i, (t, ms, inp, out, c, kind) in enumerate(cards):
        x = 40 + i * (cw + 24)
        rrect(d, (x, 80, x + cw, 400), 20, fill="white", outline="#dde3ea")
        d.rounded_rectangle((x, 80, x + cw, 140), 20, fill=c)
        d.rectangle((x, 120, x + cw, 140), fill=c)
        text(d, (x + 20, 94), t, 28, "white", True)
        # glyph
        gx, gy = x + 24, 160
        rrect(d, (gx, gy, gx + 150, gy + 100), 8, fill="#e9eef4")
        d.rectangle((gx + 20, gy + 40, gx + 100, gy + 85), fill="#c7d2de")
        if kind == "click":
            d.ellipse((gx + 54, gy + 56, gx + 66, gy + 68), fill=GREEN)
        elif kind == "text":
            text(d, (gx + 75, gy + 14), '"tuk-tuk"', 16, NAVY, True, "ma")
        elif kind == "ex":
            d.rectangle((gx + 16, gy + 36, gx + 104, gy + 88), outline=TEAL, width=3)
        else:
            text(d, (gx + 75, gy + 14), "80 lớp", 16, NAVY, True, "ma")
        arrow(d, (gx + 160, gy + 50), (gx + 200, gy + 50), c)
        rrect(d, (gx + 208, gy, gx + 208 + 90, gy + 100), 8, fill="#e9eef4")
        if kind in ("click",):
            d.polygon([(gx + 222, gy + 80), (gx + 232, gy + 44), (gx + 262, gy + 38), (gx + 284, gy + 70), (gx + 270, gy + 88)], fill=c)
        else:
            d.rectangle((gx + 222, gy + 40, gx + 284, gy + 86), outline=c, width=4)
        text(d, (x + 20, 280), ms, 20, NAVY, True)
        text(d, (x + 20, 340 if "\n" not in ms else 350), f"vào: {inp}", 17, GRAY)
        text(d, (x + 20, 364 if "\n" not in ms else 374), f"ra: {out}", 17, GRAY)
    im.save(HERE / "model_families.png")


# --------------------------------------------------------------------------- CVAT mock: màn hình job
def cvat_job_ui():
    W, H = 1500, 860
    im = Image.new("RGB", (W, H), "#e9edf2")
    d = ImageDraw.Draw(im)
    rrect(d, (20, 20, W - 20, 70), 6, fill="white", outline="#cdd5de")
    text(d, (40, 33), "Task #12  >  Job #34      Menu    Undo    Redo    Save", 20, NAVY, True)
    rrect(d, (W - 420, 28, W - 290, 62), 6, fill="#f0f3f7", outline="#cdd5de")
    text(d, (W - 405, 35), "Filters", 18, NAVY)
    # toolbar trái
    rrect(d, (20, 90, 90, 760), 6, fill="white", outline="#cdd5de")
    for k, lbl in enumerate(["Cur", "Box", "Poly", "AI", "Undo"]):
        d.rounded_rectangle((30, 106 + k * 70, 80, 150 + k * 70), 6, fill="#f0f3f7")
        text(d, (55, 128 + k * 70), lbl, 15, NAVY, True, "mm")
    # canvas
    img = Image.open(ROOT / "data/images/img_017.jpg").convert("RGB")
    img = fit(img, 900, 640)
    im.paste(img, (110, 100))
    dd = ImageDraw.Draw(im)
    for (x1, y1, x2, y2, c, lab) in [(420, 330, 690, 640, ORANGE, "three_wheeler 0.86"), (150, 330, 330, 630, GREEN, "person 0.89"),
                                     (170, 330, 470, 600, PURPLE, "three_wheeler 0.36")]:
        dd.rectangle((x1, y1, x2, y2), outline=c, width=4)
        dd.rectangle((x1, y1 - 24, x1 + 200, y1), fill=c)
        text(dd, (x1 + 6, y1 - 20), lab, 15, "white", True)
    text(dd, (110, 748), "<  img_017.jpg   12 / 24   >            (F: ảnh sau · D: ảnh trước)", 18, NAVY)
    # panel phải
    rrect(d, (1030, 90, 1480, 790), 6, fill="white", outline="#cdd5de")
    text(d, (1050, 104), "Objects        [mắt] = ẩn/hiện tất cả", 19, NAVY, True)
    rows = [("#1 three_wheeler", ORANGE, "conf 0.86"), ("#2 person", GREEN, "conf 0.89"), ("#3 three_wheeler", PURPLE, "conf 0.36")]
    for k, (n, c, a) in enumerate(rows):
        y = 150 + k * 78
        rrect(d, (1050, y, 1460, y + 66), 6, fill="#f7f9fb", outline="#dbe2ea")
        d.rectangle((1050, y, 1060, y + 66), fill=c)
        text(d, (1074, y + 8), n, 18, NAVY, True)
        text(d, (1074, y + 36), f"{a}   ☐ can_review", 16, GRAY)
    rrect(d, (1050, 400, 1460, 520), 6, fill="#fff7ec", outline="#ffd9a8")
    text(d, (1064, 410), "Thuộc tính của box đang chọn", 17, NAVY, True)
    text(d, (1064, 440), "conf: 0.86 (máy điền, KHÔNG sửa)", 16, GRAY)
    text(d, (1064, 468), "☐ can_review  (tick khi không chắc QUY TẮC)", 16, GRAY)
    text(d, (1050, 545), "Phím: Tab · Del · Ctrl+1..9 · N · F · Ctrl+S", 17, NAVY, True)
    text(d, (1050, 575), "(F1 xem đầy đủ)", 16, GRAY)
    # callout số
    marks = [(640, 45, "1", "Filters: lọc theo nhãn / thuộc tính / source"), (98, 300, "2", "AI Tools (nút AI): Interactor SAM, Detectors"), (560, 480, "3", "Box máy (màu theo lớp, kèm conf)"),
             (1452, 108, "4", "Ẩn hết nhãn máy → quét ảnh TRƯỚC (bước 0)"), (1452, 404, "5", "conf & can_review"), (400, 760, "6", "Chuyển ảnh")]
    for x, y, n, _ in marks:
        d.ellipse((x - 16, y - 16, x + 16, y + 16), fill=RED)
        text(d, (x, y), n, 18, "white", True, "mm")
    ly = 800
    legend = "  ".join(f"({n}) {t}" for _, _, n, t in marks[:3])
    text(d, (20, 818), "Sơ đồ minh hoạ bố cục (không phải ảnh chụp màn hình; tên nút có thể khác theo phiên bản).   " + "   ".join(f"({n}) {t}" for _, _, n, t in marks[3:]), 15, GRAY)
    text(d, (20, 796), legend, 15, GRAY)
    im.save(HERE / "cvat_job_ui.png")


# --------------------------------------------------------------------------- CVAT mock: Project labels Raw
def cvat_labels_raw():
    W, H = 1300, 600
    im = Image.new("RGB", (W, H), "#e9edf2")
    d = ImageDraw.Draw(im)
    rrect(d, (30, 20, W - 30, H - 20), 10, fill="white", outline="#cdd5de")
    text(d, (60, 40), "Create a new project", 28, NAVY, True)
    text(d, (60, 92), "Name", 18, GRAY)
    rrect(d, (60, 118, 700, 160), 6, fill="white", outline="#9aa6b4")
    text(d, (74, 128), "day08-22001234", 20, NAVY)
    text(d, (60, 190), "Labels", 18, GRAY)
    for k, (t, on) in enumerate([("Constructor", False), ("Raw", True)]):
        x = 60 + k * 150
        rrect(d, (x, 220, x + 140, 262), 6, fill=TEAL if on else "#f0f3f7", outline="#9aa6b4")
        text(d, (x + 70, 241), t, 20, "white" if on else NAVY, True, "mm")
    rrect(d, (60, 272, 840, 520), 8, fill="#14213d")
    code = ['[', '  {"name": "person", "type": "any",', '   "attributes": [', '     {"name": "conf", "input_type": "text", ...},',
            '     {"name": "can_review", "input_type": "checkbox", ...}]},', '  {"name": "three_wheeler", ...},', '  {"name": "ignore_region", ...}', ']']
    for k, ln in enumerate(code):
        text(d, (78, 288 + k * 26), ln, 17, "#9be9c8")
    rrect(d, (880, 272, 1250, 400), 10, fill="#fff7ec", outline="#ffd9a8")
    text(d, (896, 286), "1. Mở cvat/labels_generated.json", 16, NAVY, True)
    text(d, (896, 316), "2. Copy TOÀN BỘ nội dung", 16, NAVY, True)
    text(d, (896, 346), "3. Dán vào ô Raw", 16, NAVY, True)
    rrect(d, (880, 420, 1050, 466), 8, fill="#2f7cf6")
    text(d, (965, 443), "Save", 20, "white", True, "mm")
    rrect(d, (1070, 420, 1250, 466), 8, fill="#f0f3f7", outline="#9aa6b4")
    text(d, (1160, 443), "Submit & Open", 18, NAVY, True, "mm")
    text(d, (60, 540), "Sơ đồ minh hoạ (không phải ảnh chụp màn hình). Tên nhãn phải TRÙNG TỪNG CHỮ với taxonomy.json.", 16, GRAY)
    im.save(HERE / "cvat_project_labels.png")


# --------------------------------------------------------------------------- CVAT mock: upload flow
def cvat_upload_flow():
    W, H = 1500, 520
    im = Image.new("RGB", (W, H), "#e9edf2")
    d = ImageDraw.Draw(im)
    shots = [("1  Task → Actions ⋮", 0), ("2  Chọn format & file", 1), ("3  Kiểm tra sau khi nạp", 2)]
    cw = (W - 80 - 2 * 40) // 3
    for i, (t, k) in enumerate(shots):
        x = 40 + i * (cw + 40)
        rrect(d, (x, 60, x + cw, 480), 12, fill="white", outline="#cdd5de")
        text(d, (x + 20, 20), t, 24, NAVY, True)
        if k == 0:
            rrect(d, (x + 30, 90, x + cw - 30, 150), 8, fill="#f7f9fb", outline="#dbe2ea")
            text(d, (x + 46, 108), "#12  batch-review", 20, NAVY, True)
            text(d, (x + cw - 150, 108), "Actions  ⋮", 18, NAVY)
            rrect(d, (x + cw - 250, 160, x + cw - 30, 440), 8, fill="white", outline="#cdd5de")
            for j, lbl in enumerate(["Upload annotations", "Export task dataset", "Automatic annotation", "Backup Task", "View analytics", "Delete"]):
                if j == 0:
                    d.rectangle((x + cw - 248, 168, x + cw - 32, 206), fill="#e6f1ff")
                text(d, (x + cw - 236, 176 + j * 40), lbl, 18, NAVY if j != 0 else "#1d5fd1", j == 0)
        elif k == 1:
            text(d, (x + 24, 90), "Upload annotations", 22, NAVY, True)
            text(d, (x + 24, 140), "Import format", 17, GRAY)
            rrect(d, (x + 24, 166, x + cw - 24, 208), 6, fill="white", outline="#9aa6b4")
            text(d, (x + 38, 176), "COCO 1.0 (box)  |  CVAT 1.1 (XML)", 17, NAVY)
            text(d, (x + 24, 236), "Annotation file", 17, GRAY)
            rrect(d, (x + 24, 262, x + cw - 24, 340), 6, fill="#f7f9fb", outline="#9aa6b4")
            text(d, (x + 36, 290), "pre_traps_import.coco.json", 17, NAVY, True)
            rrect(d, (x + cw - 130, 380, x + cw - 24, 424), 8, fill="#2f7cf6")
            text(d, (x + cw - 77, 402), "OK", 20, "white", True, "mm")
        else:
            img = fit(ROOT / "data/images/img_017.jpg", cw - 40, 220)
            im.paste(img, (x + 20, 90))
            dd = ImageDraw.Draw(im)
            dd.rectangle((x + 190, 150, x + 330, 300), outline=ORANGE, width=4)
            dd.rectangle((x + 40, 160, x + 110, 300), outline=GREEN, width=4)
            text(dd, (x + 24, 330), "✔ Đếm số box ở ảnh đầu = số trong viz/?", 18, NAVY, True)
            text(dd, (x + 24, 362), "✔ Có box (không phải 0)?  Đúng lớp?", 18, NAVY, True)
            text(dd, (x + 24, 394), "✘ Hai bộ box chồng nhau → xoá hết,", 18, RED, True)
            text(dd, (x + 24, 422), "   rồi nạp lại MỘT lần.", 18, RED, True)
    text(d, (40, 490), "Sơ đồ minh hoạ (không phải ảnh chụp màn hình). Ảnh thật của menu Actions: docs/assets/cvat/cvat-actions-menu.jpg", 15, GRAY)
    im.save(HERE / "cvat_upload_flow.png")


# --------------------------------------------------------------------------- checklist
def review_checklist():
    W, H = 1600, 430
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    text(d, (40, 18), "Checklist review: thứ tự cố định cho MỖI ảnh", 28, NAVY, True)
    steps = [("0", "Quét ảnh\nTRƯỚC", "ẩn nhãn máy,\nđếm vật bằng mắt", GRAY, "—"), ("1", "Bỏ sót?", "vật nào chưa\ncó box?", RED, "added"),
             ("2", "Thừa?", "box nào không\ncó vật?", ORANGE, "deleted"), ("3", "Sai nhãn?", "tuk-tuk bị gọi\ncar / truck?", PURPLE, "edited*"),
             ("4", "Lỏng?", "box / mask ôm\nđủ vật chưa?", TEAL, "edited")]
    cw = (W - 80 - 4 * 24) // 5
    for i, (n, t, sub, c, state) in enumerate(steps):
        x = 40 + i * (cw + 24)
        rrect(d, (x, 80, x + cw, 330), 16, fill="#f7f9fb", outline=c, width=3)
        d.ellipse((x + 16, 94, x + 64, 142), fill=c)
        text(d, (x + 40, 118), n, 28, "white", True, "mm")
        for k, ln in enumerate(t.split("\n")):
            text(d, (x + 80, 98 + k * 30), ln, 26, NAVY, True)
        for k, ln in enumerate(sub.split("\n")):
            text(d, (x + 20, 175 + k * 26), ln, 20, GRAY)
        rrect(d, (x + 20, 270, x + 20 + 140, 310), 20, fill=c)
        text(d, (x + 90, 290), state, 20, "white", True, "mm")
        if i < 4:
            arrow(d, (x + cw + 2, 205), (x + cw + 22, 205), GRAY)
    text(d, (40, 350), "Không lỗi → accepted.   * Đổi lớp tính là xoá + thêm trong review_stats.py.   Nghỉ 3–5' sau mỗi ~10 ảnh.", 20, NAVY)
    text(d, (40, 384), "Điểm tin cậy (conf) chỉ để ưu tiên nhìn kỹ, KHÔNG để bỏ qua box điểm cao. Lỗi bỏ sót không hiện trên bất kỳ điểm số nào.", 19, GRAY)
    im.save(HERE / "review_checklist.png")


# --------------------------------------------------------------------------- hai đường vào CVAT
def two_paths():
    W, H = 1600, 520
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    text(d, (40, 16), "Hai cách đưa nhãn máy vào CVAT", 28, NAVY, True)
    # lane A
    rrect(d, (30, 70, W - 30, 250), 16, fill="#eefaf7", outline=TEAL, width=3)
    text(d, (50, 82), "A · OFFLINE → IMPORT  (mặc định, chạy mọi nơi)", 22, TEAL, True)
    boxes = [("Máy bạn / Colab", "prelabel.py\nYOLO26 · YOLOE · SAM"), ("pre.coco.json\n(+ .xml cho polygon)", "box · mask · conf"), ("CVAT", "Actions →\nUpload annotations"),
             ("Review", "checklist 4 bước\n+ ảnh bẫy"), ("Export COCO 1.0", "submission/labels/")]
    bw = (W - 100 - 4 * 40) // 5
    for i, (t, s) in enumerate(boxes):
        x = 50 + i * (bw + 40)
        rrect(d, (x, 120, x + bw, 230), 12, fill="white", outline=TEAL, width=2)
        for k, ln in enumerate(t.split("\n")):
            text(d, (x + bw // 2, 134 + k * 22), ln, 19, NAVY, True, "ma")
        for k, ln in enumerate(s.split("\n")):
            text(d, (x + bw // 2, 182 + k * 20), ln, 16, GRAY, False, "ma")
        if i < 4:
            arrow(d, (x + bw + 4, 175), (x + bw + 36, 175), TEAL)
    # lane B
    rrect(d, (30, 280, W - 30, 490), 16, fill="#f4f1ff", outline=PURPLE, width=3)
    text(d, (50, 292), "B · TRỰC TIẾP TRONG CVAT  (tuỳ chọn trong 4 giờ)", 22, PURPLE, True)
    cols = [("B1  AI Tools", "Interactor SAM: click trái = thêm,\nclick phải = bớt → mask"), ("B2  Automatic annotation", "model có sẵn trong CVAT\nLabels matching · Clean previous: TẮT"),
            ("B3  cvat-cli", "model CỦA BẠN chạy trên máy\ntask auto-annotate / agent")]
    cw = (W - 100 - 2 * 40) // 3
    for i, (t, s) in enumerate(cols):
        x = 50 + i * (cw + 40)
        rrect(d, (x, 335, x + cw, 470), 12, fill="white", outline=PURPLE, width=2)
        text(d, (x + 20, 348), t, 22, NAVY, True)
        for k, ln in enumerate(s.split("\n")):
            text(d, (x + 20, 392 + k * 24), ln, 18, GRAY)
    im.save(HERE / "two_paths.png")


# --------------------------------------------------------------------------- cây quyết định QC
def decision_tree():
    W, H = 1500, 470
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    text(d, (40, 16), "Chọn Release / Hold / Rework: đi từ trên xuống, dừng ở câu đầu tiên đúng", 26, NAVY, True)
    rows = [("1", "Guideline còn câu hỏi CHƯA chốt, hoặc dữ liệu hỏng / không dùng được?", "HOLD", ORANGE, "làm rõ rồi quay lại"),
            ("2", "Ảnh lỗi > 2%  HOẶC  một loại lỗi lặp lại (>= 3 lỗi và >= 50%)  HOẶC  bắt bẫy < 80%  HOẶC  gán lại mù dưới ngưỡng?", "REWORK", RED, "sửa lỗi, đo lại, ghi vòng sau"),
            ("3", "Không rơi vào hai câu trên", "RELEASE", GREEN, "kèm qc_report_auto.md + REPORT.md")]
    for i, (n, q, res, c, note) in enumerate(rows):
        y = 80 + i * 120
        rrect(d, (40, y, 1000, y + 96), 14, fill="#f7f9fb", outline="#cdd5de")
        d.ellipse((56, y + 24, 104, y + 72), fill=GRAY)
        text(d, (80, y + 48), n, 26, "white", True, "mm")
        words, line, ln = q.split(" "), "", []
        for w_ in words:
            if d.textlength(line + w_ + " ", font=font(20)) > 860:
                ln.append(line); line = ""
            line += w_ + " "
        ln.append(line)
        for k, t in enumerate(ln[:3]):
            text(d, (124, y + 14 + k * 28), t, 20, NAVY)
        arrow(d, (1008, y + 48), (1060, y + 48), c)
        rrect(d, (1070, y + 6, 1460, y + 90), 14, fill=c)
        text(d, (1265, y + 34), res, 30, "white", True, "mm")
        text(d, (1265, y + 68), note, 17, "white", False, "mm")
        if i < 2:
            text(d, (80, y + 100), "không", 16, GRAY, True, "mm")
    im.save(HERE / "decision_tree.png")


if __name__ == "__main__":
    for f in (hero, roadmap, families, cvat_job_ui, cvat_labels_raw, cvat_upload_flow, review_checklist, two_paths, decision_tree):
        f()
        print("ok", f.__name__)
