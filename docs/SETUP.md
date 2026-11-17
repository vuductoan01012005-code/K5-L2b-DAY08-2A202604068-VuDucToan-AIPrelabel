# Cài đặt môi trường (Part 0: làm TRƯỚC buổi học)

> **Cách nhanh nhất:** mở **[setup_wizard.html](setup_wizard.html)** (tải file về rồi mở bằng trình duyệt). Chọn hệ điều hành (Windows, WSL2, Mac chip Apple, Mac Intel, Ubuntu) và RAM (8 GB hay 16 GB trở lên); trang tự nhận diện máy, đề xuất chạy local hay Colab, rồi hiện từng bước kèm lệnh để copy. Trang này dưới đây là bản đọc.

Mục tiêu: sau trang này, `python scripts/check_env.py` in **SẴN SÀNG** và bạn chạy được YOLO26 trên một ảnh.
Mất khoảng 30–45 phút (phần lớn là chờ tải) nên **làm ở nhà trước buổi học**, không tính vào 4 giờ. Làm ở **thư mục gốc của repo**. Máy yếu: bỏ qua trang này, dùng [Google Colab](COLAB.md).

## 0. Bạn cần gì

| Thứ | Bắt buộc? | Ghi chú |
|---|---|---|
| Python **3.10 – 3.13** | có | **Khuyến nghị 3.12** (3.13 đã có wheel sẵn trên PyPI/Colab; 3.14+ có thể chưa có bản cài sẵn cho torch/opencv). Kiểm tra: `python --version` (macOS/Linux: `python3 --version`) |
| `git` | có | YOLOE tự cài CLIP từ GitHub qua git |
| ~8 GB ổ trống | có | trọng số bài chính ~0.35 GB (bài nâng cao thêm ~6 GB), cộng thư viện ~4 GB |
| GPU | **không** | Có CUDA hay Apple MPS thì nhanh hơn; CPU vẫn làm được (xem mục 5) |
| Docker + CVAT | có | CVAT **chạy bằng Docker trên máy bạn** (không dùng bản web): cài theo [CVAT_AI.md](CVAT_AI.md) mục 0; cần ~5–8 GB RAM khi chạy |
| Tài khoản HuggingFace | tuỳ chọn | chỉ cho SAM 3 (mục 4) |

## 1. Tạo môi trường ảo và cài thư viện

macOS / Linux:

```bash
cd K5-L23-Day08-AI-Assisted-Labeling        # thư mục repo của bạn
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Windows: cài trước **Python 3.12** (hoặc 3.10–3.13 từ python.org, tick "Add to PATH") và **Git** (https://git-scm.com, kèm Git Bash). Rồi dùng PowerShell như dưới, hoặc Git Bash / WSL với khối macOS/Linux:

```powershell
cd K5-L23-Day08-AI-Assisted-Labeling        # thư mục repo của bạn
python -m venv .venv                        # nếu có Python Launcher muốn chỉ định 3.12: py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

> Nếu PowerShell chặn script: `Set-ExecutionPolicy -Scope Process Bypass`, rồi kích hoạt lại.

Máy có **GPU NVIDIA**: cài torch đúng bản CUDA **trước** `requirements.txt` (chọn lệnh trên https://pytorch.org/get-started/locally/), ví dụ:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Dùng `uv` (nhanh hơn, tuỳ chọn): `uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt`.

## 2. Tải trọng số

```bash
python scripts/download_models.py --set core     # YOLO26m, YOLOE-26s-seg, SAM 2.1 base (~0.35 GB): ĐỦ cho bài chính
python scripts/download_models.py --set sam1     # (tuỳ chọn) SAM 1 bản nhẹ mobile_sam (~40 MB); SAM bắt buộc, bạn chọn phiên bản
# Nâng cao (không bắt buộc): --set hf (Grounding DINO + OWLv2, ~2 GB) · --set sam3
```

Máy yếu / chỉ CPU: dùng `--set small` (n-size + SAM 2.1 tiny). Khi đó mọi lệnh có `--size m` đổi thành `--size n`, và `--sam sam2.1_t.pt` (hoặc `mobile_sam.pt` của SAM 1 nếu máy rất yếu).

Trọng số vào thư mục `models/` (không commit; đã có trong `.gitignore`). **YOLOE** lần chạy đầu tải thêm bộ mã hoá chữ (`mobileclip2_b.ts`) vào `models/` và tự cài CLIP từ git nếu máy chưa có `clip`: cần mạng và `git`.

## 3. Kiểm tra

```bash
python scripts/check_env.py
```

Đọc từng dòng: `PASS` ổn, `WARN` có thể bỏ qua, `FAIL` phải sửa. Rồi thử nhanh:

```bash
python scripts/prelabel.py --model yolo26 --size m --images img_011.jpg --conf 0.3 --out work/smoke
```

Kỳ vọng: in vài dòng "Số box theo lớp: person=…, motorbike=…, car=…" và tạo `work/smoke/viz/img_011.jpg` (mở ra xem: có box màu chồng lên ảnh).
Lỗi thường gặp ở mục 6.

## 4. SAM 3 (nâng cao, không bắt buộc)

SAM 3 là model **gated** trên HuggingFace: phải xin quyền và đăng nhập mới tải được.

1. Tạo tài khoản https://huggingface.co, vào https://huggingface.co/facebook/sam3 → đọc điều khoản → **Request access** (chờ được duyệt, có thể mất vài giờ).
2. Tạo **access token** loại *read* tại https://huggingface.co/settings/tokens.
3. Đăng nhập (token **không** được commit, không dán vào file trong repo):
   ```bash
   hf auth login           # hoặc đặt biến môi trường HF_TOKEN (bash: export HF_TOKEN=...; PowerShell: $env:HF_TOKEN="...")
   ```
4. `python scripts/download_models.py --set sam3` rồi `python scripts/prelabel.py --model sam3 ...` (xem [ADVANCED.md](ADVANCED.md) mục 2).

Chưa được cấp quyền? Bỏ qua SAM 3 (chọn SAM 1 hoặc 2.1), ghi vào `model_log.csv` dòng `sam3` với `decision = skipped (chưa có quyền)`.
Bạn vẫn đạt đủ yêu cầu bằng SAM 2.1.

## 5. Máy không có GPU (CPU only) — hoặc dùng Google Colab

**Colab (GPU T4 miễn phí) là lựa chọn tốt nhất cho máy yếu**: chạy model trên Colab, tải file nhãn về nạp vào CVAT trên máy bạn. Hướng dẫn: [COLAB.md](COLAB.md) (notebook `notebooks/day08_colab.ipynb`).
Nếu không dùng Colab:

- Dùng `--set small`, `--size n`, `--imgsz 640` (nhỏ hơn mặc định 1024, nhanh ~2.5×, nhưng bỏ sót vật nhỏ nhiều hơn: ghi nhận xét này vào `model_log.csv`).
- Bỏ Grounding DINO và OWLv2 nếu quá chậm; YOLOE vẫn đủ làm open-vocabulary.
- SAM 2.1 mask cho cả 8 ảnh seg mất vài phút trên CPU: chạy `--images` theo từng ảnh.
- Thêm `--device cpu` vào mọi lệnh nếu `auto` chọn nhầm GPU.

## 6. Lỗi hay gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` (hoặc numpy, cv2…) | Chưa kích hoạt venv: `source .venv/bin/activate`; hoặc quên `pip install -r requirements.txt` |
| `No module named 'clip'` / lỗi `git+https://github.com/ultralytics/CLIP.git` | Máy chưa có `git` hoặc bị chặn GitHub. Cài git; thử `pip install git+https://github.com/ultralytics/CLIP.git` bằng tay |
| Tải trọng số chậm / lỗi mạng | Chạy lại `download_models.py` (tự bỏ qua file đã tải). Nếu lớp học dùng chung mạng: xin file từ giảng viên và đặt vào `models/` |
| `MPS backend out of memory` / lỗi dtype trên Mac | Thêm `--device cpu` cho lệnh đó (OWLv2 đã tự chạy CPU) |
| `CUDA out of memory` | `--size s` hoặc `--imgsz 640`; đóng chương trình khác dùng GPU |
| `Cannot access gated repo ... facebook/sam3` | Chưa được cấp quyền / chưa `hf auth login` (mục 4) |
| `AttributeError: Sam3Model` | `pip install -U "transformers>=5.0"` |
| `check_env.py` báo Python 3.14+ (hoặc thiếu wheel) | Tạo lại venv bằng Python 3.12 (`py -3.12 -m venv .venv` hoặc `python3.12 -m venv .venv`) |
| Chạy xong nhưng `0 box` | Đọc dòng "Tên thô KHÔNG có trong ánh xạ": có thể ánh xạ nhãn sai; hoặc `--conf` quá cao |
