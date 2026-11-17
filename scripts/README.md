# Scripts của lab

Chạy **từ thư mục gốc repo** (`python scripts/<tên>.py`), venv đã kích hoạt. Mọi script có `--help`. Lớp, ánh xạ nhãn, prompt và ngưỡng đọc từ `submission/taxonomy.json`:
đổi taxonomy thì **chỉ sửa file đó**, không sửa code.

## Thứ tự dùng

| Part | Script | Việc | Đầu ra chính |
|---|---|---|---|
| 0 | `check_env.py` | kiểm môi trường | in PASS / WARN / FAIL |
| 0 | `download_models.py --set core` | tải trọng số | `models/` |
| 0, 1 | `make_cvat_labels.py` | nhãn dán vào CVAT | `cvat/labels_generated.json` |
| 1 | `prepare_workdirs.py` | copy ảnh thành các thư mục Task (`--advanced`: thêm pilot/seg/selfcheck) | `work/tasks/*` |
| 2 | `prelabel.py` | chạy một model → COCO + ảnh xem nhanh | `<out>/pre.coco.json`, `viz/`, `work/runs.csv` |
| 5 | `sam2_points.py` | SAM (bản bất kỳ) với click | `<out>/<name>.jpg` (→ `submission/masks/`) |
| 2, 3 | `merge_prelabels.py` | gộp nhiều model, áp ngưỡng, loại trùng | `pre.coco.json` |
| 3 | `threshold_sweep.py` | ngưỡng theo lớp | `qc/threshold_sweep.*` |
| 4 | `inject_traps.py` | cài ảnh bẫy | `prelabel/pre_traps_import.coco.json`, `pre_traps.coco.json`, `traps_key.json` |
| 4 | `review_stats.py` | accepted/edited/deleted/added, so với gold, bắt bẫy | `qc/review_stats.*` |
| nâng cao | `coco_to_cvat_xml.py` | COCO → CVAT XML (polygon sạch) | `.xml` |
| 3–6 | `validate_labels.py` | kiểm file COCO export từ CVAT | in màn hình |
| nâng cao | `self_consistency.py` | so hai bộ nhãn (gán lại mù / bạn học) | `qc/self_check.*` |
| 6 | `build_release.py` | nhãn cuối 24 ảnh batch | `release/` |
| 6 | `qc_sample.py`, `qc_report.py` | mẫu QC, số đo, gợi ý quyết định | `qc/qc_sample.csv`, `qc_review.csv`, `qc_report_auto.md` |
| 6 | `check_submission.py` | tự kiểm bài nộp | in LỖI / CẢNH BÁO |
| tuỳ chọn | `cvat_function_yolo26.py` | hàm cho `cvat-cli` (CVAT trực tiếp) | xem docs/CVAT_AI.md B3 |
| tuỳ chọn | `viz.py` | vẽ bất kỳ file COCO lên ảnh | thư mục ảnh |

## `prelabel.py` — cờ quan trọng

| Cờ | Ý nghĩa |
|---|---|
| `--model` | `yolo26` · `yoloe` (bài chính) · `gdino` · `owlv2` · `sam3` (nâng cao) |
| `--size` | cỡ YOLO: `n s m l x` (YOLOE: `s m l`); mặc định `m` / `s` |
| `--split` / `--images` | `pilot` (8 ảnh thử model), `gold`, `batch`, `seg`, `all` (hoặc danh sách tên ảnh) |
| `--conf`, `--thr "person=0.4,car=0.5"` | ngưỡng mặc định, ngưỡng riêng từng lớp |
| `--classes` | chỉ **giữ** các lớp này (ví dụ `three_wheeler`) |
| `--prompts 'tuk-tuk=three_wheeler,...'` | open-vocab: cụm chữ → lớp (mặc định lấy từ `taxonomy.json`) |
| `--segmenter` | `none` · `native` (mask của model) · `sam` (box → SAM; `sam2` là tên cũ) |
| `--sam` | trọng số SAM: SAM 1 `mobile_sam.pt` / `sam_b.pt`; SAM 2.1 `sam2.1_t/s/b/l.pt` (SAM 3: dùng `--model sam3`). **SAM bắt buộc**, bạn chọn phiên bản |
| `--map motorcycle=motorbike` | ánh xạ tên thô → lớp, bổ sung `coco_map` |
| `--device` | `auto` · `cpu` · `mps` · `cuda` |

Cuối mỗi lần chạy, script in **tên thô không có trong ánh xạ** (box bị bỏ im lặng nếu bạn không đọc) và **số box theo lớp**.

## Định dạng file

Mọi file nhãn là **COCO 1.0**: `images`, `categories` (id theo thứ tự lớp trong taxonomy), `annotations` với `bbox` `[x,y,w,h]`,
`segmentation` (danh sách polygon phẳng; rỗng với box), `attributes` (`conf`, `can_review`) và `score` (điểm máy, chỉ có ở pre-label).
Script map lớp theo **tên** và ảnh theo **tên file** (bỏ tiền tố thư mục).

## `review_stats.py` — hợp đồng chấm

Ghép box máy với box cuối **cùng lớp**, theo IoU giảm dần (ngưỡng `match_iou` = 0.5):

| Trạng thái | Điều kiện | Mẫu số của tỉ lệ |
|---|---|---|
| `accepted` | IoU ≥ 0.9 (mask: ≥ 0.85) | box máy |
| `edited` | 0.5 ≤ IoU < ngưỡng accepted | box máy |
| `deleted` | box máy không ghép được (kể cả sai lớp) | box máy |
| `added` | box cuối không ghép được | đối tượng sau duyệt |

Vùng `ignore_region` không bị chấm. So với gold: TP / FP / FN cùng lớp, IoU ≥ 0.5; **residual** = lỗi của máy còn trong nhãn của bạn, **new** = lỗi do bạn tạo ra.
Bẫy: `missing` đúng khi box gốc có lại (cùng lớp, IoU ≥ 0.5); `extra` đúng khi box ma biến mất; `wrong_class` đúng khi lớp gốc trở lại; `loose` đúng khi box cùng lớp có IoU với box gốc ≥ 0.75 (hoặc đã rời box lỗi IoU < 0.8 và vẫn đạt IoU với box gốc ≥ 0.5).

## Script kiểm được gì, không kiểm được gì

Kiểm được: tên file, lớp có trong taxonomy, box trong ảnh, box quá nhỏ, đủ / thừa ảnh, polygon, độ lệch giữa hai bộ nhãn, các số đo QC.
**Không** kiểm được: box có ôm sát vật thật không, lớp có đúng thực tế không, vật mà cả bạn lẫn máy cùng bỏ sót (ngoài gold), việc bạn có theo checklist thật không.
