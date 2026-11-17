# Submission — bài cá nhân

## Repo bài nộp

Mỗi học viên **một repo public**: fork repo đề, đổi tên theo mẫu.

| Mẫu tên repo | Ví dụ |
|---|---|
| `K5-L23-DAY08-[MSHV]-[HoTen]-AIPrelabel` | `K5-L23-DAY08-22001234-NguyenVanA-AIPrelabel` |

`HoTen` không dấu, viết liền. Theo hướng dẫn lớp nếu giảng viên đặt quy ước khác.

## Trước khi nộp

1. **Đổi dòng tiêu đề của `README.md`** (dòng bắt đầu bằng `# Day 08 Lab —`, không phải dòng đầu tiên của file):
   ```markdown
   # Day 08 Lab — 22001234 - Nguyễn Văn A
   ```
2. Điền [STUDENT.md](STUDENT.md) (kể cả "Kết quả tóm tắt").
3. Nộp link repo lên LMS **https://aithucchien.vinuni.edu.vn** hoặc kênh giảng viên/coach chỉ định.

Hạn nộp do giảng viên/coach thông báo.

## Cấu trúc bắt buộc

```text
README.md                          # có dòng tiêu đề: # Day 08 Lab — <MSHV> - <Họ và tên>
STUDENT.md                         # thông tin + kết quả tóm tắt
data/ATTRIBUTION.csv               # giữ nguyên
submission/
├── taxonomy.json                  # KHÔNG đổi mục rules
├── REPORT.md                      # báo cáo chính (8 mục): recipe, error profile, bẫy, mask, calibration, QC + quyết định, ghi chú cá nhân
├── model_log.csv                  # >= 4 lần chạy, >= 3 model (YOLO26, YOLOE, SAM), có nhận xét
├── time_log.csv                   # thời gian thật: gold_manual, batch_review
├── guidelines/
│   └── guideline_v1.1_calibrated.md   # CHỈ phần bổ sung so với v1.0
├── prelabel/
│   ├── pre_traps_import.coco.json # file đã nạp vào CVAT (24 ảnh batch, có bẫy)
│   ├── pre_traps.coco.json        # đầy đủ (batch + gold), dùng cho review_stats
│   └── traps_key.json             # đáp án bẫy
├── labels/
│   ├── gold_manual.json           # 4 ảnh gán tay
│   └── batch_reviewed.json        # 24 ảnh sau review
├── masks/                         # >= 3 ảnh mask SAM (sam2_points.py)
├── release/                       # python scripts/build_release.py
│   ├── final_det.coco.json
│   ├── release_summary.csv
│   └── release_manifest.json
└── qc/
    ├── threshold_sweep.{csv,md,png}
    ├── review_stats.{json,md}
    ├── calibration_decisions.csv  # >= 3 dòng, mỗi quyết định một dòng
    └── qc_sample.csv, qc_review.csv, qc_report_auto.md
```

Bài nâng cao (tuỳ chọn) có thể thêm `labels/pilot_reviewed.json`, `labels/seg_reviewed.json`, `labels/self_check_blind.json`, `qc/self_check.*`, `qc/review_stats_seg.*`, `prelabel/seg_pre.coco.json`.
**Không** commit: `work/`, `models/`, `*.pt`, token, mật khẩu, ảnh ngoài bộ dữ liệu.

## Checklist trước khi nộp

- [ ] Repo **public**, tên đúng quy tắc, mở được khi chưa đăng nhập.
- [ ] `README.md` tiêu đề đúng `MSHV - Họ và tên`; `STUDENT.md` điền đủ, kể cả kết quả tóm tắt.
- [ ] ≥ 4 commit bằng tài khoản của bạn, theo từng Part.
- [ ] `python scripts/validate_labels.py --all` sạch.
- [ ] `model_log.csv` (có dòng **SAM**, ghi rõ phiên bản) và `time_log.csv` là số thật; mỗi dòng log có nhận xét.
- [ ] `calibration_decisions.csv` ≥ 3 dòng; nội dung đã vào `guideline_v1.1_calibrated.md`, mỗi ca khó có ảnh ví dụ.
- [ ] `REPORT.md` điền đủ 8 mục: tỉ lệ kèm mẫu số, quyết định Release / Hold / Rework, giới hạn mẫu nhỏ, 3 quy tắc giảm thiên kiến.
- [ ] `submission/masks/` có ≥ 3 ảnh.
- [ ] `python scripts/check_submission.py` không còn `LỖI`.
- [ ] Không có token, `work/`, trọng số, hay dữ liệu ngoài phạm vi.
- [ ] Đã nộp link repo trước hạn.
