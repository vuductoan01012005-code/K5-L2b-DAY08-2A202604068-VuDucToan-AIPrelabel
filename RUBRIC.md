# Rubric — 100 điểm

Điểm dựa trên **bằng chứng trong repo và chất lượng phân tích**, **không** dựa vào việc số đo đẹp hay được Release.
Bắt bẫy 60% mà phân tích trung thực nguyên nhân tốt hơn 100% mà không giải thích được.

| # | Hạng mục | Điểm | Bằng chứng |
|---|---|---:|---|
| 1 | Cài đặt, tái lập, git | 8 | `check_submission.py`, commit, `STUDENT.md`, README |
| 2 | Gold gán tay và dùng label contract | 10 | `labels/gold_manual.json`, `time_log.csv`, `REPORT.md` mục 8 |
| 3 | Tour model và recipe | 20 | `model_log.csv`, `REPORT.md` mục 1 |
| 4 | Chọn ngưỡng theo lớp | 8 | `qc/threshold_sweep.*`, `REPORT.md` mục 1 |
| 5 | Review và đo (batch, bẫy, AI error profile) | 28 | `labels/batch_reviewed.json`, `qc/review_stats.*`, `REPORT.md` mục 2–4 |
| 6 | Mask SAM | 8 | `masks/*.jpg`, `REPORT.md` mục 5 |
| 7 | Calibration | 10 | `calibration_decisions.csv`, `guideline_v1.1_calibrated.md` |
| 8 | QC và báo cáo | 8 | `qc/qc_review.csv`, `REPORT.md` mục 7, `release/` |

## 1. Cài đặt, tái lập, git (8)
- 3: `check_submission.py` không còn `LỖI`; repo public, tên đúng, README tiêu đề đúng.
- 3: ≥ 4 commit theo từng Part, bằng tài khoản của bạn; không commit `work/`, trọng số, token.
- 2: `STUDENT.md` đầy đủ; `REPORT.md` mục 8 ghi rõ công cụ AI đã dùng và cách kiểm tra.
- *Mất điểm:* lộ token; commit file nặng; commit gộp một lần cuối.

## 2. Gold và label contract (10)
- 4: `gold_manual.json` hợp lệ (đủ 4 ảnh, qua `validate_labels`), gán **trước** khi thấy nhãn máy (ghi trung thực nếu bị nhiễm).
- 3: `can_review` dùng đúng nghĩa (ca **không chắc quy tắc**); có ghi ca khó theo ảnh (`img_0xx`).
- 3: `time_log.csv` có `gold_manual` thật.
- *Mất điểm:* gold giống hệt pre-label đến từng pixel (nghi dùng AI); tick `can_review` cho mọi box.

## 3. Tour model và recipe (20)
- 6: ≥ 4 lần chạy, ≥ 3 model khác nhau (YOLO26, YOLOE, **SAM bắt buộc**); thông số đủ để chạy lại; ghi rõ phiên bản SAM.
- 8: `what_went_wrong` **cụ thể** (ảnh, vật, loại lỗi), có so sánh giữa ngưỡng / model / prompt.
- 6: `REPORT.md` mục 1: lớp nào dùng model nào và **vì sao**, **phiên bản SAM đã chọn và lý do**; hiểu được lỗi nào **không** sửa bằng ngưỡng (tuk-tuk → car/truck).
- *Mất điểm:* chỉ chạy lệnh trong hướng dẫn, không quan sát; log không khớp `work/runs.csv`.

## 4. Chọn ngưỡng (8)
- 4: `threshold_sweep` chạy, ngưỡng **theo lớp**, chuỗi `--thr` dùng đúng ở Part 4.
- 4: nói rõ giới hạn: chọn và chấm trên cùng gold (4 ảnh), lớp ít vật.
- *Mất điểm:* một ngưỡng chung không lý do.

## 5. Review và đo (28)
- 5: `batch_reviewed.json` hợp lệ (24 ảnh), đã nạp đúng `pre_traps`.
- 8: đọc **đúng** `review_stats` (mẫu số, vì sao loại vùng bẫy khỏi profile, precision-recall của máy so với gold); trả lời máy có đáng dùng không dựa trên `time_log` + `accepted`.
- 8: phân tích bẫy lọt (loại nào, vì sao, bước checklist nào bị bỏ) và liên hệ thiên kiến tự động hóa.
- 7: 3 quy tắc giảm thiên kiến **kiểm tra được bằng số hoặc việc làm**.
- *Mất điểm:* mở `traps_key.json` trước; sửa tay số đo; nói "accepted cao = máy tốt" mà không nhắc tới sót (added, recall trên gold).

## 6. Mask SAM (8)
- 3: ≥ 3 ảnh mask trong `masks/`, ghi rõ phiên bản SAM và số click.
- 3: phân tích: mask gồm người lái / hàng hoá không, đối chiếu guideline 4.2.
- 2: quy tắc / lỗi mask rút ra. *(Cộng điểm phân tích nếu thử Interactor SAM trong CVAT và so sánh.)*

## 7. Calibration (10)
- 4: `calibration_decisions.csv` ≥ 3 dòng, mỗi dòng có nguồn và quyết định; có ít nhất một `loi_may` và một `cau_hoi_mo`.
- 4: `guideline_v1.1` **thật sự chứa** quy tắc của các quyết định; mỗi ca khó có ảnh ví dụ; lịch sử thay đổi đúng.
- 2: phân biệt "guideline mơ hồ" và "tôi / máy sai".
- *Mất điểm:* quyết định không vào v1.1; v1.1 mâu thuẫn với nhãn đã nộp.

## 8. QC và báo cáo (8)
- 3: mẫu QC và cách chọn; **hai cách đếm tỉ lệ lỗi kèm mẫu số**.
- 3: quyết định theo cây Hold → Rework → Release, **khớp số liệu**; nêu giới hạn thống kê của mẫu nhỏ.
- 2: `REPORT.md` đủ mục, đọc được, không chép nguyên output của script.
- *Mất điểm:* báo Release khi số liệu không đạt; chỉ ghi % không mẫu số.

## Trừ điểm chung
- Thiếu thành phần: chỉ chấm phần có bằng chứng.
- **Số liệu giả** (nhãn, thời gian, bẫy, log model): 0 điểm phần liên quan, báo giảng viên.
- Nộp muộn, sai tên repo, sai link: theo [RULES.md](RULES.md).
- **Bài nâng cao** ([docs/ADVANCED.md](docs/ADVANCED.md)): tối đa +5 điểm thưởng, không bù được phần cốt lõi bị thiếu.
