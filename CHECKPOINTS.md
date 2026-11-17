# Checkpoints — Day 08 (240 phút trên lớp)

Mốc tính từ lúc bắt đầu buổi, là điểm **tự kiểm tra**, không phải hạn nộp. Cài đặt (Part 0) làm **trước** buổi học. Deadline do giảng viên/coach thông báo.

| Mốc | Mục tiêu | Tự kiểm tra |
|---|---|---|
| Trước lớp | **CP0** Cài đặt | `check_env.py` SẴN SÀNG (hoặc Colab chạy được); Project CVAT có 8 nhãn; đã đọc label contract |
| 0:25 | **CP1** Gold | `gold_manual.json` qua `validate_labels` (4 ảnh); chưa hề xem pre-label của 4 ảnh gold; có `time_log` |
| 0:55 | **CP2** Tour model | `model_log.csv` ≥ 4 dòng có nhận xét (YOLO26, YOLOE, **SAM**); đã chạy `--split all --conf 0.05` |
| 1:05 | **CP3** Ngưỡng | `threshold_sweep.md` có ngưỡng theo lớp |
| 1:15 | Nghỉ 10' | |
| 2:50 | **CP4** Review | `batch_reviewed.json` (24 ảnh); `review_stats.md` có bắt bẫy % |
| 3:05 | **CP5** Mask | ≥ 3 ảnh trong `submission/masks/` |
| 3:50 | **CP6** Calibration + QC | `calibration_decisions.csv` ≥ 3, `guideline_v1.1`, `REPORT.md` có quyết định |
| 4:00 | Nộp | `check_submission.py` sạch; đã nộp link |

## Cần hiểu ở mỗi mốc
- **CP1:** vì sao gold phải gán **trước** khi thấy máy? Neo nhãn làm sai số đo nào?
- **CP2:** vì sao YOLO26 gọi tuk-tuk là `car`? Open-vocabulary khác closed-set ở đâu?
- **CP3:** vì sao chọn ngưỡng **theo lớp**? "Điểm 0.9" có phải "90% đúng" không? Chọn và chấm trên cùng gold gây thiên lệch gì?
- **CP4:** `accepted` cao nhưng bắt bẫy thấp nghĩa là gì? Lỗi nào của máy **không hiện** trên điểm số?
- **CP5:** "một click, ba mask": ai quyết mask gồm gì? Mask đúng có cứu được nhãn lớp sai không?
- **CP6:** guideline mơ hồ khác "tôi làm sai" ở đâu? Vì sao 0 lỗi trong vài ảnh chưa chứng minh "dưới 2%"?

Chi tiết lệnh: [GUIDE.md](GUIDE.md). Bài nâng cao: [docs/ADVANCED.md](docs/ADVANCED.md).
