# Quy tắc làm bài

## Hình thức

**Bài cá nhân.** Mỗi người một repo public, một bộ nhãn, một báo cáo, các commit của chính mình (tên repo: xem [SUBMISSION.md](SUBMISSION.md)).
Được **thảo luận khái niệm** với bạn học (IoU là gì, cách nạp nhãn vào CVAT, lỗi cài đặt). **Không** chia sẻ file nhãn, báo cáo, hay đáp án ảnh bẫy.

## Gold phải sạch

- **Part 1 làm trước mọi thứ có AI.** Từ lúc nhận ảnh gold đến khi export `gold_manual.json`: **không** chạy model nào lên ảnh gold, **không** nhìn pre-label của ảnh gold, **không** nhờ AI "nhìn ảnh rồi cho biết nên gán gì".
- Nếu lỡ nhìn pre-label của gold trước khi gán: **ghi thật vào `REPORT.md` mục 8**, báo đó là gold bị nhiễm, và phân tích số đo theo hướng đó. (Không tính là gian lận; che giấu thì tính.)
- Bài chính chọn ngưỡng **trên chính gold** (Part 3): nói rõ trong báo cáo rằng số đo máy trên gold vì thế lạc quan. Bài nâng cao có pilot riêng.

## Sử dụng AI

- **Được** dùng AI (chatbot, Copilot…) để giải thích khái niệm, gợi ý viết script, sửa chính tả. **Ghi công cụ và cách bạn kiểm tra** trong `REPORT.md` mục 8.
- Các model pre-label trong repo là **một phần của bài**: dùng ở Part 2 trở đi đúng như hướng dẫn.
- **Không** để AI/mô hình viết hộ **kết luận** (phân tích số đo, quyết định Release/Hold/Rework, quy tắc guideline). Bạn phải giải thích được từng câu trong file mình nộp.

## Tính trung thực của số liệu

- **Không bịa** nhãn, số đo, dòng log, thời gian, quyết định chưa làm. Số xấu mà trung thực (bắt bẫy 60%, báo cáo **Rework**) vẫn được điểm tốt; **số đẹp mà bịa thì 0 điểm phần liên quan** và báo giảng viên.
- `time_log.csv`: ghi thời gian **thật** (bấm đồng hồ), không ước lượng sau.
- **Ảnh bẫy là honor system:** không mở `traps_key.json` trước khi export `batch_reviewed.json`; không sửa `pre_traps.coco.json` hay `traps_key.json`.
- Không sửa tay `review_stats.*`, `threshold_sweep.*`, `self_check.*`, `qc_report_auto.md`, `release_manifest.json`: đó là đầu ra của script.
- Không đổi mục `rules` trong `submission/taxonomy.json` để "đạt" số đo. Được đổi `version`, `classes`, `attributes`, `prompts`, `coco_map` (và phải khớp guideline v1.1).
- Được sửa lỗi script nếu tìm thấy bug; ghi rõ sửa gì và vì sao trong `REPORT.md` mục 8.
- Không sửa, thêm bớt, đổi tên ảnh trong `data/images/`; không sửa `data/split.csv`.

## Dữ liệu, model và giấy phép

- 36 ảnh lấy từ Wikimedia Commons theo giấy phép mở (CC BY, CC BY-SA, CC0); giữ nguyên `data/ATTRIBUTION.csv`. Không đưa ảnh riêng tư vào repo; **không** gửi ảnh lên dịch vụ AI đám mây (VLM…) trừ khi giảng viên cho phép: ảnh gửi đi là dữ liệu bị chuyển giao.
- Trọng số model **không commit** (`models/`, `*.pt` đã trong `.gitignore`). Thư mục `data/cache/` (kết quả chạy sẵn) **được** commit và chỉ là phương án dự phòng: nếu dùng, ghi vào `REPORT.md` mục 8. Mỗi model có giấy phép riêng (ví dụ Ultralytics YOLO: AGPL-3.0 hoặc giấy phép doanh nghiệp; SAM 3 có điều khoản truy cập riêng): bài này dùng cho **học tập**; trước khi dùng thương mại phải kiểm tra lại.
- **Không commit token / mật khẩu** (HuggingFace, CVAT). Lộ token là lỗi nghiêm trọng: thu hồi token ngay và báo giảng viên. `check_submission.py` có quét sơ bộ nhưng không thay được việc bạn tự kiểm.
- Không commit thư mục `work/`.

## Git và nộp bài

- Repo **public**. Commit **theo từng Part** (≥ 4 commit), bằng tài khoản của bạn.
- Bài được chấm theo commit cuối cùng trước hạn; commit sau hạn tính là nộp muộn.
- Thời hạn và kênh nộp do giảng viên/coach công bố; nộp muộn bị trừ điểm.
