# Dữ liệu

- `images/img_001.jpg` … `img_036.jpg`: 36 ảnh đường phố / chợ ở Việt Nam và Thái Lan (cạnh dài tối đa 1024 px), **chưa có nhãn**.
- `split.csv`: ảnh nào thuộc tập nào.

| Tập | Ảnh | Vai trò |
|---|---|---|
| `gold` (4) | 001, 012, 020, 036 | bạn gán tay từ đầu, làm chuẩn |
| `batch` (24) | còn lại | lô thật: pre-label → review → QC; cột `seg = 1` (8 ảnh: 003, 004, 009, 010, 013, 017, 023, 031) dùng ở bài nâng cao |
| `pilot` (8) | 002, 007, 011, 015, 016, 019, 030, 032 | **thử model ở Part 2** (không làm nhãn); pilot riêng ở bài nâng cao |

- `ATTRIBUTION.csv`: tên gốc, tác giả, giấy phép (CC BY, CC BY-SA, CC0) và liên kết Wikimedia Commons của từng ảnh.
- `cache/`: kết quả YOLO26 / YOLOE (ngưỡng 0.05, 36 ảnh) chạy sẵn trên máy giảng viên (Apple MPS), **chỉ là phương án dự phòng** khi không chạy được model.
- `fetch_candidates.py`: script **giảng viên** dùng để tìm ứng viên ảnh. Học viên không cần chạy.

Thành phần: ~28% ảnh có xe ba bánh (tuk-tuk, xích lô, ba gác: lớp `three_wheeler`, nằm ngoài COCO), có ảnh ban đêm, ảnh đông đúc (xe máy dày đặc), ảnh mờ do chuyển động, vật bị che, vật ở xa.

Ảnh dùng cho mục đích học tập; khi chia sẻ lại phải giữ ghi công và điều kiện giấy phép của từng ảnh (CC BY-SA yêu cầu chia sẻ tương tự).
**Không sửa, đổi tên hay thêm bớt ảnh**: script dựa vào tên file và `split.csv`.
