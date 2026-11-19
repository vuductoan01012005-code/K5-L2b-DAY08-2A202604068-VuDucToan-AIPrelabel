# Thông tin học viên

- Họ và tên: Vũ Đức Toàn
- MSHV:2A202604068
- Lớp (L2 / L3 / khác): L2b
- Tài khoản GitHub: https://github.com/vuductoan01012005-code
- Repo bài nộp (URL):https://github.com/vuductoan01012005-code/K5-L2b-DAY08-2A202604068-VuDucToan-AIPrelabel
- Commit chốt bài (hash): b22d0482cedbf8051bcd36f24ebb969e30f43ae9
- Máy: Windows 11 + PowerShell 5, không GPU (chạy CPU; một số lần chạy đầu trên CUDA vì có GPU máy học — model_log ghi CUDA ở r01–r06, CPU ở r07–r12, Part 5 r13/r14 để trống thiết bị theo yêu cầu)
- CVAT (Docker): v2.74.1

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 429/435 = 98.6%, 3/435 = 0.7%, 3/435 = 0.7% (mẫu số: 435 box máy); added: 5/437 = 1.1% (mẫu số: 437 đối tượng sau duyệt; 2 trong số đó là đổi lớp, sót thật ≈ 3)
- So với gold: precision / recall của **máy** 23/66 = 34.8% (KTC 95%: 24%–47%) / 23/28 = 82.1% (KTC 95%: 64%–92%)
- Bắt bẫy: 4/12 = 33.3% (KTC 95%: 14%–61%), chưa đạt ngưỡng 80% (10/12)
- Thời gian: gán tay và duyệt đều **khoảng 1 phút/ảnh**: `gold_manual` 4/4 = 1.0 phút/ảnh, `batch_review` 24/24 = 1.0 phút/ảnh.
- Lỗi QC: 100.0% theo ảnh (8/8), 6.8% theo đối tượng (15/219); cận trên Wilson 95% của tỉ lệ ảnh lỗi thật là 100% — mẫu 8 ảnh quá nhỏ để chứng minh ngưỡng 2%.
- **Quyết định:** **Rework** — lý do một dòng: ảnh QC lỗi 8/8 vượt ngưỡng 2%, bắt bẫy 4/12 = 33.3% (< 80%), và guideline v1.0 mới chốt được một phần (câu 1/2/4/5 chưa chốt), nên không thể Release bằng số đo hiện có; việc cần làm: sửa 15 lỗi QC + box < 20 px còn sót, duyệt lại theo bước 0, ghi `times` thật và kiểm lại trên mẫu ≥ 20 ảnh.
