# Quy trình review nhãn máy (Part 4)

Máy gán sẵn, **bạn là chốt chặn chất lượng**. Hai kỹ năng khác nhau: vẽ chính xác và **bắt lỗi của máy**. Trang này là quy trình bạn phải theo
(và phải **tự đo** mình có theo đúng không bằng ảnh bẫy).

## 1. Vì sao cần quy trình: thiên kiến tự động hóa

Nhãn máy càng trông hợp lý, mắt bạn càng ít nhìn lại ảnh. Bốn nguyên nhân khiến lỗi lọt:

| Nguyên nhân | Biểu hiện |
|---|---|
| Neo nhãn | mắt tìm vật theo box có sẵn, không quét cả ảnh |
| Điểm cao | `conf 0.9` gợi cảm giác "đúng" (nhưng **điểm không phải xác suất đúng**) |
| Lỗi nhỏ | lệch vài pixel khó thấy |
| Mệt mỏi | sau hàng chục ảnh, chú ý giảm |

Hai loại lỗi của người duyệt: **bỏ qua** (máy không vẽ thì bạn không kiểm: *lỗi bỏ sót* là lỗi khó thấy nhất) và **làm theo** (nhận gợi ý sai của máy).

## 2. Checklist bốn bước cho MỖI ảnh (thứ tự cố định)

![Checklist review](assets/review_checklist.png)

**Bước 0 — Quét ảnh trước.** Ẩn hết nhãn máy. Nhìn ảnh ~15 giây, tự đếm vật theo lớp (nhẩm: "6 người, 4 xe máy, 1 tuk-tuk, vài xe ở xa").
Chỉ khi xong mới hiện nhãn máy.

| # | Câu hỏi | Việc làm | Trạng thái cuối |
|---|---|---|---|
| 1 | **Bỏ sót**: còn vật nào chưa có box? (đặc biệt xa, bị che, mép ảnh, tuk-tuk / xích lô) | vẽ thêm (`N`) | `added` |
| 2 | **Thừa**: box nào không có vật thật? (bóng, tượng, phản chiếu, cột, biển) | xoá (`Del`) | `deleted` |
| 3 | **Sai nhãn**: đúng lớp chưa? (tuk-tuk bị gọi `car`/`truck`; xe ba gác thành `motorbike`) | đổi nhãn (`Ctrl+số`) | `edited` (xem 4) |
| 4 | **Lỏng**: box ôm đủ vật, không dư nền? mask bám biên? | kéo lại | `edited` |

Không lỗi → giữ nguyên (`accepted`). Gặp ca bạn **không chắc quy tắc** → tick `can_review` (và ghi vào bảng calibration sau).

> Trong `review_stats.py`, **đổi lớp tính là xoá + thêm** (máy sai lớp = một box máy bị xoá và một box mới): đó là hợp đồng chấm của slide.
> Con số riêng "đổi lớp" được in thêm ở báo cáo.

## 3. Quy tắc làm việc (phải theo)

1. **Quét trước, nhìn nhãn máy sau** (bước 0) trên **mọi** ảnh. Với ảnh đông (> 30 vật), quét theo góc phần tư.
2. **Duyệt theo checklist cố định**, không nhảy cóc theo hứng.
3. **Nghỉ**: sau mỗi ~10 ảnh nghỉ 3–5 phút (mệt mỏi là một nguyên nhân lỗi lọt).
4. **Không mở `traps_key.json`** trước khi export `batch_reviewed.json`. Đó là đáp án của ảnh bẫy.
5. **Không hạ chuẩn vì điểm cao:** `conf` chỉ giúp bạn *ưu tiên nhìn kỹ* box điểm thấp, không bao giờ để *bỏ qua* box điểm cao.
6. **Ghi giờ**: ghi vào `submission/time_log.csv` thời gian từng bước (xem dưới). Số này quyết định pre-label có đáng dùng không.

## 4. Ảnh bẫy: đo chính bạn

`inject_traps.py` chèn **12 lỗi đã biết** vào pre-label của batch, trộn lẫn và **không đánh dấu**:

| Loại bẫy | Ví dụ | Bạn phải |
|---|---|---|
| `missing` | máy bỏ sót một box đúng | vẽ thêm |
| `extra` | một box "ma" ở chỗ không có vật | xoá |
| `wrong_class` | gán sai lớp (ví dụ `car` → `truck`) | sửa lớp |
| `loose` | box lệch / lỏng rõ | kéo lại |

`review_stats.py --key ...` tính **tỉ lệ bắt lỗi** = bẫy bạn sửa đúng / tổng bẫy. Dưới **80%** (tức < 10/12) thì duyệt lại lô (và giải thích vì sao lọt: loại bẫy nào, ảnh nào, bạn đã bỏ bước nào của checklist).
Bẫy **không** thay thế QC mẫu (Part 6); nó đo **người duyệt**, còn QC đo **nhãn**. Chỉ 12 bẫy nên một bẫy lọt đổi kết quả: đọc số đo cùng danh sách bẫy lọt, đừng chỉ đọc %. Bẫy bị chấm theo box **gốc** của máy: nếu bạn sửa đúng nhưng khác box gốc thì ghi chú trong REPORT.

## 5. Ghi giờ (time_log.csv)

Cột: `step, images, minutes, note`. Các `step` bắt buộc: `gold_manual` (gán tay 4 ảnh gold),
`batch_review` (duyệt 24 ảnh batch). Bấm đồng hồ thật, **không ước lượng sau**. Thêm dòng `break` nếu nghỉ dài (không tính vào phút làm).
Dùng để trả lời trong REPORT: *một ảnh gán tay hết bao lâu, duyệt hết bao lâu; pre-label có tiết kiệm không, tiết kiệm bao nhiêu %?* Lưu ý: 4 ảnh gold bạn gán tay trước nên đã quen kiểu cảnh này: nói rõ thiên lệch đó khi so thời gian.

## 6. Sau khi export: đo

```bash
python scripts/validate_labels.py --all
python scripts/review_stats.py --pre submission/prelabel/pre_traps.coco.json --final submission/labels/batch_reviewed.json \
    --gold submission/labels/gold_manual.json --key submission/prelabel/traps_key.json
```

Đọc `submission/qc/review_stats.md`: accepted / edited / deleted / added; precision-recall của **máy** so với gold; tỉ lệ bắt bẫy (đo **bạn**).

> Vì sao 4 ảnh gold **không** nằm trong Task review: bạn vừa tự gán tay chúng, duyệt lại ngay sẽ đo trí nhớ chứ không đo thiên kiến. *(Tuỳ chọn nâng cao)*: sau ≥ 24 giờ, tạo Task `gold-review` từ `work/tasks/gold/` + pre-label gold, duyệt, export `submission/labels/gold_reviewed.json`, rồi chạy lại `review_stats.py --final submission/labels/gold_reviewed.json --splits gold --out submission/qc/review_stats_gold ...` để có thêm **residual / new error** (lỗi của máy bạn giữ lại / lỗi bạn tạo ra). Cách đọc từng số: [CALIBRATION_QC.md](CALIBRATION_QC.md).

### Ba quy tắc giảm thiên kiến của riêng bạn (ghi vào REPORT.md)

Sau khi thấy số đo, viết **3 quy tắc** (mỗi quy tắc một câu, **kiểm tra được bằng số hoặc việc làm**) để lần duyệt sau tốt hơn.
Ví dụ hay: "Ẩn nhãn máy và đếm vật trước ở 100% ảnh (kiểm bằng log)". Ví dụ dở: "Cố gắng cẩn thận hơn".
