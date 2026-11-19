# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 435; đối tượng sau duyệt: 437; ngưỡng accepted: IoU >= 0.9
- Đã **loại 19 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 429 | 429/435 = 98.6% | 435 box máy |
| edited (sửa) | 3 | 3/435 = 0.7% | 435 box máy |
| deleted (xoá) | 3 | 3/435 = 0.7% | 435 box máy |
| added (vẽ thêm) | 5 | 5/437 = 1.1% | 437 đối tượng sau duyệt |

Trong số deleted + added, **2 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 3.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 252 | 0 | 0 | 0 |
| bicycle | 7 | 1 | 1 | 1 |
| motorbike | 89 | 1 | 0 | 1 |
| car | 50 | 0 | 0 | 0 |
| bus | 3 | 0 | 0 | 1 |
| truck | 7 | 0 | 0 | 1 |
| three_wheeler | 21 | 1 | 2 | 1 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 28 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 23 | 43 | 5 | 34.8% (KTC 95%: 24–47%) | 82.1% (KTC 95%: 64–92%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 4/12 = 33.3% (KTC 95%: 14–61%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 2/3
  - wrong_class: 1/3
  - loose: 0/3
  - missing: 1/3
  - lọt: img_003.jpg (wrong_class), img_013.jpg (extra), img_023.jpg (loose), img_021.jpg (loose), img_026.jpg (missing), img_017.jpg (loose), img_024.jpg (wrong_class), img_005.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
