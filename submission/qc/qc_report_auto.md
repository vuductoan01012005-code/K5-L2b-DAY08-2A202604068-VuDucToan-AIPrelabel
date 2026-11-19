# Báo cáo QC tự động (nháp)

> **Gợi ý** do script tính; bạn xác nhận. Script chỉ đếm theo phiếu kiểm bằng mắt người.

## Mẫu
- 8 ảnh (3 ca khó + 5 ngẫu nhiên), 219 đối tượng.
- taxonomy v1.0

## Số đo
| Chỉ số | Giá trị | Mẫu số |
|---|---|---|
| Lỗi **theo đối tượng** | 15/219 = 6.8% | 219 đối tượng trong mẫu |
| **Ảnh có lỗi** (dùng cho ngưỡng) | 8/8 = 100.0% | 8 ảnh trong mẫu |
| Máy: accepted / edited / deleted | 429/435 = 98.6% / 3/435 = 0.7% / 3/435 = 0.7% | 435 box máy |
| Máy: added | 5/437 = 1.1% | 437 đối tượng sau duyệt |
| Bắt lỗi bẫy | 4/12 = 33.3% | 12 bẫy |

## Top 3 loại lỗi
| Loại lỗi | Số lỗi | Ảnh ví dụ |
|---|---|---|
| thua | 8 | img_003.jpg, img_013.jpg, img_014.jpg |
| sai_lop | 5 | img_003.jpg, img_006.jpg, img_024.jpg |
| sai_vi_tri_kich_thuoc | 2 | img_021.jpg, img_035.jpg |

**Lỗi hệ thống**: 'thua' lặp lại -> sửa quy tắc / recipe, không chỉ sửa từng ảnh.

## Giới hạn thống kê
- 8 ảnh lỗi trong 8 ảnh mẫu: cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là **100%**. Ngưỡng 2% không thể được **chứng minh** bằng mẫu nhỏ như vậy: hãy nói rõ giới hạn này trong báo cáo.

## Gợi ý quyết định
**REWORK**

- ảnh có lỗi 100.0% > 2%
- lỗi hệ thống: 'thua' chiếm 8/15 lỗi
- bắt bẫy 33% < 80% -> lô duyệt chưa đáng tin

Cây quyết định: Hold (guideline chưa rõ / dữ liệu hỏng) -> Rework (> ngưỡng, lỗi lặp, bắt bẫy / nhất quán thấp) -> Release.
