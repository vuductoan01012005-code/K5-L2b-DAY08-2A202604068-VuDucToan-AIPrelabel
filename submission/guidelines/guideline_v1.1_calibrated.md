# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202604068 - Vũ Đức Toàn

> **Chỉ phần thay đổi so với v1.0** (không copy cả file). Mỗi ca có ảnh ví dụ `img_0xx` và khớp một dòng trong `qc/calibration_decisions.csv` (`case_id`).

## 1. Câu hỏi mở (mục 6 của v1.0)

| # | Câu hỏi (mục 6 v1.0) | Quyết định v1.1 | Trạng thái | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | **Chưa chốt.** Không có ảnh nào trong 8 ảnh QC đủ rõ động cơ/bàn đạp. Tạm quy: thấy bàn đạp + khung đạp thường → `bicycle`; thấy yếm máy + động cơ 2 bánh → `motorbike`; còn lại tick `can_review`. | Chưa chốt; cần ảnh gần hơn (v1.2) | — | — |
| 2 | Khối xe máy đỗ dày đặc *gần* camera: từng xe hay `ignore_region`? Ngưỡng cụ thể? | **Chốt quy trình, chưa chốt ngưỡng số**: tách được từng vật rõ → gán từng vật; không tách được → một `ignore_region` bao cả cụm. | Đã áp dụng quy trình; ngưỡng cho xe máy gần camera chưa có vật chứng (img_018 41 đối tượng / img_024 đêm cuối) | img_018.jpg, img_024.jpg | CAL-04 |
| 3 | Hàng nhô ra: box xe có gồm hàng không? Mask thì sao? | **Chốt một phần (box)**: khung/giỏ/thùng *gắn cố định* trên xe → thuộc xe; hàng/giỏ/thùng *treo/roi* xe → không thuộc xe; nhô xa > 20% theo 4.1.5 → không gom. Mask chưa có vật chứng. | Box đã áp dụng; mask chưa chốt | img_014.jpg | CAL-03 |
| 4 | Người trong xe ba bánh có mui: ngưỡng "còn thấy ≥ 20%" (4.1.2) có hợp lý? Hai người đánh giá lệch nhau bao nhiêu? | **Giữ nguyên ngưỡng 20%** + cách đọc cố định: chỉ tính phần thấy được; < 20% diện tích ước tính → không gán; ≥ 20% → gán `person` riêng. | Chưa đo được độ lệch giữa hai người (không có gán lại mù) | img_013.jpg | — |
| 5 | Kính chắn gió / cửa sổ xe: thuộc mask xe hay không? | **Chưa chốt cho mask.** Giữ quyết định hiện tại của v1.0 (kính thuộc vật khi nhìn thấy được), chưa ghi thêm quy tắc. | Chưa chốt; cần thêm mẫu mask | img_013.jpg | — |
| 6 | Box máy chồng nhau quá nhiều ở ảnh đêm: gộp hay giữ? | **Chốt: giữ một box duy nhất mỗi vật**; bỏ box chồng trùng (IoU ≥ 0,4) vì giữ cả hai là trùng đối tượng, không phải hai vật. | Đã áp dụng | img_003.jpg, img_021.jpg | CAL-06 |
| 7 | Mức tin cậy tối thiểu (`conf`) để nhận box máy không cần sửa: có đặt không? | **Chốt: không đặt.** `conf` chỉ để xếp thứ tự duyệt, không thay việc nhìn ảnh. Box `conf` 0,4–0,6 vẫn có thể đúng; box `conf` cao nhưng sai lớp/vị trí vẫn phải sửa. | Đã áp dụng | img_003.jpg, img_024.jpg | CAL-04, CAL-05 |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_006.jpg | Xe van/minivan 4 bánh một mui bị ngụy thành xe 3 bánh | Dán `three_wheeler` vào xe van 4 bánh | Giữ `car`; van/minivan 4 bánh có mui là `car` theo định nghĩa mục 2, bất kể `conf` | Mục 2: van/minivan 4 bánh = `car`; mục 4.3: không dùng `conf` quyết định lớp; không thấy cấu trúc → `can_review` (`CAL-01`, nguồn `can_review`) |
| img_014.jpg | Giỏ/thùng treo trên gánh hàng rong bị dán nhãn `three_wheeler` (#130, #131) | YOLOE --prompts `three_wheeler` | Xóa cả hai box; chỉ giữ `#123 person`; một `three_wheeler` chỉ khi thấy được khung/mũi + 3 bánh/trục + người ngồi | Mục 2 (`three_wheeler` cần thấy cấu trúc); mục 4.3 (không thấy cấu trúc → `can_review`) (`CAL-02`, nguồn `loi_may`) |
| img_014.jpg | Gánh hàng rong: giỏ/thùng *treo/roi* xe bị coi là một phần của xe | Box theo khung gánh / box xe có thể lồng hàng treo | Đối với xe: hàng/giỏ/thùng gắn cố định → thuộc xe; treo/roi xe → không thuộc xe, nhô xa > 20% → không gom (áp dụng 4.1.5) | Mục 4.1.5 (mở rộng): treo/roi xe không thuộc box xe (`CAL-03`, nguồn `cau_hoi_mo`) |
| img_018.jpg, img_024.jpg | Ảnh đêm dày đặc (img_018: 41 đối tượng) + box rất nhỏ ở xa | YOLO26m ra nhiều box `conf` thấp | Tách được từng vật rõ → gán từng vật; không tách được hoặc cao < 20 px → một `ignore_region` bao cụm; không dùng `conf` làm ngưỡng loại trừ | Mục 4.1.6 (giữ nguyên quy tắc, không hạ ngưỡng theo `conf`) (`CAL-04`, nguồn `cau_hoi_mo`) |
| img_018.jpg | Box cao 16–17 px (#152 bicycle 16,6 px; #183 person 16,2 px) | Box máy còn sót dù dưới ngưỡng | Xóa; không vẽ từng vật cao < 20 px; cụm ở xa không tách được → `ignore_region` | Mục 4.1.6 (nhấn mạnh chính sách < 20 px) (`CAL-08`, nguồn `can_review`) |
| img_003.jpg, img_024.jpg | Person bị ghi `bicycle`, car bị ghi `truck` (bẫy `wrong_class` lọt) | YOLO26m `conf` 0,5–0,6 nhưng sai lớp | Sửa lớp: `#5` → `person`, `#289`/`#287` → `car`; không để `conf` quyết định lớp; không thấy đặc trưng cấu trúc → `can_review` | Mục 4.3 (nhấn mạnh: `conf` không quyết định lớp; thiếu đặc trưng → `can_review`) (`CAL-05`, nguồn `bay_lot`) |
| img_003.jpg, img_021.jpg | Box trùng/tách hai nửa trên cùng một xe; box lọc IoU ≈ 0,62 với box chuẩn (bẫy `loose` lọt) | Merge `three_wheeler` chồng nhau; box `loose` | Giữ một box/người trong nhóm chồng trùng; box phải ôm sát vật (IoU ≥ 0,9 với box chuẩn theo ngưỡng của lab, mục 4.2 Equ); nếu chỉ đạt IoU ≈ 0,6 → sửa ngay từ lần duyệt đầu | Mục 4.1.1 (box ôm sát), mục 4.2 Equ (IoU ≥ 0,9 là ngưỡng accepted — chỉ đọc hiểu, không đổi ngưỡng của lab) (`CAL-06`, nguồn `bay_lot`) |
| img_013.jpg | Box `motorbike` phía trong xe tuk-tuk (khoang xe trống) | Box máy thừa (bẫy `extra` lọt) | Xóa; không vẽ box khi không có vật; khoang xe trống không phải vật; quét ảnh trước khi nhìn nhãn máy | Mục 4.1 (không vẽ khi không có vật) (`CAL-07`, nguồn `bay_lot`) |
| img_035.jpg | Một xích lô bị tách thành 2 box `three_wheeler` (#446 + #447 — #447 chỉ ôm mui của người đạp) | YOLOE `three_wheeler` | Gộp/sửa thành 1 box cả xe + mui; người đạp là `person` riêng theo 4.1.4; tách thành 2 box là thừa | Mục 2 (một xe = một `three_wheeler`, gồm mui) (`CAL-09`, nguồn `loi_may`) |
| img_006.jpg, img_035.jpg | Người xa bị ghi `bicycle`/`motorbike`; box chồng `car` lồng một xe van | YOLO26m `conf` 0,35–0,55 | Sửa thành `person` (#431); thiếu đặc trưng cấu trúc → `can_review`; không dùng `conf` quyết định lớp | Mục 4.3 (thiếu đặc trưng → `can_review`) (`CAL-10`, nguồn `loi_may`) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | **Đã chốt và áp dụng:** quy trình tách mỗi vật → `ignore_region` theo cụm khi không tách được (mục 2: van 4 bánh = `car`); không đặt mức `conf` tối thiểu (câu 7); một xe = một box `three_wheeler` kể cả mui; bỏ box chồng trùng (câu 6); thêm quy tắc "treo/roi xe không thuộc xe" cho box (câu 3, box) và "không dùng `conf` quyết định lớp; thiếu đặc trưng cấu trúc → `can_review`" (mục 4.3). **Chưa chốt:** câu 1 (xe đạp điện), câu 2 (ngưỡng số cho khối xe máy gần camera), câu 4 (chuẩn 20% và độ lệch đánh giá), câu 5 (kính/mask) — vẫn giữ quyết định tạm thời của v1.0 và ghi rõ để v1.2 xử lý | CAL-01..CAL-10 |
