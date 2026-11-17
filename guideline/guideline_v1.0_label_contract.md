# Label contract — Phố châu Á (v1.0)

> Đây là bản guideline **giảng viên viết sẵn** cho lô ảnh Day 08. Nó đủ để bắt đầu gán nhãn, nhưng **cố ý chưa trả lời hết**
> (xem mục 6). Bạn sẽ dùng nó để gán tay tập gold, rồi dùng các lỗi của máy để **calibration** nó thành `v1.1`.
> Đừng sửa file này: copy sang `submission/guidelines/guideline_v1.1_calibrated.md` rồi sửa bản của bạn.

- Phiên bản: v1.0
- Loại nhãn: **detection** (bounding box) cho 4 + 24 ảnh (gold + batch); **mask** (SAM) cho 3 vật khó ở Part 5 (bài nâng cao: polygon cho 8 ảnh `seg`)
- Công cụ: CVAT. Xuất COCO 1.0

---

## 1. Mục đích và phạm vi

- **Mục đích:** huấn luyện detector / segmenter đếm và phân loại phương tiện, người trên đường phố Đông Nam Á
  (camera giao thông, dashcam). Model cần nhận ra **xe ba bánh địa phương** (tuk-tuk, xích lô, ba gác) vì model COCO có sẵn gọi chúng là car / truck / motorbike.
- **Phạm vi:** ảnh ngoài trời ban ngày và ban đêm, đường phố, chợ; không dùng ảnh trong nhà.
- **Nguyên tắc chung:** gán **mọi** vật thuộc lớp mà **nhìn thấy được** và **đủ lớn** (mục 4). Máy gợi ý, người quyết định.

## 2. Lớp (mỗi vật thuộc đúng một lớp)

| Lớp | Định nghĩa | Gồm | Không gồm (dễ nhầm) |
|---|---|---|---|
| `person` | Một người, từ đầu đến chân (phần nhìn thấy). | người đi bộ, người ngồi bán hàng, người lái / người ngồi sau xe | tượng, ma-nơ-canh, người trong tranh / biển quảng cáo / phản chiếu |
| `bicycle` | Xe đạp thường, đạp bằng chân. | xe đạp chở hàng, xe đạp có giỏ | xe đạp điện (xem mục 6), xe đạp gấp gọn dựng trong cửa hàng nhưng không ai dùng vẫn tính |
| `motorbike` | Xe máy, xe tay ga, mô tô 2 bánh có động cơ. | xe đang đỗ, xe chở hàng | **người ngồi trên xe** (người là box `person` riêng) |
| `car` | Ô tô con, taxi, SUV, bán tải, minivan. | taxi màu, xe công ty | xe ba bánh (→ `three_wheeler`) |
| `bus` | Xe buýt, xe khách lớn. | minibus chở khách | xe buýt trên biển quảng cáo |
| `truck` | Xe tải có buồng lái và thùng hàng, ≥ 4 bánh. | xe tải nhỏ | **xe ba gác / xe ba bánh chở hàng** (→ `three_wheeler`) |
| `three_wheeler` | Xe **ba bánh**: tuk-tuk (Thái), xích lô (cyclo), xe ba gác / xe ba bánh chở hàng. | cả xe lẫn khung mái / thùng hàng gắn liền xe | xe máy kéo rơ-moóc nhỏ có 4 bánh (hiếm, ghi `can_review`) |
| `ignore_region` | **Vùng** không gán từng vật: đám đông xe / người dày đặc, mờ, không tách được từng cá thể. | khối xe máy đỗ sát nhau ở xa | không dùng để "lờ đi" vật rõ ràng chỉ vì khó vẽ |

**Vật không thuộc lớp nào** (cây, biển báo, thùng hàng, tượng…): **không gán, không vẽ ignore**.

## 3. Thuộc tính

| Thuộc tính | Giá trị | Cách dùng |
|---|---|---|
| `conf` | chuỗi (ví dụ `0.87`), tự điền khi nạp pre-label | Điểm tin cậy của máy. Bạn **không** sửa. Nhãn bạn vẽ tay để trống. Chỉ dùng để xếp thứ tự duyệt, **không** thay việc nhìn ảnh. |
| `can_review` | tick / không | Tick khi bạn **không chắc quy tắc** (không phải khi vật khó vẽ). Mỗi ca tick phải được ghi vào bảng ca khó (mục 5) hoặc vào calibration. |

## 4. Quy tắc hình học

### 4.1 Box (detection)

1. Box là hình chữ nhật **ôm sát phần nhìn thấy** của vật: cạnh box chạm điểm ngoài cùng của vật theo từng phía, sai số cho phép ≤ ~2 px ở ảnh 1024 px.
2. Vật **bị che một phần**: vẽ box quanh **phần nhìn thấy**, không đoán phần bị che. Vật bị che > 80% (còn nhìn thấy < 20% diện tích) thì **không gán**.
3. Vật **bị cắt mép ảnh**: vẽ box đến mép ảnh (không vẽ ra ngoài ảnh).
4. **Người ngồi trên xe**: hai box riêng, một `person` (cả người), một `motorbike` / `bicycle` / ... (cả xe). Hai box chồng nhau là bình thường.
5. **Hàng hoá chở trên xe**: không tính vào box xe nếu nhô ra khỏi xe hơn ~20% chiều dài xe (xem mục 6, câu hỏi 3).
6. **Vật quá nhỏ**: box cao < **20 px** → không vẽ từng vật. Nếu là cả cụm nhỏ nằm sát nhau (≥ 5 vật) thì vẽ **một** `ignore_region` bao cụm.
7. **Hình in, tượng, phản chiếu, ảnh trên biển quảng cáo**: không gán.

### 4.2 Mask / polygon (Part 5: 3 vật; nâng cao: 8 ảnh `seg`)

1. Polygon bám **đường biên của phần nhìn thấy**; đỉnh đủ dày để lệch biên ≤ ~3 px, không cần chi tiết nhỏ hơn bánh xe / tay lái.
2. **Lỗ thủng** (khoảng trống nhìn thấy xuyên qua khung xích lô, giữa tay và thân người): **không** vẽ lỗ nếu lỗ nhỏ hơn ~5% diện tích vật; lỗ lớn thì bỏ ra khỏi polygon.
3. **Bóng đổ** không thuộc vật. **Kính xe** thuộc vật (xem mục 6, câu hỏi 5).
4. Hai vật chạm nhau: mỗi vật một polygon, **không chồng lên nhau**; vùng chạm thuộc vật ở phía trước.

### 4.3 Chọn lớp khi nhìn không rõ

- Không chắc lớp (ví dụ xe ba bánh ở xa, mờ): chọn lớp **bạn tin nhất** và tick `can_review`. Không chọn lớp ngẫu nhiên để "cho có".
- Hai lớp có thể đúng (xe ba gác có mui / xe tải nhỏ): theo đếm **số bánh nhìn thấy**; không đếm được → `can_review`.

## 5. Ca khó đã chốt (bảng sống: thêm dòng ở v1.1)

| Ca | Quyết định | Lý do |
|---|---|---|
| Xích lô: khách ngồi trước, người đạp ngồi sau | 1 `three_wheeler` (cả xe + mui), mỗi người một `person` | Người là vật riêng; xe là vật riêng |
| Người bán hàng dắt xe đạp chở đầy hàng | `person` + `bicycle` (box xe ôm cả giỏ hàng gắn cố định) | Giỏ gắn cố định thuộc xe |
| Xe máy đỗ sát nhau thành hàng, vật ở xa ít thấy | Vật rõ → gán từng xe; phần ở xa, cao < 20 px → `ignore_region` | Mục 4.1.6 |
| Tuk-tuk có hành khách ngồi nhìn thấy qua cạnh hở | Gán `person` nếu còn nhìn thấy ≥ 20% người | Quy tắc che chung (4.1.2) |

## 6. Câu hỏi CÒN MỞ (bạn phải chốt trong v1.1 sau khi nhìn lỗi của máy)

1. **Xe đạp điện / xe máy điện** nhỏ: `bicycle` hay `motorbike`? (Guideline chưa nói.)
2. **Khối xe máy đỗ dày đặc gần camera** (không phải ở xa): từng xe hay `ignore_region`? Ngưỡng cụ thể?
3. **Hàng nhô ra**: xe đạp / ba gác chở tre, rổ nhô xa: box xe có gồm hàng không? Mask thì sao?
4. **Người ngồi trong xe ba bánh có mui che** (chỉ thấy thấp thoáng): ngưỡng "còn thấy ≥ 20%" ở mục 4.1.2 có hợp lý không? Hai người đánh giá "20%" bằng mắt sẽ lệch nhau bao nhiêu?
5. **Kính chắn gió / cửa sổ xe**: thuộc mask xe (nhìn xuyên qua thấy nội thất / người) hay không?
6. **Box máy chồng nhau quá nhiều** trong ảnh đêm: gộp hay giữ?
7. Mức **tin cậy tối thiểu** để chấp nhận box máy không cần sửa (chỉ là `conf`, chưa là quy tắc chất lượng): có đặt không?

Mỗi câu chốt xong phải có **ảnh ví dụ** (`img_0xx`) và ghi vào bảng ca khó + lịch sử thay đổi.

## 7. QC và phiên bản

- Ngưỡng QC: xem `submission/taxonomy.json` mục `rules` (**không sửa**): IoU chấp nhận 0.9 (mask 0.85), ảnh lỗi ≤ 2%, bắt bẫy ≥ 80% (≥ 10/12).
- Người quyết ca mơ hồ: bạn (lab cá nhân); ghi từng quyết định vào `calibration_decisions.csv`.
- Mọi thay đổi làm đổi nhãn → tăng phiên bản.

| Phiên bản | Thay đổi | Lý do (ca nào) |
|---|---|---|
| v1.0 | Bản giảng viên viết sẵn | — |
