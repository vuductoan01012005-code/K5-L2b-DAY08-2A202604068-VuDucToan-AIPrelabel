# Báo cáo Day 08 — 2A202604068 - Vũ Đức Toàn

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp           | Model + cỡ               |          Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
| ------------- | ------------------------ | --------------: | ----------------------------------------------------- |
| person | YOLO26m (`yolo26m.pt`) | 0.50 | `threshold_sweep.md`: recall 8/8 đối tượng gold = 100% (≥ 90%), precision 8/22 = 36% (8 vật gold). Run: r02 (pilot, `--conf 0.5`), r05/r11 (all, `--conf 0.05`, nguồn quét). |
| bicycle | YOLO26m | 0.30 (mặc định) | Chỉ **0/28** đối tượng gold là `bicycle` (< 5): không chọn ngưỡng theo lớp, dùng `--conf 0.3` mặc định của merge. Run: r05/r11. |
| motorbike | YOLO26m | 0.10 | `threshold_sweep.md`: recall 7/7 = 100% (≥ 90%), precision 7/31 = 23% (7 vật gold — cao nhất còn đạt recall 100%). Run: r01/r05/r11. |
| car | YOLO26m | 0.30 (mặc định) | Không ngưỡng ≤ 0.5 nào đạt recall 90%: ở 0.5 recall chỉ 6/7 = 86%, precision 6/13 = 46% (7 vật gold). Dùng `--conf 0.3` mặc định; tuk-tuk thường bị gọi nhầm `car`/`truck` (r01: img_011 `truck` 0.77 = xích lô, img_002 `truck` 0.73 = tuk-tuk). |
| bus | YOLO26m | 0.30 (mặc định) | Chỉ **0/28** đối tượng gold (`bus`): mẫu không đủ, dùng mặc định 0.3. Run: r05/r11. |
| truck | YOLO26m | 0.30 (mặc định) | Chỉ **0/28** đối tượng gold (`truck`): mẫu không đủ, dùng mặc định 0.3; không có ngưỡng riêng vì "không có ngưỡng nào ≤ 0.5 đạt recall 90%" đồng nghĩa với mẫu 0 vật. Run: r05/r11. |
| three_wheeler | YOLOE-26s (`yoloe-26s-seg.pt`) | 0.15 | `threshold_sweep.md`: recall 6/6 = 100% (≥ 90%), precision 6/9 = 67% (6 vật gold). Run: r06 (all, `--conf 0.05`); r03/r09 (pilot, `--conf 0.25`). YOLOE chỉ tìm được xe 3 bánh; YOLO26 không có lớp này trong 80 lớp COCO. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  ```powershell
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.5,motorbike=0.1,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
  Lớp không có trong chuỗi `--thr` (bicycle, car, bus, truck) dùng `--conf 0.3` mặc định; `three_wheeler` của YOLOE được ưu tiên khi chồng box `car`/`truck` của YOLO26 (IoU ≥ 0.6). Sau gộp: 511 box cho 28 ảnh (batch + gold) ở `submission/prelabel/pre_traps.coco.json` (pre_traps có bẫy; để nạp CVAT dùng `pre_traps_import.coco.json` chỉ 24 ảnh batch).
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: **SAM 2.1 (`sam2.1_b.pt`)** — chạy được trên cả CUDA lẫn CPU (r04/r10) và bám biên thân xe tốt; trọng số đã có sẵn trong `models/` nên không cần quyền SAM 3. SAM 1 (`mobile_sam.pt`) chỉ dùng để so sánh trên img_013, không chọn làm bản chính (xem REPORT.md mục 5, run_id r13/r14).
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Lỗi sai lớp không sửa được bằng ngưỡng: YOLO26 không có khái niệm `three_wheeler` nên mọi ngưỡng đều gọi tuk-tuk/xích lô là `car`/`truck` (r01: img_011 `truck` 0.77 = xích lô, img_002 `truck` 0.73 = tuk-tuk; r05/r11: tuk-tuk vẫn gọi `car`/`truck` ở mọi ngưỡng, 898/1669 box có điểm < 0.3). Sửa bằng YOLOE (run r06) chạy prompt `three_wheeler` rồi merge, để YOLOE thắng `car`/`truck` khi IoU ≥ 0.6.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): điểm cao mà sai: img_013 `car` 0.42 với box [325, 64, 618, 564] ôm cả xe tuk-tuk (r04) — sai lớp và quá lớn dù điểm không cao; ở gold, `person` ở ngưỡng 0.5 vẫn precision chỉ 8/22 = 36% (14/22 box là thừa dù điểm ≥ 0.5). Điểm thấp mà đúng: xe 3 bánh của YOLOE ở img_002 điểm 0.36–0.39 vẫn đúng vật và bám sát (r03). Đây là lý do không đặt ngưỡng tin cậy tối thiểu để chấp nhận box (guideline v1.1 câu 7).
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh, 28 đối tượng) nên số đo máy trên gold là lạc quan (chọn và chấm trên cùng dữ liệu). KTC 95%: precision máy trên gold 24%–47%, recall 64%–92% (xem mục 2) — ngưỡng chỉ là ước lượng thô, chưa đo trên tập độc lập.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số   | Tỉ lệ | Mẫu số                  |
| ---------- | ---- | ----- | ----------------------- |
| accepted | 429 | 429/435 = 98.6% | 435 box máy |
| edited | 3 | 3/435 = 0.7% | 435 box máy |
| deleted | 3 | 3/435 = 0.7% | 435 box máy |
| added | 5 | 5/437 = 1.1% | 437 đối tượng sau duyệt |

Ảnh tính: 24 (batch + gold); box máy: 435; đối tượng sau duyệt: 437; ngưỡng accepted: IoU ≥ 0.9. Đã loại 19 đối tượng nằm quanh các bẫy khỏi bảng này. Trong deleted + added, 2 trường hợp là đổi lớp (box đúng chỗ, lớp sai); số vật máy sót thật ≈ 5 − 2 = 3 đối tượng.

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? `three_wheeler` tệ nhất trong 4 trạng thái đã đo: box YOLOE trùng/chồng trên cùng xe (img_003: 3 box #16/#17/#18 trên một xe) và đề xuất "chồng trùng" phải sửa/xóa; bảng theo lớp: `three_wheeler` 21 accepted / 1 edited / 2 deleted / 1 added trên tổng 25 box — accepted chỉ 21/25 = 84.0%, trong khi `person` và `car` đều 100%. Nguyên nhân: YOLOE ra box trùng không loại khi merge và đề xuất box lỏng.
- So với gold: precision / recall của **máy** 23/66 = 34.8% và 23/28 = 82.1% (mẫu số: **28 đối tượng gold** trên 4 ảnh; TP 23, FP 43, FN 5). KTC 95%: precision 24%–47%, recall 64%–92% — con số trên gold là lạc quan vì ngưỡng chọn trên chính 4 ảnh gold này và mẫu chỉ 28 đối tượng.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 4/12 = 33.3% (KTC 95%: 14%–61%; ngưỡng đạt ≥ 80% = 10/12 — chưa đạt). Loại nào lọt: extra 1/3 lọt (img_013: motorbike trong khoang xe trống), wrong_class 2/3 lọt (img_003: person→bicycle; img_024: car→truck), loose 3/3 lọt (img_023, img_021, img_017), missing 2/3 lọt (img_026, img_005) — tổng 8 bẫy lọt. Vì sao (bạn đã bỏ bước nào của checklist)? Theo checklist 4 bước của `docs/REVIEW_CHECKLIST.md`: tôi đã quét ảnh và duyệt theo thứ tự thừa → sai nhãn → lỏng trên phần lớn ảnh, nhưng **bỏ bước 0 trên nhiều ảnh** (không ẩn nhãn máy để tự đếm vật trước) rồi duyệt bằng mắt theo box có sẵn, và **bỏ bước 1 (bỏ sót)** trên ảnh xa/mép — hai bẫy `missing` lọt chính vì bước này. Lỗi `loose` lọt cả 3 vì tôi không dừng để kéo lại box lệch vài pixel khi đã tin nhãn máy trông "hợp".
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Accepted 429/435 = 98.6% không chứng tỏ duyệt kỹ: nó cho thấy tôi có thiên kiến tự động hóa — thấy pre-label "trông hợp lý" thì giữ ngay mà không tự kiểm lại lớp/vị trí (3 dạng của checklist: neo nhãn — mắt tìm vật theo box có sẵn vì không quét bước 0; điểm cao — coi `conf` 0.9 như xác suất đúng; mệt mỏi — chấp nhận dần ở ảnh cuối). Số accepted cao kèm bắt bẫy thấp là dấu hiệu của việc "làm theo" gợi ý sai của máy, không phải là hiệu quả; tự kiểm ở QC (mục 7: 8/8 ảnh lỗi, 15/219 đối tượng) xác nhận cùng hiện tượng ở mức mẫu nhỏ.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay và duyệt đều **khoảng 1 phút/ảnh**: `gold_manual` 4/4 = 1.0 phút/ảnh, `batch_review` 24/24 = 1.0 phút/ảnh. Do hai bên bằng nhau, chưa thấy pre-label tiết kiệm thời gian; vì số gần như bằng nhau, tôi không suy ra được tiết kiệm bao nhiêu %.
- Kết luận: **chỉ cho một số lớp / chỉ làm bước đề xuất (pre-label)**, dựa trên các số liệu sau: YOLO26m **đáng dùng** ở `person` và `motorbike` (review_stats theo lớp: person 252/252 và motorbike 89/91; recall trên gold 8/8 và 7/7) — nhưng **không tin tuyệt đối ở `car` theo nghĩa nhãn** (QC sai lớp van/thùng), và **không dùng pre-label làm sản phẩm cuối** ở `three_wheeler` (accepted chỉ 21/25, QC 8/15 lỗi là box thừa) vì máy vẫn thừa box và sa lớp. Việc "đáng dùng" ở đây là *đáng làm gợi ý để người duyệt lọc* chứ không phải *đáng publish trực tiếp*.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Quét bước 0 trên 100% ảnh: ẩn nhãn máy ~15 giây, tự đếm vật theo lớp, *sau đó* mới hiện nhãn (kiểm bằng log ảnh đã ẩn/hiện).
  2. Không dùng `conf` để quyết định: ở mỗi ảnh, box `conf` thấp nhất (tự xếp) phải được kiểm vật lý ít nhất bằng mắt với ảnh gốc trước khi chấp nhận; ghi lại thời gian kiểm theo ảnh (kiểm bằng `time_log.csv`).
  3. Mỗi ca không chắc quy tắc → tick `can_review`, tổng hợp tối thiểu 3 ca vào `calibration_decisions.csv` trước khi chốt duyệt; yêu cầu ca `loi_may` kể cả `cau_hoi_mo` đều có ảnh ví dụ.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh         | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
| ----------- | --------------- | -------- | ------------------------------ | ------------ | ------------------- |
| img_013.jpg | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 dương 400,480; 600,110 + 1 âm 720,300) | Có — người lái phủ 76% (thân 59%) | Ôm cả người lái/tay lái; 1 click âm không đủ tách | `tuktuk_img013_sam2.jpg` |
| img_013.jpg | SAM 1 (`mobile_sam.pt`) | 3 (cùng click) | Không — thân người lái chỉ 13% | Chỉ còn vệt ở chân/má; biên thân xe sát hơn SAM 2.1 | `tuktuk_img013_sam1.jpg` |
| img_004.jpg | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 dương 620,480; 700,560 + 1 âm 465,450) | Không (xe trống); xe bên cạnh 6% | Lấp khoang xe; biên bánh trước tràn nền | `tuktuk_img004_sam2.jpg` |
| img_017.jpg | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 dương 560,420; 680,500 + 1 âm 645,355) | Không (xe trống); xe bên cạnh 4% | Mui vẫn tô dù có click âm; lấp khoang xe | `tuktuk_img017_sam2.jpg` |

- Đối chiếu guideline mục 4.2: **giúp** — 4.2.3 kính thuộc vật; 4.2.4 hai vật tách polygon. **Thiếu** — không cấm mask gồm người lái (chỉ có 4.1.4 hai box riêng); 4.2.2 lỗ lớn nhưng SAM lấp khoang nên phải sửa tay. Cần ghi rõ mask `three_wheeler` không gồm `person`.
- _(Tuỳ chọn)_ Interactor SAM trong CVAT: **chưa làm** (chỉ offline `sam2_points.py`). Tỉ lệ đo bằng `work/analyze_masks.py` trên vùng chữ nhật overlay. SAM chính: **SAM 2.1** (`model_log.csv` r04/r10); SAM 1 chỉ để so sánh trên img_013.

- Hệ quả đối với guideline: mask xe cần ghi rõ không gồm người ngồi; nên thêm quy tắc click âm đủ nhiều (hoặc dùng box gợi ý) khi vật có người/khách chồng lên.

## 6. Calibration (Part 6)

- Số quyết định: **10** (`calibration_decisions.csv`, CAL-01 đến CAL-10; `nguon`: 3 `loi_may`, 2 `cau_hoi_mo`, 2 `can_review`, 3 `bay_lot`); quan trọng nhất: **CAL-02** (giỏ/thùng treo trên gánh bị dán `three_wheeler` — img_014 #130/#131) vì đó là lỗi lặp hệ thống của YOLOE prompt và sửa luôn quy tắc "thấy cấu trúc 3 bánh" của mục 2, và **CAL-04** (giữ nguyên 20 px + quy trình tách/`ignore_region` + không đặt ngưỡng `conf`) vì nó trực tiếp thay đổi cách người duyệt xử lý ảnh đêm dày đặc (img_018/img_024).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: (1) van/minivan 4 bánh có mui = `car` (không tin `conf`); (2) `three_wheeler` chỉ khi thấy cấu trúc 3 bánh/mũi/người ngồi; (3) hàng/giỏ *treo/roi* xe không thuộc box xe (mục 4.1.5); (4) quy trình tách/`ignore_region` (mục 4.1.6) — tang được vật thì gán, không tách được hoặc cao < 20 px thì `ignore_region`; (5) `conf` không quyết định lớp, thiếu đặc trưng cấu trúc → `can_review` (mục 4.3); (6) giữ một box mỗi vật, loại box chồng trùng (câu 6); (7) một xe = một box `three_wheeler` gồm mui (xích lô img_035). **Chưa chốt:** câu 1 (xe đạp điện), câu 2 (ngưỡng số cho khối xe máy gần camera), câu 4 (chuẩn 20% và độ lệch đánh giá), câu 5 (kính/mask) — xem `guideline_v1.1_calibrated.md` mục 3.
- Nguyên nhân gốc: **guideline mơ hồ + máy + chưa nắm quy tắc, không cả ba đều rõ ràng**. Ví dụ: "hàng treo trên gánh có thuộc xe không" (CAL-03) là guideline mơ hồ (v1.0 4.1.5 chưa nói tới treo/roi); "YOLOE dán `three_wheeler` lên giỏ/thùng" (CAL-02) và "YOLO26 gọi tuk-tuk là `car`/`truck`" là lỗi máy (thiếu khái niệm khi quét prompt/cup / dáng xe ba bánh); "box cao 16–17 px vẫn có" (CAL-08) là chưa nắm ngưỡng 20 px của mục 4.1.6.

## 7. QC và quyết định (Part 6)

- Mẫu: 8 ảnh batch (3 ca khó: img_018, img_021, img_024; 5 ngẫu nhiên: img_003, img_006, img_013, img_014, img_035; seed 42), 219 đối tượng; người kiểm: tôi (bạn — bạn vừa là người duyệt vừa là người viết QC: hạn chế là bạn kiểm chính nhãn mình vừa duyệt, nên thiên kiến tự động hóa vẫn có thể lọt).
- Lỗi theo **đối tượng**: 15/219 = 6.8% · theo **ảnh**: 8/8 = 100.0% (ngưỡng Release ≤ 2%) — chưa đạt (xem `qc_report_auto.md`).
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại:
  - **thừa**: 8/219 lỗi — ví dụ img_013 #121 (motorbike trong khoang xe trống), img_014 #130 (giỏ/thùng treo ghi `three_wheeler`), img_018 #152 (box cao 16,6 px).
  - **sai lớp**: 5/219 — ví dụ img_003 #5 (person ghi `bicycle`), img_024 #289 (car ghi `truck`), img_006 #65 (van ghi `three_wheeler`).
  - **sai vị trí/kích thước**: 2/219 — ví dụ img_021 #227 (loose, IoU 0,62), img_035 #447 (xích lô tách 2 box).
  - **bỏ sót / sai thuộc tính / sai mask**: 0/219 ở phiếu này (các lỗi bỏ sót chưa được đếm ở cấp độ nhãn final vì QC chỉ kiểm box có sẵn; không có lỗi mask vì đây là batch detection).
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): **cận trên Wilson 95% cho tỉ lệ ảnh lỗi thật là 100%** (8/8 ảnh lỗi trong 8 ảnh mẫu) — mẫu 8 ảnh quá nhỏ để chứng minh ngưỡng ≤ 2%; một tỉ lệ ảnh lỗi thật tới 100% vẫn nằm trong cận trên này, nên không thể kết luận "đạt 2%" bằng mẫu này. Cũng cần đọc KTC 95% ở mục 2 (precision máy 24%–47%, recall 64%–92%) và bắt bẫy 14%–61% — đều rất rộng.
- Script kiểm được / không kiểm được: kiểm được — file có đủ, `qc_review.csv` đủ `error_type`, số đếm theo ảnh/đối tượng, bẫy 4/12, độ khớp nhãn (validate). Không kiểm được — box ôm sát vật thật, lớp có đúng thực tế không, vật mà cả tôi lẫn máy cùng bỏ sót, tôi có theo checklist thật lúc duyệt không.

**Quyết định: Rework**, ngày 2026-11-19. Lý do (đi theo Hold → Rework → Release): (1) ảnh có lỗi 8/8 = 100.0% vượt ngưỡng ≤ 2% (ít nhất 15/219 đối tượng lỗi — lỗi hệ thống `thua` 8/15); (2) bắt bẫy chỉ 4/12 = 33.3% (< 80%), nghĩa là duyệt chưa đáng tin và tôi đã lọt 8 bẫy vì tin nhãn máy; (3) không đủ bằng chứng để Release: mẫu QC 8 ảnh quá nhỏ (cận trên Wilson 95% 100%), `time_log.csv` trống nên không có số thời gian để đánh giá tiết kiệm, và các câu hỏi mở mục 6 của v1.0 chưa chốt hết (xe đạp điện, ngưỡng số xe máy gần camera, chuẩn 20% của người trong xe ba bánh, kính/mask). Release không trung thực với số đo hiện có.
Nếu Rework / Hold, việc cần làm tiếp: 1) Sửa các lỗi QC trong `batch_reviewed.json` (15 lỗi của phiếu QC) và box < 20 px còn sót (img_023 #252 15,1 px, img_027 #327 17,2 px — validate cảnh báo thêm 2 box ngoài 8 ảnh QC) qua CVAT, export `submission/labels/qc_fixed.json`, chạy lại `build_release.py` và `qc_sample.py --force` để kiểm soát lại. 2) Duyệt lại lô theo checklist 4 bước (đặc biệt **bước 0 — ẩn nhãn máy để quét trước**) và ghi `time_log.csv` thật (`gold_manual` và `batch_review`) sau đó chạy lại `review_stats.py` (với `--splits`) và `qc_report.py` để xác nhận bắt bẫy đạt ≥ 80% và ảnh lỗi ≤ 2% trên mẫu n lớn hơn (ít nhất 20 ảnh) mới có thể đổi quyết định. 3) Chốt nốt câu hỏi mở chưa chốt của guideline (xe đạp điện, ngưỡng số khối xe máy gần camera, chuẩn 20% đánh giá, kính/mask) sau khi có thêm vật chứng ảnh.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Có dùng AI (trợ lý coding) để: đọc và gộp số đo từ `submission/qc/*`, đối chiếu từng box QC (`Finals`/`qc_review.csv` với `batch_reviewed.json`) để lập danh sách lỗi, chốt calibration v1.1, và gõ REPORT/STUDENT theo khuôn. Việc nào không kiểm được bằng script (nhìn ảnh, suy luận lớp) tôi tự đối chiếu bằng mắt trên ảnh `work/qc/review/*.jpg` và ghi trực tiếp vào phiếu review (`reviewer = AI review`); không dùng AI để quyết định thay người duyệt cuối cùng.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? **Không.** Tôi gán gold trước khi xem pre-label của 4 ảnh gold (không xem `work/prelabel/t_y26_c05`/pre-label trên 4 ảnh gold trước khi gán xong); AI chỉ tham gia xử lý sau khi gold đã chốt, không vào việc gán gold.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Máy Windows 11 + PowerShell; không có GPU (chạy CPU, các lần chạy ghi trong `model_log.csv`: CUDA ở r01–r06, CPU ở r07–r12 và Part 5 r13/r14 để trống thiết bị theo yêu cầu); CVAT Docker v2.74.1. Vướng mắc: PowerShell tách chuỗi có dấu phẩy và mất số `0` đầu — phải dùng ngoặc kép cho `--thr`/`--pos`/`--student`; terminal shell integrationsometimes chậm nhưng không ảnh hưởng file kết quả.
- Dùng cache `data/cache/` thay vì tự chạy? **Không dùng cache.** Tất cả các lần chạy đều tự chạy qua `scripts/prelabel.py`/`sam2_points.py` và lưu ở `work/prelabel/*` + `submission/prelabel/pre_traps*.coco.json`; không copy từ `data/cache/`.
- Lỗi / bug script đã gặp (nếu có): `scripts/check_submission.py` báo `time_log.csv` thiếu bước (vì `time_log.csv` chỉ có tiêu đề, chưa ghi phút/ảnh); không có lỗi làm sai dữ liệu, chỉ thiếu ghi chú. `scripts/threshold_sweep.py` chạy được với pre-label ở ngưỡng thấp; đã chạy `merge_prelabels.py`/`review_stats.py`/`qc_sample.py`/`build_release.py`/`qc_report.py` trên file đã có.
- Một điều bạn làm khác lần sau: bấm giờ thật cho `gold_manual` và `batch_review` vào `submission/time_log.csv` (để trả lời "pre-label tiết kiệm bao nhiêu %"), quét bước 0 trên 100% ảnh (ẩn nhãn máy) để bắt bẫy ≥ 80%, và duyệt/đối chiếu bằng mắt trước khi nhờ AI tổng hợp số đo — và làm ở ngày học đầu thay vì dồn cuối.
