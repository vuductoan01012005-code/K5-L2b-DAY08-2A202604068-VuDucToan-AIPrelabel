---
day: "D08"
title: "Day 08 Lab — AI hỗ trợ gán nhãn: Pre-label, Review & Calibration"
description: "Thực hành quy trình gán nhãn có AI hỗ trợ trên 36 ảnh phố châu Á: gán tay 4 ảnh gold độc lập, khảo sát YOLO26 · YOLOE · SAM, quét ngưỡng theo lớp, gộp pre-label kèm 12 ảnh bẫy để đo thiên kiến tự động hóa khi review trên CVAT, tách mask SAM cho ca khó, calibration thành guideline v1.1 và ra quyết định Release / Hold / Rework."
outcomes:
  - "Hoàn thiện bộ 4 ảnh `gold_manual.json` gán tay hoàn toàn độc lập trên CVAT (không AI) và vượt qua `python scripts/validate_labels.py submission/labels/gold_manual.json --expect gold`."
  - "Chạy thực nghiệm YOLO26, YOLOE-26s và SAM (`sam2.1_b.pt` / `sam2.1_t.pt` / `mobile_sam.pt`) qua `prelabel.py`, ghi log đầy đủ vào `submission/model_log.csv` và chọn ngưỡng theo lớp đạt mục tiêu recall >= 90% bằng `threshold_sweep.py`."
  - "Gộp pre-label và cài 12 bẫy nhân tạo theo MSHV bằng `merge_prelabels.py` + `inject_traps.py`, thực hiện review 24 ảnh batch trong CVAT (`pre_traps_import.coco.json`), đo tỷ lệ bắt bẫy >= 80% và AI error profile bằng `review_stats.py`."
  - "Tạo tối thiểu 3 mask phân đoạn đối tượng khó bằng `sam2_points.py`, chạy `build_release.py`, `qc_sample.py`, `qc_report.py`, chốt `guideline_v1.1_calibrated.md`, `submission/REPORT.md` với quyết định Release / Hold / Rework và đạt 0 lỗi với `python scripts/check_submission.py`."
prerequisites:
  - "Đã nắm thao tác tạo Project, Task, vẽ Rectangle và export/import COCO 1.0 trên CVAT."
  - "Có Python 3.10–3.13, Git và kết nối internet (lần đầu tải trọng số YOLO26/YOLOE/SAM) hoặc tài khoản Google Colab cho máy không đủ RAM."
  - "Đã đọc `guideline/guideline_v1.0_label_contract.md` trước khi mở ảnh và chuẩn bị MSHV để làm hạt giống sinh bẫy."
requiredTools:
  - "Python 3.10–3.13 cùng `requirements-local.txt` (khi inference trên Colab và dùng CVAT Docker local) hoặc `requirements.txt` (`torch`, `ultralytics`, `pillow`, `matplotlib`, `transformers`)"
  - "CVAT Docker local (`http://localhost:8080`, kiểm chứng thực tế trên v2.74.1 / tương thích v2.81)"
  - "Git, trình duyệt web và tùy chọn Google Colab (`notebooks/day08_colab.ipynb`) cho bước inference nặng"
commonErrors:
  - "CVAT báo lỗi khi tạo nhãn từ `Raw` → Phải dùng `cvat/labels_generated.json` mới nhất (sinh từ `python scripts/make_cvat_labels.py`, mọi thuộc tính đều có mảng `values` không rỗng) rồi bấm **Save** trước khi **Submit & Open**."
  - "Import `pre_traps_import.coco.json` vào CVAT xong mở Job vẫn thấy 0 shape → Khi `Import mode` đang ở `Replace`, sau khi bấm **OK** phải bấm thêm **Replace annotations** trên hộp thoại xác nhận; đồng thời nhớ nạp `pre_traps_import.coco.json` chứ không nạp `pre_traps.coco.json`."
  - "PowerShell cắt mất số `0` đầu MSHV hoặc tách tham số có dấu phẩy (`--classes`, `--thr`, `--pos`) → Bắt buộc bọc ngoặc kép quanh giá trị chuỗi, ví dụ `--student \"00123456\"`, `--classes \"person,motorbike,car\"`, `--thr \"person=0.35,motorbike=0.5,car=0.35\"`, `--pos \"400,480\"`."
  - "Lỗi encoding `cp1252` trên Windows khi pipe hoặc đọc CSV xuất từ Excel → Hệ thống script và `check_submission.py` đã ép UTF-8 mặc định; khi lưu CSV từ Excel nhớ giữ định dạng UTF-8 (dấu `,` hoặc `;` đều được parser tự nhận diện)."
  - "`check_submission.py` báo lỗi thư mục `models/` đang bị commit → Xóa file trọng số `.pt/.onnx/.ts` khỏi git index (`git rm --cached ...`); `.gitignore` chỉ giữ `models/.gitkeep`."
requiresSubmission: true
workMode: "individual"
---

# Day 08 Lab — AI hỗ trợ gán nhãn: Pre-label, Review & Calibration

Bài thực hành cá nhân này đưa bạn đi trọn chu trình gán nhãn hiện đại trong **4 giờ trên lớp** (kèm **~45 phút chuẩn bị trước lớp**) trên bộ **36 ảnh đường phố châu Á** (`data/images/img_001.jpg` đến `img_036.jpg`). Thay vì vẽ tay mọi khung hình hay tin tuyệt đối vào kết quả từ mô hình, bạn tự tay thiết lập chuẩn đối chứng trên **4 ảnh gold**, chạy ba họ model từ đơn giản đến phức tạp (**YOLO26 → SAM → YOLOE**), tối ưu ngưỡng theo từng lớp, cài **12 ảnh bẫy** để tự đo mức độ **thiên kiến tự động hóa** khi duyệt **24 ảnh batch** trên CVAT, rồi kiểm định QC để ra quyết định **Release / Hold / Rework** và cập nhật guideline lên `v1.1`.

:::image{src="docs/assets/hero.png" alt="Minh họa quy trình Day 08 gồm pre-label đa model, kiểm soát thiên kiến trên CVAT và vòng lặp calibration" caption="Từ dữ liệu phố châu Á có xe ba bánh tuk-tuk ngoài danh mục COCO tới pipeline pre-label, gài bẫy chống automation bias và chốt QC."}
:::

Khoảng **28% ảnh** trong bộ dữ liệu xuất hiện tuk-tuk, xích lô hoặc xe ba gác (`three_wheeler`), vốn không nằm trong 80 lớp COCO chuẩn nên detector truyền thống dễ bỏ sót hoặc gán nhầm thành `car` hay `truck`. Lộ trình sáu pha dưới đây tách rạch ròi giữa phần việc của mô hình (tạo ứng viên nhanh) và phần việc của con người (chốt ngữ nghĩa, phát hiện bẫy và sửa quy tắc).

:::image{src="docs/assets/roadmap.png" alt="Sơ đồ lộ trình Day 08 qua sáu pha thực hành trong 240 phút trên lớp" caption="Phân bổ thời gian 240 phút trên lớp: Pha 1 (25′) · Pha 2 (30′) · Pha 3 (10′) · Nghỉ (10′) · Pha 4 (95′) · Pha 5 (15′) · Pha 6 + Nộp bài (55′)."}
:::

## Pha 0 — Chuẩn bị môi trường, trọng số và Project CVAT (Trước lớp, ~45 phút)

Trước buổi học, bạn fork repository đề gốc về tài khoản cá nhân ở chế độ **public** đúng tên `K5-L23-DAY08-[MSHV]-[HoTen]-AIPrelabel`, clone về máy, thiết lập môi trường Python `3.10–3.13` (khuyến nghị `3.12`) và cấu hình `git config user.name` cùng `user.email`. Để chuẩn bị nhanh trước giờ học, bạn có thể dùng [docs/setup_wizard.html](docs/setup_wizard.html) và công cụ độc lập [docs/day08_setup.py](docs/day08_setup.py) (`download --profile 8gb|16gb`, `check [--light]`) song song với `scripts/download_models.py` và `scripts/check_env.py`.

:::repository{url="https://github.com/VinUni-AI20k/K5-L23-Day08-AI-Assisted-Labeling" label="Repo gốc Day 08 — K5-L23-Day08-AI-Assisted-Labeling"}
- [Repository gốc Day 08](https://github.com/VinUni-AI20k/K5-L23-Day08-AI-Assisted-Labeling)
:::

Tùy cấu hình máy, bạn chọn một trong ba đường cài đặt trước khi sinh bộ nhãn CVAT:

- **Máy local đủ RAM/GPU (`requirements.txt`):** cài `requirements.txt`, tải `--set core` (`yolo26m.pt`, `yoloe-26s-seg.pt`, `sam2.1_b.pt`) vào `models/`, rồi chạy `python scripts/check_env.py --cvat http://localhost:8080` để nhận `SẴN SÀNG.`.
- **Máy 8 GB RAM / chỉ CPU (local nhẹ):** tải `python scripts/download_models.py --set small` (`yolo26n.pt`, `yoloe-26s-seg.pt`, `sam2.1_t.pt`) kèm `--set sam1` (`mobile_sam.pt`), rồi thay `--size n --sam sam2.1_t.pt` (hoặc `mobile_sam.pt`) khi chạy để đạt ~0.4–4 giây/ảnh trên CPU mà vẫn chừa đủ RAM cho Docker CVAT.
- **Colab + local nhẹ (`requirements-local.txt`):** cài `requirements-local.txt` trên máy cho script đo/kiểm tra và đẩy inference Pha 2–3 lên Google Colab ([docs/COLAB.md](docs/COLAB.md), `notebooks/day08_colab.ipynb`). Khi đó `check_env.py` báo thiếu `torch`/`ultralytics` là bình thường — chỉ cần thư viện nhẹ + CVAT đạt `PASS`, hoặc chạy `python docs/day08_setup.py check --light` ra `SẴN SÀNG cho buổi học.`.

:::code-tabs{title="Cài đặt môi trường, tải trọng số và tạo bộ nhãn + thư mục CVAT"}
```powershell {title="Windows (PowerShell)"}
git clone https://github.com/<github-cua-ban>/K5-L23-DAY08-<MSHV>-<HoTen>-AIPrelabel.git
cd K5-L23-DAY08-<MSHV>-<HoTen>-AIPrelabel
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python scripts/download_models.py --set core   # máy 8 GB / CPU: --set small rồi --set sam1
python scripts/check_env.py --cvat http://localhost:8080
python scripts/make_cvat_labels.py
python scripts/prepare_workdirs.py
```
```bash {title="macOS / Ubuntu (Bash)"}
git clone https://github.com/<github-cua-ban>/K5-L23-DAY08-<MSHV>-<HoTen>-AIPrelabel.git
cd K5-L23-DAY08-<MSHV>-<HoTen>-AIPrelabel
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python scripts/download_models.py --set core   # máy 8 GB / CPU: --set small rồi --set sam1
python scripts/check_env.py --cvat http://localhost:8080
python scripts/make_cvat_labels.py
python scripts/prepare_workdirs.py
```
:::

Sau khi `prepare_workdirs.py` tạo `work/tasks/gold/` (`4` ảnh) và `work/tasks/review/` (`24` ảnh), bạn mở **CVAT chạy bằng Docker tại `http://localhost:8080`** (**không dùng `cvat.ai`**) theo [docs/CVAT_AI.md](docs/CVAT_AI.md) để tạo Project `day08-<MSHV>`. Ở **Labels → Raw**, dán toàn bộ `cvat/labels_generated.json`, bấm **Save** (hoặc **Done**), rà 8 nhãn ở **Constructor** rồi **Submit & Open**. Cuối Pha 0, cập nhật `STUDENT.md`, dòng tiêu đề `# Day 08 Lab — <MSHV> - <Họ và tên>` trong `README.md` và commit bước chuẩn bị.

:::image{src="walkthrough_images/01_cvat_project_labels_saved.png" alt="Màn hình Create a new project trên CVAT hiển thị JSON nhãn đã dán vào tab Raw và nút Save" caption="Dán toàn bộ mảng nhãn từ cvat/labels_generated.json vào mục Raw rồi bấm Save trước khi Submit & Open."}
:::

:::image{src="walkthrough_images/01b_cvat_project_created.png" alt="Trang Project trên CVAT hiển thị đủ 8 nhãn màu sau khi tạo thành công" caption="Project day08-00123456 hiển thị đủ 8 nhãn trong Constructor, sẵn sàng tạo hai Task gold và batch."}
:::

:::checklist{title="Tín hiệu sẵn sàng trước Pha 1" tone="info"}
- [ ] Local full (`requirements.txt`): `python scripts/check_env.py --cvat http://localhost:8080` báo `SẴN SÀNG.`.
- [ ] Hoặc Colab (`requirements-local.txt`): thư viện nhẹ + CVAT đạt và `python docs/day08_setup.py check --light` báo `SẴN SÀNG cho buổi học.` (`check_env.py` thiếu `torch`/`ultralytics` là bình thường).
- [ ] `python scripts/prepare_workdirs.py` tạo xong `work/tasks/gold/` (4 ảnh) và `work/tasks/review/` (24 ảnh).
- [ ] Nếu chạy local: có trọng số trong `models/` (`--set core` hoặc `--set small` + `--set sam1`); dùng Colab thì bỏ qua.
- [ ] Project CVAT Docker (`http://localhost:8080`) hiển thị đủ 8 nhãn kèm `conf` và `can_review`.
:::

## Pha 1 — Gán tay 4 ảnh gold độc lập (0:00 → 0:25 · 25 phút)

### Bản chất vấn đề: vì sao phải khóa tập gold trước khi chạm vào model AI?

Nếu bạn chạy model YOLO26 hay YOLOE lên cả 36 ảnh trước khi gán tập gold, bộ não sẽ bị neo vào những khung bao có sẵn do thuật toán vẽ ra. Hậu quả là bộ nhãn gold — vốn đóng vai trò thước đo chân lý để chọn ngưỡng ở Pha 3 và tính Precision/Recall khách quan ở Pha 4 — bị nhiễm chính những thiên kiến bỏ sót hoặc sai lớp của máy. Bốn ảnh gold (`work/tasks/gold/`, đại diện cả đường phố đông người lẫn cảnh có tuk-tuk) phải phản ánh trung thực cách hiểu của **người gán** dựa thuần túy trên [guideline/guideline_v1.0_label_contract.md](guideline/guideline_v1.0_label_contract.md).

### Giải nghĩa quy tắc vàng trên bộ nhãn 8 lớp

Bộ nhãn của bài giữ nguyên 7 lớp mục tiêu (`person`, `bicycle`, `motorbike`, `car`, `bus`, `truck`, `three_wheeler`) và 1 vùng đặc biệt `ignore_region`. Mọi khung đối tượng chuẩn có chiều cao tối thiểu `20 px` (`min_box_height_px = 20`); cụm người hoặc xe quá nhỏ ở nền xa, hay đám đông dính sát nhau không thể tách từng cá thể, được gom bằng một box `ignore_region` để script chấm không phạt nhầm máy lẫn người. Khi gặp tình huống lưỡng lự giữa hai lớp hoặc vật bị che khuất nửa thân, bạn tự quyết định theo v1.0 rồi tick thuộc tính `can_review` thành `true` ngay trên box đó để đánh dấu mang ra rà ở Pha 6.

### Lựa chọn thiết kế: ghi lại thời gian thật và không mở `work/prelabel/`

Trong `submission/time_log.csv`, dòng `gold_manual` đo trực tiếp tốc độ gán tay nguyên thủy (số phút thực tế chia 4 ảnh). Ở cuối Pha 4, bạn so sánh con số này với tốc độ duyệt lô có pre-label (`batch_review`, phút trên 24 ảnh). Tuy nhiên, người gán gold trước thường đã làm quen bối cảnh phố châu Á nên tốc độ ở lô batch được hưởng một phần hiệu ứng quen tay; việc hiểu rõ cả hai yếu tố giúp bạn đánh giá trung thực giá trị của AI trong mục 4 của `REPORT.md`.

:::callout{tone="warning" title="Tuyệt đối không dùng AI trong Pha 1"}
Không chạy `prelabel.py`, không bật AI Interactor trong CVAT và không tham khảo nhãn mẫu trên `data/cache/` trước khi hoàn tất export `submission/labels/gold_manual.json`. Tập gold chỉ có giá trị kiểm định khi được tạo hoàn toàn bằng mắt và tay của bạn.
:::

### Các bước thực hiện Pha 1

1. Đọc kỹ một lượt `guideline/guideline_v1.0_label_contract.md`, đặc biệt mục quy định vùng `ignore_region`, quy tắc bao người lái `person` riêng với xe và cách phân biệt `three_wheeler`.
2. Trên Project CVAT vừa tạo ở Pha 0, tạo Task mới đặt tên `gold-manual` và tải lên đúng **4 ảnh** trong `work/tasks/gold/`.
3. Mở Job của `gold-manual`, dùng phím `N` (Rectangle) gán đầy đủ đối tượng cho cả 4 ảnh, để trống `conf`, gắn cờ `can_review` cho các ca biên và bấm `Ctrl+S` lưu sau mỗi ảnh. Bấm đồng hồ ghi tổng số phút làm việc vào dòng `gold_manual` của `submission/time_log.csv`.

:::image{src="walkthrough_images/02_cvat_gold_manual_workspace.png" alt="Không gian gán nhãn thủ công CVAT Task gold-manual trên ảnh img_012.jpg với box three_wheeler, car, person và thuộc tính can_review" caption="Gán tay trực tiếp trong Task gold-manual trên CVAT: box ôm sát phần nhìn thấy, để trống conf và tick can_review khi gặp ca phân vân."}
:::

4. Vào **Actions → Export task dataset**, chọn định dạng **COCO 1.0**, **tắt Save images**, bấm **OK** rồi chuyển sang trang **Requests** để tải file `.zip` khi tiến trình đạt 100%.

:::image{src="walkthrough_images/03_cvat_export_coco_requests.png" alt="Trang Requests trên CVAT xác nhận hoàn tất Export Annotations định dạng COCO 1.0 cho Task #23 đạt 100%" caption="Chờ tiến trình Export Annotations đạt Finished 100% trong mục Requests, giải nén lấy annotations/instances_default.json."}
:::

5. Trích file `annotations/instances_default.json` trong gói `.zip` lưu đè thành `submission/labels/gold_manual.json`, sau đó chạy lệnh kiểm tra cấu trúc COCO, đủ 4 ảnh gold và chiều cao box tối thiểu.

:::code-tabs{title="Kiểm tra file gold_manual.json sau khi export từ CVAT"}
```powershell {title="Windows (PowerShell)"}
python scripts/validate_labels.py submission/labels/gold_manual.json --expect gold
```
```bash {title="macOS / Ubuntu (Bash)"}
python scripts/validate_labels.py submission/labels/gold_manual.json --expect gold
```
:::

### Checkpoint Pha 1 và bước tiếp theo

Lệnh `validate_labels.py` với cờ `--expect gold` phải kết thúc với dòng `[submission\labels\gold_manual.json] 4 ảnh, <số đối tượng>: OK` (trong kiểm chứng thực tế trên 4 ảnh gold đạt `62` đối tượng, không có lỗi schema, thiếu ảnh hay tên lớp lạ; mọi cảnh báo chiều cao `< 20 px` phải được rà lại xem có nên chuyển sang `ignore_region` không). Khi `gold_manual.json` đã khóa và commit, bạn chính thức mở khóa bước dùng mô hình AI ở Pha 2.

## Pha 2 — Tour 3 model pre-label: YOLO26, YOLOE và SAM (0:25 → 0:55 · 30 phút)

### Bản chất vấn đề: mỗi họ mô hình giải một bài toán nhận diện khác nhau

Không có một mô hình đơn lẻ nào vừa nhanh, vừa hiểu mọi khái niệm địa phương vừa vẽ được mặt nạ sắc nét. Khi áp dụng **YOLO26** (`yolo26m.pt`, detector danh mục cố định 80 lớp COCO) lên ảnh phố Bangkok hay Hà Nội, mô hình bắt rất chắc `person`, `motorbike` và `car`, nhưng không hề có khái niệm `three_wheeler` nên bỏ sót hoàn toàn hoặc gọi nhầm tuk-tuk thành `car` hay `truck`. Ngược lại, **YOLOE-26s** (`yoloe-26s-seg.pt`, open-vocabulary theo prompt chữ `--classes three_wheeler`) tìm được tuk-tuk nhờ ngữ nghĩa ngôn ngữ-thị giác, nhưng box sinh ra thường lỏng hơn, dễ chồng hai box vào một xe hoặc gộp nhầm xe đẩy hàng rong. Cuối cùng, **SAM** (Segment Anything — chọn `sam2.1_b.pt`, `mobile_sam.pt` hoặc SAM 3 có token) không đoán tên lớp mà biến bounding box hoặc điểm click thành polygon/mask bám sát biên vật thể, đánh đổi thời gian xử lý lâu hơn đáng kể trên CPU.

:::image{src="docs/assets/model_families.png" alt="So sánh ba họ model pre-label gồm detector đóng YOLO26, open-vocabulary YOLOE và segmenter SAM" caption="Vai trò bổ trợ của ba họ mô hình: YOLO26 làm nền nhanh cho lớp phổ biến, YOLOE vá lớp lạ như three_wheeler, SAM xuất polygon chuẩn."}
:::

### Giải nghĩa tác động của ngưỡng và cảnh báo tham số PowerShell

Trong `scripts/prelabel.py`, tham số `--conf` kiểm soát cân bằng giữa **độ phủ (recall)** và **nhiễu nền (false positive)**: nâng `--conf` từ `0.25` lên `0.5` giúp giảm mạnh box rác ở nền xa nhưng có thể làm mất hơn 35% người đi bộ hoặc xe máy nhỏ. Mọi lượt chạy qua `prelabel.py` đều tự động lưu JSON kết quả (`pre.coco.json`), ảnh trực quan hóa trong `viz/` và thống kê tốc độ + số box vào `work/runs.csv`. Một lưu ý kỹ thuật quan trọng trên Windows: PowerShell tách chuỗi chứa dấu phẩy thành danh sách đối số rời rạc nếu không bọc trong dấu ngoặc kép, nên bắt buộc phải viết `--classes "person,motorbike,car"`.

### Lựa chọn thiết kế: tách cache theo lượt chạy và ghi log trung thực

Thay vì ghi đè một thư mục đầu ra duy nhất, bạn giữ riêng từng thư mục (`t_y26`, `t_y26_c05`, `t_yoloe`, `t_sam`) trên tập 8 ảnh `pilot` (hoặc hai ảnh tùy chọn `img_004.jpg`, `img_013.jpg` khi chạy SAM trên CPU để không chờ lâu), rồi chép số liệu vào `submission/model_log.csv` kèm hai cột quan trọng là `what_went_wrong` và `decision`. Với máy 8 GB RAM / chỉ CPU đã tải `--set small` + `--set sam1`, bạn đổi `--size m` thành `--size n` (thêm `--imgsz 640` nếu cần) và đổi `--sam sam2.1_b.pt` sang `--sam sam2.1_t.pt` hoặc `mobile_sam.pt` để chạy ~0.4–4 giây/ảnh, đồng thời ghi rõ phần đánh đổi bỏ sót vật nhỏ ở nền xa vào `model_log.csv`. Nếu máy quá chậm, hãy mở `notebooks/day08_colab.ipynb` trên Google Colab GPU T4 để xuất cùng cấu trúc `work/prelabel/` rồi tải về giải nén vào repo.

### Các bước thực hiện Pha 2

1. Chạy thử nghiệm 4 lượt model trên tập `pilot` (`8` ảnh) và hai ảnh thử SAM bằng nhóm lệnh dưới đây (lưu ý dấu ngoặc kép cho danh sách lớp trên PowerShell).

:::code-tabs{title="Chạy Tour 3 model: YOLO26 (conf 0.25 vs 0.5), YOLOE và YOLO26+SAM"}
```powershell {title="Windows (PowerShell)"}
python scripts/prelabel.py --model yolo26 --size m --split pilot --conf 0.25 --out work/prelabel/t_y26
python scripts/prelabel.py --model yolo26 --size m --split pilot --conf 0.5 --out work/prelabel/t_y26_c05
python scripts/prelabel.py --model yoloe --classes three_wheeler --split pilot --conf 0.25 --out work/prelabel/t_yoloe
python scripts/prelabel.py --model yolo26 --size m --classes "person,motorbike,car" --images img_004.jpg img_013.jpg --conf 0.3 --segmenter sam --sam sam2.1_b.pt --out work/prelabel/t_sam
```
```bash {title="macOS / Ubuntu (Bash)"}
python scripts/prelabel.py --model yolo26 --size m --split pilot --conf 0.25 --out work/prelabel/t_y26
python scripts/prelabel.py --model yolo26 --size m --split pilot --conf 0.5 --out work/prelabel/t_y26_c05
python scripts/prelabel.py --model yoloe --classes three_wheeler --split pilot --conf 0.25 --out work/prelabel/t_yoloe
python scripts/prelabel.py --model yolo26 --size m --classes "person,motorbike,car" --images img_004.jpg img_013.jpg --conf 0.3 --segmenter sam --sam sam2.1_b.pt --out work/prelabel/t_sam
```
:::

2. Mở ảnh đối chiếu trong `work/prelabel/t_y26/viz/`, `work/prelabel/t_y26_c05/viz/`, `work/prelabel/t_yoloe/viz/` và `work/prelabel/t_sam/viz/` để so sánh trực tiếp lỗi bỏ sót, sai lớp và độ khít viền polygon.

:::image{src="walkthrough_images/04_model_tour_yolo26_vs_yoloe_img002.jpg" alt="So sánh ảnh img_002.jpg giữa YOLO26m ở conf 0.25 bỏ sót tuk-tuk và YOLOE-26s phát hiện đúng three_wheeler với độ tin cậy 0.93" caption="Đối chiếu thực tế trên img_002.jpg: YOLO26m bỏ sót hoàn toàn chiếc tuk-tuk xanh ở tiền cảnh, trong khi YOLOE-26s bắt đúng three_wheeler (0.93) dù vẫn sinh box thừa 0.37 ở xe bán tải."}
:::

:::image{src="walkthrough_images/05_prelabel_sam_img013.jpg" alt="Kết quả chạy YOLO26m kết hợp SAM 2.1 trên ảnh img_013.jpg với đường polygon bao quanh người và xe máy nhưng khung tuk-tuk bị gắn nhầm nhãn car" caption="Kết quả t_sam trên img_013.jpg: SAM cắt mask người và xe sát viền nhưng hộp đầu vào từ YOLO26 vẫn gán nhầm tuk-tuk thành car (0.42)."}
:::

3. Mở `work/runs.csv`, chuyển các dòng chạy thực tế vào `submission/model_log.csv` và điền đầy đủ nhận xét ngắn gọn ở ba cột cuối (`what_went_right`, `what_went_wrong`, `decision`).

### Checkpoint Pha 2 và bước tiếp theo

Pha 2 hoàn thành khi `submission/model_log.csv` có ít nhất 4 dòng thuộc tối thiểu 3 biến thể/model khác nhau (kèm 1 lượt có `sam`), không để trống cột `what_went_wrong`, và bạn chỉ rõ được ít nhất một tình huống sai mà tăng hay giảm `--conf` đơn thuần không thể sửa. Tiếp theo, ta chạy quét một lượt ở ngưỡng thấp để chọn bộ ngưỡng riêng cho từng lớp ở Pha 3.

## Pha 3 — Quét ngưỡng theo lớp trên tập gold (0:55 → 1:05 · 10 phút)

### Bản chất vấn đề: vì sao một ngưỡng chung 0.3 làm hỏng chất lượng lô nhãn?

Áp dụng cùng một ngưỡng tin cậy cho mọi lớp giống như đặt cùng một chiều cao thanh chắn cho cả người đi bộ lẫn xe tải: lớp có nhiều biến dạng và kích thước nhỏ như `person` thường cần ngưỡng mềm hơn để giữ độ phủ, trong khi `motorbike` chồng lấn dày đặc ở đường phố châu Á dễ sinh hàng loạt box giả nếu hạ ngưỡng quá thấp. Không có ngưỡng hoàn hảo cho mọi mục tiêu: ngưỡng thấp tạo nhiều box thừa khiến người duyệt mất công xóa, còn ngưỡng cao làm rơi vật thật vào vùng bỏ sót rất khó phát hiện bằng mắt. Bằng cách chạy inference toàn tập một lần duy nhất ở ngưỡng đáy `--conf 0.05` (`--no-viz`), bạn lưu được toàn bộ ứng viên vào đĩa rồi dùng `scripts/threshold_sweep.py` quét 19 mức ngưỡng (từ `0.10` đến `0.95`, bước `0.05`) trên tập gold trong vài giây mà không phải chạy lại mạng nơ-ron.

### Giải nghĩa ngưỡng mục tiêu 90% recall, quy tắc mẫu nhỏ và trần 0.5

Script `threshold_sweep.py` tính precision, recall và F1 cho từng lớp ở mỗi mức ngưỡng (ghép cùng lớp với `match_iou = 0.5`), rồi tìm ngưỡng tin cậy **cao nhất (không vượt quá `--max-thr 0.5`)** mà recall trên 4 ảnh gold vẫn đạt tối thiểu `90%` (`recall_target = 0.9`). Nếu một lớp trên 4 ảnh gold có ít hơn 5 cá thể (`--min-objects 5`, ví dụ kiểm chứng thực tế chỉ có 4 `bicycle`, 1 `bus`, 1 `truck`, 4 `three_wheeler`), script tự động từ chối chọn ngưỡng riêng và khuyến nghị giữ **mặc định (`0.3`)** vì thống kê trên 1–4 vật không có độ tin cậy. Quan trọng hơn, những con số precision/recall in trong `threshold_sweep.md` luôn là phép đo **in-sample** (chọn ngưỡng trên chính 4 ảnh gold rồi đo lại trên chính 4 ảnh đó), nên luôn lạc quan hơn hiệu năng thật trên 24 ảnh batch.

### Lựa chọn thiết kế: nạp cả hai nguồn ứng viên qua `--pred`

Ta xuất hai tệp ứng viên nền gồm `work/prelabel/y26_all/pre.coco.json` (YOLO26m chạy trên `all`, ghi nhận `1.669` box trong thử nghiệm thực tế) và `work/prelabel/yoloe_all/pre.coco.json` (YOLOE-26s tìm riêng `three_wheeler`, `125` box). Với máy 8 GB RAM / chỉ CPU, bạn thay `--size m` bằng `--size n` khớp trọng số `--set small` đã tải (hoặc dùng phương án dự phòng từ `data/cache/` nếu máy/mạng quá yếu và ghi rõ "dùng cache" vào `REPORT.md` mục 8). Tham số `--pred` của `threshold_sweep.py` nhận đồng thời cả hai tệp trên cùng dòng lệnh để script tổng hợp ứng viên theo lớp đối chiếu với `--gold submission/labels/gold_manual.json` và in sẵn chuỗi `--thr "..."` chuẩn bị cho Pha 4.

### Các bước thực hiện Pha 3

1. Chạy hai lượt cache ở ngưỡng thấp `--conf 0.05` trên toàn bộ 36 ảnh (`--split all --no-viz`), rồi chạy `threshold_sweep.py` truyền cả hai tệp qua `--pred` và đối chiếu với `submission/labels/gold_manual.json`.

:::code-tabs{title="Cache ứng viên conf 0.05 và quét ngưỡng theo lớp trên 4 ảnh gold"}
```powershell {title="Windows (PowerShell)"}
python scripts/prelabel.py --model yolo26 --size m --split all --conf 0.05 --out work/prelabel/y26_all --no-viz
python scripts/prelabel.py --model yoloe --classes three_wheeler --split all --conf 0.05 --out work/prelabel/yoloe_all --no-viz
python scripts/threshold_sweep.py --pred work/prelabel/y26_all/pre.coco.json work/prelabel/yoloe_all/pre.coco.json --gold submission/labels/gold_manual.json
```
```bash {title="macOS / Ubuntu (Bash)"}
python scripts/prelabel.py --model yolo26 --size m --split all --conf 0.05 --out work/prelabel/y26_all --no-viz
python scripts/prelabel.py --model yoloe --classes three_wheeler --split all --conf 0.05 --out work/prelabel/yoloe_all --no-viz
python scripts/threshold_sweep.py --pred work/prelabel/y26_all/pre.coco.json work/prelabel/yoloe_all/pre.coco.json --gold submission/labels/gold_manual.json
```
:::

2. Mở ba sản phẩm sinh ra tại `submission/qc/threshold_sweep.md`, `submission/qc/threshold_sweep.csv` và biểu đồ `submission/qc/threshold_sweep.png` để đọc đường cong Precision–Recall từng lớp và chép chuỗi `--thr "..."` ở cuối file báo cáo.

:::image{src="walkthrough_images/06_threshold_sweep.png" alt="Biểu đồ 7 đường cong Precision-Recall theo ngưỡng tin cậy từ threshold_sweep.py với vạch đỏ tại các lớp đủ mẫu" caption="Đồ thị quét ngưỡng trên 4 ảnh gold: person=0.35, motorbike=0.5, car=0.35 đạt >= 90% recall; các lớp dưới 5 mẫu giữ ngưỡng mặc định."}
:::

:::callout{tone="warning" title="Kiểm tra giới hạn lạc quan trước giờ giải lao"}
Đọc kỹ dòng cảnh báo mẫu nhỏ trong `threshold_sweep.md` và chép lại nhận xét vào mục 1 của `REPORT.md`. Đừng sửa tăng ngưỡng cho các lớp chỉ có 1–4 mẫu trong gold như `bicycle`, `bus`, `truck` hay `three_wheeler`.
:::

### Checkpoint Pha 3 và bước tiếp theo

Bạn chốt Pha 3 khi trong `submission/qc/` đã có đủ `.csv`, `.md`, `.png` từ `threshold_sweep.py` và chuỗi tham số gợi ý (ví dụ `--thr "person=0.35,motorbike=0.5,car=0.35"`). Sau **10 phút nghỉ giải lao (1:05 → 1:15)** để mắt thoát khỏi màn hình, bạn bắt đầu pha trọng tâm là gộp pre-label và duyệt 24 ảnh batch có bẫy.

## Pha 4 — Gộp pre-label, cài 12 bẫy và duyệt lô 24 ảnh trong CVAT (1:15 → 2:50 · 95 phút)

### Bản chất vấn đề: vì sao phải gài bẫy nhân tạo vào pre-label trước khi review?

Hiện tượng nguy hiểm nhất trong gán nhãn có AI hỗ trợ là **automation bias** (thiên kiến tự động hóa): hộp pre-label nhìn càng gọn và tự nhiên, người duyệt càng dễ lướt qua ảnh và bấm chấp nhận mà không tự kiểm chứng bằng mắt. Nếu chỉ báo cáo tỉ lệ `accepted = 90%`, ta không phân biệt nổi giữa trường hợp "mô hình rất tốt" với trường hợp "người duyệt buồn ngủ bấm qua loa". Nhằm đo lường trực diện độ tỉnh táo của chính bạn, `inject_traps.py` lấy mã sinh viên làm hạt giống ngẫu nhiên để cấy **12 lỗi nhân tạo** (3 `wrong_class`, 3 `loose`, 3 `extra`, 3 `missing`) rải vào lô 24 ảnh batch, rồi `review_stats.py` kiểm xem bạn phát hiện được bao nhiêu phần trăm số bẫy này (`trap_catch_min = 80%`, tức từ `10/12` trở lên).

### Giải nghĩa thứ tự ưu tiên file, `pre_traps.coco.json` vs `pre_traps_import.coco.json` và số 0 đầu

Trong `merge_prelabels.py`, thứ tự tệp truyền vào quyết định quyền ưu tiên: tệp đứng trước thắng tệp đứng sau khi hai box chồng nhau với `IoU >= 0.6`. Vì vậy, bạn phải đặt `work/prelabel/yoloe_all/pre.coco.json` **trước** `work/prelabel/y26_all/pre.coco.json` để `three_wheeler` của YOLOE loại bỏ box `car`/`truck` nhận nhầm từ YOLO26 trên cùng chiếc tuk-tuk. Sau đó, lệnh `inject_traps.py` tạo ra cùng lúc hai bản COCO: `submission/prelabel/pre_traps.coco.json` (giữ đủ 28 ảnh `batch + gold` kèm điểm số dự đoán để tính số đo AI error profile và precision/recall trên gold) và `submission/prelabel/pre_traps_import.coco.json` (chỉ chứa đúng **24 ảnh batch** để nạp sạch vào Task `batch-review` trên CVAT). Do đó, khi tải lên CVAT bạn **chỉ được chọn `pre_traps_import.coco.json`**. Ngoài ra, trên Windows PowerShell, luôn bọc `--student "00123456"` trong ngoặc kép trùng khít 100% với chuỗi MSHV ở dòng tiêu đề `README.md` để không bao giờ bị mất số `0` đứng đầu.

### Lựa chọn thiết kế: quy trình duyệt 4 bước chống nhìn gợi ý trước

Để không rơi vào bẫy `missing` (vật bị xóa mất nhãn máy) hoặc `loose` (box bị nới lỏng thêm phần nền), bạn bắt buộc thực hiện checklist gồm bốn bước cố định: (1) **quét toàn ảnh** bằng mắt theo chiều trái-phải và đếm nhẩm tổng số người/xe trước khi chú ý tới màu sắc các khung bao (dùng nút hình con mắt ở đầu bảng *Objects* để ẩn hết nhãn trước); (2) đối chiếu từng vật bạn thấy để **thêm box sót (`N`)** hoặc **xóa box ma (`Del`)**; (3) soi **nhãn và 4 cạnh viền** từng box để sửa sai lớp (như `car` → `three_wheeler`), kéo khít box bị hở nền (`loose`) và xoá box trùng lặp nếu YOLOE vẽ đè hai khung lên cùng một tuk-tuk; (4) dùng nút **Filters** trên thanh công cụ CVAT lọc theo từng lớp hoặc gắn cờ `can_review` cho các ca nửa che khuất cần đem ra Calibration ở Pha 6.

:::image{src="docs/assets/review_checklist.png" alt="Sơ đồ 4 bước checklist duyệt nhãn chống automation bias gồm quét ảnh trước, sửa thiếu-thừa, chỉnh lớp và kiểm viền 4 cạnh" caption="Checklist 4 bước bắt buộc cho mỗi bức ảnh khi review pre-label nhằm chặn đứng lỗi sót vật và box lỏng."}
:::

### Các bước thực hiện Pha 4

1. Gộp kết quả từ `yoloe_all` (ưu tiên trước) và `y26_all` với bộ ngưỡng từ Pha 3 vào `work/prelabel/merged_review/pre.coco.json`, sau đó cài 12 bẫy bằng đúng MSHV của bạn (thay `00123456` bằng mã của bạn, giữ nguyên ngoặc kép và **không** mở `submission/prelabel/traps_key.json` trước khi duyệt xong).

:::code-tabs{title="Gộp pre-label theo ngưỡng từng lớp và tiêm 12 ảnh bẫy theo MSHV"}
```powershell {title="Windows (PowerShell)"}
python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.35,motorbike=0.5,car=0.35" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
python scripts/inject_traps.py --pre work/prelabel/merged_review/pre.coco.json --student "00123456"
```
```bash {title="macOS / Ubuntu (Bash)"}
python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.35,motorbike=0.5,car=0.35" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
python scripts/inject_traps.py --pre work/prelabel/merged_review/pre.coco.json --student "00123456"
```
:::

2. Trên Project CVAT đã có ở Pha 0, tạo Task mới đặt tên `batch-review` và tải lên toàn bộ **24 ảnh** trong `work/tasks/review/`.
3. Vào **Actions → Upload annotations** trên Task `batch-review`, chọn **Import format: COCO 1.0**, chọn tệp `submission/prelabel/pre_traps_import.coco.json`, giữ **Import mode: Replace** (hoặc chuyển **Append** nếu muốn giữ nhãn cũ) rồi bấm **OK**.

:::callout{tone="danger" title="Lưu ý quan trọng trên CVAT 2.74.1 sau khi bấm OK"}
Khi `Import mode` đang để `Replace`, sau khi bấm **OK** trên hộp thoại Upload annotations, CVAT sẽ hiện thêm một hộp thoại xác nhận **Replace existing annotations?**. Phải bấm tiếp nút **Replace annotations** thì nhãn mới được nạp thật vào Task (nếu đóng nhầm hộp thoại thứ hai, Job vẫn có 0 shape).
:::

:::image{src="walkthrough_images/07_cvat_upload_pre_traps_import.png" alt="Hộp thoại Import annotation to task trên CVAT chọn định dạng COCO 1.0 và tệp pre_traps_import.coco.json" caption="Nạp đúng tệp submission/prelabel/pre_traps_import.coco.json theo định dạng COCO 1.0 trong màn hình Actions → Upload annotations."}
:::

4. Mở Job của `batch-review`, bấm đồng hồ bắt đầu và duyệt lần lượt cả 24 ảnh theo checklist 4 bước (với 6 ảnh xe máy dày `005, 018, 021, 024, 028, 029`, cụm xe nhỏ `< 20 px` gom bằng `ignore_region`). Dùng thanh công cụ **Filters** bên phải trên cùng nếu muốn rà riêng từng lớp hoặc kiểm box có điểm `conf` thấp, và tick `can_review` ở panel **Objects** cho các ca biên. Nhớ `Ctrl+S` thường xuyên và ghi tổng thời gian thực tế vào dòng `batch_review` của `submission/time_log.csv`.

:::image{src="walkthrough_images/08_cvat_batch_review_job_ui.png" alt="Màn hình Job review 24 ảnh trong CVAT hiển thị ảnh img_013.jpg với nhãn three_wheeler, thuộc tính conf=0.63, can_review và nút Filters trên thanh công cụ" caption="Giao diện duyệt batch-review trong CVAT: dùng bảng Objects bên phải xem conf/can_review và nút Filters trên thanh top bar."}
:::

5. Khi hoàn tất 24 ảnh, vào **Actions → Export task dataset** (COCO 1.0, tắt Save images), lấy `annotations/instances_default.json` từ gói `.zip` lưu thành `submission/labels/batch_reviewed.json`, rồi chạy ngay kiểm tra toàn bộ nhãn và tính thống kê review với đầy đủ `--pre`, `--final`, `--gold`, `--key`.

:::code-tabs{title="Kiểm tra nhãn batch sau duyệt và tính AI error profile + tỉ lệ bắt bẫy"}
```powershell {title="Windows (PowerShell)"}
python scripts/validate_labels.py --all
python scripts/review_stats.py --pre submission/prelabel/pre_traps.coco.json --final submission/labels/batch_reviewed.json --gold submission/labels/gold_manual.json --key submission/prelabel/traps_key.json
```
```bash {title="macOS / Ubuntu (Bash)"}
python scripts/validate_labels.py --all
python scripts/review_stats.py --pre submission/prelabel/pre_traps.coco.json --final submission/labels/batch_reviewed.json --gold submission/labels/gold_manual.json --key submission/prelabel/traps_key.json
```
:::

### Checkpoint Pha 4 và bước tiếp theo

Lệnh `validate_labels.py --all` phải báo `OK` cho cả `gold_manual.json` (4 ảnh) lẫn `batch_reviewed.json` (24 ảnh), đồng thời `submission/qc/review_stats.md` ghi rõ bảng 4 trạng thái (`accepted`, `edited`, `deleted`, `added`), Precision/Recall so với gold và tỉ lệ bắt bẫy **>= 10/12 (80%)** (mốc kiểm chứng E2E đạt `11/12 = 91.7%`). Nếu chưa đạt `80%`, bạn cần mở lại những ảnh bẫy bị lọt để siết lại cách duyệt trước khi sang Pha 5 tách mask SAM.

## Pha 5 — Tách mask SAM cho 3 đối tượng khó (2:50 → 3:05 · 15 phút)

### Bản chất vấn đề: vì sao box chữ nhật bất lực trước vật thể rỗng ruột và chen chúc?

Với xe tuk-tuk có mái che khung sắt hở nền hoặc hai người ngồi sát nhau trên cùng một băng ghế, bounding box chữ nhật luôn chứa phần lớn diện tích nền, ghế trống hoặc phần thân của đối tượng khác nằm phía sau. Bài toán instance segmentation ở mức pixel giải quyết vấn đề này bằng **các cặp điểm kích hoạt dương (`--pos`) và điểm loại trừ âm (`--neg`)** đưa vào mạng Segment Anything: một click đơn có thể tương ứng cả **bộ phận, toàn bộ vật hay cả nhóm** ("một click, ba mask hợp lệ"), nên chính **label contract** quyết định phạm vi mask mong muốn chứ không phải mô hình tự đoán. Điểm dương kéo mặt nạ mở rộng lên thân xe mục tiêu, còn điểm âm đặt lên người lái hoặc nền shop phía sau giúp cắt bớt phần thừa tại biên tiếp xúc.

### Giải nghĩa lựa chọn phiên bản SAM, quy chuẩn mục 4.2 và cú pháp `sam2_points.py`

Trong `scripts/sam2_points.py`, bạn chọn phiên bản qua `--sam sam2.1_b.pt` (SAM 2.1 base), `--sam mobile_sam.pt` (SAM 1 bản gọn nhẹ ~40 MB chạy nhanh trên CPU) hoặc `--sam sam_b.pt` (hay dùng SAM 3 nếu có quyền tải mô hình), kèm tùy chọn thu gọn bằng khung `--box "x1,y1,x2,y2"`. Theo quy ước ở **mục 4.2** của `guideline_v1.0_label_contract.md`, mask của phương tiện như `three_wheeler` chỉ ôm phần khung xe, mái che và bánh xe; người lái (`person`) và hàng hóa nhô hẳn ra ngoài phải được tách riêng bằng điểm `--neg`. Về mặt cú pháp CLI, script tự đọc ảnh từ `data/images/` nên `--image` chỉ truyền tên tệp (`img_013.jpg`), mỗi tọa độ click phải truyền bằng một cờ `--pos "x,y"` hoặc `--neg "x,y"` lặp lại nhiều lần (luôn bọc ngoặc kép để PowerShell không tách dấu phẩy và không cắt số `0` đầu), còn tên ảnh kết quả điều khiển qua `--out submission/masks --name <ten_file>`.

### Lựa chọn thiết kế: so sánh trực tiếp SAM 2.1 vs MobileSAM trên cùng một xe

Thay vì chỉ bấm lấy đủ 3 file ảnh rồi dừng, bạn nên chạy thử cả `sam2.1_b.pt` và `mobile_sam.pt` trên cùng một bộ tọa độ click ở chiếc tuk-tuk `img_013.jpg` để ghi lại nhận xét định tính rõ ràng vào mục 5 của `REPORT.md`: model nào bám viền nan hoa, bánh xe và mái vòm tốt hơn, đồng thời kiểm xem chỉ đặt **1 điểm âm** lên người lái đã đủ tách hẳn người lái ra khỏi mask hay vẫn còn dính phần tay/chân.

### Các bước thực hiện Pha 5

1. Chọn 3 ảnh có đối tượng khó tách (ví dụ `img_013.jpg`, `img_004.jpg`, `img_017.jpg`), mở ảnh xem tọa độ pixel thật trên kích thước gốc (`1024 × 682`) để xác định điểm dương trên thân/mái xe và điểm âm lên người lái hoặc nền phía sau.
2. Chạy `scripts/sam2_points.py` theo đúng từng cờ `--pos` / `--neg` và `--name` để xuất ít nhất 3 ảnh `.jpg` vào thư mục `submission/masks/`.

:::code-tabs{title="Tách mask 3 đối tượng khó bằng SAM 2.1 và đối chiếu với MobileSAM"}
```powershell {title="Windows (PowerShell)"}
python scripts/sam2_points.py --image img_013.jpg --pos "400,480" --pos "600,110" --neg "720,300" --sam sam2.1_b.pt --out submission/masks --name tuktuk_img013_sam2
python scripts/sam2_points.py --image img_013.jpg --pos "400,480" --pos "600,110" --neg "720,300" --sam mobile_sam.pt --out submission/masks --name tuktuk_img013_sam1
python scripts/sam2_points.py --image img_004.jpg --pos "620,480" --pos "700,560" --neg "465,450" --sam sam2.1_b.pt --out submission/masks --name tuktuk_img004_sam2
python scripts/sam2_points.py --image img_017.jpg --pos "560,420" --pos "680,500" --neg "645,355" --sam sam2.1_b.pt --out submission/masks --name tuktuk_img017_sam2
```
```bash {title="macOS / Ubuntu (Bash)"}
python scripts/sam2_points.py --image img_013.jpg --pos "400,480" --pos "600,110" --neg "720,300" --sam sam2.1_b.pt --out submission/masks --name tuktuk_img013_sam2
python scripts/sam2_points.py --image img_013.jpg --pos "400,480" --pos "600,110" --neg "720,300" --sam mobile_sam.pt --out submission/masks --name tuktuk_img013_sam1
python scripts/sam2_points.py --image img_004.jpg --pos "620,480" --pos "700,560" --neg "465,450" --sam sam2.1_b.pt --out submission/masks --name tuktuk_img004_sam2
python scripts/sam2_points.py --image img_017.jpg --pos "560,420" --pos "680,500" --neg "645,355" --sam sam2.1_b.pt --out submission/masks --name tuktuk_img017_sam2
```
:::

3. Mở từng ảnh xuất trong `submission/masks/` để kiểm tra vị trí chấm xanh (`--pos`), chấm đỏ (`--neg`), số lượng polygon in ra trên terminal, quan sát xem mask có còn dính phần người lái/hành khách không (thử thêm 1–2 `--neg` nếu cần) và ghi bảng so sánh vào mục 5 của `submission/REPORT.md`.

:::image{src="walkthrough_images/09_sam_points_mask_tuktuk.jpg" alt="Ảnh img_013.jpg với mask SAM 2.1 phủ màu hồng trên thân xe tuk-tuk nhờ hai điểm xanh dương và một điểm đỏ âm đặt lên người lái" caption="Mask tách bằng sam2_points.py trên img_013.jpg: hai điểm xanh (400,480; 600,110) phủ toàn xe tuk-tuk, còn một điểm đỏ (720,300) mới khoét bớt lưng áo người lái chứ chưa tách hết đầu và chân — lý do cần thêm click âm hoặc --box."}
:::

### Checkpoint Pha 5 và bước tiếp theo

Thư mục `submission/masks/` phải có ít nhất **3 tệp `.jpg`** rõ nét (đủ dấu click và vùng phủ bám xe) kèm bảng đối chiếu vào mục 5 của `submission/REPORT.md` và dòng ghi nhận chạy SAM trong `submission/model_log.csv`. Bước cuối cùng trên lớp ở Pha 6 là ghép bản release, rút mẫu QC bằng mắt và chốt đặc tả v1.1.

## Pha 6 — Calibration, QC theo mẫu và chốt quyết định Release / Hold / Rework (3:05 → 3:50 · 45 phút)

### Bản chất vấn đề: vì sao không thể kết thúc bài ngay sau khi bấm Save trên CVAT?

Ngay cả khi người duyệt đạt 100% bẫy ở Pha 4, bản thân người duyệt vẫn có điểm mù lặp lại — ví dụ quen tay bỏ qua người bán hàng ngồi khuất sau sạp chuối, hoặc bỏ qua hai khung `three_wheeler` chồng đè mà YOLOE sinh ra ở cạnh nhau. Kiểm định QC cuối lô (`qc_sample.py` + `qc_report.py`) rút ra tổ hợp **tối đa 3 ảnh `can_review` + 5 ảnh ngẫu nhiên (`--seed 42`)** trên lô 24 ảnh để soi lại từng ID đối tượng và đối chiếu ngưỡng chất lượng `error_rate_img_max = 2%`. Song song đó, quá trình Calibration bắt buộc bạn biến mọi vướng mắc đã gặp thành quy tắc cụ thể trong `guideline_v1.1_calibrated.md`, khép lại câu hỏi mở còn bỏ ngỏ ở v1.0.

### Giải nghĩa cây quyết định `Hold → Rework → Release` và giới hạn cận trên Wilson 95%

Khi đọc đầu ra của `qc_report.py` tại `submission/qc/qc_report_auto.md`, bạn luôn đi theo đúng thứ tự **Hold → Rework → Release** thay vì tự ý chọn `Release` cho đẹp hồ sơ. Cụ thể: chọn **Hold** nếu guideline vẫn còn ca chưa chốt được định nghĩa (`--guideline-gap > 0`) hoặc dữ liệu ảnh gốc có lỗi (`--data-problem > 0`); chọn **Rework** nếu tỉ lệ ảnh lỗi vượt quá `2%`, xuất hiện lỗi hệ thống (`>= 3` lỗi cùng loại chiếm `>= 50%`), bắt bẫy dưới `80%`, hoặc độ nhất quán tự kiểm không đạt; chỉ chọn **Release** khi đồng thời sạch cả hai tầng trên. Đặc biệt, ngay cả khi mẫu 7 ảnh cho `0` lỗi (`0%`), cận trên khoảng tin cậy Wilson 95% vẫn lên tới **`35%`** (và với `4/7` ảnh lỗi như mẫu E2E thực tế, cận trên đạt tới **`84%`**) — mẫu nhỏ chỉ phát hiện được lỗi lớn chứ không thể chứng minh tuyệt đối lô sạch dưới 2%.

:::image{src="docs/assets/decision_tree.png" alt="Cây quyết định QC theo thứ tự kiểm Hold (guideline gap hoặc ảnh hỏng), Rework (ảnh lỗi > 2%, lỗi hệ thống, bẫy < 80%) và Release" caption="Cây quyết định Hold → Rework → Release đi kèm báo cáo tự động qc_report_auto.md và luận giải tại REPORT.md."}
:::

### Lựa chọn thiết kế: ghi nhận trung thực Rework kèm ca lỗi mắt thấy trên overlay

Để phiếu kiểm `qc_review.csv` có cơ sở kiểm chứng lại, `qc_sample.py` sinh sẵn bộ ảnh đánh số ID từng hộp (`work/qc/review/img_0xx.jpg`). Nếu phát hiện hộp duplicate từ YOLOE hay người đi bộ bị sót ở rìa ảnh, hãy ghi đúng mã `object_id` và một trong các giá trị hợp lệ (`khong`, `bo_sot`, `thua`, `sai_lop`, `sai_vi_tri_kich_thuoc`, `sai_thuoc_tinh`, `sai_mask`) vào `qc_review.csv`. Nếu quyết định cuối cùng là **Rework**, bạn viết thẳng lý do kèm danh sách việc cần sửa ở bước tiếp theo (hoặc sửa các ảnh lỗi trên CVAT, xuất thành `submission/labels/qc_fixed.json`, chạy lại `build_release.py` để ghi nhận vòng lặp sửa lỗi).

### Các bước thực hiện Pha 6

1. Tạo gói release candidate từ `batch_reviewed.json` (tự động ghép `qc_fixed.json` nếu có), rồi rút mẫu kiểm QC 5 ảnh ngẫu nhiên kết hợp tối đa 3 ảnh có `can_review`.

:::code-tabs{title="Sinh bản release candidate và rút mẫu QC có đánh số ID đối tượng"}
```powershell {title="Windows (PowerShell)"}
python scripts/build_release.py
python scripts/qc_sample.py --n 5 --seed 42
```
```bash {title="macOS / Ubuntu (Bash)"}
python scripts/build_release.py
python scripts/qc_sample.py --n 5 --seed 42
```
:::

2. Mở toàn bộ các ảnh trong `work/qc/review/`, đối chiếu từng box với guideline để điền đủ dòng cho mọi ảnh trong `submission/qc/qc_review.csv`, rồi chạy `python scripts/qc_report.py` để sinh `submission/qc/qc_report_auto.md`.

:::image{src="walkthrough_images/10_qc_review_overlay.jpg" alt="Ảnh work/qc/review/img_013.jpg hiển thị ID từng bounding box sau duyệt làm lộ hai khung #115 và #116 three_wheeler chồng đè lên nhau" caption="Ảnh kiểm QC work/qc/review/img_013.jpg: việc in ID từng box giúp phát hiện nhanh lỗi trùng lặp #115 và #116 three_wheeler (thua) để ghi vào qc_review.csv và calibration."}
:::

3. Hoàn thiện tối thiểu **3 quyết định** vào `submission/qc/calibration_decisions.csv` (phải có đủ ca từ `loi_may`, `cau_hoi_mo` ở mục 6 v1.0 và `can_review` hoặc `bay_lot`), rồi chuyển phần bổ sung (gồm ảnh ví dụ `img_0xx.jpg` và lịch sử `v1.0 → v1.1`) vào `submission/guidelines/guideline_v1.1_calibrated.md`.
4. Viết đầy đủ **8 mục** trong `submission/REPORT.md` (mọi tỉ lệ bắt buộc kèm mẫu số như `431 box máy`, `61 đối tượng gold`, `123 đối tượng QC`, `12 bẫy`, nêu rõ cận trên Wilson 95% và quyết định `Release / Hold / Rework`), cùng bản tóm tắt cuối cùng trong `STUDENT.md`.

:::callout{tone="info" title="Tự kiểm tính đầy đủ dữ liệu trước khi đóng pha"}
Bảo đảm không còn ký hiệu `<<` chưa điền trong `README.md`, `STUDENT.md`, `submission/REPORT.md` và `submission/guidelines/guideline_v1.1_calibrated.md`; toàn bộ các bảng CSV đều dùng mã hóa UTF-8 và khớp số liệu từ `submission/qc/`.
:::

### Checkpoint Pha 6 và bước tiếp theo

Pha 6 hoàn tất khi thư mục `submission/release/` có `final_det.coco.json` + `release_manifest.json`, `submission/qc/qc_report_auto.md` khớp phiếu `qc_review.csv`, `guideline_v1.1_calibrated.md` đồng bộ với `calibration_decisions.csv` và `REPORT.md` hoàn chỉnh cả 8 mục.

## Nộp bài & Kiểm tra cuối cùng (3:50 → 4:00 · 10 phút)

Ở chặng cuối, bạn rà soát toàn bộ gói sản phẩm cá nhân theo chuẩn [SUBMISSION.md](SUBMISSION.md) và [RULES.md](RULES.md) trước khi đẩy commit cuối lên GitHub và nộp link lên LMS (`https://aithucchien.vinuni.edu.vn`). Trước hết, kiểm tra dòng tiêu đề trong `README.md` phải khớp chính xác mẫu `# Day 08 Lab — <MSHV> - <Họ và tên>`, điền nốt commit hash và bảng "Kết quả tóm tắt" vào `STUDENT.md`, đồng thời bảo đảm thư mục `submission/` có đủ các tệp bắt buộc: `taxonomy.json` (giữ nguyên mục `rules`), `REPORT.md` (8 mục), `model_log.csv`, `time_log.csv`, `guidelines/guideline_v1.1_calibrated.md`, `prelabel/{pre_traps_import.coco.json, pre_traps.coco.json, traps_key.json}`, `labels/{gold_manual.json, batch_reviewed.json}`, `masks/*.jpg` (≥ 3 ảnh), `release/{final_det.coco.json, release_summary.csv, release_manifest.json}` và `qc/{threshold_sweep.*, review_stats.*, calibration_decisions.csv, qc_sample.csv, qc_review.csv, qc_report_auto.md}`.

Tiếp theo, bạn chạy `python scripts/check_submission.py` và `git status` để tự kiểm lỗi trước khi `git push`. Script `check_submission.py` tự động gọi `validate_labels.py --all` trong môi trường UTF-8 an toàn trên Windows, đối khớp MSHV trong `traps_key.json` với `README.md`, rà `git ls-files` nhằm chặn commit nhầm `work/`, `models/` hoặc các tệp trọng số `.pt/.pth/.onnx/.ts`, và kiểm tra tối thiểu `4` commit theo từng Part bằng chính tài khoản Git của bạn. Lưu ý rằng `check_submission.py` kiểm tính hợp lệ cấu trúc chứ không chấm nội dung lập luận hay độ ôm sát của box: bạn vẫn phải tự đối chiếu bảng số liệu trong `REPORT.md` với đầu ra thực tế từ `submission/qc/`.

:::code-tabs{title="Kiểm tra toàn bộ bài nộp và đối chiếu trạng thái Git trước khi push"}
```powershell {title="Windows (PowerShell)"}
python scripts/check_submission.py
git status
```
```bash {title="macOS / Ubuntu (Bash)"}
python scripts/check_submission.py
git status
```
:::

Nếu giảng viên yêu cầu đóng gói thêm bản nộp dự phòng trên LMS theo chuẩn thư mục cá nhân `K5-DAY08-HoVaTen-MSSV/`, tuyệt đối vẫn phải giữ nguyên cấu trúc repo gốc `K5-L23-DAY08-[MSHV]-[HoTen]-AIPrelabel` ở chế độ **public** trên GitHub để giữ trọn lịch sử commit qua các pha. Tuyệt đối không đưa mật khẩu CVAT, token Hugging Face hay ảnh chụp chứa email cá nhân vào bất kỳ tệp hoặc lịch sử commit nào.

:::checklist{title="Kiểm tra bàn giao Day 08 trước khi push" tone="success"}
- [ ] Dòng tiêu đề `README.md` có dạng `# Day 08 Lab — <MSHV> - <Họ và tên>`, khớp `traps_key.json`.
- [ ] `python scripts/check_submission.py` trả về `0 lỗi, 0 cảnh báo. Sạch lỗi: có thể nộp.` (`exit code 0`).
- [ ] `git status` có >= 4 commit theo từng Part, không có trọng số `.pt / .pth / .onnx / .ts`, `work/` hay token bí mật.
- [ ] Đã đẩy đủ commit lên repo public `K5-L23-DAY08-[MSHV]-[HoTen]-AIPrelabel` và nộp đúng link trên LMS.
:::
