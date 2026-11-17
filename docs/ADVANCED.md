# Bài nâng cao (tuỳ chọn, ngoài 4 giờ)

Mọi mục dưới đây **không bắt buộc**. Làm khi còn thời gian hoặc muốn đào sâu; ghi kết quả vào `REPORT.md` (mục Ghi chú) nếu muốn được ghi nhận.
Cài thêm: `python scripts/download_models.py --set hf` (Grounding DINO, OWLv2) · `--set sam3` (cần quyền).

## 1. Grounding DINO và OWLv2: open-vocab "nặng"

**Là gì.** Hai model gõ-chữ-ra-box khác họ với YOLOE: Grounding DINO (text-grounded, Transformer, `IDEA-Research/grounding-dino-tiny`) và OWLv2 (`google/owlv2-base-patch16-ensemble`). Chậm hơn YOLOE ~3–10×.

```bash
python scripts/prelabel.py --model gdino --classes three_wheeler --split pilot --conf 0.3 --out work/prelabel/gdino_pilot
python scripts/prelabel.py --model owlv2 --classes three_wheeler --split pilot --conf 0.2 --out work/prelabel/owlv2_pilot
```

> Grounding DINO có thêm `--text-thr` (ngưỡng text, mặc định 0.25). OWLv2 trên Mac tự chạy CPU, chậm hơn nhưng ổn.
> Ngưỡng điểm của ba model **không so trực tiếp được** với nhau: 0.3 ở model này ≠ 0.3 ở model kia.

**Nhìn gì.** Lập bảng nhỏ cho `img_002`, `img_011`, `img_016`... (ảnh pilot có xe ba bánh): số tuk-tuk đúng (bạn tự đếm), YOLOE tìm được bao nhiêu, gdino, owlv2; ai có box thừa; ai nhanh nhất (`sec_per_img`).
**Không có model "thắng tuyệt đối"**: chọn model theo lớp và theo ngưỡng của chính **dữ liệu của bạn** (pilot).

---


## 2. SAM 3: gõ chữ, ra mọi instance + mask

**Là gì.** SAM 3 (Meta, 2025) nhận **cụm danh từ** và trả mask cho **mọi** vật khớp, không cần click. Yếu ở khái niệm hiếm, vật bị che (slide: cgF1 65.0 so với 74.2 của người).
Model gated: xin quyền trước ([SETUP.md](SETUP.md) mục 4).

```bash
python scripts/download_models.py --set sam3
python scripts/prelabel.py --model sam3 --classes three_wheeler --images img_004.jpg img_013.jpg --conf 0.4 --out work/prelabel/sam3_demo
```

> Đường chạy SAM 3 trong repo này chưa được kiểm chứng trên máy giảng viên (cần quyền truy cập). Nếu API của `transformers` khác bản ghi trong script,
> báo lỗi đầy đủ cho giảng viên thay vì tự sửa; ghi `decision = skipped` vào `model_log.csv`.

**Nhìn gì (nếu chạy được).** Mask có tách riêng từng tuk-tuk khi xếp sát nhau (img_004)? Có gom cả người lái vào xe không? So với B2 (box → SAM 2.1): bớt được bước nào? Còn lỗi nào?

---


**Xin quyền SAM 3:** [SETUP.md](SETUP.md) mục 4.

## 3. Pilot riêng để chọn ngưỡng (đỡ phụ thuộc vào gold)

Bài chính chọn ngưỡng trên **gold**, nên số đo máy trên gold lạc quan. Cách làm chặt hơn:

```bash
python scripts/prepare_workdirs.py --advanced     # thêm work/tasks/pilot/ (8 ảnh)
python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json \
    --conf 0.1 --iou 0.6 --split pilot --out work/prelabel/merged_pilot/pre.coco.json
# Task "pilot": nạp merged_pilot/pre.coco.json, duyệt, export submission/labels/pilot_reviewed.json
python scripts/threshold_sweep.py --pred work/prelabel/merged_pilot/pre.coco.json --gold submission/labels/pilot_reviewed.json
```
Giới hạn: nhãn chuẩn pilot dựng từ pre-label ngưỡng 0.10 rồi sửa, nên vật điểm < 0.10 mà bạn không tự vẽ thì vắng mặt. Rồi **dùng ngưỡng này** cho Part 4 và nói gold giờ là chuẩn "sạch".

## 4. Mask cho cả 8 ảnh `seg` và review mask

```bash
python scripts/prelabel.py --model yolo26 --size m --classes "person,bicycle,motorbike,car,bus,truck" --split seg --thr "<ngưỡng>" --conf 0.3 \
    --segmenter sam --sam sam2.1_b.pt --out work/prelabel/seg_y26
python scripts/prelabel.py --model yoloe --classes three_wheeler --split seg --conf 0.25 --segmenter sam --sam sam2.1_b.pt --out work/prelabel/seg_yoloe
python scripts/merge_prelabels.py work/prelabel/seg_yoloe/pre.coco.json work/prelabel/seg_y26/pre.coco.json \
    --conf 0.25 --iou 0.6 --split seg --out submission/prelabel/seg_pre.coco.json
python scripts/coco_to_cvat_xml.py --coco submission/prelabel/seg_pre.coco.json --shape polygon --out work/cvat_import/seg_pre.xml
```
Task `seg-review` từ `work/tasks/seg/`, nạp XML (**Upload annotations → CVAT 1.1**), review mask bằng checklist 4 bước, export `submission/labels/seg_reviewed.json`, rồi:
```bash
python scripts/validate_labels.py submission/labels/seg_reviewed.json --expect seg --mode seg
python scripts/review_stats.py --mode seg --pre submission/prelabel/seg_pre.coco.json --final submission/labels/seg_reviewed.json --splits batch --out submission/qc/review_stats_seg
```
**Chọn SAM 3:** `python scripts/prelabel.py --model sam3 --split seg --conf 0.4 --out work/prelabel/seg_sam3`, copy `pre.coco.json` thành `submission/prelabel/seg_pre.coco.json`.

## 5. Dùng AI trực tiếp trong CVAT

B1 (Interactor SAM), B2 (Automatic annotation), B3 (`cvat-cli` + `scripts/cvat_function_yolo26.py`): xem [CVAT_AI.md](CVAT_AI.md) mục 6. Ghi cách làm, kết quả, và so với offline.

## 6. Gán lại mù để đo nhất quán của chính bạn

Sau ≥ 60 phút kể từ lúc duyệt xong, **gán tay không AI**, theo guideline **v1.0**, 8 ảnh batch:

```bash
python scripts/prepare_workdirs.py --advanced        # work/tasks/selfcheck/
# Task "selfcheck-blind" -> gán -> export submission/labels/self_check_blind.json
python scripts/self_consistency.py --a submission/labels/batch_reviewed.json --b submission/labels/self_check_blind.json
```
Đạt khi IoU TB ≥ 0.85, cùng lớp ≥ 95% **và** box không ghép được ≤ 10%. Mỗi dòng `self_check_cases.csv` là ứng viên cho `calibration_decisions.csv` (`nguon = self_check`). Có bạn học: đổi file nhau, chạy `--a` của bạn, `--b` của họ.

## 7. Mở rộng cho GPU mạnh
- Thử `--size l/x` của YOLO26, `sam2.1_l.pt`, `--imgsz 1280`; so recall vật nhỏ.
- Đo precision/recall của nhiều model trên gold bằng `threshold_sweep.py` (gộp bằng `--pred` nhiều file).
- Gán lại gold sau ≥ 24 giờ **có** pre-label rồi `review_stats.py --splits gold` để đo residual / new error ([REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md)).

## 8. Phương án dự phòng: cache kết quả
`data/cache/` chứa kết quả YOLO26 / YOLOE (ngưỡng 0.05, 36 ảnh) đã chạy sẵn trên máy giảng viên (Apple MPS). Chỉ dùng khi không thể chạy model (xem GUIDE.md).
