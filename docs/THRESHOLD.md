# Chọn ngưỡng tin cậy theo lớp (Part 3, 10 phút)

**Một việc duy nhất:** tìm ngưỡng `conf` cho **từng lớp** để pre-label ít box thừa mà không bỏ sót thêm.

Không có ngưỡng hoàn hảo, chỉ có đánh đổi: ngưỡng **thấp** → nhiều box thừa (mất công xoá); ngưỡng **cao** → bỏ sót (lỗi lọt vào dataset, khó thấy).

## Làm gì

1. Quét ngưỡng trên kết quả thô của hai model, với **nhãn gán tay** của bạn làm chuẩn ([GUIDE.md](../GUIDE.md) Part 3):
   ```bash
   python scripts/threshold_sweep.py --pred work/prelabel/y26_all/pre.coco.json work/prelabel/yoloe_all/pre.coco.json \
       --gold submission/labels/gold_manual.json
   ```
   (Script chỉ dùng 4 ảnh có trong file gold; không cần gộp trước.)
2. Mở `submission/qc/threshold_sweep.md` (bảng ngưỡng gợi ý) và `threshold_sweep.png` (precision / recall theo ngưỡng).
3. Copy chuỗi `--thr "..."` in ra vào lệnh `merge_prelabels.py` ở Part 4, và ghi ngưỡng cùng lý do vào `REPORT.md` mục 1.

## Đọc kết quả

- **Ngưỡng gợi ý** = ngưỡng **cao nhất (tối đa 0.5)** mà recall của lớp đó còn ≥ 0.9. Nghĩa là ít box thừa nhất trong khi chưa bỏ sót thêm.
- Lớp có **< 5 vật** trong 4 ảnh gold (thường bus, truck, bicycle) **không được chọn ngưỡng riêng**: script ghi "dùng ngưỡng mặc định" (0.3). Đó là đúng, vì vài vật không đủ để chọn ngưỡng.
- Dòng *"không ngưỡng nào ≤ 0.5 đạt recall 90%"*: model không biết lớp đó đủ tốt (ví dụ tuk-tuk với YOLO26): nói trong báo cáo, không ép chọn.

## Hai giới hạn phải nói trong báo cáo

- **Chọn và chấm trên cùng dữ liệu:** bạn chọn ngưỡng trên gold rồi (ở Part 4) đo máy trên gold, nên precision / recall của máy trên gold là **lạc quan**. Muốn chặt hơn: pilot riêng ở [ADVANCED.md](ADVANCED.md) mục 3.
- **Mẫu nhỏ:** chỉ 4 ảnh gold. Lớp ít vật (xem cột `n_gold` trong `threshold_sweep.csv`) thì ngưỡng chỉ là ước lượng thô.
