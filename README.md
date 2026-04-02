# face-mask-tracking-system
Hệ thống phát hiện và theo dõi người đeo khẩu trang theo thời gian sử dụng YOLO và ByteTrack
##  📂 Cấu trúc dự án & mô tả
##  🔹 1. Root (thư mục gốc)
- README.md
  - Giới thiệu đề tài
  - Hướng dẫn cài đặt & chạy hệ thống
  - Mô tả cấu trúc project
- requirements.txt
  - Danh sách thư viện cần cài
  - Cài bằng:
    - pip install -r requirements.txt
- .gitignore
  - Bỏ qua file không cần push (dataset lớn, cache, log…)
##  ⚙️ 2. configs/ – Cấu hình hệ thống
- configs/
  - config.py
    - Chứa tham số hệ thống:
      - ALERT_TIME = 30
      - CONFIDENCE_THRESHOLD = 0.5
- tracker.yaml
  - Cấu hình cho ByteTrack:
    - ngưỡng tracking
    - IOU threshold
    - buffer size
##  📊 3. data/ – Dữ liệu (Hồng Ngọc)
- data/
  - raw/
    - Dữ liệu gốc:
      -  video quay
      -  ảnh chưa xử lý
  - processed/
    - Dữ liệu đã tiền xử lý:
      - resize
      - clean
      - chuẩn hóa
  - images/
    - Ảnh dùng để train YOLO
      - images/train/   → ảnh train
      - images/val/     → ảnh validation
  - labels/
    - Nhãn tương ứng với ảnh
      - labels/train/   → label train
      - labels/val/     → label val
  - dataset.yaml
    - File config dataset cho YOLO:
      - train: images/train
      - val: images/val

      - names:
        - 0: mask
        - 1: no_mask
##  🛠 4. scripts/ – Script xử lý dữ liệu (Hồng Ngọc)
- scripts/
  - extract_frames.py
    - Cắt video → ảnh
  - preprocess.py
    - Resize, normalize ảnh
  - split_dataset.py
    - Chia train / validation
##  🤖 5. models/ – Model (Quốc Trường)
- models/
  - weights/
    - Chứa model đã train:
      - best.pt
  - train_logs/
    - Log quá trình train:
      - loss
      - accuracy
      - biểu đồ
##  🧠 6. src/ – Code chính hệ thống
- src/
- 🔹 detection/ (Quốc Trường)
- yolo_detector.py
  - Load model YOLO
  - Detect mask / no_mask
  - Output:
    -bounding box
    -class
- 🔹 tracking/ (Kim Ngân – CORE)
- tracker.py
- Tracking bằng ByteTrack
- Gán ID cho từng người
- 🔹 timer/ (Kim Ngân)
- timer.py
  - Đếm thời gian theo từng ID
  - Reset khi mất tracking
- 🔹 alert/ (Duy Hưng)
  - alert.py
    - Xử lý cảnh báo:
    - 30s → alert
    - hiển thị text / âm thanh
- 🔹 visualization/ (Duy Hưng)
  - draw.py
    - Vẽ:
      - bounding box
      - ID
      - timer
      - alert
- 🔹 main.py ⭐
  - File chạy chính của hệ thống
  - Kết nối toàn bộ:
    - YOLO → Tracking → Timer → Alert → Hiển thị
##  🎥 7. demo/ – Demo hệ thống (Duy Hưng)
- demo/
  - input_videos/
    - Video đầu vào để test
  - output_videos/
    - Video sau khi detect + tracking
  - demo_app.py
    - Chạy demo:
      - OpenCV hoặc Streamlit
## 📈 8. evaluation/ – Đánh giá (Duy Hưng)
- evaluation/
  - metrics.py
    -Tính:
      -Precision
      -Recall
      -mAP
  -confusion_matrix.py
    -Vẽ confusion matrix
  -results/
    -Lưu:
    -bảng kết quả
    -biểu đồ
##  📓 9. notebooks/ – Notebook 
- notebooks/
  - data_analysis.ipynb( (Hồng Ngọc)
    -Phân tích dữ liệu
  - training.ipynb( Quốc Trường)
    -Train thử model
📄 10. reports/ – Báo cáo
- figures: chứa hình ảnh
- bib: chứa nguồn tham khảo
- .tex: code latex
