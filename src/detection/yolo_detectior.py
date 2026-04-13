from ultralytics import YOLO

# 1. Khởi tạo mô hình YOLO11
model = YOLO("yolo11n.pt") 

# 2. Chạy nhận diện trên 1 tấm ảnh cụ thể
# save=True: Tự động lưu ảnh kết quả vào thư mục 'runs/detect/predict'
# conf=0.5: Chỉ hiển thị các đối tượng có độ tin cậy trên 50%
results = model.predict(source="with_mask_1017.jpg", save=True, conf=0.5)
model.trai
# 3. Hiển thị kết quả ra màn hình (tùy chọn)
for result in results:
    result.show()