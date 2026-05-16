# auto_create_labels.py
import cv2
from pathlib import Path
from tqdm import tqdm

def detect_faces_and_create_labels(images_dir, labels_dir, confidence_threshold=0.5):
    """
    Tự động phát hiện mặt và tạo labels
    Sử dụng OpenCV DNN face detector
    """
    
    # Tải face detection model
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách ảnh
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(images_path.glob(ext))
    
    if len(image_files) == 0:
        print(f"❌ Không tìm thấy ảnh trong {images_dir}")
        return 0
    
    print(f"📁 Tìm thấy {len(image_files)} ảnh")
    print("⚠️ Lưu ý: Labels tự động chỉ phát hiện mặt, KHÔNG phân biệt mask/no_mask")
    print("   Mặc định tất cả đều là 'mask' (class 0). Bạn cần chỉnh sửa thủ công sau!\n")
    
    stats = {
        'total_faces': 0,
        'no_face': 0,
        'success': 0
    }
    
    for img_file in tqdm(image_files, desc="Đang xử lý"):
        # Đọc ảnh
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện mặt
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        if len(faces) == 0:
            stats['no_face'] += 1
            print(f"\n⚠️ Không tìm thấy mặt trong: {img_file.name}")
            continue
        
        # Tạo label file
        label_file = labels_path / img_file.with_suffix('.txt').name
        
        with open(label_file, 'w') as f:
            for (x, y, w_box, h_box) in faces:
                # Chuyển sang YOLO format (chuẩn hóa)
                x_center = (x + w_box/2) / w
                y_center = (y + h_box/2) / h
                width = w_box / w
                height = h_box / h
                
                # Mặc định là mask (class 0) - bạn cần sửa lại thành no_mask nếu cần
                class_id = 0
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                stats['total_faces'] += 1
        
        stats['success'] += 1
    
    # In kết quả
    print("\n" + "="*50)
    print("KẾT QUẢ TỰ ĐỘNG LABELING:")
    print(f"✅ Đã xử lý: {stats['success']}/{len(image_files)} ảnh")
    print(f"👤 Tổng số mặt phát hiện: {stats['total_faces']}")
    print(f"⚠️ Ảnh không có mặt: {stats['no_face']}")
    print(f"📁 Labels được lưu tại: {labels_dir}")
    print("="*50)
    print("\n💡 GỢI Ý:")
    print("   - Dùng LabelImg để chỉnh sửa labels thủ công")
    print("   - Kiểm tra và đổi class từ mask sang no_mask nếu cần")
    
    return stats['success']

if __name__ == "__main__":
    # Tạo labels cho train
    print("="*50)
    print("TẠO LABELS CHO TRAIN")
    print("="*50)
    detect_faces_and_create_labels("data/images/train", "data/labels/train")
    
    print("\n" + "="*50)
    print("TẠO LABELS CHO VAL")
    print("="*50)
    detect_faces_and_create_labels("data/images/val", "data/labels/val")