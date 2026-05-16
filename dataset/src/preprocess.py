# src/preprocess.py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
sys.path.append(str(Path(__file__).parent))
from utils import check_blurry

class ImagePreprocessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size
        
    def remove_noise(self, image):
        """Loại bỏ nhiễu"""
        # Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised
    
    def resize_with_padding(self, image):
        """Resize ảnh giữ tỷ lệ và thêm padding"""
        h, w = image.shape[:2]
        scale = self.target_size[0] / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Tạo ảnh vuông với padding đen
        square_img = np.zeros((self.target_size[0], self.target_size[1], 3), 
                             dtype=np.uint8)
        y_offset = (self.target_size[0] - new_h) // 2
        x_offset = (self.target_size[1] - new_w) // 2
        square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return square_img, (x_offset, y_offset, scale)
    
    def enhance_contrast(self, image):
        """Cải thiện contrast bằng CLAHE"""
        # Chuyển sang LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Áp dụng CLAHE cho channel L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Ghép lại và chuyển về BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def gamma_correction(self, image, gamma=1.5):
        """Điều chỉnh độ sáng tối"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def preprocess(self, image, apply_gamma=False):
        """Pipeline tiền xử lý hoàn chỉnh"""
        # 1. Loại bỏ nhiễu
        img = self.remove_noise(image)
        
        # 2. Kiểm tra ảnh mờ
        is_blurry, blur_value = check_blurry(img)
        if is_blurry:
            return None, blur_value
        
        # 3. Gamma correction (tùy chọn)
        if apply_gamma:
            img = self.gamma_correction(img)
        
        # 4. Cải thiện contrast
        img = self.enhance_contrast(img)
        
        # 5. Resize
        img, padding_info = self.resize_with_padding(img)
        
        return img, blur_value
    
    def process_directory(self, input_dir, output_dir, min_blur_threshold=50):
        """Xử lý toàn bộ thư mục (bao gồm cả thư mục con)"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Lấy tất cả file ảnh từ thư mục chính và các thư mục con
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']:
            # Tìm trong thư mục chính
            image_files.extend(input_path.glob(ext))
            # Tìm trong tất cả thư mục con (rglob = recursive glob)
            image_files.extend(input_path.rglob(ext))
        
        # Loại bỏ trùng lặp (nếu có)
        image_files = list(set(image_files))
        
        print(f"📁 Tìm thấy {len(image_files)} ảnh trong {input_dir}")
        
        # Hiển thị cấu trúc thư mục con nếu có
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        if subdirs:
            print(f"📂 Thư mục con tìm thấy: {', '.join([d.name for d in subdirs])}")
        
        stats = {'processed': 0, 'blurry': 0, 'failed': 0}
        
        for img_file in tqdm(image_files, desc="Đang xử lý ảnh"):
            try:
                # Đọc ảnh
                image = cv2.imread(str(img_file))
                if image is None:
                    stats['failed'] += 1
                    print(f"\n⚠️ Không đọc được: {img_file.name}")
                    continue
                
                # Tiền xử lý
                processed_img, blur_value = self.preprocess(image)
                
                # Kiểm tra độ mờ
                if processed_img is None or blur_value < min_blur_threshold:
                    stats['blurry'] += 1
                    print(f"\n📷 Ảnh mờ: {img_file.parent.name}/{img_file.name} (blur: {blur_value:.2f})")
                    continue
                
                # Tạo tên file duy nhất (thêm tên thư mục cha để tránh trùng)
                parent_name = img_file.parent.name
                if parent_name == input_path.name:
                    # Ảnh nằm trực tiếp trong thư mục chính
                    output_filename = img_file.name
                else:
                    # Ảnh nằm trong thư mục con
                    output_filename = f"{parent_name}_{img_file.name}"
                
                # Lưu ảnh đã xử lý
                output_file = output_path / output_filename
                cv2.imwrite(str(output_file), processed_img)
                stats['processed'] += 1
                
            except Exception as e:
                stats['failed'] += 1
                print(f"\n❌ Lỗi xử lý {img_file.name}: {str(e)}")
        
        # In thống kê
        print("\n" + "="*50)
        print("THỐNG KÊ XỬ LÝ:")
        print(f"✅ Đã xử lý thành công: {stats['processed']}")
        print(f"⚠️ Ảnh bị mờ (loại bỏ): {stats['blurry']}")
        print(f"❌ Lỗi: {stats['failed']}")
        print(f"📁 Kết quả lưu tại: {output_dir}")
        print("="*50)
        
        # Hiển thị 5 ảnh đã xử lý đầu tiên
        if stats['processed'] > 0:
            print("\n📸 Ảnh đã xử lý (5 ảnh đầu):")
            processed_files = list(output_path.glob("*"))[:5]
            for f in processed_files:
                size = f.stat().st_size / 1024
                print(f"   ✅ {f.name} ({size:.1f} KB)")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Tiền xử lý ảnh cho face mask detection')
    parser.add_argument('--input', type=str, required=True, help='Thư mục đầu vào')
    parser.add_argument('--output', type=str, required=True, help='Thư mục đầu ra')
    parser.add_argument('--size', type=int, default=640, help='Kích thước ảnh đầu ra')
    parser.add_argument('--blur_threshold', type=float, default=50, 
                       help='Ngưỡng phát hiện ảnh mờ')
    
    args = parser.parse_args()
    
    # Khởi tạo preprocessor
    preprocessor = ImagePreprocessor(target_size=(args.size, args.size))
    
    # Xử lý
    stats = preprocessor.process_directory(
        args.input, 
        args.output, 
        min_blur_threshold=args.blur_threshold
    )

if __name__ == "__main__":
    main()