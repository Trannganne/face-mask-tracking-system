# dataset/src/preprocess.py
# Nhiệm vụ: Hồng Ngọc — feature/preprocess_data
# Tiền xử lý ảnh: resize, normalize, noise removal, contrast, augmentation

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import random

# ============================================================
# HÀM KIỂM TRA ẢNH MỜ (thay thế utils.check_blurry)
# ============================================================
def check_blurry(image, threshold=50):
    """Kiểm tra ảnh có bị mờ không bằng Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = blur_value < threshold
    return is_blurry, blur_value


# ============================================================
# CLASS TIỀN XỬ LÝ ẢNH
# ============================================================
class ImagePreprocessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    def remove_noise(self, image):
        """Loại bỏ nhiễu bằng Non-local Means Denoising"""
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised

    def resize_with_padding(self, image):
        """Resize ảnh giữ tỷ lệ và thêm padding đen"""
        h, w = image.shape[:2]
        scale = self.target_size[0] / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        square_img = np.zeros(
            (self.target_size[0], self.target_size[1], 3), dtype=np.uint8
        )
        y_offset = (self.target_size[0] - new_h) // 2
        x_offset = (self.target_size[1] - new_w) // 2
        square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return square_img, (x_offset, y_offset, scale)

    def enhance_contrast(self, image):
        """Cải thiện contrast bằng CLAHE trên kênh L của LAB"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def gamma_correction(self, image, gamma=1.5):
        """Điều chỉnh độ sáng/tối bằng Gamma Correction"""
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype("uint8")
        return cv2.LUT(image, table)

    def normalize(self, image):
        """Chuẩn hóa pixel về [0, 1] — trả về float32"""
        return image.astype(np.float32) / 255.0

    def preprocess(self, image, apply_gamma=False):
        """Pipeline tiền xử lý hoàn chỉnh cho 1 ảnh"""
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

        # 5. Resize về target_size
        img, padding_info = self.resize_with_padding(img)

        return img, blur_value

    def process_directory(self, input_dir, output_dir, min_blur_threshold=50):
        """Xử lý toàn bộ thư mục ảnh (bao gồm thư mục con)"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']:
            image_files.extend(input_path.glob(ext))
            image_files.extend(input_path.rglob(ext))
        image_files = list(set(image_files))

        print(f"📁 Tìm thấy {len(image_files)} ảnh trong {input_dir}")
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        if subdirs:
            print(f"📂 Thư mục con: {', '.join([d.name for d in subdirs])}")

        stats = {'processed': 0, 'blurry': 0, 'failed': 0}

        for img_file in tqdm(image_files, desc="Đang xử lý ảnh"):
            try:
                image = cv2.imread(str(img_file))
                if image is None:
                    stats['failed'] += 1
                    continue

                processed_img, blur_value = self.preprocess(image)

                if processed_img is None or blur_value < min_blur_threshold:
                    stats['blurry'] += 1
                    continue

                parent_name = img_file.parent.name
                if parent_name == input_path.name:
                    output_filename = img_file.name
                else:
                    output_filename = f"{parent_name}_{img_file.name}"

                cv2.imwrite(str(output_path / output_filename), processed_img)
                stats['processed'] += 1

            except Exception as e:
                stats['failed'] += 1
                print(f"\n❌ Lỗi {img_file.name}: {e}")

        print("\n" + "="*50)
        print("THỐNG KÊ XỬ LÝ:")
        print(f"✅ Thành công : {stats['processed']}")
        print(f"⚠️  Ảnh mờ    : {stats['blurry']}")
        print(f"❌ Lỗi        : {stats['failed']}")
        print(f"📁 Lưu tại   : {output_dir}")
        print("="*50)
        return stats


# ============================================================
# CLASS AUGMENTATION — Main 4
# ============================================================
class DataAugmentor:
    """
    Tăng cường dữ liệu bằng các phép biến đổi:
    - HorizontalFlip  : lật ngang
    - Rotate          : xoay ±15 độ
    - Brightness      : thay đổi độ sáng
    - Contrast        : thay đổi độ tương phản
    """

    def augment(self, image):
        """Áp dụng augmentation ngẫu nhiên cho 1 ảnh (input: BGR uint8)"""
        img = image.copy()

        # 1. Lật ngang (50%)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        # 2. Xoay ngẫu nhiên ±15 độ (50%)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT)

        # 3. Thay đổi độ sáng (50%)
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # 4. Thay đổi contrast (40%)
        if random.random() < 0.4:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.randint(-20, 20)     # brightness offset
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return img

    def augment_directory(self, input_dir, output_dir, aug_per_image=2):
        """
        Augment toàn bộ thư mục:
        - Mỗi ảnh gốc tạo thêm aug_per_image ảnh mới
        - Lưu vào output_dir
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_files.extend(input_path.glob(ext))
        image_files = list(set(image_files))

        print(f"📁 Augmenting {len(image_files)} ảnh × {aug_per_image} = "
              f"{len(image_files) * aug_per_image} ảnh mới")

        count = 0
        for img_file in tqdm(image_files, desc="Augmenting"):
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            # Giữ lại ảnh gốc
            cv2.imwrite(str(output_path / img_file.name), image)

            # Tạo thêm aug_per_image ảnh biến đổi
            for i in range(aug_per_image):
                aug_img = self.augment(image)
                aug_name = f"{img_file.stem}_aug{i}{img_file.suffix}"
                cv2.imwrite(str(output_path / aug_name), aug_img)
                count += 1

        print(f"✅ Đã tạo {count} ảnh augmented → {output_dir}")
        return count


# ============================================================
# MAIN — chạy từ command line
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Tiền xử lý + Augmentation ảnh face mask'
    )
    parser.add_argument('--input',  required=True, help='Thư mục ảnh đầu vào')
    parser.add_argument('--output', required=True, help='Thư mục ảnh đầu ra')
    parser.add_argument('--size',   type=int, default=640, help='Kích thước resize')
    parser.add_argument('--blur_threshold', type=float, default=50)
    parser.add_argument('--augment', action='store_true',
                        help='Bật augmentation sau khi preprocess')
    parser.add_argument('--aug_per_image', type=int, default=2,
                        help='Số ảnh aug tạo ra từ mỗi ảnh gốc')
    args = parser.parse_args()

    # Bước 1: Preprocess
    preprocessor = ImagePreprocessor(target_size=(args.size, args.size))
    preprocessor.process_directory(
        args.input, args.output,
        min_blur_threshold=args.blur_threshold
    )

    # Bước 2: Augmentation (nếu bật)
    if args.augment:
        aug_output = args.output + "_aug"
        augmentor = DataAugmentor()
        augmentor.augment_directory(args.output, aug_output, args.aug_per_image)


if __name__ == "__main__":
    main()