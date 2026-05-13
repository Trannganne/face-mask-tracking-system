
# import cv2
# import os
# import glob

# def convert_yolo_to_classification(image_dir, label_dir, output_base_dir):
#     # Tạo các thư mục đầu ra
#     for i in range(3):
#         os.makedirs(os.path.join(output_base_dir, str(i)), exist_ok=True)

#     # Lấy danh sách tất cả file .jpg (không phân biệt hoa thường)
#     image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
#     print(f"🔍 Đang quét thư mục: {image_dir}")
#     print(f"📸 Tìm thấy tổng cộng {len(image_files)} file ảnh.")

#     count = 0
#     for img_path in image_files:
#         # Lấy tên file chuẩn (loại bỏ đuôi .jpg cuối cùng)
#         filename = os.path.splitext(os.path.basename(img_path))[0]
#         lbl_path = os.path.join(label_dir, f"{filename}.txt")

#         if not os.path.exists(lbl_path):
#             # Debug: Nếu không thấy file nhãn thì in ra để kiểm tra
#             # print(f"❌ Không tìm thấy nhãn cho: {filename}")
#             continue

#         img = cv2.imread(img_path)
#         if img is None: 
#             print(f"⚠️ Không thể đọc ảnh: {img_path}")
#             continue
            
#         h, w, _ = img.shape

#         with open(lbl_path, 'r') as f:
#             for i, line in enumerate(f.readlines()):
#                 data = line.split()
#                 if not data: continue
                
#                 try:
#                     cls = int(data[0])
#                     x_center, y_center, width, height = map(float, data[1:])

#                     x1 = int((x_center - width/2) * w)
#                     y1 = int((y_center - height/2) * h)
#                     x2 = int((x_center + width/2) * w)
#                     y2 = int((y_center + height/2) * h)

#                     x1, y1 = max(0, x1), max(0, y1)
#                     x2, y2 = min(w, x2), min(h, y2)

#                     face = img[y1:y2, x1:x2]
                    
#                     if face.size > 0:
#                         save_path = os.path.join(output_base_dir, str(cls), f"{filename}_face_{i}.jpg")
#                         cv2.imwrite(save_path, face)
#                         count += 1
#                 except Exception as e:
#                     print(f"🔥 Lỗi khi xử lý dòng trong file {lbl_path}: {e}")

#     print(f"\n✅ Hoàn thành! Đã cắt thành công {count} ảnh khuôn mặt.")

# # --- KIỂM TRA ĐƯỜNG DẪN CỦA BẠN TẠI ĐÂY ---
# # Hãy dùng dấu r'...' để tránh lỗi đường dẫn Windows


# # CÁCH DÙNG:
# # 1. Thực hiện cho tập Train
# #convert_yolo_to_classification('D:/deep_learning/DoAn/face-mask-tracking-system/dataset/images/train', 'D:/deep_learning/DoAn/face-mask-tracking-system/dataset/labels/train', 'D:/deep_learning/DoAn/Dataset_Mask/train')
# # 2. Thực hiện cho tập Val
# #convert_yolo_to_classification('D:/deep_learning/DoAn/face-mask-tracking-system/dataset/images/val', 'D:/deep_learning/DoAn/face-mask-tracking-system/dataset/labels/val', 'D:/deep_learning/DoAn/Dataset_Mask/val')
# # 3. Thực hiện cho tập Test
# convert_yolo_to_classification('D:/deep_learning/DoAn/train/images', 'D:/deep_learning/DoAn/train/labels', 'D:/deep_learning/DoAn/Dataset_Mask_Incorrect/test')

# Cắt riêng class chỉ định (không đeo mask) để test nhanh
import cv2
import os
import glob

def convert_yolo_to_class0(image_dir, label_dir, output_base_dir):
    # Chỉ tạo thư mục cho class 0
    output_dir = os.path.join(output_base_dir, "1")
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(f"🔍 Đang quét thư mục: {image_dir}")
    print(f"📸 Tìm thấy tổng cộng {len(image_files)} file ảnh.")

    count = 0

    for img_path in image_files:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(label_dir, f"{filename}.txt")

        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Không thể đọc ảnh: {img_path}")
            continue

        h, w, _ = img.shape

        with open(lbl_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                data = line.strip().split()
                if not data:
                    continue

                try:
                    cls = int(data[0])

                    # 👉 CHỈ lấy class 2
                    if cls != 1:
                        continue

                    x_center, y_center, width, height = map(float, data[1:])

                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)

                    # Clamp về biên ảnh
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # 👉 Tránh bbox lỗi
                    if x2 <= x1 or y2 <= y1:
                        continue

                    face = img[y1:y2, x1:x2]

                    if face.size > 0:
                        save_path = os.path.join(output_dir, f"{filename}_face_{i}.jpg")
                        cv2.imwrite(save_path, face)
                        count += 1

                except Exception as e:
                    print(f"🔥 Lỗi tại {lbl_path}: {e}")

    print(f"\n✅ Hoàn thành! Đã cắt {count} ảnh thuộc class 0.")


# ===== CÁCH DÙNG =====
convert_yolo_to_class0(
    r'D:\deep_learning\DoAn\locdulieu\test\images',
    r'D:\deep_learning\DoAn\locdulieu\test\labels',
    'D:/deep_learning/DoAn/Dataset_Mask_Incorrect/test'
)