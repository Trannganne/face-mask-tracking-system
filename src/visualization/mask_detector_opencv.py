import torch
import cv2
import time
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
import soundfile as sf

# Lưu ý: Cần có class MaskDetectionCNN trong cùng file 
# hoặc import từ file model 

def process_video_pipeline(model_path, video_input, video_output='output.mp4'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Khởi tạo và Load Model đã train
    # Giả sử MaskDetectionCNN đã được định nghĩa ở trên
    model = MaskDetectionCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Khởi tạo Face Detector (MTCNN) và Tracker
    mtcnn = MTCNN(keep_all=True, device=device)
    
    # Giả định lớp PersonTracker đã được code bởi thành viên khác
    from your_team_module import PersonTracker 
    tracker = PersonTracker()
    
    # 3. Cấu hình Video I/O
    cap = cv2.VideoCapture(video_input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    # 4. Transform chuẩn hóa ảnh đầu vào cho model
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']
    colors = [(0, 165, 255), (0, 255, 0), (0, 0, 255)] # Cam, Xanh, Đỏ

    start_time = time.time()
    frame_count = 0

    print("🎬 Đang xử lý video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = time.time() - start_time
        frame_count += 1

        # BƯỚC A: Tìm khuôn mặt
        boxes, probs = mtcnn.detect(img_rgb)
        detections = []

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.9: continue
                
                x1, y1, x2, y2 = map(int, box)
                face = img_rgb[max(0, y1):y2, max(0, x1):x2]
                if face.size == 0: continue

                # BƯỚC B: Dự đoán bằng CNN của bạn
                face_pil = Image.fromarray(face)
                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(face_tensor)
                    class_id = torch.argmax(output).item()
                    conf = torch.softmax(output, dim=1)[0][class_id].item()

                detections.append(([x1, y1, x2, y2], conf, class_id))

        # BƯỚC C: Cập nhật Tracking (Hàm này do người khác viết)
        tracked_objects = tracker.update(detections, current_time)

        # BƯỚC D: Vẽ kết quả lên Frame
        for obj in tracked_objects:
            track_id = obj['track_id']
            bx1, by1, bx2, by2 = map(int, obj['bbox'])
            cid = obj['class_id']
            no_mask_t = obj['mask_time']
            warn = obj['warning']

            color = colors[cid]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {class_names[cid]}", (bx1, by1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if cid == 2: # Không đeo khẩu trang
                cv2.putText(frame, f"Time: {no_mask_t:.1f}s", (bx1, by2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if warn:
                cv2.putText(frame, "WARNING!", (bx1, by2+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(frame)
        if frame_count % 30 == 0:
            print(f"  Đã xử lý {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Hoàn thành! Video lưu tại: {video_output}")