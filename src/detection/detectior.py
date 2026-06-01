import torch
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

CLASS_NAMES = ["Incorrect_Mask", "With_Mask", "Without_Mask"]
COLORS = {0: (0, 165, 255), 1: (0, 0, 255), 2: (0, 255, 0)}  # cam / xanh / đỏ

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class FaceDetector:
    def __init__(self, device):
        self.mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7])
        self.device = device

    def detect_and_classify(self, frame_bgr, mask_model):
        """
        Trả về list dict:
        [{'bbox': [x1,y1,x2,y2], 'cls_id': int, 'conf': float}, ...]
        """
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        img_pil   = Image.fromarray(frame_rgb)
        boxes, probs = self.mtcnn.detect(img_pil)
        results = []
        if boxes is None:
            return results

        h, w = frame_bgr.shape[:2]
        for box, prob in zip(boxes, probs):
            if prob < 0.85:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = img_pil.crop((x1, y1, x2, y2))
            inp = transform(face_crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = mask_model(inp)
                probs_all  = torch.softmax(out, 1)[0]   
                cls_id = out.argmax(1).item()
                conf   = torch.softmax(out, 1).max().item()
                # Xác suất từng class
                prob_dict  = {
                    CLASS_NAMES[i]: round(probs_all[i].item() * 100, 1)
                    for i in range(len(CLASS_NAMES))
                }

            results.append({
                "bbox":   [x1, y1, x2, y2],
                "cls_id": cls_id,
                "conf":   conf,
                "prob_dict": prob_dict
            })
        return results