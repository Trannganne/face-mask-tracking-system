import cv2
import numpy as np
from tensorflow.keras.models import load_model
import urllib.request
from pathlib import Path

class MaskDetectorOpenCV:
    def __init__(self, 
                 model_path: str = "../../models/mask_detector.h5",
                 face_conf: float = 0.5,
                 mask_conf: float = 0.5,
                 frame_resize: int = 600):
        
        self.face_conf = face_conf
        self.mask_conf = mask_conf
        self.frame_resize = frame_resize

        print("[INFO] Đang load mask model...")
        self.maskNet = load_model(model_path)

        # DNN Face Detector
        self.face_detector_dir = Path("src/visualization/face_detector")
        self.prototxt = self.face_detector_dir / "deploy.prototxt"
        self.weights = self.face_detector_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        self._download_if_needed()
        self.faceNet = cv2.dnn.readNetFromCaffe(str(self.prototxt), str(self.weights))

    def _download_if_needed(self):
        self.face_detector_dir.mkdir(parents=True, exist_ok=True)
        if not self.prototxt.exists() or not self.weights.exists():
            print("[INFO] Tự động tải DNN Face Detector...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                str(self.prototxt))
            urllib.request.urlretrieve(
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                str(self.weights))
            print("[INFO] Tải DNN Face Detector xong!")

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        scale = self.frame_resize / max(h, w)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

        blob = cv2.dnn.blobFromImage(resized, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        results = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > self.face_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                face_crop = cv2.resize(face_crop, (224, 224))
                face_crop = face_crop.astype("float32") / 255.0
                face_crop = np.expand_dims(face_crop, axis=0)

                (mask_prob, no_mask_prob) = self.maskNet.predict(face_crop, verbose=0)[0]

                label = "MASK" if mask_prob > self.mask_conf else "NO MASK"
                conf_mask = mask_prob if label == "MASK" else no_mask_prob

                results.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "confidence": float(conf_mask),
                    "mask_prob": float(mask_prob)
                })
        return results