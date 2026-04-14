from src.visualization.mask_detector_opencv import MaskDetectorOpenCV
import cv2

detector = MaskDetectorOpenCV()      # load model + DNN

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] Face Mask Detection - Evaluation Mode (OpenCV DNN)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    detections = detector.detect(frame)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 255, 0) if det["label"] == "MASK" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = f"{det['label']} {det['confidence']*100:.1f}%"
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detection - Evaluation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
