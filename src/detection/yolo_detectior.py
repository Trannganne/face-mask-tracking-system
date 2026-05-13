import cv2
from ultralytics import YOLO

from tracking import update_tracker
from timer import update_timer
from alert import play_alert

# load model đã train
model = YOLO("runs/detect/train/weights/best.pt")

# mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    boxes = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        boxes.append([x1, y1, x2, y2, conf, cls])

    # tracking
    tracks = update_tracker(boxes, frame)

    for (x1, y1, x2, y2, track_id) in tracks:

        # tìm class tương ứng
        cls = None
        for b in boxes:
            if abs(b[0] - x1) < 20 and abs(b[1] - y1) < 20:
                cls = b[5]
                break

        if cls is None:
            continue

        # 1 = with_mask
        is_violation = (cls == 1)

        duration, alert = update_timer(track_id, is_violation)

        # vẽ box
        color = (0,255,0) if not is_violation else (0,0,255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        text = f"ID {track_id} - {int(duration)}s"
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if alert:
            play_alert()

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# from ultralytics import YOLO
# import time
# import pygame
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import os

# # init sound
# pygame.mixer.init()

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# sound_path = os.path.join(BASE_DIR, "..", "timer", "mixkit-sound-alert-in-hall-1006.wav")

# pygame.mixer.music.load(sound_path)

# # load model COCO
# model = YOLO("yolo11n.pt")

# # tracker
# tracker = DeepSort(max_age=30)

# time_dict = {}
# THRESHOLD = 10  # test nhanh 10s thôi

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)[0]

#     detections = []

#     for box in results.boxes:
#         cls = int(box.cls[0])

#         if cls != 0:  # chỉ lấy person
#             continue

#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])

#         detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))

#     tracks = tracker.update_tracks(detections, frame=frame)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         track_id = track.track_id
#         l, t, w, h = map(int, track.to_ltrb())

#         # 👉 giả lập: person = không đeo mask
#         is_violation = True

#         if track_id not in time_dict:
#             time_dict[track_id] = time.time()

#         duration = time.time() - time_dict[track_id]

#         # vẽ
#         color = (0,0,255)
#         cv2.rectangle(frame, (l,t), (l+w,t+h), color, 2)
#         cv2.putText(frame, f"ID {track_id} - {int(duration)}s",
#                     (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         if duration > THRESHOLD:
#             if not pygame.mixer.music.get_busy():
#                 pygame.mixer.music.play()

#     cv2.imshow("Demo Tracking + Timer", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
