from collections import defaultdict
import numpy as np
import soundfile as sf
#import sounddevice as sd
from deep_sort_realtime.deepsort_tracker import DeepSort

import time
import threading
import os
import pygame
from deep_sort_realtime.deepsort_tracker import DeepSort

CLASS_NAMES = ["Đeo sai", "Đeo đúng", "Không đeo"]
ALERT_SECONDS = 10

_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
ALERT_SOUND = os.path.join(_ASSETS_DIR, "alert.wav")

# ── Khởi tạo pygame mixer 1 lần duy nhất ────────────────────────────────────
pygame.mixer.init()
_alert_lock    = threading.Lock()
_alert_playing = False


def _play_alert():
    global _alert_playing
    try:
        sound = pygame.mixer.Sound(ALERT_SOUND)
        with _alert_lock:
            _alert_playing = True
        sound.play()
        # Chờ cho đến khi phát xong
        while pygame.mixer.get_busy():
            time.sleep(0.05)
    except Exception as e:
        print(f"❌ Lỗi phát âm thanh: {e}")
    finally:
        with _alert_lock:
            _alert_playing = False

    #return results

def _stop_alert():
    """Dừng âm thanh ngay lập tức."""
    try:
        pygame.mixer.stop()
    except Exception:
        pass
    global _alert_playing
    with _alert_lock:
        _alert_playing = False


def _trigger_alert():
    global _alert_playing
    with _alert_lock:
        if _alert_playing:
            return  # Đang phát → bỏ qua
    threading.Thread(target=_play_alert, daemon=True).start()


class PersonTracker:
    def __init__(self, alert_seconds=ALERT_SECONDS, sound_path=None):
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.ALERT_SECONDS = alert_seconds
        self.states = {}
        if sound_path:
            global ALERT_SOUND
            ALERT_SOUND = sound_path

    def update(self, detections, frame_rgb, current_time=None):
        if current_time is None:
            current_time = time.time()

        ds_input = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            ds_input.append(([x1, y1, x2-x1, y2-y1], d["conf"], d["cls_id"]))

        tracks = self.tracker.update_tracks(ds_input, frame=frame_rgb)
        results = []

        # Kiểm tra xem có ai đang ở trạng thái cần cảnh báo không
        any_alert = False

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid    = track.track_id
            bbox   = [int(v) for v in track.to_ltrb()]
            
            cls_id = track.get_det_class()
            if cls_id is None:
                cls_id = 2

            if tid not in self.states:
                self.states[tid] = {
                    "cls_id":     cls_id,
                    "start_time": current_time,
                    "alerted":    False
                }
            st = self.states[tid]

            # Class đổi → reset hoàn toàn
            if st["cls_id"] != cls_id:
                st["cls_id"]     = cls_id
                st["start_time"] = current_time
                st["alerted"]    = False

            duration = current_time - st["start_time"]
            alert = (cls_id in [0, 1]) and (duration > self.ALERT_SECONDS)

            capture_now = False

            if alert:
                any_alert = True  # ← còn người cần cảnh báo

            if alert and not st["alerted"]:
                st["alerted"] = True
                capture_now = True

                print(f"⚠️  CẢNH BÁO! ID:{tid} — {CLASS_NAMES[cls_id]} — {duration:.1f}s")
                _trigger_alert()

            results.append({
                "track_id": tid,
                "bbox":     bbox,
                "cls_id":   cls_id,
                "duration": duration,
                "alert":    alert,
                "captured":capture_now
            })

        # Nếu không còn ai cần cảnh báo → dừng âm thanh ngay
        if not any_alert:
            _stop_alert()

        return results
