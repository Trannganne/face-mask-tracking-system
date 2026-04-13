from collections import defaultdict
import numpy as np
import soundfile as sf
import sounddevice as sd
from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_iou_distance=0.7
        )

        # Lưu trạng thái từng người
        self.person_states = defaultdict(lambda: {
            'start_time': None,
            'total_mask_time': 0,
            'last_state': None,
            'warned': False
        })

        # Tạo âm thanh cảnh báo (440Hz - nốt La)
        self.alert_sound = self.generate_alert_sound()

    def generate_alert_sound(self, duration=1.0, frequency=440):
        """Tạo âm thanh cảnh báo"""
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)
        return wave
    
    def play_alert(self):
        """Phát âm thanh cảnh báo"""
        try:
            sf.write('alert.wav', self.alert_sound, 22050)       # Ghi file
            sd.play(self.alert_sound, 22050)                     # Phát ra loa
            sd.wait()                                            # Chờ phát xong mới tiếp tục
            print("🚨 CẢNH BÁO: Phát hiện người đeo khẩu trang quá 20 giây!")
        except Exception as e:
            print(f"🚨 CẢNH BÁO: Không thể phát âm thanh! Lỗi: {e}")

    def update(self, detections, current_time):
        """
        Cập nhật tracking và kiểm tra cảnh báo
        Args:
            detections: List of (bbox, confidence, class_id)
            current_time: Thời gian hiện tại
        Returns:
            List of tracked objects với cảnh báo
        """
        # Convert sang định dạng DeepSORT
        ds_detections = []
        for bbox, conf, class_id in detections:
            x1, y1, x2, y2 = bbox
            ds_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))

        # Cập nhật tracker
        tracks = self.tracker.update_tracks(ds_detections, frame=None)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()
            class_id = track.get_det_class()
            state = self.person_states[track_id]

            # Class: 0=đeo sai, 1=đeo đúng, 2=không đeo
            if class_id == 1:  # Đeo đúng → đếm giờ, cảnh báo nếu quá 20s
                if state['start_time'] is None:
                    state['start_time'] = current_time

                mask_duration = current_time - state['start_time']
                state['total_mask_time'] = mask_duration

                if mask_duration > 20 and not state['warned']:
                    self.play_alert()
                    state['warned'] = True

                warning = mask_duration > 20

            else:  # Đeo sai hoặc không đeo → reset bộ đếm
                state['start_time'] = None
                state['total_mask_time'] = 0
                state['warned'] = False
                mask_duration = 0
                warning = False

            state['last_state'] = class_id

            results.append({
                'track_id': track_id,
                'bbox': bbox,
                'class_id': class_id,
                'mask_time': mask_duration,
                'warning': warning
            })

        return results
