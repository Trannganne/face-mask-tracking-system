
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30)

def update_tracker(boxes, frame):
    """
    boxes: [[x1,y1,x2,y2,conf,class], ...]
    """
    tracks = tracker.update_tracks(boxes, frame=frame)

    results = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()

        results.append((int(l), int(t), int(w), int(h), track_id))

    return results