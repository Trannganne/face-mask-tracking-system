import time

# lưu thời gian bắt đầu vi phạm
time_dict = {}

THRESHOLD = 20  # giây

def update_timer(person_id, is_violation):
    """
    person_id: ID từ tracker
    is_violation: True nếu không đeo khẩu trang
    """

    if is_violation:
        if person_id not in time_dict:
            time_dict[person_id] = time.time()
            return 0, False
        else:
            duration = time.time() - time_dict[person_id]
            return duration, duration > THRESHOLD
    else:
        # reset nếu đeo lại mask
        if person_id in time_dict:
            del time_dict[person_id]
        return 0, False