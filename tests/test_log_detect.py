import numpy as np
import cv2
from sem_psd.core import detect_blobs_log, preprocess, make_roi_mask

def test_detect_blobs_log_simple():
    gray = np.zeros((200,200), np.uint8) + 30
    cv2.circle(gray, (60,60), 8, 200, -1)
    cv2.circle(gray, (140,60), 12, 210, -1)
    lev = preprocess(gray, 2.0, 0.0, 0.1, 0.3)
    roi = make_roi_mask(gray.shape, 0, 0, 0, 0)
    res = detect_blobs_log(
        gray, lev, um_per_px=0.1,
        dmin_um=1.0, dmax_um=10.0,
        threshold_rel=0.03, minsep_um=0.8,
        roi_mask=roi, min_rel_contrast=0.0
    )
    # очікуємо хоча б 1–2 об'єкти
    assert len(res) >= 1
    assert all(d > 0 for _, d, _ in res)
