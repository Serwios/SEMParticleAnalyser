import numpy as np
import cv2
from sem_psd.core import morph_open, morph_close, fill_small_holes, split_touching_watershed

def test_morph_open_close():
    img = np.zeros((100,100), np.uint8)
    cv2.rectangle(img, (20,20), (80,80), 255, -1)
    noisy = img.copy()
    cv2.circle(noisy, (10,10), 2, 255, -1)  # сміття
    after_open = morph_open(noisy, open_um=0.5, um_per_px=0.1)
    assert after_open.sum() <= noisy.sum()
    after_close = morph_close(after_open, closing_um=0.5, um_per_px=0.1)
    assert after_close.sum() >= after_open.sum()
