import numpy as np
import cv2
from sem_psd.core import measure_components

def test_measure_components_basic():
    lev = np.zeros((128,128), np.uint8) + 150
    obj = np.zeros_like(lev)
    cv2.circle(obj, (64,64), 12, 255, -1)
    res = measure_components(
        obj, min_d_um=1.0, max_d_um=100.0, min_circ=0.1,
        um_per_px=0.1, lev_img=lev, min_rel_contrast=-1.0
    )
    assert len(res) == 1
    cnt, d_um, circ = res[0]
    assert d_um > 0 and 0.0 < circ <= 1.0

def test_measure_components_contrast_filter():
    lev = np.zeros((128,128), np.uint8) + 100
    obj = np.zeros_like(lev)
    cv2.circle(obj, (64,64), 12, 255, -1)
    # зробимо “об’єкт темніший за фон” — rel_contrast буде негативним
    lev[obj==255] = 80
    res = measure_components(
        obj, min_d_um=1.0, max_d_um=100.0, min_circ=0.1,
        um_per_px=0.1, lev_img=lev, min_rel_contrast=0.05
    )
    assert len(res) == 0  # відсіяно контрастним порогом
