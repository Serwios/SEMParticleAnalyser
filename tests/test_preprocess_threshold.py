import numpy as np
from sem_psd.core import preprocess, make_roi_mask, threshold_pair

def test_preprocess_changes_contrast(small_gray):
    lev = preprocess(small_gray, clahe_clip=2.0, tophat_um=0.0, um_per_px=0.1, level_strength=0.3)
    assert lev.shape == small_gray.shape
    # перевіримо, що гістограма відрізняється (хоч би дисперсія)
    assert lev.var() != small_gray.var()

def test_threshold_pair_otsu_and_adaptive(small_gray):
    roi = make_roi_mask(small_gray.shape, 0.0, 0.0, 0.0, 0.0)
    bwB, bwD = threshold_pair(small_gray, roi, method="otsu")
    assert bwB.shape == small_gray.shape and bwD.shape == small_gray.shape
    assert set(np.unique(bwB)) <= {0, 255}
    bwB2, _ = threshold_pair(small_gray, roi, method="adaptive", block_size=31, C=-10)
    assert bwB2.shape == small_gray.shape
