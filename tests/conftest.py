import numpy as np
import cv2
import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(0)

@pytest.fixture
def small_gray(rng):
    # 256x256 синтетика з легким шумом
    img = np.zeros((256, 256), np.uint8)
    cv2.circle(img, (64, 64), 18, 180, -1)
    cv2.circle(img, (128, 128), 12, 210, -1)
    cv2.circle(img, (192, 60), 10, 200, -1)
    noise = (rng.normal(0, 5, img.shape)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img
