import numpy as np
from pathlib import Path

import pytest
from PIL import Image, TiffImagePlugin
from sem_psd.core import imread_gray, parse_um_per_px_from_text, scale_from_metadata

def test_imread_gray_uint16(tmp_path: Path):
    # створюємо 16-бітне TIFF
    p = tmp_path / "u16.tif"
    arr = (np.linspace(0, 65535, 256*256, dtype=np.uint16).reshape(256,256))
    Image.fromarray(arr).save(p)
    g = imread_gray(str(p))
    assert g.dtype == np.uint8
    assert g.shape == (256, 256)
    assert g.max() == 255

def test_parse_um_per_px_from_text_pixelwidth():
    txt = "SomeHeader PixelWidth = 1.25e-07 other"
    um = parse_um_per_px_from_text(txt)
    assert um == pytest.approx(0.125, rel=1e-6)  # 1.25e-7 m/px → 0.125 µm/px

def test_scale_from_metadata_hfw_resolution(tmp_path: Path):
    p = tmp_path / "meta.tif"
    img = Image.new("L", (200, 100), 0)

    # Записуємо текст прямо в TIFF-тег ImageDescription (270)
    tiffinfo = TiffImagePlugin.ImageFileDirectory_v2()
    tiffinfo[270] = "HFW = 0.0005\nResolutionX = 200"  # 0.5 мм та 200 пікселів по ширині

    img.save(p, tiffinfo=tiffinfo)

    um = scale_from_metadata(str(p))
    assert um == pytest.approx((0.0005 * 1e6) / 200, rel=1e-6)  # 2.5 µm/px