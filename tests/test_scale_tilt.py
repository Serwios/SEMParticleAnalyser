import pytest
from sem_psd.addons import resolve_scale_xy, effective_um_per_px_for_isotropic_kernels, tilt_corr_factor

def test_resolve_scale_xy_variants():
    assert resolve_scale_xy(0.2) == (0.2, 0.2)
    assert resolve_scale_xy(None, 0.1, 0.2) == (0.1, 0.2)
    with pytest.raises(ValueError):
        resolve_scale_xy(None, None, None)

def test_effective_um_per_px_and_tilt():
    assert effective_um_per_px_for_isotropic_kernels(0.1, 0.4) == pytest.approx(0.2, rel=1e-6)
    assert tilt_corr_factor(0.0) == pytest.approx(1.0)
    assert tilt_corr_factor(60.0) == pytest.approx(2.0, rel=1e-6)
