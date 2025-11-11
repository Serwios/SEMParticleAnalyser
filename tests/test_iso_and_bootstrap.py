import numpy as np
from sem_psd.core import stats_from_diams
from sem_psd.addons import iso9276_weighted_means, bootstrap_ci_percentile, bootstrap_ci_mean

def test_stats_and_iso_means():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    st = stats_from_diams(d)
    assert st["particles"] == 4 and st["D50"] == 2.5
    iso = iso9276_weighted_means(d)
    assert iso["D32"] > 0 and iso["D43"] > iso["D32"]

def test_bootstrap_cis_seeded():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    lo1, hi1 = bootstrap_ci_percentile(x, q=50, n=2000, alpha=0.1, seed=123)
    lo2, hi2 = bootstrap_ci_percentile(x, q=50, n=2000, alpha=0.1, seed=123)
    assert (lo1, hi1) == (lo2, hi2)  # детермінізм при фіксованому seed
    lom, him = bootstrap_ci_mean(x, n=2000, alpha=0.1, seed=123)
    assert lom < np.mean(x) < him
