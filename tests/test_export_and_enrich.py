import numpy as np
import cv2
from sem_psd.addons import extended_metrics, enrich_results, ParticleRow, write_csv_extended

def test_extended_metrics_um_domain():
    img = np.zeros((100,100), np.uint8)
    cnt = cv2.ellipse2Poly((50,50), (15,10), 0, 0, 360, 6).reshape(-1,1,2)
    m = extended_metrics(cnt, umx=0.1, umy=0.2, tilt_deg=30.0)
    assert m["area_um2"] > 0 and m["perimeter_um"] > 0
    assert 0.0 < m["roundness"] <= 1.0
    assert 0.0 < m["solidity"] <= 1.0

def test_enrich_and_csv(tmp_path):
    cnt = np.array([[[10,10]],[[20,10]],[[20,20]],[[10,20]]], dtype=np.int32)
    results = [(cnt, 2.0, 0.8)]
    rows = enrich_results(results, umx=0.1, umy=0.1, tilt_deg=0.0)
    assert isinstance(rows, list) and isinstance(rows[0], ParticleRow)
    csvp = tmp_path / "ext.csv"
    write_csv_extended(str(csvp), rows, unit="µm")
    text = csvp.read_text(encoding="utf-8")
    assert "idx" in text and "circularity" in text and "aspect_ratio" in text
