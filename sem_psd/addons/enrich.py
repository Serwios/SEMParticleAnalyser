"""
Result enrichment utilities.

Converts raw detection results into structured
particle records with extended geometric metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from sem_psd.addons.metrics import extended_metrics


@dataclass
class ParticleRow:
    """Structured record of a single particle with extended metrics."""
    idx: int
    d_um: float
    circ: float
    area_um2: float
    perimeter_um: float
    d_eq_area_um: float
    d_eq_perim_um: float
    feret_min_um: float
    feret_max_um: float
    aspect_ratio: float
    ellipse_minor_um: float
    ellipse_major_um: float
    roundness: float
    solidity: float


def enrich_results(
    results: List[Tuple[np.ndarray, float, float]],
    umx: float,
    umy: float,
    tilt_deg: float = 0.0,
) -> List[ParticleRow]:
    """Compute extended metrics for all detected particles."""
    rows: List[ParticleRow] = []
    for i, (cnt, d_um, circ) in enumerate(results):
        m = extended_metrics(cnt, umx, umy, tilt_deg)
        rows.append(ParticleRow(
            idx=i, d_um=float(d_um), circ=float(circ),
            area_um2=m["area_um2"], perimeter_um=m["perimeter_um"],
            d_eq_area_um=m["d_eq_area_um"], d_eq_perim_um=m["d_eq_perim_um"],
            feret_min_um=m["feret_min_um"], feret_max_um=m["feret_max_um"],
            aspect_ratio=m["aspect_ratio"],
            ellipse_minor_um=m["ellipse_minor_um"], ellipse_major_um=m["ellipse_major_um"],
            roundness=m["roundness"], solidity=m["solidity"],
        ))
    return rows
