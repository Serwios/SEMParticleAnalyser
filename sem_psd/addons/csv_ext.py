"""
Extended CSV export utilities.

Writes particle metrics with extended geometry
and shape descriptors to a UTF-8 CSV file.
"""

from __future__ import annotations
from typing import List
import csv

from sem_psd.addons.enrich import ParticleRow


def write_csv_extended(path: str, particles: List[ParticleRow], unit: str = "µm") -> None:
    """Write a CSV file with extended particle metrics."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "idx",
            f"d_{unit}",
            "circularity",
            f"area_{unit}^2",
            f"perimeter_{unit}",
            f"d_eq_area_{unit}",
            f"d_eq_perim_{unit}",
            f"feret_min_{unit}",
            f"feret_max_{unit}",
            "aspect_ratio",
            f"ellipse_minor_{unit}",
            f"ellipse_major_{unit}",
            "roundness",
            "solidity",
        ])
        # Data rows
        for r in particles:
            writer.writerow([
                r.idx,
                r.d_um,
                r.circ,
                r.area_um2,
                r.perimeter_um,
                r.d_eq_area_um,
                r.d_eq_perim_um,
                r.feret_min_um,
                r.feret_max_um,
                r.aspect_ratio,
                r.ellipse_minor_um,
                r.ellipse_major_um,
                r.roundness,
                r.solidity,
            ])
