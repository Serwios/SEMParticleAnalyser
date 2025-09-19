# SEM Particle Analyser (NanoScope PSD)

**SEM Particle Analyser (NanoScope PSD)** is a desktop application for **particle size distribution (PSD) analysis** of scanning electron microscopy (SEM) images.  
It provides a reproducible, interactive, and user-friendly workflow for extracting particle statistics from high-resolution microscopy data.

---

## ✨ Key Features

- **Wide image support**  
  - 8-bit and 16-bit grayscale SEM images.  
  - Formats: TIFF, PNG, JPEG, BMP.  

- **Automatic scale detection**  
  - Reads µm/pixel calibration from TIFF metadata.  
  - Manual scale override available.  

- **Flexible preprocessing**  
  - Region of interest (ROI) cropping (exclude top, bottom, left, right).  
  - Background leveling & CLAHE contrast enhancement.  
  - Top-hat filtering to highlight small bright particles.  

- **Adaptive or Otsu thresholding**  
  - Fully configurable block size and offset `C`.  

- **Morphological operations**  
  - Opening / closing in real-world units (µm).  
  - Automatic hole filling.  

- **Watershed segmentation**  
  - Split touching particles by adjustable neck size & minimum segment size.  

- **Interactive GUI (PySide6)**  
  - Preview tabs: **Original, Leveled, Threshold, Overlay, Histogram, Cumulative PSD**.  
  - Particle statistics: Count, D10, D50, D90, mean, std, min, max.  
  - Table of particle diameters.  

- **Interactive exclusions**  
  - Toggle *Remove blobs mode* → click on a contour to exclude (red).  
  - Click again to restore.  
  - Excluded particles are grey/italic in the table.  
  - Statistics, plots, and CSV export consider only active (green) particles.  

- **Export options**  
  - CSV of particle diameters.  
  - Overlay images with contours (PNG/JPEG).  

---

## 📦 Requirements

- Python ≥ 3.9  
- PySide6  
- OpenCV (cv2)  
- Pillow  
- NumPy  
- Matplotlib  

Install dependencies:

```bash
pip install PySide6 opencv-python pillow numpy matplotlib
