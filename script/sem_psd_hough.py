# sem_psd_hough.py — Spherical PSD via HoughCircles + greedy selection (no keep_big/keep_small)
# Scale priority:
#   1) TIFF metadata: PixelWidth (m/px) or HFW/ResolutionX -> µm/px
#   2) --scale_text "500um" | "1mm" | "20nm"  + auto bar length (px)
#   3) --scale µm/px manual override

import argparse
from pathlib import Path
import re, math, csv
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ---------- I/O ----------
def imread_gray(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img)

# ---------- Metadata dump ----------
def dump_tiff_metadata_text(image_path: str) -> str:
    try:
        pil = Image.open(image_path)
    except Exception as e:
        return f"[ERROR opening TIFF: {e}]"
    texts = []
    try:
        for tag, val in getattr(pil, "tag_v2", {}).items():
            if isinstance(val, bytes):
                s = val.decode(errors="ignore")
            elif isinstance(val, (list, tuple)):
                s = " ".join([v.decode(errors="ignore") if isinstance(v, bytes) else str(v) for v in val])
            else:
                s = str(val)
            texts.append(f"[{tag}] {s}")
    except Exception:
        pass
    try:
        for k, v in (pil.info or {}).items():
            if isinstance(v, bytes):
                v = v.decode(errors="ignore")
            texts.append(f"[{k}] {v}")
    except Exception:
        pass
    return "\n".join(texts)

# ---------- Metadata scale ----------
def parse_um_per_px_from_text(txt: str) -> float | None:
    if not txt:
        return None
    m = re.search(r"PixelWidth\s*=\s*([0-9eE\.\-\+]+)", txt)
    if m:
        try:
            px_m = float(m.group(1))
            if px_m > 0:
                return px_m * 1e6  # µm/px
        except Exception:
            pass
    m_hfw = re.search(r"(HorFieldsize|HFW)\s*=\s*([0-9eE\.\-\+]+)", txt)
    m_rx  = re.search(r"(ResolutionX|Resolutionx)\s*=\s*([0-9]+)", txt)
    if m_hfw and m_rx:
        try:
            hfw_m = float(m_hfw.group(2))
            resx  = int(m_rx.group(2))
            if hfw_m > 0 and resx > 0:
                return (hfw_m * 1e6) / float(resx)  # µm/px
        except Exception:
            pass
    return None

def scale_from_metadata(image_path: str) -> float | None:
    return parse_um_per_px_from_text(dump_tiff_metadata_text(image_path) or "")

# ---------- Scale bar autodetect (for --scale_text) ----------
def detect_scale_bar(gray: np.ndarray, band_ratio: float = 0.18):
    h, w = gray.shape
    band_h = max(40, int(h * band_ratio))
    roi = gray[h - band_h : h, :]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(roi)
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 35, -5)
    lines = cv2.HoughLinesP(thr, 1, np.pi/180, threshold=120,
                            minLineLength=int(0.2*w), maxLineGap=10)
    best = None; best_len = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx < 30 or dy > 8:  # nearly horizontal
                continue
            L = math.hypot(dx, dy)
            if L > best_len:
                best_len, best = L, (x1, y1, x2, y2)
    if best is None:
        return None, roi
    x1, y1, x2, y2 = best
    ymid = (y1 + y2) // 2
    y0 = max(0, int(ymid) - 15)
    y1b = min(roi.shape[0] - 1, int(ymid) + 15)
    band = thr[y0:y1b, :]
    band = cv2.dilate(band, np.ones((3,3), np.uint8), 1)
    cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        length_px = int(math.hypot(x2 - x1, y2 - y1))
        return length_px, roi
    c = max(cnts, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(c)
    return ww, roi

def parse_scale_text_to_um(scale_text: str) -> float:
    s = scale_text.strip().lower().replace(" ", "")
    m = re.match(r"([0-9\.]+)(nm|um|µm|mm)$", s)
    if not m:
        raise ValueError("Bad --scale_text. Examples: 500um, 1mm, 20nm")
    val = float(m.group(1)); unit = m.group(2)
    if unit in ("um", "µm"): return val
    if unit == "mm": return val * 1000.0
    if unit == "nm": return val / 1000.0
    raise ValueError("Unsupported unit in --scale_text")

# ---------- ROI ----------
def make_roi_mask(shape, top=0.02, bottom=0.22, left=0.0, right=0.0):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    t = int(h*top); b = int(h*(1.0-bottom))
    l = int(w*left); r = int(w*(1.0-right))
    mask[t:b, l:r] = 255
    return mask

# ---------- Preprocess ----------
def preprocess_for_edges(gray: np.ndarray) -> np.ndarray:
    bg = cv2.GaussianBlur(gray, (0,0), 25)             # rolling-ball
    leveled = cv2.addWeighted(gray, 1.5, bg, -0.5, 0)
    leveled = cv2.medianBlur(leveled, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(leveled)

# ---------- Hough + scoring ----------
def hough_candidates(gray_for_edges: np.ndarray,
                     roi_mask: np.ndarray,
                     rmin_px: float, rmax_px: float,
                     minDist: int, param1=110, param2=25):
    g = cv2.bitwise_and(gray_for_edges, gray_for_edges, mask=roi_mask)
    blur = cv2.GaussianBlur(g, (9,9), 2)
    cir = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                           minDist=minDist, param1=param1, param2=param2,
                           minRadius=int(rmin_px), maxRadius=int(rmax_px))
    return [] if cir is None else np.round(cir[0]).astype(int).tolist()

def ring_score(x:int,y:int,r:int, edges:np.ndarray, gray:np.ndarray) -> float:
    H,W = edges.shape
    if x-r<0 or y-r<0 or x+r>=W or y+r>=H:
        return 0.0
    ring = np.zeros_like(edges, np.uint8)
    cv2.circle(ring,(x,y), r, 255, 2)
    e_vals = edges[ring==255]
    g_vals = gray[ring==255]
    if e_vals.size==0 or g_vals.size==0: return 0.0
    return float(e_vals.mean()) + float(g_vals.std())

# ---------- Greedy selection (no keep_big/keep_small) ----------
def greedy_select(cands, edges, lev,
                  min_score=35.0,
                  exclude_frac=0.85,   # радіус зони, яку "викошуємо" після прийняття кола
                  max_circles=500):
    """
    1) score = mean(Canny_on_ring) + std(gray_on_ring)
    2) сортуємо за score↓
    3) приймаємо коло, якщо score >= min_score і його центр НЕ лежить
       у вже зайнятій зоні; після прийняття зафарбовуємо заповнений диск
       радіуса exclude_frac*r у масці окупації.
    """
    if not cands:
        return []
    # порахувати бали
    scored = []
    for x,y,r in cands:
        s = ring_score(x,y,r, edges, lev)
        if s >= min_score:  # миттєвий відсів дуже слабких
            scored.append((x,y,r,s))
    if not scored:
        return []

    scored.sort(key=lambda t: t[3], reverse=True)

    H,W = edges.shape
    occ = np.zeros((H,W), np.uint8)  # маска-окупація
    selected = []
    for x,y,r,s in scored:
        if occ[y, x]:   # центр вже "зайнятий" іншим прийнятим колом
            continue
        selected.append((x,y,r))
        # «викошуємо» зону навколо прийнятого кола
        rr = max(3, int(exclude_frac * r))
        cv2.circle(occ, (x,y), rr, 255, thickness=-1)
        if len(selected) >= max_circles:
            break
    return selected


# ---------- PSD ----------
def stats_from_diams(d_um: np.ndarray) -> dict:
    return {
        "particles": int(d_um.size),
        "D10": float(np.percentile(d_um, 10)) if d_um.size else 0.0,
        "D50": float(np.percentile(d_um, 50)) if d_um.size else 0.0,
        "D90": float(np.percentile(d_um, 90)) if d_um.size else 0.0,
        "mean": float(np.mean(d_um)) if d_um.size else 0.0,
        "std": float(np.std(d_um)) if d_um.size else 0.0,
        "min": float(np.min(d_um)) if d_um.size else 0.0,
        "max": float(np.max(d_um)) if d_um.size else 0.0,
    }

def plot_psd(d_um, st, out_hist: Path, out_cum: Path, title: str):
    if not d_um.size: return
    plt.figure(figsize=(8,5))
    plt.hist(d_um, bins=40, edgecolor="black", alpha=0.7)
    for n,c in [("D10","green"),("D50","red"),("D90","purple")]:
        plt.axvline(st[n], color=c, linestyle="--", label=f"{n}={st[n]:.2f} µm")
    plt.xlabel("Particle diameter (µm)"); plt.ylabel("Count"); plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(out_hist, dpi=180); plt.close()

    plt.figure(figsize=(8,5))
    s = np.sort(d_um); cum = np.arange(1, s.size+1)/s.size*100.0
    plt.plot(s, cum)
    for n,c in [("D10","green"),("D50","red"),("D90","purple")]:
        plt.axvline(st[n], color=c, linestyle="--", label=f"{n}={st[n]:.2f} µm")
    plt.xlabel("Particle diameter (µm)"); plt.ylabel("Cumulative %"); plt.title("Cumulative PSD")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(out_cum, dpi=180); plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Spherical PSD via Hough + greedy selection")
    ap.add_argument("image")

    # scale
    ap.add_argument("--scale", type=float, default=None, help="µm/px manual override")
    ap.add_argument("--scale_text", type=str, default=None,
                    help='Scale bar text (e.g., "500um", "1mm", "20nm"); bar length (px) autodetected')
    ap.add_argument("--save_scale_debug", type=str, default=None)

    # metadata dump
    ap.add_argument("--meta_debug", action="store_true", help="Print all textual TIFF metadata and exit")
    ap.add_argument("--meta_dump_to", type=str, default=None, help="Save textual TIFF metadata to file and continue")

    # ROI / Hough / greedy selection params
    ap.add_argument("--exclude_bottom_ratio", type=float, default=0.22)
    ap.add_argument("--exclude_top_ratio", type=float, default=0.02)
    ap.add_argument("--exclude_left_ratio", type=float, default=0.00)
    ap.add_argument("--exclude_right_ratio", type=float, default=0.00)
    ap.add_argument("--rmin_um", type=float, default=None)
    ap.add_argument("--rmax_um", type=float, default=None)
    ap.add_argument("--min_edge_grad", type=float, default=10.0, help="pre-filter: ring Canny mean + gray std >= this")
    ap.add_argument("--min_score", type=float, default=25.0, help="stop when best candidate score falls below this")
    ap.add_argument("--nms_center_frac", type=float, default=0.6, help="NMS: min center distance as fraction of min radius")
    ap.add_argument("--nms_radius_tol", type=float, default=0.35, help="NMS: consider duplicate if radii differ < this (relative)")
    ap.add_argument("--exclude_frac", type=float, default=0.85, help="частка радіуса, яку забиваємо після прийняття кола (0.7..1.0)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max_circles", type=int, default=500)
    args = ap.parse_args()

    p = Path(args.image)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.meta_debug:
        print("===== TIFF METADATA DUMP =====")
        print(dump_tiff_metadata_text(str(p)))
        print("===== END ====="); return
    if args.meta_dump_to:
        Path(args.meta_dump_to).write_text(dump_tiff_metadata_text(str(p)), encoding="utf-8", errors="ignore")
        print(f"[meta] saved textual TIFF metadata to: {args.meta_dump_to}")

    gray = imread_gray(str(p))

    # --- SCALE: metadata -> scale_text -> manual ---
    um_per_px = scale_from_metadata(str(p))
    if um_per_px:
        print(f"[meta-scale] {um_per_px:.6f} µm/px from TIFF metadata")
    if um_per_px is None and args.scale_text:
        bar_um = parse_scale_text_to_um(args.scale_text)
        length_px, roi = detect_scale_bar(gray)
        if args.save_scale_debug:
            cv2.imwrite(args.save_scale_debug, cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR))
        if length_px is None:
            raise SystemExit("Scale bar not found; cannot compute scale from --scale_text.")
        um_per_px = bar_um / float(length_px)
        print(f"[bar-scale] {length_px:.0f} px = {bar_um:.3f} µm → {um_per_px:.6f} µm/px")
    if um_per_px is None and args.scale is not None:
        um_per_px = float(args.scale); print(f"[manual-scale] {um_per_px:.6f} µm/px")
    if um_per_px is None:
        raise SystemExit("No scale: use TIFF with PixelWidth/HFW metadata, or provide --scale_text / --scale.")

    # --- ROI & preprocess ---
    roi_mask = make_roi_mask(gray.shape, args.exclude_top_ratio, args.exclude_bottom_ratio,
                             args.exclude_left_ratio, args.exclude_right_ratio)
    lev = preprocess_for_edges(gray)
    edges = cv2.Canny(lev, 40, 120)

    # --- Radii (px) ---
    H,W = gray.shape
    if args.rmin_um is not None and args.rmax_um is not None:
        rmin_px = max(5.0, args.rmin_um / um_per_px)
        rmax_px = max(10.0, args.rmax_um / um_per_px)
    else:
        rmin_px = max(6.0, 0.02 * min(H, W))   # 2%..25% від меншої сторони
        rmax_px = max(12.0, 0.25 * min(H, W))

    # --- Hough ---
    cands = hough_candidates(lev, roi_mask, rmin_px, rmax_px, minDist=int(0.8*rmin_px))
    # швидкий відсів дуже слабких
    pre = []
    for x,y,r in cands:
        s = ring_score(x,y,r, edges, lev)
        if s >= args.min_edge_grad:
            pre.append((x,y,r))
    # --- Greedy selection until score falls under threshold ---
    selected = greedy_select(pre, edges, lev,
                             min_score=args.min_score,
                             exclude_frac=args.exclude_frac,
                             max_circles=args.max_circles)

    # --- Overlay & PSD ---
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    diams = []
    for x,y,r in selected:
        cv2.circle(overlay, (x,y), r, (0,255,0), 2)
        cv2.rectangle(overlay, (x-r,y-r), (x+r,y+r), (0,255,0), 1)
        diams.append(2.0 * r * um_per_px)
    diams = np.array(diams, float)

    overlay_path = out_dir / f"annotated_{p.stem}.png"
    cv2.imwrite(str(overlay_path), overlay)

    print("=== RESULTS ===")
    print(f"Image   : {p}")
    print(f"Overlay : {overlay_path}")
    print(f"Accepted circles: {len(selected)}")

    if diams.size:
        st = stats_from_diams(diams)
        for k in ["particles","D10","D50","D90","mean","std","min","max"]:
            v = st[k]; print(f"{k:>9}: {v:.3f}" if k!="particles" else f"{k:>9}: {v}")
        hist_path = out_dir / f"hist_{p.stem}.png"
        cum_path  = out_dir / f"cum_{p.stem}.png"
        plot_psd(diams, st, hist_path, cum_path, f"PSD: {p.name}")
        print(f"Hist    : {hist_path}")
        print(f"Cum     : {cum_path}")
        csv_path = out_dir / f"diameters_{p.stem}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["diameter_um"])
            for d in diams: w.writerow([f"{d:.6f}"])
        print(f"CSV     : {csv_path}")
    else:
        print("No circles accepted. Try lowering --min_score or --min_edge_grad, or set --rmin_um/--rmax_um.")
if __name__ == "__main__":
    main()
