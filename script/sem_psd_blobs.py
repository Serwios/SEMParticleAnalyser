# sem_psd_blobs.py — PSD із нерегулярних «білих» частинок (blob-аналіз, без навчання)
# - Підтримка 8/16-bit TIFF (OpenCV/Pillow fallback)
# - Масштаб: --scale (µm/px) або з метаданих (PixelWidth/HFW/ResolutionX), якщо є
# - Автовибір полярності: робимо дві бінарки (bright/dark) і беремо ту, де адекватно багато компонентів у діапазоні розмірів
# - Фільтри: діаметр (µm), круглість (circularity), морфологічне змикання
# - Вивід: overlay з контурами, CSV діаметрів (еквівалентний діаметр по площі), гістограма й кумулятив, проміжні дебаг-стадії

import argparse, math, csv, re
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ========== I/O, filenames ==========
def safe_stem(st: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', st)

def imread_gray(path: str) -> np.ndarray:
    """Читає 8/16-бітне зображення та повертає uint8灰."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Pillow fallback
        pil = Image.open(path)
        if pil.mode not in ("L", "I;16", "I;16B", "I;16L"):
            pil = pil.convert("L")
        arr = np.array(pil)
        if arr.dtype == np.uint16:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr.astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

# ========== Метадані (опц.) ==========
def dump_tiff_metadata_text(image_path: str) -> str:
    try:
        pil = Image.open(image_path)
    except Exception as e:
        return f"[ERROR opening TIFF: {e}]"
    out = []
    try:
        for tag, val in getattr(pil, "tag_v2", {}).items():
            if isinstance(val, bytes):
                s = val.decode(errors="ignore")
            elif isinstance(val, (list, tuple)):
                s = " ".join([v.decode(errors="ignore") if isinstance(v, bytes) else str(v) for v in val])
            else:
                s = str(val)
            out.append(f"[{tag}] {s}")
    except Exception:
        pass
    try:
        for k, v in (pil.info or {}).items():
            if isinstance(v, bytes):
                v = v.decode(errors="ignore")
            out.append(f"[{k}] {v}")
    except Exception:
        pass
    return "\n".join(out)

def parse_um_per_px_from_text(txt: str) -> float | None:
    if not txt: return None
    m = re.search(r"PixelWidth\s*=\s*([0-9eE\.\-\+]+)", txt)
    if m:
        try:
            px_m = float(m.group(1))
            if px_m > 0: return px_m * 1e6
        except Exception:
            pass
    m_hfw = re.search(r"(HorFieldsize|HFW)\s*=\s*([0-9eE\.\-\+]+)", txt)
    m_rx  = re.search(r"(ResolutionX|Resolutionx)\s*=\s*([0-9]+)", txt)
    if m_hfw and m_rx:
        try:
            hfw_m = float(m_hfw.group(2))
            resx  = int(m_rx.group(2))
            if hfw_m > 0 and resx > 0:
                return (hfw_m * 1e6) / float(resx)
        except Exception:
            pass
    return None

def scale_from_metadata(image_path: str) -> float | None:
    return parse_um_per_px_from_text(dump_tiff_metadata_text(image_path))

# ========== ROI ==========
def make_roi_mask(shape, top=0.02, bottom=0.22, left=0.0, right=0.0):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    t = int(h*top); b = int(h*(1.0-bottom))
    l = int(w*left); r = int(w*(1.0-right))
    mask[max(0,t):max(0,b), max(0,l):max(0,r)] = 255
    return mask

# ========== Препроцес ==========
def preprocess(gray: np.ndarray, clahe_clip=2.0, tophat_um=0.0, um_per_px=0.05, level_strength=0.3) -> np.ndarray:
    lev = gray.copy()
    if level_strength and level_strength > 0:
        bg = cv2.GaussianBlur(gray, (0,0), 25)
        lev = cv2.addWeighted(gray, 1.0 + level_strength, bg, -level_strength, 0)
    if tophat_um and tophat_um > 0:
        rpx = max(1, int(tophat_um / um_per_px))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rpx+1, 2*rpx+1))
        lev = cv2.morphologyEx(lev, cv2.MORPH_TOPHAT, ker)
    lev = cv2.medianBlur(lev, 3)
    if clahe_clip and clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8,8))
        lev = clahe.apply(lev)
    return lev


# ========== Порогування (автополярність) ==========
def threshold_pair(lev, roi_mask, method="otsu", block_size=31, C=-10):
    if method.lower() == "adaptive":
        if block_size % 2 == 0: block_size += 1
        bwB = cv2.adaptiveThreshold(lev, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block_size, C)
    else:
        _, bwB = cv2.threshold(lev, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bwB = cv2.bitwise_and(bwB, bwB, mask=roi_mask)
    bwD = 255 - bwB
    return bwB, bwD

def morph_open(bw, open_um, um_per_px):
    if open_um <= 0: return bw
    r = max(1, int(open_um / um_per_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker, iterations=1)

def split_touching_watershed(bw, um_per_px, min_neck_um=0.12, min_seg_d_um=0.20):
    # bw: 255 = об’єкт
    obj = (bw > 0).astype(np.uint8)
    # distance transform (invert для максимумів усередині)
    dist = cv2.distanceTransform(obj, cv2.DIST_L2, 3)
    # придушити дуже тонкі «шийки»
    neck_px = max(1, int(min_neck_um / um_per_px))
    dist_smooth = cv2.GaussianBlur(dist, (0,0), sigmaX=0.5*neck_px)
    # піки як насінини
    # нормалізація для порогування локальних максимумів
    dn = cv2.normalize(dist_smooth, None, 0, 1.0, cv2.NORM_MINMAX)
    # насінини: точки, де dist вище за локальний медіан у вікні ~neck_px
    k = max(3, 2*neck_px+1)
    med = cv2.medianBlur((dn*255).astype(np.uint8), k)
    seeds = ((dn*255).astype(np.uint8) > (med + 5)).astype(np.uint8)
    # очищення насінин
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    # маркери
    num, markers = cv2.connectedComponents(seeds)
    if num < 2:
        return bw  # нічого різати
    # watershed потребує 3-канальний фон
    markers = markers.astype(np.int32)
    # маска заборони: за межі об’єкта виходити не можна
    mask3 = cv2.cvtColor((obj*255), cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask3, markers)
    # збираємо сегменти тільки всередині колишнього об’єкта
    seg = np.zeros_like(obj, np.uint8)
    for lab in range(1, markers.max()+1):
        m = (markers == lab) & (obj.astype(bool))
        if not m.any(): continue
        # мін. діаметр сегмента
        A = float(m.sum())
        d_px = 2.0 * math.sqrt(A/ math.pi)
        d_um = d_px * um_per_px
        if d_um >= min_seg_d_um:
            seg[m] = 255
    return seg

def count_reasonable_components(bw, um_per_px, dmin_um, dmax_um):
    num, lbl, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    def area_from_d_um(d_um):
        if d_um is None: return None
        r_px = max(1.0, (d_um/2.0) / um_per_px)
        return math.pi * r_px * r_px
    # трохи розширюємо межі (нехай не зрізає «майже граничні»)
    amin = area_from_d_um(dmin_um*0.6 if dmin_um else None)
    amax = area_from_d_um(dmax_um*1.5 if dmax_um else None)
    k = 0
    for i in range(1, num):
        a = st[i, cv2.CC_STAT_AREA]
        if amin is not None and a < amin:  continue
        if amax is not None and a > amax:  continue
        k += 1
    return k

# ========== Морфологія ==========
def morph_close(bw, closing_um, um_per_px):
    if closing_um <= 0:
        return bw
    rad_px = max(1, int(closing_um / um_per_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rad_px+1, 2*rad_px+1))
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker, iterations=1)

def fill_small_holes(mask: np.ndarray, max_frac: float) -> np.ndarray:
    # очікує двійкову маску: 255 = об’єкт, 0 = фон
    filled = mask.copy()
    contours, hier = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return filled
    hier = hier[0]
    for i,(cnt,h) in enumerate(zip(contours, hier)):
        # тільки зовнішні контури
        if h[3] != -1:
            continue
        area = cv2.contourArea(cnt)
        # пройтись по дочірніх (внутрішніх) контурах і залити «малі» отвори
        child = h[2]
        while child != -1:
            hole_cnt = contours[child]
            hole_area = cv2.contourArea(hole_cnt)
            if area > 0 and (hole_area/area) <= max_frac:
                cv2.drawContours(filled, [hole_cnt], -1, 255, thickness=-1)
            child = hier[child][0]   # next sibling
    return filled


# ========== Вимірювання компонентів ==========
def measure_components(bw, min_d_um, max_d_um, min_circ,
                       um_per_px, lev_img, min_rel_contrast):
    """
    Аналіз компонентів у бінарці bw.
    - Відкидає за розміром, круглістю, локальним контрастом.
    - lev_img: вирівняне зображення (градації сірого) для вимірювання контрасту.
    - min_rel_contrast: (mean_in - mean_bg)/255 має бути >= цього порогу.
    Повертає:
      results = [(contour, equiv_d_um, circ)],
      labels_color, keep_size_mask, keep_shape_mask
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    H, W = bw.shape
    # кольорове представлення для debug
    color = np.zeros((H, W, 3), np.uint8)
    rng = np.random.default_rng(123)
    for i in range(1, num):
        color[labels == i] = rng.integers(60, 255, size=(1,3), dtype=np.uint8)

    # пороги площі з діаметра
    def area_from_d_um(d_um):
        if d_um is None: return None
        r_px = max(0.5, (d_um/2.0) / um_per_px)
        return math.pi * r_px * r_px
    area_min = area_from_d_um(min_d_um) if min_d_um else None
    area_max = area_from_d_um(max_d_um) if max_d_um else None

    keep_size_mask = np.zeros_like(bw)
    keep_shape_mask = np.zeros_like(bw)

    results = []
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        if area_min is not None and a < area_min:  continue
        if area_max is not None and a > area_max:  continue
        keep_size_mask[labels == i] = 255

        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        roi = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:  continue
        cnt = max(cnts, key=cv2.contourArea)
        # відновлюємо координати контуру в системі зображення
        cnt[:,0,0] += x
        cnt[:,0,1] += y

        A = float(cv2.contourArea(cnt))
        if A < 1.0: continue
        P = float(cv2.arcLength(cnt, True))
        circ = (4.0*math.pi*A) / (P*P + 1e-9)
        if circ < min_circ:
            continue

        # === Локальний контраст ===
        mask_obj = np.zeros_like(bw)
        cv2.drawContours(mask_obj, [cnt], -1, 255, thickness=-1)

        ring = np.zeros_like(bw)
        d_px_est = 2.0 * math.sqrt(A / math.pi)
        ring_thick = max(3, int(0.15 * d_px_est))
        cv2.drawContours(ring, [cnt], -1, 255, thickness=ring_thick)
        ring = cv2.subtract(ring, mask_obj)

        vals_in = lev_img[mask_obj==255]
        vals_bg = lev_img[ring==255]
        if vals_bg.size < 30:
            cv2.drawContours(ring, [cnt], -1, 255, thickness=ring_thick*2)
            ring = cv2.subtract(ring, mask_obj)
            vals_bg = lev_img[ring==255]

        mean_in = float(vals_in.mean()) if vals_in.size else 0.0
        mean_bg = float(vals_bg.mean()) if vals_bg.size else 0.0
        rel_contrast = (mean_in - mean_bg) / 255.0

        if rel_contrast < min_rel_contrast:
            continue  # відкидаємо «напівсвітлі» плями

        # еквівалентний діаметр
        d_px = 2.0 * math.sqrt(A / math.pi)
        d_um = d_px * um_per_px

        keep_shape_mask[labels == i] = 255
        results.append((cnt, d_um, circ))

    return results, color, keep_size_mask, keep_shape_mask



# ========== PSD ==========
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

# ========== MAIN ==========
def main():
    ap = argparse.ArgumentParser(description="SEM PSD via blob analysis (автополярність, 16-bit TIFF, дебаг)")
    ap.add_argument("image")
    # масштаб
    ap.add_argument("--scale", type=float, default=None, help="µm/px. Якщо не задано, пробуємо зчитати з TIFF metadata.")
    # ROI
    ap.add_argument("--exclude_bottom_ratio", type=float, default=0.22)
    ap.add_argument("--exclude_top_ratio", type=float, default=0.02)
    ap.add_argument("--exclude_left_ratio", type=float, default=0.00)
    ap.add_argument("--exclude_right_ratio", type=float, default=0.00)
    # фільтри
    ap.add_argument("--min_d_um", type=float, default=0.01)
    ap.add_argument("--max_d_um", type=float, default=10.0)
    ap.add_argument("--closing_um", type=float, default=0.12)
    ap.add_argument("--min_circ", type=float, default=0.10)
    # поріг
    ap.add_argument("--thr", choices=["otsu","adaptive"], default="otsu")
    ap.add_argument("--block_size", type=int, default=31)
    ap.add_argument("--block_C", type=int, default=-10)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    # вивід
    ap.add_argument("--out", default="results")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--meta_debug", action="store_true", help="Роздрукувати всі текстові TIFF-метадані і вийти")

    ap.add_argument("--min_rel_contrast", type=float, default=0.15,
                    help="мін. відносний контраст до локального фону: (mean_in - mean_bg)/255 ≥ val")
    ap.add_argument("--tophat_um", type=float, default=0.0,
                    help="радіус SE для white-tophat (µm). 0 = вимкнено")
    ap.add_argument("--level_strength", type=float, default=0.3,
                    help="сила rolling-ball вирівнювання (0 = викл, 0.2..0.5 = помірно, >0.6 = агресивно)")
    ap.add_argument("--hole_fill_frac", type=float, default=0.6,
                    help="заповнювати внутрішні порожнини, якщо їхня площа ≤ frac * площі частинки")
    ap.add_argument("--open_um", type=float, default=0.08, help="радіус SE для MORPH_OPEN (µm); 0 = вимкнено")

    # >>> argparse (додай)
    ap.add_argument("--split_touching", action="store_true",
                    help="розщеплювати злиплі частинки watershed-ом")
    ap.add_argument("--min_neck_um", type=float, default=0.12,
                    help="мін. товщина перемички (умовна), дрібніші не ріжемо")
    ap.add_argument("--min_seg_d_um", type=float, default=0.20,
                    help="мін. еквівалентний діаметр сегмента після розщеплення")

    args = ap.parse_args()

    p = Path(args.image)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    sstem = safe_stem(p.stem)
    dbg_dir = out_dir / f"debug_{sstem}" if args.debug else None
    if dbg_dir: dbg_dir.mkdir(parents=True, exist_ok=True)

    if args.meta_debug:
        print("===== TIFF METADATA =====")
        print(dump_tiff_metadata_text(str(p)))
        print("===== END =====")
        return

    gray = imread_gray(str(p))
    if dbg_dir: cv2.imwrite(str(dbg_dir / "01_gray.png"), gray)

    # масштаб
    um_per_px = args.scale
    if um_per_px is None:
        um_per_px = scale_from_metadata(str(p))
        if um_per_px:
            print(f"[meta-scale] {um_per_px:.6f} µm/px from TIFF metadata")
    if um_per_px is None:
        raise SystemExit("No scale: provide --scale (µm/px) or TIFF with PixelWidth/HFW metadata.")

    # ROI, препроцес
    roi_mask = make_roi_mask(gray.shape, args.exclude_top_ratio, args.exclude_bottom_ratio,
                             args.exclude_left_ratio, args.exclude_right_ratio)
    if dbg_dir: cv2.imwrite(str(dbg_dir / "02_roi_mask.png"), roi_mask)

    lev = preprocess(gray, args.clahe_clip, args.tophat_um, um_per_px, args.level_strength)

    if dbg_dir: cv2.imwrite(str(dbg_dir / "03_leveled.png"), lev)

    # поріг із автополярністю (беремо варіант, де більше «розумних» компонентів у діапазоні розмірів)
    bwB, bwD = threshold_pair(lev, roi_mask, method=args.thr, block_size=args.block_size, C=args.block_C)
    kB = count_reasonable_components(bwB, um_per_px, args.min_d_um, args.max_d_um)
    kD = count_reasonable_components(bwD, um_per_px, args.min_d_um, args.max_d_um)
    bw = bwB if kB >= kD else bwD
    if dbg_dir:
        cv2.imwrite(str(dbg_dir / "04_thresh_bright.png"), bwB)
        cv2.imwrite(str(dbg_dir / "04_thresh_dark.png"),   bwD)
        cv2.imwrite(str(dbg_dir / "04_thresh.png"),        bw)

    # морфологічне змикання
    bw_m = morph_close(bw, args.closing_um, um_per_px)
    bw_m = morph_open(bw_m, args.open_um, um_per_px)
    bw_m = fill_small_holes(bw_m, args.hole_fill_frac)
    if args.split_touching:
        bw_m = split_touching_watershed(bw_m, um_per_px,
                                        min_neck_um=args.min_neck_um,
                                        min_seg_d_um=args.min_seg_d_um)
    if dbg_dir: cv2.imwrite(str(dbg_dir / "05b_split.png"), bw_m)

    # вимірювання
    results, labels_color, size_keep_mask, shape_keep_mask = measure_components(
        bw_m, args.min_d_um, args.max_d_um, args.min_circ, um_per_px,
        lev_img=lev, min_rel_contrast=args.min_rel_contrast
    )

    if dbg_dir:
        cv2.imwrite(str(dbg_dir / "06_labels_color.png"), labels_color)
        cv2.imwrite(str(dbg_dir / "07_after_size.png"),   size_keep_mask)
        cv2.imwrite(str(dbg_dir / "08_after_shape.png"),  shape_keep_mask)

    # оверлей і діаметри
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    diams = []
    for cnt, d_um, circ in results:
        diams.append(d_um)
        cv2.drawContours(overlay, [cnt], -1, (0,255,0), 2)
    overlay_path = out_dir / f"annotated_{sstem}.png"
    ok = cv2.imwrite(str(overlay_path), overlay)
    if not ok: print(f"[WARN] imwrite failed: {overlay_path}")

    print("=== RESULTS ===")
    print(f"Image   : {p}")
    print(f"Overlay : {overlay_path}")

    csv_path = out_dir / f"diameters_{sstem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["diameter_um"])
        for d in diams: w.writerow([f"{d:.6f}"])
    print(f"CSV     : {csv_path}")

    d = np.array(diams, float)
    if d.size:
        st = stats_from_diams(d)
        for k in ["particles","D10","D50","D90","mean","std","min","max"]:
            v = st[k]; print(f"{k:>9}: {v:.3f}" if k!="particles" else f"{k:>9}: {v}")
        hist_path = out_dir / f"hist_{sstem}.png"
        cum_path  = out_dir / f"cum_{sstem}.png"
        plot_psd(d, st, hist_path, cum_path, f"PSD: {p.name}")
        print(f"Hist    : {hist_path}")
        print(f"Cum     : {cum_path}")
    else:
        print("No particles accepted. Tune --min_d_um/--max_d_um/--min_circ/--closing_um or thresholding params.")

if __name__ == "__main__":
    main()
