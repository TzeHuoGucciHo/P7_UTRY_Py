from __future__ import annotations
import os, json, sys
from typing import Dict, Optional, Tuple, List
import cv2
import numpy as np

from .segmenter import SilhouetteSegmenter
from .geometry import (
    ellipse_circumference, mask_bbox,
    width_at_row_centerclip, width_at_row,
    width_at_row_centerclip_window
)

# ---------- helpers ----------
def _save(path: str, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def _save_txt(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _centroid_x(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    return float(xs.mean()) if xs.size else mask.shape[1] / 2.0

def _save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _overlay(img_bgr, mask_u8, rows, widths_px, semiaxes_cm, circumf_cm, out_path, show_clip=False):
    vis = img_bgr.copy()
    tint = np.zeros_like(vis); tint[mask_u8 > 0] = (0,255,0)
    vis = cv2.addWeighted(vis, 1.0, tint, 0.35, 0)

    colors = {"chest": (0,128,255), "waist": (255,0,255), "hip": (255,128,0),
              "shoulder": (0,255,255), "thigh": (0,200,100), "crotch": (180,180,0)}
    h, w = mask_u8.shape[:2]
    if show_clip:
        cx = _centroid_x(mask_u8)
        keep_local = {"chest":0.72, "waist":0.74, "hip":0.83, "shoulder":0.95, "thigh":0.78}
        for k, y in rows.items():
            if k not in keep_local: continue
            row = int(max(0, min(h-1, y)))
            xs = np.where(mask_u8[row, :] > 0)[0]
            if xs.size < 2: continue
            left, right = xs[0], xs[-1]; half = (right - left)/2.0
            new_half = half * keep_local[k]
            x0 = int(max(0, cx - new_half)); x1 = int(min(w-1, cx + new_half))
            cv2.rectangle(vis, (x0, row-3), (x1, row+3), (0,255,255), 1)

    for k, y in rows.items():
        c = colors.get(k, (0,255,255))
        row = int(max(0, min(h-1, y)))
        cv2.line(vis, (0,row), (w-1,row), c, 2, cv2.LINE_AA)
        label = k
        if k in ("chest","waist","hip","thigh"):
            a_cm, b_cm = semiaxes_cm.get(k, (float("nan"), float("nan")))
            circ = circumf_cm.get(k, float("nan"))
            label = f"{k}: a={a_cm:.1f}cm, b={b_cm:.1f}cm, C≈{circ:.1f}cm"
        cv2.putText(vis, label, (10, max(20, row-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    _save(out_path, vis)

def _parse_pairs(s: Optional[str]) -> Dict[str, float]:
    d = {}
    if not s: return d
    for p in s.split(","):
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                d[k.strip().lower()] = float(v)
            except:
                pass
    return d

def _keep_largest_component(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return (m * 255).astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return cleaned

def _crop_and_resize_to_height(
    img_bgr: np.ndarray, mask_u8: np.ndarray,
    target_h: int = 2000, margin_ratio: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = mask_bbox(mask_u8)
    H, W = img_bgr.shape[:2]

    bbox_h = max(1, y1 - y0)
    bbox_w = max(1, x1 - x0)
    dy = int(round(bbox_h * margin_ratio))
    dx = int(round(bbox_w * margin_ratio))

    x0c = max(0, x0 - dx); x1c = min(W - 1, x1 + dx)
    y0c = max(0, y0 - dy); y1c = min(H - 1, y1 + dy)

    crop_img  = img_bgr[y0c:y1c+1, x0c:x1c+1]
    crop_mask = mask_u8[y0c:y1c+1, x0c:x1c+1]

    _, yy0, _, yy1 = mask_bbox(crop_mask)
    crop_bbox_h = max(1, yy1 - yy0)

    scale = float(target_h) / float(crop_bbox_h)
    new_w = max(1, int(round(crop_img.shape[1] * scale)))
    new_h = max(1, int(round(crop_img.shape[0] * scale)))

    img_norm  = cv2.resize(crop_img,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_norm = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_norm, mask_norm

def _smooth_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.dilate(m, kernel, iterations=1)
    m = cv2.erode(m,  kernel, iterations=1)
    return (m > 127).astype(np.uint8) * 255

# ---------- width profiles ----------
def _rolling_median(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return a.copy()
    k2 = k // 2
    out = np.empty_like(a)
    for i in range(len(a)):
        j0 = max(0, i-k2); j1 = min(len(a)-1, i+k2)
        block = np.sort(a[j0:j1+1])
        m = len(block)//2
        out[i] = block[m] if len(block)%2==1 else int(round((block[m-1]+block[m])/2))
    return out

def _width_profile(mask_u8: np.ndarray, keep: float, half_window: int = 12) -> np.ndarray:
    """Width per row using center-clip median in a vertical window."""
    h = mask_u8.shape[0]
    cx = _centroid_x(mask_u8)
    vals = [width_at_row_centerclip_window(mask_u8, y, cx, keep, half_window=half_window) for y in range(h)]
    return np.asarray(vals, dtype=np.int32)

def _band_indices(mask_u8: np.ndarray, rel_lo: float, rel_hi: float) -> Tuple[int,int,int,int]:
    x0, y0, x1, y1 = mask_bbox(mask_u8)
    H = max(1, y1 - y0)
    a = int(round(y0 + rel_lo * H))
    b = int(round(y0 + rel_hi * H))
    if b < a: a, b = b, a
    return y0, y1, max(y0,a), min(y1,b)

def _argmin_in_band(arr: np.ndarray, a: int, b: int) -> int:
    if b < a: return (a+b)//2
    idx = int(np.argmin(arr[a:b+1]))
    return a + idx

def _argmax_in_band(arr: np.ndarray, a: int, b: int) -> int:
    if b < a: return (a+b)//2
    idx = int(np.argmax(arr[a:b+1]))
    return a + idx

def _estimate_rows_from_profiles(mask_f: np.ndarray, mask_s: np.ndarray) -> Dict[str,int]:
    """
    Robust row picker using smoothed width profiles + anatomy bands.
    Returns rows dict with shoulder, chest, waist, hip, crotch, thigh.
    """
    # profiles
    prof_f = _width_profile(mask_f, keep=0.80, half_window=12)
    prof_s = _width_profile(mask_s, keep=0.80, half_window=12)
    prod   = prof_f.astype(np.float32) * prof_s.astype(np.float32)

    # smooth (rolling median)
    prof_f = _rolling_median(prof_f, 9)
    prof_s = _rolling_median(prof_s, 9)
    prod   = _rolling_median(prod.astype(np.int32), 9)

    y0, y1, a, b = _band_indices(mask_f, 0.0, 1.0)
    H = max(1, y1 - y0)

    rows = {}

    # Shoulder: 18–24% band, pick local minimum of |d/dy(prod)| to avoid armpit kinks, then slight downward bias
    _, _, s_lo, s_hi = _band_indices(mask_f, 0.18, 0.24)
    deriv = np.abs(np.gradient(prod.astype(np.float32)))
    s_row = _argmin_in_band(deriv, s_lo, s_hi) + int(0.01 * H)
    rows["shoulder"] = int(max(y0, min(y1, s_row)))

    # Chest: 24–34% band, pick maximum of product profile
    _, _, c_lo, c_hi = _band_indices(mask_f, 0.24, 0.34)
    rows["chest"] = _argmax_in_band(prod, c_lo, c_hi)

    # Waist: 45–62% band, pick minimum of product profile
    _, _, w_lo, w_hi = _band_indices(mask_f, 0.45, 0.62)
    rows["waist"] = _argmin_in_band(prod, max(w_lo, rows["chest"]+int(0.05*H)), w_hi)

    # Hip: 64–82% band, max BELOW waist (>= waist+4%H) and above crotch guess
    _, _, h_lo, h_hi = _band_indices(mask_f, 0.64, 0.82)
    h_lo = max(h_lo, rows["waist"] + int(0.04*H))
    rows["hip"] = _argmax_in_band(prod, h_lo, h_hi)

    # Crotch: search 80–95% band for minimum of product followed by increase (use plain minimum as proxy)
    _, _, k_lo, k_hi = _band_indices(mask_f, 0.80, 0.95)
    rows["crotch"] = _argmin_in_band(prod, max(k_lo, rows["hip"] + int(0.03*H)), k_hi)

    # Thigh: a bit below crotch ( +10%H ), clamped to bottom
    rows["thigh"] = int(min(y1, rows["crotch"] + int(0.10 * H)))

    # Enforce order and clamp
    order = ["shoulder","chest","waist","hip","crotch","thigh"]
    for i in range(1, len(order)):
        prev, cur = order[i-1], order[i]
        if rows[cur] <= rows[prev]:
            rows[cur] = rows[prev] + max(1, int(0.02*H))
    for k in order:
        rows[k] = int(max(y0, min(y1, rows[k])))

    return rows, {"prof_front": prof_f.tolist(), "prof_side": prof_s.tolist(), "prof_prod": prod.tolist()}

# ---------- main ----------
def compute(
    front_bgr: np.ndarray,
    side_bgr:  np.ndarray,
    height_cm: Optional[float] = None,
    debug_dir: Optional[str] = None,
    save_masks: bool = False,
    prefer_backend: str = "deeplabv3",
    device: str = "cuda",
    setup_load: Optional[str] = None,
    setup_save: Optional[str] = None,
    calibrate_many: Optional[str] = None,
    aruco_mm: Optional[float] = None,      # ignored (marker-less)
    target_height_px: int = 2000,
    crop_margin_ratio: float = 0.02,
    no_profile_scale: bool = False,
) -> Dict:
    """
    Profile-guided pipeline:
      - Segment → largest component → crop+normalize → smooth
      - Build smoothed width profiles (front, side, product)
      - Pick rows from anatomical bands (shoulder/chest/waist/hip/crotch/thigh)
      - Scale from HEIGHT (profile only fallback)
      - Measure widths (center-clip, window median), compute ellipse circumferences
    """
    # 1) Segmentation
    segmenter = SilhouetteSegmenter(prefer_backend=prefer_backend, device=device)
    mask_f_raw, backend_f = segmenter(front_bgr)
    mask_s_raw, backend_s = segmenter(side_bgr)

    # 2) Remove small islands
    mask_f_raw = _keep_largest_component(mask_f_raw)
    mask_s_raw = _keep_largest_component(mask_s_raw)

    # 3) Normalize
    front_norm, mask_f = _crop_and_resize_to_height(front_bgr, mask_f_raw, target_h=target_height_px, margin_ratio=crop_margin_ratio)
    side_norm,  mask_s = _crop_and_resize_to_height(side_bgr,  mask_s_raw,  target_h=target_height_px, margin_ratio=crop_margin_ratio)

    # 4) Smooth masks
    mask_f = _smooth_mask(mask_f)
    mask_s = _smooth_mask(mask_s)

    # 5) Width profiles → rows
    rows, profiles = _estimate_rows_from_profiles(mask_f, mask_s)

    # 6) Scale (HEIGHT > PROFILE)
    ppc = None
    ppc_height = None
    ppc_profile = None
    if setup_load and not no_profile_scale:
        try:
            prof = _load_json(setup_load)
            ppc_profile = float(prof.get("ppc_ref", 0.0)) or None
        except Exception:
            ppc_profile = None

    if height_cm and height_cm > 0:
        ppc_height = float(target_height_px) / float(height_cm)
        ppc = ppc_height
        if ppc_profile:
            rel = abs(ppc_height - ppc_profile) / ppc_height
            if rel > 0.02:
                print(f"[warn] profile ppc_ref ({ppc_profile:.6f}) differs from height scale ({ppc_height:.6f}) by {rel*100:.1f}% — using HEIGHT.",
                      file=sys.stderr)
    else:
        if no_profile_scale:
            raise RuntimeError("No scale: --no-profile-scale was set but --height-cm not provided.")
        if ppc_profile:
            ppc = ppc_profile
        else:
            raise RuntimeError("No scale: pass --height-cm (recommended) or provide a profile with ppc_ref.")

    # 7) Widths (window median, center-clip)
    cx_f = _centroid_x(mask_f);  cx_s = _centroid_x(mask_s)
    keep = {"chest":0.72, "waist":0.74, "hip":0.83, "thigh":0.78}

    widths_front = {
        "chest":    width_at_row_centerclip_window(mask_f, rows["chest"],    cx_f, keep["chest"], half_window=16),
        "waist":    width_at_row_centerclip_window(mask_f, rows["waist"],    cx_f, keep["waist"], half_window=16),
        "hip":      width_at_row_centerclip_window(mask_f, rows["hip"],      cx_f, keep["hip"],   half_window=16),
        "shoulder": width_at_row(mask_f, rows["shoulder"]),
        "thigh":    width_at_row_centerclip_window(mask_f, rows["thigh"],    cx_f, keep["thigh"], half_window=16),
    }
    widths_side = {
        "chest": width_at_row_centerclip_window(mask_s, rows["chest"], cx_s, keep["chest"], half_window=16),
        "waist": width_at_row_centerclip_window(mask_s, rows["waist"], cx_s, keep["waist"], half_window=16),
        "hip":   width_at_row_centerclip_window(mask_s, rows["hip"],   cx_s, keep["hip"],   half_window=16),
        "thigh": width_at_row_centerclip_window(mask_s, rows["thigh"], cx_s, keep["thigh"], half_window=16),
    }

    # 8) Circumferences (ellipse)
    semiaxes_cm, circumf_cm = {}, {}
    for k in ("chest","waist","hip","thigh"):
        a_cm = (widths_front[k] / 2.0) / ppc
        b_cm = (widths_side[k]  / 2.0) / ppc
        semiaxes_cm[k] = (float(a_cm), float(b_cm))
        circumf_cm[k]  = float(ellipse_circumference(a_cm, b_cm))

    # 9) Lengths
    shoulder_width_cm = float(widths_front["shoulder"] / ppc)
    _, _, _, y1f = mask_bbox(mask_f)
    inseam_px = max(0, y1f - rows["crotch"])
    inseam_cm = float(inseam_px / ppc)

    # 10) Debug profiles (CSV-like text & quickline PNGs)
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        if save_masks:
            _save(os.path.join(debug_dir, "front_mask.png"), mask_f)
            _save(os.path.join(debug_dir, "side_mask.png"),  mask_s)
        _overlay(front_norm, mask_f, rows, widths_front, semiaxes_cm, circumf_cm,
                 os.path.join(debug_dir, "front_overlay.png"), show_clip=True)
        _overlay(side_norm,  mask_s, rows, widths_side,  semiaxes_cm, circumf_cm,
                 os.path.join(debug_dir, "side_overlay.png"),  show_clip=True)

        # write profiles
        _save_txt(os.path.join(debug_dir, "profiles.csv"),
                  "y,front_width,side_width,prod\n" +
                  "\n".join(f"{i},{profiles['prof_front'][i]},{profiles['prof_side'][i]},{profiles['prof_prod'][i]}"
                            for i in range(len(profiles['prof_front']))))

        # tiny PNG plots without external libs
        def _quickplot(vals: List[int], path: str):
            h, w = 300, 600
            img = np.full((h,w,3), 255, np.uint8)
            v = np.asarray(vals, dtype=np.float32)
            if v.max() <= 0:
                _save(path, img);
                return
            v = (v / v.max()) * (h-20)
            points = []
            for i in range(w):
                y = int((i/ (w-1)) * (len(vals)-1))
                py = h-10 - int(v[y])
                points.append((i, py))
            for i in range(1,len(points)):
                cv2.line(img, points[i-1], points[i], (60,60,200), 2)
            _save(path, img)

        _quickplot(profiles['prof_front'], os.path.join(debug_dir, "profile_front.png"))
        _quickplot(profiles['prof_side'],  os.path.join(debug_dir, "profile_side.png"))
        _quickplot(profiles['prof_prod'],  os.path.join(debug_dir, "profile_prod.png"))

    # 11) QA
    qa = {"normalization": {
        "target_h_px": int(target_height_px),
        "crop_margin_ratio": float(crop_margin_ratio),
        "ppc_used_px_per_cm": float(ppc),
        "ppc_source": ("height" if (height_cm and height_cm>0) else ("profile" if ppc_profile else "none"))
    }}

    if setup_save:
        _save_json(setup_save, {"ppc_ref": float(ppc)})

    return {
        "pixels_per_cm": float(ppc),
        "backend_front": backend_f,
        "backend_side":  backend_s,
        "rows_px": rows,
        "widths_px_front": widths_front,
        "widths_px_side": widths_side,
        "semiaxes_cm": semiaxes_cm,
        "chest_cm":  circumf_cm["chest"],
        "waist_cm":  circumf_cm["waist"],
        "hip_cm":    circumf_cm["hip"],
        "thigh_cm":  circumf_cm["thigh"],
        "shoulder_width_cm": shoulder_width_cm,
        "inseam_cm": inseam_cm,
        "qa": qa,
    }
