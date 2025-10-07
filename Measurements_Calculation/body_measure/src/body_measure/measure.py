from __future__ import annotations
import os, json
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from .segmenter import SilhouetteSegmenter
from .pose import measurement_rows_from_mask
from .geometry import (
    ellipse_circumference, mask_bbox,
    width_at_row_centerclip, width_at_row,
    width_at_row_centerclip_window
)

# ---------- helpers ----------
def _save(path: str, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

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

    colors = {"chest": (0,128,255), "waist": (255,0,255), "hip": (255,128,0), "shoulder": (0,255,255), "thigh": (0,200,100)}
    h, w = mask_u8.shape[:2]
    if show_clip:
        cx = _centroid_x(mask_u8)
        keep_local = {"chest":0.70, "waist":0.72, "hip":0.85, "shoulder":0.95, "thigh":0.78}
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
        if k in ("chest","waist","hip","thigh"):
            a_cm, b_cm = semiaxes_cm.get(k, (float("nan"), float("nan")))
            circ = circumf_cm.get(k, float("nan"))
            label = f"{k}: a={a_cm:.1f}cm, b={b_cm:.1f}cm, C≈{circ:.1f}cm, width_px={widths_px.get(k,0)}"
        elif k == "shoulder":
            label = f"{k}: width_px={widths_px.get(k,0)}"
        else:
            label = k
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

def _crop_and_resize_to_height(
    img_bgr: np.ndarray, mask_u8: np.ndarray,
    target_h: int = 2000, margin_ratio: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop to silhouette bbox (extrema on both axes) + small margin, then resize so
    the bbox height equals target_h. Returns (img_norm, mask_norm).
    """
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

    # recompute bbox in crop for precise scale
    _, yy0, _, yy1 = mask_bbox(crop_mask)
    crop_bbox_h = max(1, yy1 - yy0)

    scale = float(target_h) / float(crop_bbox_h)
    new_w = max(1, int(round(crop_img.shape[1] * scale)))
    new_h = max(1, int(round(crop_img.shape[0] * scale)))

    img_norm  = cv2.resize(crop_img,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_norm = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_norm, mask_norm

def _refine_waist_row_by_local_min(mask_u8, y_guess: int, band_ratio: float = 0.12, keep: float = 0.70) -> int:
    x0, y0, x1, y1 = mask_bbox(mask_u8)
    H = max(1, y1 - y0)
    ys = np.arange(max(y0, y_guess - int(max(10, band_ratio*H))), min(y1, y_guess + int(max(10, band_ratio*H)) + 1))
    if ys.size == 0: return y_guess
    cx = _centroid_x(mask_u8)
    vals = [width_at_row_centerclip(mask_u8, int(y), cx, keep) for y in ys]
    return int(ys[int(np.argmin(vals))])

def _smooth_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    m = cv2.medianBlur(m, 5)
    kernel = np.ones((5,5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (m > 127).astype(np.uint8) * 255

# ---------- main ----------
def compute(
    front_bgr: np.ndarray,
    side_bgr:  np.ndarray,
    height_cm: Optional[float] = None,
    debug_dir: Optional[str] = None,
    save_masks: bool = False,
    prefer_backend: str = "deeplabv3",
    device: str = "cpu",
    setup_load: Optional[str] = None,
    setup_save: Optional[str] = None,
    calibrate_many: Optional[str] = None,
    aruco_mm: Optional[float] = None,      # ignored (marker-less)
    target_height_px: int = 2000,
    crop_margin_ratio: float = 0.02,
) -> Dict:
    """
    Marker-less normalization:
      - Segment
      - Crop to bbox (extrema) + margin
      - Resize so bbox height = target_height_px
      - Scale precedence:
          * if height_cm:     ppc = target_height_px / height_cm  (PER-USER, CORRECT)
          * elif setup_load:  ppc = profile.ppc_ref               (FALLBACK ONLY)
          * else:             error
      - Measure on normalized masks (waist local-min, windowed widths, shoulder full width)
    """
    # 1) Segment originals
    segmenter = SilhouetteSegmenter(prefer_backend=prefer_backend, device=device)
    mask_f_raw, backend_f = segmenter(front_bgr)
    mask_s_raw, backend_s = segmenter(side_bgr)

    # 2) Normalize (crop + resize)
    front_norm, mask_f = _crop_and_resize_to_height(front_bgr, mask_f_raw, target_h=target_height_px, margin_ratio=crop_margin_ratio)
    side_norm,  mask_s = _crop_and_resize_to_height(side_bgr,  mask_s_raw,  target_h=target_height_px, margin_ratio=crop_margin_ratio)

    # 3) Smooth masks for stability
    mask_f = _smooth_mask(mask_f)
    mask_s = _smooth_mask(mask_s)

    # 4) Rows (on normalized front) + waist refinement
    rows = measurement_rows_from_mask(mask_f)
    rows["waist"] = _refine_waist_row_by_local_min(mask_f, rows["waist"], band_ratio=0.12, keep=0.70)

    # 5) Scale (HEIGHT overrides PROFILE)
    ppc = None
    profile_ppc = None
    if setup_load:
        try:
            prof = _load_json(setup_load)
            profile_ppc = float(prof.get("ppc_ref", 0.0)) or None
        except Exception:
            profile_ppc = None

    if height_cm and height_cm > 0:
        ppc = float(target_height_px) / float(height_cm)
    elif profile_ppc:
        ppc = float(profile_ppc)
    else:
        raise RuntimeError("No scale: pass --height-cm (recommended) or provide a profile with ppc_ref as fallback.")

    # 6) Widths (windowed median for torso/leg; shoulder = full width)
    cx_f = _centroid_x(mask_f);  cx_s = _centroid_x(mask_s)
    keep = {"chest":0.70, "waist":0.72, "hip":0.85, "thigh":0.78}

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

    # 7) Circumferences (ellipse)
    semiaxes_cm, circumf_cm = {}, {}
    for k in ("chest","waist","hip","thigh"):
        a_cm = (widths_front[k] / 2.0) / ppc
        b_cm = (widths_side[k]  / 2.0) / ppc
        semiaxes_cm[k] = (float(a_cm), float(b_cm))
        circumf_cm[k]  = float(ellipse_circumference(a_cm, b_cm))

    # 8) Lengths
    shoulder_width_cm = float(widths_front["shoulder"] / ppc)
    _, _, _, y1f = mask_bbox(mask_f)
    inseam_px = max(0, y1f - rows["crotch"])
    inseam_cm = float(inseam_px / ppc)

    # 9) Optional global calibration vs. known measurements
    gt = _parse_pairs(calibrate_many)
    if gt:
        est = {"chest": circumf_cm.get("chest"), "hip": circumf_cm.get("hip"),
               "thigh": circumf_cm.get("thigh"), "inseam": inseam_cm, "shoulder": shoulder_width_cm}
        ratios = [gt[k]/est[k] for k in gt.keys() if (k in est and est[k] and gt[k] > 0)]
        if ratios:
            s = float(np.median(ratios))
            ppc /= max(1e-9, s)
            # recompute at new scale
            for k in ("chest","waist","hip","thigh"):
                a_cm = (widths_front[k] / 2.0) / ppc
                b_cm = (widths_side[k]  / 2.0) / ppc
                semiaxes_cm[k] = (float(a_cm), float(b_cm))
                circumf_cm[k]  = float(ellipse_circumference(a_cm, b_cm))
            shoulder_width_cm = float(widths_front["shoulder"] / ppc)
            inseam_cm = float(inseam_px / ppc)

    # 10) Debug on normalized images
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        if save_masks:
            _save(os.path.join(debug_dir, "front_mask.png"), mask_f)
            _save(os.path.join(debug_dir, "side_mask.png"),  mask_s)
        _overlay(front_norm, mask_f, rows,
                 {k: widths_front.get(k,0) for k in ["chest","waist","hip","shoulder","thigh"]},
                 {k: semiaxes_cm.get(k,(np.nan,np.nan)) for k in ["chest","waist","hip","thigh"]},
                 {k: circumf_cm.get(k,np.nan) for k in ["chest","waist","hip","thigh"]},
                 os.path.join(debug_dir, "front_overlay.png"), show_clip=True)
        _overlay(side_norm,  mask_s, rows,
                 {k: widths_side.get(k,0) for k in ["chest","waist","hip","thigh"]},
                 {k: semiaxes_cm.get(k,(np.nan,np.nan)) for k in ["chest","waist","hip","thigh"]},
                 {k: circumf_cm.get(k,np.nan) for k in ["chest","waist","hip","thigh"]},
                 os.path.join(debug_dir, "side_overlay.png"), show_clip=True)

    # 11) QA
    qa = {"normalization": {
        "target_h_px": int(target_height_px),
        "crop_margin_ratio": float(crop_margin_ratio),
        "ppc_used_px_per_cm": float(ppc),
        "ppc_source": ("height" if (height_cm and height_cm>0) else ("profile" if profile_ppc else "none"))
    }}
    if setup_load and profile_ppc:
        qa["setup_profile_ppc_ref"] = float(profile_ppc)
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
