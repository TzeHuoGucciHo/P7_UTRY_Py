from __future__ import annotations
import os, json, time
from typing import Dict, Optional, Tuple, Callable
import cv2
import numpy as np

# =======================
# QUICK TOGGLE — vælg ÉN kombination
# =======================

# --- Global (grid 82) ---
PARAM_TARGET_HEIGHT_PX = 2200
PARAM_CROP_MARGIN_RATIO = 0.0017
KEEP_BY_KEY = {"chest": 0.8395764121252157, "waist": 0.6557841821537503, "hip": 0.9373690683077852, "thigh": 0.7908863001992785}
ARMR_BY_KEY = {"chest": 0.019982132976907208, "waist": 0.02256598041367644, "hip": 0.04173145394123699, "thigh": 0.017981314598465695}

from .segmenter import SilhouetteSegmenter
from .pose import measurement_rows_from_mask
from .geometry import (
    ellipse_circumference, mask_bbox,
    width_at_row, width_at_row_centerclip, width_at_row_centerclip_window,
)

try:
    import mediapipe as mp
    _LM = mp.solutions.pose.PoseLandmark
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

# ---------- utils ----------
def _save(path: str, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def _save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _centroid_x(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    return float(xs.mean()) if xs.size else mask.shape[1] / 2.0

def _keep_largest_component(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return (m * 255).astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    filled = m.copy()
    h, w = filled.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    bg = filled.copy()
    cv2.floodFill(bg, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(bg)
    out = cv2.bitwise_or(filled, holes)
    return (out > 0).astype(np.uint8) * 255

def _smooth_mask(mask: np.ndarray) -> np.ndarray:
    m = _fill_holes(mask)
    k = np.ones((3,3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = cv2.dilate(m, k, iterations=1)
    m = cv2.erode(m,  k, iterations=1)
    return (m > 127).astype(np.uint8) * 255

def _crop_and_resize_to_height(
    img_bgr: np.ndarray, mask_u8: np.ndarray,
    target_h: int = 2000, margin_ratio: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
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
    return img_norm, mask_norm, (x0c, y0c, x1c, y1c)

def _overlay(img_bgr, mask_u8, rows, out_path):
    vis = img_bgr.copy()
    #tint = np.zeros_like(vis); tint[mask_u8 > 0] = (0,255,0)
    #vis = cv2.addWeighted(vis, 1.0, tint, 0.35, 0)
    colors = {"shoulder":(0,255,255), "chest":(0,128,255), "waist":(255,0,255),
              "hip":(255,128,0), "crotch":(180,180,0), "thigh":(0,200,100)}
    h, w = mask_u8.shape[:2]
    for k, y in rows.items():
        y = int(max(0, min(h-1, y)))
        cv2.line(vis, (0,y), (w-1,y), colors.get(k,(255,255,255)), 2, cv2.LINE_AA)
        cv2.putText(vis, k, (10, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, colors.get(k, (255, 255, 255)), 3,
                    cv2.LINE_AA)
    _save(out_path, vis)

# ---------- arm suppression ----------
def _arm_suppressed(mask: np.ndarray, h_frac_radius: float) -> np.ndarray:
    _, y0, _, y1 = mask_bbox(mask)
    H = max(1, y1 - y0)
    r = max(1, int(round(h_frac_radius * H)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    m = _fill_holes(mask)
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return (opened > 127).astype(np.uint8) * 255

def _precompute_torso_masks(mask_f: np.ndarray, mask_s: np.ndarray, arm_r_by_key: Dict[str,float]) -> Tuple[Dict[str,np.ndarray], Dict[str,np.ndarray]]:
    mf: Dict[str,np.ndarray] = {}
    ms: Dict[str,np.ndarray] = {}
    for k, r in arm_r_by_key.items():
        mf[k] = _arm_suppressed(mask_f, r)
        ms[k] = _arm_suppressed(mask_s, r)
    return mf, ms

def _width_centerclip_from_cache(mask_map: Dict[str,np.ndarray], key: str,
                                 y: int, cx: float, keep: float, halfw: int) -> int:
    m = mask_map[key]
    try:
        return width_at_row_centerclip_window(m, y, cx, keep, half_window=halfw)
    except Exception:
        return width_at_row_centerclip(m, y, cx, keep)

# ---------- mediapipe fast ----------
def _mp_landmarks_fast(img_bgr: np.ndarray):
    if not HAS_MEDIAPIPE:
        return {}
    h, w = img_bgr.shape[:2]
    scale = 1.0
    max_side = max(h, w)
    if max_side > 640:
        scale = 640.0 / max_side
        img_small = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr

    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    with mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=0,
        enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return {}
    hs, ws = img_small.shape[:2]
    out = {}
    inv_scale = 1.0/scale
    for lm in _LM:
        p = res.pose_landmarks.landmark[lm]
        out[lm.name.lower()] = (p.x * ws * inv_scale, p.y * hs * inv_scale, p.visibility)
    return out

def _avg_y(a, b):
    if a is None or b is None: return None
    if a[2] < 0.3 or b[2] < 0.3: return None
    return (a[1] + b[1]) / 2.0

# ---------- band search med stride ----------
def _argext_band(band: range, f: Callable[[int], float], mode: str = "max",
                 stride: int = 2, refine_radius: int = 6) -> int:
    if len(band) == 0:
        raise ValueError("empty band")
    ys_coarse = list(range(band.start, band.stop, stride))
    vals = [f(y) for y in ys_coarse]
    idx = int(np.argmax(vals)) if mode == "max" else int(np.argmin(vals))
    y_best = ys_coarse[idx]
    y0 = max(band.start, y_best - refine_radius)
    y1 = min(band.stop - 1, y_best + refine_radius)
    ys_ref = list(range(y0, y1 + 1))
    vals_ref = [f(y) for y in ys_ref]
    idx_ref = int(np.argmax(vals_ref)) if mode == "max" else int(np.argmin(vals_ref))
    return ys_ref[idx_ref]

def _rows_front(img_f: np.ndarray, mask_f: np.ndarray, width_fn_front) -> Dict[str,int]:
    ys = np.where(mask_f > 0)[0]
    if ys.size == 0: raise RuntimeError("Empty mask.")
    y0, y1 = ys.min(), ys.max()
    H = max(1, y1 - y0)

    lm = _mp_landmarks_fast(img_f) if HAS_MEDIAPIPE else {}
    ls = lm.get("left_shoulder");  rs = lm.get("right_shoulder")
    lh = lm.get("left_hip");       rh = lm.get("right_hip")
    lk = lm.get("left_knee");      rk = lm.get("right_knee")

    rows: Dict[str,int] = {}

    y_sh = _avg_y(ls, rs)
    if y_sh is None:
        band = range(int(y0+0.18*H), int(y0+0.25*H))
        if len(band) >= 3:
            rows["shoulder"] = _argext_band(band, lambda y: width_fn_front(y), mode="min", stride=2, refine_radius=4)
        else:
            rows["shoulder"] = int(y0+0.22*H)
    else:
        rows["shoulder"] = int(np.clip(y_sh, y0+0.15*H, y0+0.30*H))

    y_hp = _avg_y(lh, rh)
    if y_hp is not None:
        rows["hip"] = int(np.clip(y_hp, y0+0.55*H, y0+0.88*H))
    else:
        band0 = range(int(y0+0.62*H), int(y0+0.86*H))
        if len(band0) >= 3:
            rows["hip"] = _argext_band(band0, lambda y: width_fn_front(y), mode="max", stride=2, refine_radius=6)
        else:
            rows["hip"] = int(y0+0.74*H)

    y_kn = _avg_y(lk, rk)
    c0 = rows["hip"] + int(0.06*H)
    c1 = rows["hip"] + int(0.20*H)
    if y_kn is not None:
        c1 = min(c1, int(y_kn - 0.10*H))
    c0 = max(y0, c0); c1 = min(y1, c1)
    band_c = range(c0, c1+1) if c1 > c0 else range(int(rows["hip"]+0.12*H), int(rows["hip"]+0.12*H)+1)
    if len(band_c) >= 3:
        rows["crotch"] = _argext_band(band_c, lambda y: width_fn_front(y), mode="min", stride=2, refine_radius=6)
    else:
        rows["crotch"] = int(rows["hip"]+0.12*H)

    hf0 = rows["waist"] + int(0.06*H) if "waist" in rows else rows["hip"] - int(0.14*H)
    hf1 = min(rows["crotch"] - int(0.12*H), hf0 + int(0.12*H))
    hf0 = max(y0, hf0); hf1 = min(y1, hf1)
    band_h = range(hf0, hf1+1) if hf1 > hf0 else range(rows["hip"], rows["hip"]+1)
    if len(band_h) >= 3:
        rows["hip"] = _argext_band(band_h, lambda y: width_fn_front(y), mode="max", stride=2, refine_radius=6)
    rows["hip"] = min(rows["hip"], rows["crotch"] - int(0.10*H))

    w0 = int(rows["shoulder"] + 0.50*(rows["hip"] - rows["shoulder"]))
    w1 = int(rows["shoulder"] + 0.80*(rows["hip"] - rows["shoulder"]))
    w0 = max(y0, w0); w1 = min(y1, w1)
    band_w = range(w0, w1+1) if w1 > w0 else range(int(y0+0.60*H), int(y0+0.60*H)+1)
    if len(band_w) >= 3:
        rows["waist"] = _argext_band(band_w, lambda y: float(width_fn_front(y)), mode="min", stride=2, refine_radius=6)
    else:
        rows["waist"] = int(y0+0.60*H)
    rows["waist"] = min(rows["waist"], rows["hip"] - int(0.05*H))

    c0 = rows["shoulder"] + int(0.05*(rows["hip"]-rows["shoulder"]))
    c1 = rows["waist"]    - int(0.10*(rows["waist"]-rows["shoulder"]))
    c0 = max(y0, c0); c1 = min(y1, c1)
    band_ch = range(c0, c1+1) if c1 > c0 else range(int(rows["shoulder"] + 0.25*(rows["hip"]-rows["shoulder"])),
                                                    int(rows["shoulder"] + 0.25*(rows["hip"]-rows["shoulder"]))+1)
    if len(band_ch) >= 3:
        rows["chest"] = _argext_band(band_ch, lambda y: float(width_fn_front(y)), mode="max", stride=2, refine_radius=6)
    else:
        rows["chest"]= int(rows["shoulder"] + 0.25*(rows["hip"]-rows["shoulder"]))

    order = ["shoulder","chest","waist","hip","crotch","thigh"]
    rows["thigh"] = int(min(y1, rows["crotch"] + 0.10*H))
    for i in range(1, len(order)):
        if rows[order[i]] <= rows[order[i-1]]:
            rows[order[i]] = rows[order[i-1]] + max(1, int(0.02*H))
    rows["hip"]   = min(rows["hip"],   rows["crotch"] - int(0.10*H))
    rows["waist"] = min(rows["waist"], rows["hip"]    - int(0.05*H))
    for k in order:
        rows[k] = int(max(y0, min(y1, rows[k])))
    return rows

# ---------- MAIN ----------
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
    aruco_mm: Optional[float] = None,
    target_height_px: int = 2000,
    crop_margin_ratio: float = 0.02,
    no_profile_scale: bool = False,
    gender: str = "male",
) -> Dict:
    t_all = time.perf_counter()

    # 0) kønsspecifikke parametre
    # - 'keep' er center-clip ratio (højere = bredere vindue); 'arm_r' er arm-erosionsradius i fraktion af højde.
    # - row_bias skubber waist/hip få procentpunkter (positiv = længere ned mod fødder).
    G = {
        "male": {
            "keep":   {"chest":0.82, "waist":0.68, "hip":0.75, "thigh":0.80},
            "arm_r":  {"chest":0.026,"waist":0.028,"hip":0.020,"thigh":0.016},
            "row_bias":{"waist":0.02, "hip":-0.01}
        },
        "female": {
            "keep":   {"chest":0.80, "waist":0.68, "hip":1.30, "thigh":0.78},
            "arm_r":  {"chest":0.018,"waist":0.020,"hip":0.023,"thigh":0.018},
            "row_bias":{"waist":0.02, "hip":-0.01}  # waist lidt ned, hip en anelse op
        }
    }
    GP = G.get(gender, G["male"])

    # 1) segmentering
    t0 = time.perf_counter()
    segmenter = SilhouetteSegmenter(prefer_backend=prefer_backend, device=device)
    mask_f_raw, backend_f = segmenter(front_bgr)
    mask_s_raw, backend_s = segmenter(side_bgr)
    t_seg = (time.perf_counter() - t0) * 1000.0

    # 2) cleanup + normalisering
    t0 = time.perf_counter()
    mask_f_raw = _keep_largest_component(mask_f_raw)
    mask_s_raw = _keep_largest_component(mask_s_raw)
    front_norm, mask_f, _ = _crop_and_resize_to_height(front_bgr, mask_f_raw, target_h=target_height_px, margin_ratio=crop_margin_ratio)
    side_norm,  mask_s, _  = _crop_and_resize_to_height(side_bgr,  mask_s_raw,  target_h=target_height_px, margin_ratio=crop_margin_ratio)
    mask_f = _smooth_mask(mask_f)
    mask_s = _smooth_mask(mask_s)
    t_norm = (time.perf_counter() - t0) * 1000.0

    # 3) precompute torso-masker
    t0 = time.perf_counter()
    arm_r_by_key = GP["arm_r"]
    mask_f_torso, mask_s_torso = _precompute_torso_masks(mask_f, mask_s, arm_r_by_key)
    cx_f = _centroid_x(mask_f); cx_s = _centroid_x(mask_s)
    keep_for_rows = 0.82
    halfw_default = 16
    halfw_hip = 24
    width_fn_front = lambda y: _width_centerclip_from_cache(mask_f_torso, "hip", y, cx_f, keep_for_rows, halfw_hip)
    t_cache = (time.perf_counter() - t0) * 1000.0

    # 4) rækker
    t0 = time.perf_counter()
    if HAS_MEDIAPIPE:
        try:
            rows = _rows_front(front_norm, mask_f, width_fn_front)
            rows_source = "pose+mask_fast"
        except Exception:
            rows = measurement_rows_from_mask(mask_f)
            rows_source = "mask_fallback"
    else:
        rows = measurement_rows_from_mask(mask_f)
        rows_source = "mask_only"

    # påfør små køns-bias i rækker (beholder clamps)
    ys = np.where(mask_f > 0)[0]
    y0, y1 = int(ys.min()), int(ys.max())
    H = max(1, y1 - y0)
    if "waist" in rows:
        rows["waist"] = int(np.clip(rows["waist"] + GP["row_bias"]["waist"]*H, y0, y1))
    if "hip" in rows:
        rows["hip"]   = int(np.clip(rows["hip"]   + GP["row_bias"]["hip"]  *H, y0, y1))
    rows["hip"]   = min(rows["hip"],   rows["crotch"] - int(0.10*H))
    rows["waist"] = min(rows["waist"], rows["hip"]    - int(0.05*H))
    t_rows = (time.perf_counter() - t0) * 1000.0

    # 5) skala
    t0 = time.perf_counter()
    ppc = None; ppc_source = "none"
    if height_cm and height_cm > 0:
        ppc = float(target_height_px) / float(height_cm); ppc_source = "height"
    elif (not no_profile_scale) and setup_load:
        try:
            with open(setup_load, "r", encoding="utf-8") as f:
                prof = json.load(f)
            ppc = float(prof.get("ppc_ref", 0.0)) or None; ppc_source = "profile"
        except Exception:
            ppc = None
    if ppc is None:
        raise RuntimeError("No metric scale: pass --height-cm (recommended).")
    t_scale = (time.perf_counter() - t0) * 1000.0

    # 6) bredder (kønsspecifik keep)
    t0 = time.perf_counter()
    keep = GP["keep"]
    def TW(mask_map, key, y, cx):
        halfw = halfw_hip if key == "hip" else halfw_default
        return _width_centerclip_from_cache(mask_map, key, y, cx, keep[key], halfw)

    widths_front = {
        "chest":    TW(mask_f_torso, "chest",  rows["chest"],    cx_f),
        "waist":    TW(mask_f_torso, "waist",  rows["waist"],    cx_f),
        "hip":      TW(mask_f_torso, "hip",    rows["hip"],      cx_f),
        "shoulder": width_at_row(mask_f, rows["shoulder"]),
        "thigh":    TW(mask_f_torso, "thigh",  rows["thigh"],    cx_f),
    }
    widths_side = {
        "chest": TW(mask_s_torso, "chest", rows["chest"], cx_s),
        "waist": TW(mask_s_torso, "waist", rows["waist"], cx_s),
        "hip":   TW(mask_s_torso, "hip",   rows["hip"],   cx_s),
        "thigh": TW(mask_s_torso, "thigh", rows["thigh"], cx_s),
    }
    t_widths = (time.perf_counter() - t0) * 1000.0

    # 7) ellipse-omkredse
    t0 = time.perf_counter()
    semiaxes_cm, circumf_cm = {}, {}
    for k in ("chest","waist","hip","thigh"):
        a_cm = (widths_front[k] / 2.0) / ppc
        b_cm = (widths_side[k]  / 2.0) / ppc
        semiaxes_cm[k] = (float(a_cm), float(b_cm))
        circumf_cm[k]  = float(ellipse_circumference(a_cm, b_cm))
    t_ellipse = (time.perf_counter() - t0) * 1000.0

    # 8) længder
    t0 = time.perf_counter()
    shoulder_width_cm = float(widths_front["shoulder"] / ppc)
    chest_width_cm = float(widths_front["chest"] / ppc)  # This is the front-view width
    _, y0m, _, y1m = mask_bbox(mask_f)
    inseam_cm = float(max(0, y1m - rows["crotch"]) / ppc)
    t_lengths = (time.perf_counter() - t0) * 1000.0

    # 9) debug
    t0 = time.perf_counter()
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        if save_masks:
            _save(os.path.join(debug_dir, "front_mask.png"), mask_f)
            _save(os.path.join(debug_dir, "side_mask.png"),  mask_s)
            _save(os.path.join(debug_dir, "front_mask_torso.png"), mask_f_torso["hip"])
            _save(os.path.join(debug_dir, "side_mask_torso.png"),  mask_s_torso["hip"])
        _overlay(front_norm, mask_f, rows, os.path.join(debug_dir, "front_overlay.png"))
        _overlay(side_norm,  mask_s, rows, os.path.join(debug_dir, "side_overlay.png"))
    t_debug = (time.perf_counter() - t0) * 1000.0

    qa = {
        "normalization": {
            "target_h_px": int(target_height_px),
            "crop_margin_ratio": float(crop_margin_ratio),
            "ppc_used_px_per_cm": float(ppc),
            "ppc_source": ppc_source,
            "rows_source": rows_source,
            "gender": gender,
        },
        "timing_ms": {
            "segment": round(t_seg,1),
            "normalize": round(t_norm,1),
            "precache": round(t_cache,1),
            "rows": round(t_rows,1),
            "scale": round(t_scale,1),
            "widths": round(t_widths,1),
            "ellipse": round(t_ellipse,1),
            "lengths": round(t_lengths,1),
            "debug": round(t_debug,1),
            "total": round((time.perf_counter()-t_all)*1000.0,1),
        }
    }

    if setup_save:
        _save_json(setup_save, {"ppc_ref": float(ppc)})

    return {
        "pixels_per_cm": float(ppc),
        "backend_front": backend_f,
        "backend_side": backend_s,
        "rows_px": rows,
        "widths_px_front": widths_front,
        "widths_px_side": widths_side,
        "semiaxes_cm": semiaxes_cm,
        "chest_cm": circumf_cm["chest"],  # <-- This is CIRCUMFERENCE
        "waist_cm": circumf_cm["waist"],
        "hip_cm": circumf_cm["hip"],
        "thigh_cm": circumf_cm["thigh"],
        "shoulder_width_cm": shoulder_width_cm,
        "inseam_cm": inseam_cm,

        # --- NEW/MODIFIED KEYS ---
        # 1. Chest Front Width (snake_case)
        "chest_width_cm": chest_width_cm,  # Raw front width

        "qa": qa,
    }
