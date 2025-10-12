from __future__ import annotations
import os, json, time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

# Projektmoduler
from .segmenter import SilhouetteSegmenter
from .pose import measurement_rows_from_mask  # fallback
from .geometry import (
    ellipse_circumference, mask_bbox,
    width_at_row, width_at_row_centerclip, width_at_row_centerclip_window,
)

# -------- MediaPipe (valgfrit) --------
try:
    import mediapipe as mp
    _LM = mp.solutions.pose.PoseLandmark
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

# -------- helper-utils --------
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

# --- NEW: fill interior holes robustly (before any opening/erosion) ---
def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fills all interior holes of a binary mask (255/0) using flood fill from the border.
    Returns a 255/0 uint8 mask.
    """
    m = (mask > 0).astype(np.uint8) * 255
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    filled = m.copy()
    h, w = filled.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)  # floodFill mask must be 2 larger
    # Flood fill background from (0,0) with 255, then invert to get the holes.
    bg = filled.copy()
    cv2.floodFill(bg, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(bg)  # holes are white now
    out = cv2.bitwise_or(filled, holes)
    return (out > 0).astype(np.uint8) * 255

def _smooth_mask(mask: np.ndarray) -> np.ndarray:
    # Fill large holes first, then a light close/dilate/erode polish
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

    # skaler til target_h ud fra bbox-højde i croppet
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
    tint = np.zeros_like(vis); tint[mask_u8 > 0] = (0,255,0)
    vis = cv2.addWeighted(vis, 1.0, tint, 0.35, 0)
    colors = {"shoulder":(0,255,255), "chest":(0,128,255), "waist":(255,0,255),
              "hip":(255,128,0), "crotch":(180,180,0), "thigh":(0,200,100)}
    h, w = mask_u8.shape[:2]
    for k, y in rows.items():
        y = int(max(0, min(h-1, y)))
        cv2.line(vis, (0,y), (w-1,y), colors.get(k,(255,255,255)), 2, cv2.LINE_AA)
        cv2.putText(vis, k, (10, max(20,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors.get(k,(255,255,255)), 2, cv2.LINE_AA)
    _save(out_path, vis)

# -------- arm-suppression + center-clip --------
def _arm_suppressed(mask: np.ndarray, h_frac_radius: float) -> np.ndarray:
    _, y0, _, y1 = mask_bbox(mask)
    H = max(1, y1 - y0)
    r = max(1, int(round(h_frac_radius * H)))  # fx 0.018 → ~3.6% i diameter
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    # Fill holes BEFORE opening, so torso stays solid
    m = _fill_holes(mask)
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return (opened > 127).astype(np.uint8) * 255

def _torso_width(mask: np.ndarray, y: int, cx: float, keep: float, half_window: int, arm_r: float) -> int:
    m = _arm_suppressed(mask, arm_r)
    try:
        return width_at_row_centerclip_window(m, y, cx, keep, half_window=half_window)
    except Exception:
        return width_at_row_centerclip(m, y, cx, keep)

# -------- MediaPipe helpers --------
def _mp_landmarks(img_bgr: np.ndarray):
    if not HAS_MEDIAPIPE:
        return {}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=1,
        enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return {}
    h, w = img_bgr.shape[:2]
    out = {}
    for lm in _LM:
        p = res.pose_landmarks.landmark[lm]
        out[lm.name.lower()] = (p.x * w, p.y * h, p.visibility)
    return out

def _avg_y(a, b):
    if a is None or b is None: return None
    if a[2] < 0.3 or b[2] < 0.3: return None
    return (a[1] + b[1]) / 2.0

# -------- rækker fra pose + maske (FRONT) --------
def _rows_front(img_f: np.ndarray, mask_f: np.ndarray, width_fn_front) -> Dict[str,int]:
    ys = np.where(mask_f > 0)[0]
    if ys.size == 0: raise RuntimeError("Empty mask.")
    y0, y1 = ys.min(), ys.max()
    H = max(1, y1 - y0)

    lm = _mp_landmarks(img_f)
    ls = lm.get("left_shoulder");  rs = lm.get("right_shoulder")
    lh = lm.get("left_hip");       rh = lm.get("right_hip")
    lk = lm.get("left_knee");      rk = lm.get("right_knee")

    rows: Dict[str,int] = {}

    # SHOULDER (anker)
    y_sh = _avg_y(ls, rs)
    if y_sh is None:
        band = range(int(y0+0.18*H), int(y0+0.25*H))
        ws = np.array([width_fn_front(y) for y in band], np.float32)
        gy = band[int(np.argmin(np.abs(np.gradient(ws))))] if len(ws) > 1 else int(y0+0.22*H)
        rows["shoulder"] = int(gy)
    else:
        rows["shoulder"] = int(np.clip(y_sh, y0+0.15*H, y0+0.30*H))

    # HIP (enkeltvis fra front, ikke fusion)
    y_hp = _avg_y(lh, rh)
    if y_hp is not None:
        rows["hip"] = int(np.clip(y_hp, y0+0.55*H, y0+0.88*H))
    else:
        band0 = range(int(y0+0.62*H), int(y0+0.86*H))
        ws0 = np.array([width_fn_front(y) for y in band0], np.int32)
        rows["hip"] = band0[int(np.argmax(ws0))] if len(ws0) else int(y0+0.74*H)

    # CROTCH
    y_kn = _avg_y(lk, rk)
    c0 = rows["hip"] + int(0.06*H)
    c1 = rows["hip"] + int(0.20*H)
    if y_kn is not None:
        c1 = min(c1, int(y_kn - 0.10*H))
    c0 = max(y0, c0); c1 = min(y1, c1)
    band_c = list(range(c0, c1+1)) if c1 > c0 else [int(rows["hip"]+0.12*H)]
    wc = np.array([width_fn_front(y) for y in band_c], np.int32)
    rows["crotch"] = band_c[int(np.argmin(wc))] if len(wc) else int(rows["hip"]+0.12*H)

    # HIP (endeligt)
    hf0 = rows["waist"] + int(0.06*H) if "waist" in rows else rows["hip"] - int(0.14*H)
    hf1 = min(rows["crotch"] - int(0.12*H), hf0 + int(0.12*H))
    hf0 = max(y0, hf0); hf1 = min(y1, hf1)
    band_h = list(range(hf0, hf1+1)) if hf1 > hf0 else [rows["hip"]]
    wh = np.array([width_fn_front(y) for y in band_h], np.int32)
    rows["hip"] = band_h[int(np.argmax(wh))] if len(wh) else rows["hip"]
    rows["hip"] = min(rows["hip"], rows["crotch"] - int(0.10*H))

    # WAIST
    w0 = int(rows["shoulder"] + 0.50*(rows["hip"] - rows["shoulder"]))
    w1 = int(rows["shoulder"] + 0.80*(rows["hip"] - rows["shoulder"]))
    w0 = max(y0, w0); w1 = min(y1, w1)
    band_w = list(range(w0, w1+1)) if w1 > w0 else [int(y0+0.60*H)]
    curve = np.array([width_fn_front(y) for y in band_w], np.float32)
    if len(curve) >= 5:
        k=5; k2=k//2; sm=curve.copy()
        for i in range(len(curve)):
            j0=max(0,i-k2); j1=min(len(curve)-1,i+k2)
            sm[i]=np.median(curve[j0:j1+1])
        rows["waist"] = band_w[int(np.argmin(sm))]
    else:
        rows["waist"] = band_w[int(np.argmin(curve))] if len(curve) else int(y0+0.60*H)
    rows["waist"] = min(rows["waist"], rows["hip"] - int(0.05*H))

    # CHEST
    c0 = rows["shoulder"] + int(0.05*(rows["hip"]-rows["shoulder"]))
    c1 = rows["waist"]    - int(0.10*(rows["waist"]-rows["shoulder"]))
    c0 = max(y0, c0); c1 = min(y1, c1)
    band_ch = list(range(c0, c1+1)) if c1 > c0 else [int(rows["shoulder"] + 0.25*(rows["hip"]-rows["shoulder"]))]
    cur = np.array([width_fn_front(y) for y in band_ch], np.float32)
    if len(cur):
        if len(cur)>=5:
            k=5; k2=k//2; sm=cur.copy()
            for i in range(len(cur)):
                j0=max(0,i-k2); j1=min(len(cur)-1,i+k2)
                sm[i]=np.median(cur[j0:j1+1])
        else:
            sm=cur
        grad = np.gradient(sm) if len(sm)>1 else np.array([0],np.float32)
        thr = np.percentile(sm,70)
        idx=None
        for i in range(1,len(sm)-1):
            if sm[i]>=thr and sm[i]>=sm[i-1] and sm[i]>=sm[i+1] and grad[i]>0:
                idx=i; break
        if idx is None: idx=int(np.argmax(sm))
        rows["chest"]= band_ch[idx]
    else:
        rows["chest"]= int(rows["shoulder"] + 0.25*(rows["hip"]-rows["shoulder"]))

    # THIGH
    rows["thigh"] = int(min(y1, rows["crotch"] + 0.10*H))

    # orden + clamps
    order = ["shoulder","chest","waist","hip","crotch","thigh"]
    for i in range(1, len(order)):
        if rows[order[i]] <= rows[order[i-1]]:
            rows[order[i]] = rows[order[i-1]] + max(1, int(0.02*H))
    rows["hip"]   = min(rows["hip"],   rows["crotch"] - int(0.10*H))
    rows["waist"] = min(rows["waist"], rows["hip"]    - int(0.05*H))
    for k in order:
        rows[k] = int(max(y0, min(y1, rows[k])))
    return rows

# -------- MAIN --------
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
) -> Dict:
    t0 = time.time()

    # 1) segmentering
    segmenter = SilhouetteSegmenter(prefer_backend=prefer_backend, device=device)
    mask_f_raw, backend_f = segmenter(front_bgr)
    mask_s_raw, backend_s = segmenter(side_bgr)

    # 2) cleanup + normalisering
    mask_f_raw = _keep_largest_component(mask_f_raw)
    mask_s_raw = _keep_largest_component(mask_s_raw)

    front_norm, mask_f, _ = _crop_and_resize_to_height(front_bgr, mask_f_raw, target_h=target_height_px, margin_ratio=crop_margin_ratio)
    side_norm,  mask_s, _  = _crop_and_resize_to_height(side_bgr,  mask_s_raw,  target_h=target_height_px, margin_ratio=crop_margin_ratio)

    mask_f = _smooth_mask(mask_f)
    mask_s = _smooth_mask(mask_s)

    # 3) rækker (front; pose hvis muligt, ellers fallback)
    cx_f = _centroid_x(mask_f)
    cx_s = _centroid_x(mask_s)
    keep_for_rows = 0.82
    arm_r_by_key = {"chest":0.018, "waist":0.020, "hip":0.035, "thigh":0.018}
    halfw_default = 16
    halfw_hip = 24

    def TW_f(y:int, key:str) -> int:
        arm_r = arm_r_by_key.get(key, 0.018)
        halfw = halfw_hip if key == "hip" else halfw_default
        return _torso_width(mask_f, y, cx_f, keep_for_rows, halfw, arm_r)

    def TW_s(y:int, key:str) -> int:
        arm_r = arm_r_by_key.get(key, 0.018)
        halfw = halfw_hip if key == "hip" else halfw_default
        return _torso_width(mask_s, y, cx_s, keep_for_rows, halfw, arm_r)

    if HAS_MEDIAPIPE:
        try:
            rows = _rows_front(front_norm, mask_f, lambda y: TW_f(y, "hip"))
            rows_source = "pose+mask"
        except Exception:
            rows = measurement_rows_from_mask(mask_f)
            rows_source = "mask_fallback"
    else:
        rows = measurement_rows_from_mask(mask_f)
        rows_source = "mask_only"

    # 4) scale (px per cm)
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

    # 5) bredder (arm-suppression + center-clip)
    keep = {"chest":0.81, "waist":0.60, "hip":0.94, "thigh":0.80}

    def TW(mask, y, cx, key):
        arm_r = arm_r_by_key.get(key, 0.018)
        halfw = halfw_hip if key == "hip" else halfw_default
        m = _arm_suppressed(mask, arm_r)
        try:
            return width_at_row_centerclip_window(m, y, cx, keep[key], half_window=halfw)
        except Exception:
            return width_at_row_centerclip(m, y, cx, keep[key])

    widths_front = {
        "chest":    TW(mask_f, rows["chest"],    cx_f, "chest"),
        "waist":    TW(mask_f, rows["waist"],    cx_f, "waist"),
        "hip":      TW(mask_f, rows["hip"],      cx_f, "hip"),
        "shoulder": width_at_row(mask_f, rows["shoulder"]),
        "thigh":    TW(mask_f, rows["thigh"],    cx_f, "thigh"),
    }
    widths_side = {
        "chest": TW(mask_s, rows["chest"], cx_s, "chest"),
        "waist": TW(mask_s, rows["waist"], cx_s, "waist"),
        "hip":   TW(mask_s, rows["hip"],   cx_s, "hip"),
        "thigh": TW(mask_s, rows["thigh"], cx_s, "thigh"),
    }

    # 6) ellipse-omkredse
    semiaxes_cm, circumf_cm = {}, {}
    for k in ("chest","waist","hip","thigh"):
        a_cm = (widths_front[k] / 2.0) / ppc
        b_cm = (widths_side[k]  / 2.0) / ppc
        semiaxes_cm[k] = (float(a_cm), float(b_cm))
        circumf_cm[k]  = float(ellipse_circumference(a_cm, b_cm))

    # 7) længder
    shoulder_width_cm = float(widths_front["shoulder"] / ppc)
    _, y0m, _, y1m = mask_bbox(mask_f)
    inseam_cm = float(max(0, y1m - rows["crotch"]) / ppc)

    # 8) debug
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        if save_masks:
            _save(os.path.join(debug_dir, "front_mask.png"), mask_f)
            _save(os.path.join(debug_dir, "side_mask.png"),  mask_s)
            _save(os.path.join(debug_dir, "front_mask_torso.png"), _arm_suppressed(mask_f, arm_r_by_key["hip"]))
            _save(os.path.join(debug_dir, "side_mask_torso.png"),  _arm_suppressed(mask_s, arm_r_by_key["hip"]))
        _overlay(front_norm, mask_f, rows, os.path.join(debug_dir, "front_overlay.png"))
        _overlay(side_norm,  mask_s, rows, os.path.join(debug_dir, "side_overlay.png"))

    qa = {"normalization": {
            "target_h_px": int(target_height_px),
            "crop_margin_ratio": float(crop_margin_ratio),
            "ppc_used_px_per_cm": float(ppc),
            "ppc_source": ppc_source,
            "rows_source": rows_source,
         },
         "runtime_sec": round(time.time() - t0, 3)
    }

    if setup_save:
        _save_json(setup_save, {"ppc_ref": float(ppc)})

    return {
        "pixels_per_cm": float(ppc),
        "backend_front": backend_f,
        "backend_side":  backend_s,
        "rows_px": rows,
        "widths_px_front": widths_front,
        "widths_px_side":  widths_side,
        "semiaxes_cm": semiaxes_cm,
        "chest_cm":  circumf_cm["chest"],
        "waist_cm":  circumf_cm["waist"],
        "hip_cm":    circumf_cm["hip"],
        "thigh_cm":  circumf_cm["thigh"],
        "shoulder_width_cm": shoulder_width_cm,
        "inseam_cm": inseam_cm,
        "qa": qa,
    }
