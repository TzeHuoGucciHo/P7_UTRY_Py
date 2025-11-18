from __future__ import annotations

import os
import json
import time
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np

from .segmenter import SilhouetteSegmenter
from .pose import measurement_rows_from_mask
from .geometry import (
    ellipse_circumference,
    mask_bbox,
    width_at_row,
    width_at_row_centerclip_window,
)


# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------


def _ensure_gray(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def _segment_person(
    img_bgr: np.ndarray,
    prefer_backend: str = "deeplabv3",
    device: str = "cuda",
) -> Tuple[np.ndarray, str]:
    seg = SilhouetteSegmenter(prefer_backend=prefer_backend, device=device)
    mask_u8, backend = seg(img_bgr)
    return _ensure_gray(mask_u8), backend


def _crop_to_person(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    margin_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop omkring personens bbox med margin som fraktion af bbox.
    Returnerer (crop_img, crop_mask, (x0, y0, x1, y1) i original-coords).
    """
    x0, y0, x1, y1 = mask_bbox(mask_u8)
    h = y1 - y0 + 1
    w = x1 - x0 + 1

    margin_y = int(round(margin_ratio * h))
    margin_x = int(round(margin_ratio * w))

    H, W = mask_u8.shape[:2]
    x0c = max(0, x0 - margin_x)
    y0c = max(0, y0 - margin_y)
    x1c = min(W - 1, x1 + margin_x)
    y1c = min(H - 1, y1 + margin_y)

    crop_img = img_bgr[y0c : y1c + 1, x0c : x1c + 1]
    crop_mask = mask_u8[y0c : y1c + 1, x0c : x1c + 1]
    return crop_img, crop_mask, (x0c, y0c, x1c, y1c)


def _resize_to_target_height(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    target_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Skaler så højden af personens bbox bliver target_h pixels.
    """
    x0, y0, x1, y1 = mask_bbox(mask_u8)
    crop_h = max(1, y1 - y0 + 1)

    scale = float(target_h) / float(crop_h)
    new_w = max(1, int(round(img_bgr.shape[1] * scale)))
    new_h = max(1, int(round(img_bgr.shape[0] * scale)))

    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_resized, mask_resized


def _compute_pixels_per_cm(mask_u8: np.ndarray, height_cm: float) -> Tuple[float, int]:
    """
    Beregn pixels-per-cm ud fra vertikal bbox-udstrækning.
    Returnerer (px_per_cm, body_height_px).
    """
    if height_cm is None or height_cm <= 0:
        raise ValueError("height_cm skal angives og være > 0 for skalering.")

    x0, y0, x1, y1 = mask_bbox(mask_u8)
    body_height_px = max(1, y1 - y0 + 1)
    px_per_cm = body_height_px / float(height_cm)
    return px_per_cm, body_height_px


def _center_x_from_mask(mask_u8: np.ndarray) -> float:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return mask_u8.shape[1] * 0.5
    return float(xs.mean())


def _ellipse_circumference_from_widths(
    front_width_cm: float,
    side_width_cm: float,
) -> float:
    """
    Tværsnit modelleres som ellipse: front-bredde = én akse, side-bredde = anden.
    """
    if front_width_cm <= 0 or side_width_cm <= 0:
        return 0.0
    a = front_width_cm / 2.0
    b = side_width_cm / 2.0
    return float(ellipse_circumference(a, b))


def _overlay_debug(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    rows: Dict[str, int],
    out_path: str,
) -> None:
    vis = img_bgr.copy()
    tint = np.zeros_like(vis)
    tint[mask_u8 > 0] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 1.0, tint, 0.35, 0)

    colors = {
        "shoulder": (0, 255, 255),
        "chest": (0, 128, 255),
        "waist": (255, 0, 255),
        "hip": (255, 128, 0),
        "crotch": (180, 180, 0),
        "thigh": (0, 200, 100),
    }
    h, w = mask_u8.shape[:2]
    for key, y in rows.items():
        c = colors.get(key, (255, 255, 255))
        y_int = max(0, min(h - 1, int(y)))
        cv2.line(vis, (0, y_int), (w - 1, y_int), c, 2)
        cv2.putText(
            vis,
            key,
            (5, max(10, y_int - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            c,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(out_path, vis)


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -------------------------------------------------------------------------
# Main public API
# -------------------------------------------------------------------------


def compute(
    front_bgr: np.ndarray,
    side_bgr: np.ndarray,
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
    no_profile_scale: bool = False,  # pt. ikke brugt, men beholdes for kompatibilitet
    gender: str = "male",
) -> Dict:
    """
    Kernemåle-funktionen, som din CLI kalder.

    Returnerer bl.a.:
      - pixels_per_cm
      - backend_front, backend_side
      - rows_px
      - widths_px_front, widths_px_side
      - chest_cm, waist_cm, hip_cm, thigh_cm
      - shoulder_width_cm, chest_width_cm, inseam_cm
      - input_height_cm
      - qa (debug-info)
    """
    t_all = time.perf_counter()

    # 1) Segmentér person i begge views
    mask_front, backend_f = _segment_person(front_bgr, prefer_backend=prefer_backend, device=device)
    mask_side, backend_s = _segment_person(side_bgr, prefer_backend=prefer_backend, device=device)

    # 2) Crop omkring person med margin
    front_crop, mask_front_crop, bbox_front = _crop_to_person(front_bgr, mask_front, crop_margin_ratio)
    side_crop, mask_side_crop, bbox_side = _crop_to_person(side_bgr, mask_side, crop_margin_ratio)

    # 3) Normalisér højde i pixels (for stabile rækker)
    front_norm, mask_front_norm = _resize_to_target_height(front_crop, mask_front_crop, target_height_px)
    side_norm, mask_side_norm = _resize_to_target_height(side_crop, mask_side_crop, target_height_px)

    # 4) Pixels-per-cm
    if setup_load is not None and os.path.isfile(setup_load):
        try:
            with open(setup_load, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            ppc = float(cfg["ppc_ref"])
            _, y0_f, _, y1_f = bbox_front
            body_height_px = max(1, y1_f - y0_f + 1)
        except Exception:
            ppc, body_height_px = _compute_pixels_per_cm(mask_front_norm, height_cm or 170.0)
    else:
        if height_cm is None:
            raise ValueError("height_cm skal angives, hvis der ikke findes setup_load med ppc_ref.")
        ppc, body_height_px = _compute_pixels_per_cm(mask_front_norm, height_cm)

    if setup_save is not None:
        _save_json(setup_save, {"ppc_ref": float(ppc)})

    # 5) Række-positioner (chest, waist, hip, shoulder, crotch, thigh)
    rows = measurement_rows_from_mask(mask_front_norm)

    # 6) Width-parametre (let kønsspecifik tunings)
    gender = (gender or "male").lower()
    if gender not in ("male", "female"):
        gender = "male"

    if gender == "male":
        keep_ratios = {
            "chest": 0.84,
            "waist": 0.66,
            "hip": 0.94,
            "thigh": 0.79,
        }
    else:
        keep_ratios = {
            "chest": 0.80,
            "waist": 0.62,
            "hip": 0.96,
            "thigh": 0.77,
        }

    cx_front = _center_x_from_mask(mask_front_norm)
    cx_side = _center_x_from_mask(mask_side_norm)

    def TW(mask_u8: np.ndarray, key: str, row_y: int, cx: float) -> int:
        kr = float(keep_ratios.get(key, 0.8))
        return width_at_row_centerclip_window(
            mask_u8,
            row_y,
            center_x=cx,
            keep_ratio=kr,
            half_window=8,
        )

    # 7) Bredder i pixels (front/side)
    widths_front = {
        "chest":    TW(mask_front_norm, "chest", rows["chest"], cx_front),
        "waist":    TW(mask_front_norm, "waist", rows["waist"], cx_front),
        "hip":      TW(mask_front_norm, "hip", rows["hip"], cx_front),
        "thigh":    TW(mask_front_norm, "thigh", rows["thigh"], cx_front),
        "shoulder": width_at_row(mask_front_norm, rows["shoulder"]),
    }

    widths_side = {
        "chest": TW(mask_side_norm, "chest", rows["chest"], cx_side),
        "waist": TW(mask_side_norm, "waist", rows["waist"], cx_side),
        "hip":   TW(mask_side_norm, "hip", rows["hip"], cx_side),
        "thigh": TW(mask_side_norm, "thigh", rows["thigh"], cx_side),
    }

    # 8) Konverter bredder til cm
    widths_cm_front = {k: float(v) / ppc for k, v in widths_front.items()}
    widths_cm_side = {k: float(v) / ppc for k, v in widths_side.items()}

    chest_width_cm = widths_cm_front.get("chest", 0.0)
    shoulder_width_cm = widths_cm_front.get("shoulder", 0.0)

    # 9) Omkredse via ellipse-model
    circumf_cm: Dict[str, float] = {}
    semiaxes_cm: Dict[str, Tuple[float, float]] = {}

    for key in ("chest", "waist", "hip", "thigh"):
        fw = widths_cm_front.get(key, 0.0)
        sw = widths_cm_side.get(key, 0.0)
        if fw <= 0 or sw <= 0:
            c = 0.0
            a = b = 0.0
        else:
            a = fw / 2.0
            b = sw / 2.0
            c = ellipse_circumference(a, b)
        circumf_cm[key] = float(c)
        semiaxes_cm[key] = (float(a), float(b))

    # 10) Inseam (fra crotch-række til bund af maske i front)
    x0f, y0f, x1f, y1f = mask_bbox(mask_front_norm)
    crotch_y = rows.get("crotch", y1f)
    leg_px = max(0, int(y1f - crotch_y))
    inseam_cm = float(leg_px) / ppc

    # 11) QA/debug-info
    qa: Dict[str, Any] = {
        "body_height_px": body_height_px,
        "bbox_front": bbox_front,
        "bbox_side": bbox_side,
        "rows_px": rows,
        "widths_px_front": widths_front,
        "widths_px_side": widths_side,
        "widths_cm_front": widths_cm_front,
        "widths_cm_side": widths_cm_side,
        "gender": gender,
        "ppc": float(ppc),
    }

    if calibrate_many is not None:
        qa["calibrate_many"] = calibrate_many
    if aruco_mm is not None:
        qa["aruco_mm"] = aruco_mm

    # 12) Debug-billeder
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        if save_masks:
            cv2.imwrite(os.path.join(debug_dir, "front_mask.png"), mask_front_norm)
            cv2.imwrite(os.path.join(debug_dir, "side_mask.png"), mask_side_norm)
        _overlay_debug(front_norm, mask_front_norm, rows, os.path.join(debug_dir, "front_overlay.png"))
        _overlay_debug(side_norm, mask_side_norm, rows, os.path.join(debug_dir, "side_overlay.png"))

    t_all_end = time.perf_counter()
    qa["runtime_total_sec"] = float(t_all_end - t_all)

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
        "chest_width_cm":    chest_width_cm,   # <--- bryst-bredde i cm (front)
        "inseam_cm": inseam_cm,
        "input_height_cm": float(height_cm) if height_cm is not None else None,
        "qa": qa,
    }
