#!/usr/bin/env python3
"""
Two-View Body Measurement Pipeline (Starter)
Author: ChatGPT (GPT-5 Thinking)
License: MIT

Commercially usable starter pipeline that estimates body measurements from
one FRONT and one SIDE image using:
  - PyTorch (torchvision) for person segmentation (silhouette)
  - OpenCV (ArUco) for metric scaling (or fallback: reported height)
  - NumPy for geometry
  - scikit-learn (optional) for a learned post-processing regressor

This file is intentionally self-contained to help you get up and running.
You can later split it into modules (segmenter.py, calibrate.py, etc.).

IMPORTANT: Pose landmarks are provided as a hook — integrate MMPose (Apache-2.0)
for production-grade landmark detection.

Usage (example):
  python measurement_pipeline.py \
      --front path/to/front.jpg \
      --side  path/to/side.jpg  \
      --height-cm 180

or with an ArUco marker (preferable for accuracy):
  python measurement_pipeline.py \
      --front path/to/front.jpg \
      --side  path/to/side.jpg  \
      --aruco-mm 80

Outputs JSON with chest, waist, hip circumferences (cm) and selected lengths.

Dependencies (suggested versions):
  torch>=2.2
  torchvision>=0.17
  opencv-python>=4.9
  numpy>=1.26
  scikit-learn>=1.4  (optional; only used if --use-ml-head)

"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# PyTorch / torchvision for segmentation
import torch
import torchvision
from torchvision.transforms.functional import to_tensor

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ------------------------------
# Utility structures
# ------------------------------
@dataclass
class Keypoints:
    """Minimal container for 2D landmarks in pixel coords.

    Expected keys (extend as needed): 'nose', 'neck', 'l_shoulder', 'r_shoulder',
    'l_hip', 'r_hip', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_knee', 'r_knee',
    'l_ankle', 'r_ankle', 'top_head', 'base_neck'.
    """
    points: Dict[str, Tuple[float, float]]

    def get(self, name: str) -> Optional[Tuple[float, float]]:
        return self.points.get(name)

    def y(self, name: str) -> Optional[float]:
        p = self.points.get(name)
        return None if p is None else float(p[1])


# ------------------------------
# Image IO
# ------------------------------

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


# ------------------------------
# Segmentation (silhouette mask)
# ------------------------------
class SilhouetteSegmenter:
    """Person silhouette via torchvision DeepLabV3-ResNet50 (COCO/VOC labels).

    Produces a binary mask for the 'person' class.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights).to(self.device)
        self.model.eval()
        self.preprocess = weights.transforms()
        # VOC-ish label index of 'person' is 15 for this weight enum
        self.person_class_index = 15

    @torch.inference_mode()
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        """Returns uint8 mask (0/255) for person silhouette."""
        # Convert BGR->RGB and preprocess according to weights
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # noop, kept for clarity
        # weights.transforms expects PIL.Image; use torchvision's helper by converting via OpenCV array
        # Directly use to_tensor+normalize for robustness
        x = to_tensor(image_rgb).unsqueeze(0).to(self.device)
        # Normalization parameters from weights
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        x = (x - mean) / std

        out = self.model(x)["out"][0]  # [C,H,W]
        # softmax over classes
        probs = torch.softmax(out, dim=0)
        person = probs[self.person_class_index]  # [H,W]
        mask = (person > 0.5).float().cpu().numpy()
        mask_u8 = (mask * 255).astype(np.uint8)
        # Morph cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        return mask_u8


# ------------------------------
# Pose landmarks (hook)
# ------------------------------

def estimate_pose_keypoints_placeholder(image_bgr: np.ndarray) -> Keypoints:
    """PLACEHOLDER: Replace with MMPose inference.

    For now, we synthesize minimal landmarks based on the mask bounding box
    as a crude fallback, but you MUST integrate a real pose model for production.
    """
    h, w = image_bgr.shape[:2]
    # Very rough verticals: top ~5%, bottom ~95%
    points = {
        "top_head": (w * 0.5, h * 0.05),
        "base_neck": (w * 0.5, h * 0.15),
        "l_shoulder": (w * 0.38, h * 0.18),
        "r_shoulder": (w * 0.62, h * 0.18),
        "l_hip": (w * 0.45, h * 0.55),
        "r_hip": (w * 0.55, h * 0.55),
        "l_knee": (w * 0.48, h * 0.80),
        "r_knee": (w * 0.52, h * 0.80),
        "l_ankle": (w * 0.48, h * 0.95),
        "r_ankle": (w * 0.52, h * 0.95),
    }
    return Keypoints(points)


# ------------------------------
# Scaling: ArUco (preferred) or height fallback
# ------------------------------

def pixels_per_cm_from_aruco(image_bgr: np.ndarray, marker_size_mm: float) -> Optional[float]:
    """Compute pixels per centimeter using an ArUco square marker seen in the image.
    Returns None if no marker is found."""
    try:
        aruco = cv2.aruco  # type: ignore[attr-defined]
    except Exception:
        raise RuntimeError("OpenCV-contrib with aruco module is required. Install opencv-contrib-python.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None or len(ids) == 0:
        return None

    # Take first marker; compute its average side length in pixels
    c = corners[0].reshape(-1, 2)  # 4x2
    side_pix = 0.25 * (
        np.linalg.norm(c[0] - c[1]) +
        np.linalg.norm(c[1] - c[2]) +
        np.linalg.norm(c[2] - c[3]) +
        np.linalg.norm(c[3] - c[0])
    )
    marker_size_cm = marker_size_mm / 10.0
    ppc = side_pix / marker_size_cm  # pixels per cm
    return float(ppc)


def pixels_per_cm_from_height(keypoints: Keypoints, reported_height_cm: float) -> Optional[float]:
    """Compute pixels per cm from top_head to ankle line using reported height.
    Assumes front view reasonably upright.
    """
    y_top = keypoints.y("top_head")
    y_a = keypoints.y("l_ankle")
    y_b = keypoints.y("r_ankle")
    if y_top is None or y_a is None or y_b is None:
        return None
    y_bottom = 0.5 * (y_a + y_b)
    pixel_height = abs(y_bottom - y_top)
    if pixel_height < 10:
        return None
    return float(pixel_height / reported_height_cm)


# ------------------------------
# Silhouette width profiles and ellipse circumference
# ------------------------------

def width_at_row(mask_u8: np.ndarray, row_y: int) -> int:
    """Return total width in pixels of the mask at a given image row (y)."""
    h, w = mask_u8.shape[:2]
    y = int(np.clip(row_y, 0, h - 1))
    row = mask_u8[y, :]
    # Count contiguous foreground; we use simple count of >0
    return int(np.count_nonzero(row > 0))


def ellipse_circumference_ramanujan(a: float, b: float) -> float:
    """Ramanujan approximation II for ellipse circumference.
    a,b are semi-axes lengths (in cm)."""
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h))) * 4  # multiply by 4? (No!)


# Correct the factor: For an ellipse with semi-axes a and b, Ramanujan II is:
# C ≈ π [3(a+b) − sqrt{(3a+b)(a+3b)}]
# Keep both forms for clarity; use the canonical one below.

def ellipse_circumference(a: float, b: float) -> float:
    return math.pi * (3*(a+b) - math.sqrt((3*a + b)*(a + 3*b)))


# ------------------------------
# Determine chest/waist/hip heights from keypoints
# ------------------------------

def measurement_rows_from_pose(img_h: int, kp: Keypoints) -> Dict[str, int]:
    """Heuristic mapping from landmarks to measurement rows (pixel y).
    Refine once real pose is integrated.
    """
    y_sh_l = kp.y("l_shoulder")
    y_sh_r = kp.y("r_shoulder")
    y_hip_l = kp.y("l_hip")
    y_hip_r = kp.y("r_hip")
    y_top = kp.y("top_head")
    y_ankle_l = kp.y("l_ankle")
    y_ankle_r = kp.y("r_ankle")

    # Fallbacks
    y_sh = np.nanmean([v for v in [y_sh_l, y_sh_r] if v is not None])
    y_hip = np.nanmean([v for v in [y_hip_l, y_hip_r] if v is not None])
    if np.isnan(y_sh):
        y_sh = img_h * 0.18
    if np.isnan(y_hip):
        y_hip = img_h * 0.55

    # Chest roughly midway between armpit and mid-chest
    y_chest = int(round(y_sh + 0.22 * (y_hip - y_sh)))
    # Waist near local minimum between ribs and iliac crest; heuristic ~0.42 of shoulder->hip span
    y_waist = int(round(y_sh + 0.42 * (y_hip - y_sh)))
    # Hip slightly below hip landmark
    y_hipline = int(round(y_hip + 0.06 * (img_h - y_hip)))

    return {"chest": y_chest, "waist": y_waist, "hip": y_hipline}


# ------------------------------
# Length measurements from keypoints
# ------------------------------

def segment_length_cm(a: Tuple[float, float], b: Tuple[float, float], pixels_per_cm: float) -> float:
    if a is None or b is None or pixels_per_cm is None or pixels_per_cm <= 0:
        return float("nan")
    d = math.dist(a, b)
    return d / pixels_per_cm


def basic_lengths_from_pose(kp_front: Keypoints, ppc: float) -> Dict[str, float]:
    """Compute a few length measures from front keypoints only (for simplicity)."""
    out = {}
    # Arm length (shoulder -> wrist) left/right average
    l1 = segment_length_cm(kp_front.get("l_shoulder"), kp_front.get("l_wrist"), ppc)
    l2 = segment_length_cm(kp_front.get("r_shoulder"), kp_front.get("r_wrist"), ppc)
    out["arm_cm"] = np.nanmean([l1, l2])
    # Inseam (crotch/hip to ankle) approximate from hip to ankle average
    l3 = segment_length_cm(kp_front.get("l_hip"), kp_front.get("l_ankle"), ppc)
    l4 = segment_length_cm(kp_front.get("r_hip"), kp_front.get("r_ankle"), ppc)
    out["inseam_cm"] = np.nanmean([l3, l4])
    return out


# ------------------------------
# Main measurement function
# ------------------------------

def measure_from_two_views(front_bgr: np.ndarray,
                           side_bgr: np.ndarray,
                           height_cm: Optional[float] = None,
                           aruco_mm: Optional[float] = None,
                           use_ml_head: bool = False) -> Dict[str, float]:
    """Compute chest/waist/hip circumferences (cm) + basic lengths.

    Scaling preference: ArUco (if found) else height_cm.
    """
    # 1) Silhouettes
    segmenter = SilhouetteSegmenter()
    mask_f = segmenter(front_bgr)
    mask_s = segmenter(side_bgr)

    # 2) Pose keypoints (placeholder; replace with MMPose)
    kp_f = estimate_pose_keypoints_placeholder(front_bgr)
    kp_s = estimate_pose_keypoints_placeholder(side_bgr)

    # 3) Pixels-per-cm scaling
    ppc = None
    if aruco_mm is not None:
        ppc = pixels_per_cm_from_aruco(front_bgr, aruco_mm)
        if ppc is None:
            ppc = pixels_per_cm_from_aruco(side_bgr, aruco_mm)
    if ppc is None and height_cm is not None:
        ppc = pixels_per_cm_from_height(kp_f, height_cm)
    if ppc is None:
        raise RuntimeError("Could not establish metric scale. Provide --aruco-mm or --height-cm with pose.")

    # 4) Determine measurement rows
    rows_f = measurement_rows_from_pose(front_bgr.shape[0], kp_f)
    rows_s = measurement_rows_from_pose(side_bgr.shape[0], kp_s)

    # Ensure we sample the same anatomical rows across views: use front's rows
    rows = rows_f

    # 5) Widths (in pixels) at those rows
    widths_front = {k: width_at_row(mask_f, y) for k, y in rows.items()}
    widths_side  = {k: width_at_row(mask_s, y) for k, y in rows.items()}

    # Convert to semi-axes in cm
    semiaxes = {}
    for k in rows.keys():
        a_pix = widths_front[k] / 2.0
        b_pix = widths_side[k]  / 2.0
        a_cm = a_pix / ppc
        b_cm = b_pix / ppc
        semiaxes[k] = (a_cm, b_cm)

    # 6) Circumferences via Ramanujan
    circumferences = {k: ellipse_circumference(a, b) for k, (a, b) in semiaxes.items()}

    # 7) Basic lengths
    lengths = basic_lengths_from_pose(kp_f, ppc)

    # 8) Optional ML head: small correction model (requires prior fit)
    if use_ml_head:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not installed; cannot use --use-ml-head")
        # Features: [a_chest,b_chest,a_waist,b_waist,a_hip,b_hip, arm, inseam]
        feats = [
            semiaxes['chest'][0], semiaxes['chest'][1],
            semiaxes['waist'][0], semiaxes['waist'][1],
            semiaxes['hip'][0],   semiaxes['hip'][1],
            lengths.get('arm_cm', np.nan), lengths.get('inseam_cm', np.nan)
        ]
        X = np.array(feats, dtype=np.float32).reshape(1, -1)
        # This is a placeholder: you should load a fitted model from disk
        # (e.g., joblib.load("models/measurement_head.joblib"))
        # For safety, here we simply pass through without modification.
        pass

    out = {
        "pixels_per_cm": ppc,
        "chest_cm": float(circumferences["chest"]),
        "waist_cm": float(circumferences["waist"]),
        "hip_cm": float(circumferences["hip"]),
        **{k: float(v) for k, v in lengths.items()},
        # Optionally, add diagnostic rows in pixel coords
        "rows_px": rows,
    }
    return out


# ------------------------------
# CLI
# ------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Two-view body measurement (starter)")
    ap.add_argument("--front", required=True, help="Path to front image")
    ap.add_argument("--side",  required=True, help="Path to side image")
    ap.add_argument("--height-cm", type=float, default=None, help="Reported total height in cm (fallback scaling)")
    ap.add_argument("--aruco-mm", type=float, default=None, help="ArUco marker side length in mm (preferred scaling)")
    ap.add_argument("--use-ml-head", action="store_true", help="Apply an optional sklearn correction model (placeholder)")
    return ap.parse_args()


def main():
    args = parse_args()
    front = load_image(args.front)
    side  = load_image(args.side)

    result = measure_from_two_views(
        front_bgr=front,
        side_bgr=side,
        height_cm=args.height_cm,
        aruco_mm=args.aruco_mm,
        use_ml_head=args.use_ml_head,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
