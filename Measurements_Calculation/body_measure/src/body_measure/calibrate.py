from __future__ import annotations
import cv2, numpy as np
from typing import Optional, Tuple


def _detect_aruco(gray: np.ndarray, dict_id=int(cv2.aruco.DICT_4X4_50)) -> Tuple[list, Optional[np.ndarray]]:
    """
    Kompatibel ArUco-detektion for både ældre og nye OpenCV-API'er.
    Returnerer (corners, ids) hvor corners er liste af (4,1,2)-arrays.
    """
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(dict_id)

    # Ny API (4.7+): ArucoDetector + DetectorParameters()
    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids

    # Gammel API: detectMarkers + DetectorParameters_create()
    if hasattr(aruco, "DetectorParameters_create"):
        params = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
        return corners, ids

    # Hvis ingen af delene findes
    return [], None


def pixels_per_cm_from_aruco(image_bgr, marker_size_mm: float) -> Optional[float]:
    """
    Udleder pixels/cm ud fra første fundne ArUco-marker.
    Forventer marker-sidelængde i millimeter (fx 80 = 8 cm).
    """
    if not hasattr(cv2, "aruco"):
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids = _detect_aruco(gray)
    if ids is None or len(ids) == 0:
        return None

    # Brug første marker
    c = corners[0].reshape(-1, 2)  # (4,2), hjørner i rækkefølge
    # gennemsnitlig sidelængde i pixels (robust mod lidt rotation)
    side_pix = 0.25 * (
        np.linalg.norm(c[0] - c[1]) +
        np.linalg.norm(c[1] - c[2]) +
        np.linalg.norm(c[2] - c[3]) +
        np.linalg.norm(c[3] - c[0])
    )
    # marker_size_mm → cm
    marker_size_cm = float(marker_size_mm) / 10.0
    if marker_size_cm <= 0 or side_pix <= 0:
        return None
    return float(side_pix / marker_size_cm)  # px per cm


def pixels_per_cm_from_height_bbox(mask_u8, reported_height_cm: float) -> Optional[float]:
    """Fallback: brug maskens bbox-højde som pixel-højde for personen."""
    from .geometry import mask_bbox
    y0 = mask_bbox(mask_u8)[1]
    y1 = mask_bbox(mask_u8)[3]
    px_h = max(0, y1 - y0)
    if px_h < 10 or not reported_height_cm:
        return None
    return float(px_h / float(reported_height_cm))
