from __future__ import annotations
from typing import Dict
import numpy as np
from .geometry import mask_bbox

def measurement_rows_from_mask(mask_u8) -> Dict[str, int]:
    """
    Udled rækker relativt til silhuettens bbox (robust mod headroom/crop).
    Returnerer y-pixels for: chest, waist, hip; og ekstra: shoulder, crotch, thigh.
    """
    x0, y0, x1, y1 = mask_bbox(mask_u8)
    top, bottom = y0, y1
    H = max(1, bottom - top)

    y_shoulder = top + int(0.18 * H)
    y_hip      = top + int(0.55 * H)
    y_chest    = int(round(y_shoulder + 0.22 * (y_hip - y_shoulder)))
    y_waist    = int(round(y_shoulder + 0.42 * (y_hip - y_shoulder)))
    y_hipline  = int(round(y_hip + 0.06 * (bottom - y_hip)))

    # ekstra
    y_crotch   = y_hipline + int(0.02 * H)
    y_thigh    = y_crotch + int(0.08 * H)

    return {"chest": y_chest, "waist": y_waist, "hip": y_hipline, "shoulder": y_shoulder, "crotch": y_crotch, "thigh": y_thigh}
