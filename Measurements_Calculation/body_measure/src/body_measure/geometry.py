from __future__ import annotations
import math
import numpy as np

def ellipse_circumference(a: float, b: float) -> float:
    """Ramanujan II approximation of ellipse circumference."""
    if a <= 0 or b <= 0: return 0.0
    return math.pi * (3*(a+b) - math.sqrt((3*a + b)*(a + 3*b)))

def mask_bbox(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        h, w = mask_u8.shape[:2]
        return 0, 0, w-1, h-1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def width_at_row(mask_u8, row_y: int) -> int:
    """Full width at row measured as span (max−min+1)."""
    h, w = mask_u8.shape[:2]
    y = int(max(0, min(h-1, row_y)))
    xs = np.where(mask_u8[y, :] > 0)[0]
    if xs.size < 2:
        return int(xs.size)
    return int(xs.max() - xs.min() + 1)

def width_at_row_centerclip(mask_u8, row_y: int, center_x: float, keep_ratio: float) -> int:
    """Width within a center-clip window (span)."""
    h, w = mask_u8.shape[:2]
    y = int(max(0, min(h-1, row_y)))
    xs_all = np.where(mask_u8[y, :] > 0)[0]
    if xs_all.size == 0: return 0
    left, right = xs_all.min(), xs_all.max()
    half = (right - left) / 2.0
    cx = float(center_x)
    new_half = max(0.0, half * float(keep_ratio))
    x0 = int(max(0, min(w-1, cx - new_half)))
    x1 = int(max(0, min(w-1, cx + new_half)))
    xs = xs_all[(xs_all >= x0) & (xs_all <= x1)]
    if xs.size < 2: return int(xs.size)
    return int(xs.max() - xs.min() + 1)

def width_at_row_centerclip_window(mask_u8, row_y: int, center_x: float, keep_ratio: float, half_window: int = 8) -> int:
    """Median of span widths in a vertical window around row_y."""
    h = mask_u8.shape[0]
    y0 = max(0, row_y - half_window)
    y1 = min(h - 1, row_y + half_window)
    vals = [width_at_row_centerclip(mask_u8, y, center_x, keep_ratio) for y in range(y0, y1 + 1)]
    if not vals: return 0
    vals.sort()
    mid = len(vals)//2
    return vals[mid] if len(vals)%2==1 else int(round(0.5*(vals[mid-1]+vals[mid])))
