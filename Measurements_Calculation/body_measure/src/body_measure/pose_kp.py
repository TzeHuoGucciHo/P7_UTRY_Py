from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import cv2
import mediapipe as mp

_LM = mp.solutions.pose.PoseLandmark

def _mp_landmarks(img_bgr: np.ndarray) -> Dict[str, Tuple[float,float,float]]:
    # MediaPipe forventer RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        res = pose.process(img_rgb)
    out = {}
    if not res.pose_landmarks:
        return out
    h, w = img_bgr.shape[:2]
    for lm in _LM:
        p = res.pose_landmarks.landmark[lm]
        out[lm.name.lower()] = (p.x * w, p.y * h, p.visibility)
    return out

def _avg_y(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> Optional[float]:
    if a is None or b is None: return None
    if a[2] < 0.3 or b[2] < 0.3: return None
    return (a[1] + b[1]) / 2.0

def rows_from_pose_and_mask(img_bgr: np.ndarray, mask_u8: np.ndarray, width_fn) -> Dict[str, int]:
    """
    Returnerer rækker (y) for shoulder, chest, waist, hip, crotch, thigh.
    width_fn(y) -> torso-bredde i px (brug center-clip + arm-undertrykkelse i målemodulet).
    Fallbacks hvis enkelte landemærker mangler: vi begrænser båndene og bruger breddekurver.
    """
    h, w = mask_u8.shape[:2]
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        raise RuntimeError("Empty mask.")
    y0, y1 = ys.min(), ys.max()
    H = max(1, y1 - y0)

    lm = _mp_landmarks(img_bgr)
    ls, rs = lm.get("left_shoulder"), lm.get("right_shoulder")
    lh, rh = lm.get("left_hip"), lm.get("right_hip")
    lk, rk = lm.get("left_knee"), lm.get("right_knee")

    rows: Dict[str,int] = {}

    # anchors
    y_sh = _avg_y(ls, rs)
    y_hp = _avg_y(lh, rh)
    y_kn = _avg_y(lk, rk)

    # SHOULDER
    if y_sh is None:
        # fallback: lav gradient i 18–25% af H
        band = range(int(y0 + 0.18*H), int(y0 + 0.25*H))
        ws = np.array([width_fn(y) for y in band], np.float32)
        gy = band[int(np.argmin(np.abs(np.gradient(ws))))] if len(ws) > 1 else int(y0 + 0.22*H)
        rows["shoulder"] = int(gy)
    else:
        rows["shoulder"] = int(np.clip(y_sh, y0 + 0.15*H, y0 + 0.30*H))

    # HIP (direkte fra skelet hvis muligt)
    if y_hp is not None:
        rows["hip"] = int(np.clip(y_hp, y0 + 0.55*H, y0 + 0.88*H))
    else:
        # fallback: max bredde i 62–86%
        band = range(int(y0 + 0.62*H), int(y0 + 0.86*H))
        ws = np.array([width_fn(y) for y in band], np.int32)
        rows["hip"] = band[int(np.argmax(ws))] if len(ws) else int(y0 + 0.74*H)

    # WAIST: min bredde i båndet mellem shoulder og hip, snævert
    w0 = int(rows["shoulder"] + 0.35 * (rows["hip"] - rows["shoulder"]))
    w1 = int(rows["shoulder"] + 0.70 * (rows["hip"] - rows["shoulder"]))
    if w1 <= w0:
        w0, w1 = int(y0 + 0.48*H), int(y0 + 0.64*H)
    band = range(max(y0, w0), min(y1, w1)+1)
    ws = np.array([width_fn(y) for y in band], np.int32)
    if len(ws) >= 5:
        k=5; k2=k//2; sm=ws.copy()
        for i in range(len(ws)):
            j0=max(0,i-k2); j1=min(len(ws)-1,i+k2)
            blk=np.sort(ws[j0:j1+1]); m=len(blk)//2
            sm[i]=blk[m] if len(blk)%2==1 else int(round((blk[m-1]+blk[m])/2))
        rows["waist"] = band[int(np.argmin(sm))]
    else:
        rows["waist"] = band[int(np.argmin(ws))] if len(ws) else int(y0 + 0.56*H)

    # CROTCH: min i [hip + 6%H, hip + 20%H], men aldrig under knæ-ankeret
    c0 = rows["hip"] + int(0.06*H)
    c1 = rows["hip"] + int(0.20*H)
    if y_kn is not None:
        c1 = min(c1, int(y_kn - 0.10*H))  # hold godt over knæ
    band = range(int(np.clip(c0, y0, y1)), int(np.clip(c1, y0, y1))+1)
    ws = np.array([width_fn(y) for y in band], np.int32)
    rows["crotch"] = band[int(np.argmin(ws))] if len(ws) else int(rows["hip"] + 0.12*H)

    # CHEST: første prominente top i [shoulder + 5%, waist − 10%]
    c0 = rows["shoulder"] + int(0.05*(rows["hip"]-rows["shoulder"]))
    c1 = rows["waist"]    - int(0.10*(rows["waist"]-rows["shoulder"]))
    c0 = int(np.clip(c0, y0, y1)); c1 = int(np.clip(c1, y0, y1))
    if c1 <= c0: c1 = c0 + 1
    band = list(range(c0, c1+1))
    curve = np.array([width_fn(y) for y in band], np.float32)
    if len(curve):
        # mild median
        if len(curve)>=5:
            k=5; k2=k//2; sm=curve.copy()
            for i in range(len(curve)):
                j0=max(0,i-k2); j1=min(len(curve)-1,i+k2)
                sm[i]=np.median(curve[j0:j1+1])
        else:
            sm=curve
        grad=np.gradient(sm) if len(sm)>1 else np.array([0],np.float32)
        thr=np.percentile(sm,70)
        idx=None
        for i in range(1,len(sm)-1):
            if sm[i]>=thr and sm[i]>=sm[i-1] and sm[i]>=sm[i+1] and grad[i]>0:
                idx=i; break
        if idx is None: idx=int(np.argmax(sm))
        rows["chest"]= band[idx]
    else:
        rows["chest"]= int(rows["shoulder"] + 0.25*(rows["hip"]-rows["shoulder"]))

    # THIGH: 10 %H under crotch
    rows["thigh"] = int(min(y1, rows["crotch"] + 0.10*H))
    return rows
