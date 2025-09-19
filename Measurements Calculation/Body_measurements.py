#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Body measurements (front + side) with torso-only mask in front (robust arm removal).
Commercial-safe stack: OpenCV (BSD), MediaPipe (Apache-2.0), NumPy (BSD).

Kørsel (kendt højde):
  python Body_measurements.py --front Front.png --side Side.png --height-cm 180 --save-debug

Kalibrering med ArUco (DICT_4X4_50, marker 5.0 cm):
  python Body_measurements.py --front Front.png --side Side.png --calib aruco --marker-cm 5.0 --save-debug

Kalibrering med chessboard (indre hjørner 7x10, 25 mm firkanter):
  python Body_measurements.py --front Front.png --side Side.png --calib chessboard --cb-cols 7 --cb-rows 10 --square-mm 25 --save-debug
"""

import argparse, json
from math import sqrt, pi
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp

# ---------------- MediaPipe ----------------
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation
POSE_LM = mp_pose.PoseLandmark

# ---------------- Farver (BGR) -------------
CLR_POINTS=(40,220,255); CLR_SHOULDER=(0,165,255)
CLR_CHEST=(0,255,0); CLR_WAIST=(0,140,70); CLR_BELLY=(255,0,255); CLR_HIP=(255,0,0); CLR_HEAD=(180,105,255)
CLR_GUIDE=(120,120,120); CLR_TEXT=(255,255,255)

# ---------------- Hjælpefunktioner ----------
def detect_pose_keypoints(img_bgr: np.ndarray):
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
    if not res.pose_landmarks:
        return None, None
    h, w = img_bgr.shape[:2]
    pts = {}
    for lm in POSE_LM:
        p = res.pose_landmarks.landmark[lm]
        pts[lm.name] = (int(round(p.x*w)), int(round(p.y*h)), float(p.visibility))
    return pts, res

def person_mask(img_bgr: np.ndarray, thresh: float=0.5) -> np.ndarray:
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = seg.process(rgb)
    m = (res.segmentation_mask >= thresh).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, 1)
    return m

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Behold kun største connected component (fjerner støj/udløbere)."""
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(m)
    if num <= 2:
        return m
    areas = [(labels == i).sum() for i in range(1, num)]
    i_max = 1 + int(np.argmax(areas))
    return (labels == i_max).astype(np.uint8)

def top_bottom_from_mask(mask: np.ndarray) -> Tuple[int,int]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise RuntimeError("Segmentation mask tom.")
    return int(ys.min()), int(ys.max())

def silhouette_width_at_y(mask: np.ndarray, y: int) -> Tuple[int,int,int]:
    h,w=mask.shape[:2]
    y = int(np.clip(y, 0, h-1))
    row = (mask[y,:] > 0)
    xs = np.where(row)[0]
    if xs.size < 2: return 0,-1,-1
    return int(xs.max()-xs.min()), int(xs.min()), int(xs.max())

def main_interval_at_y(mask: np.ndarray, y: int, center_x: int, max_gap_px: int = 8):
    """
    Vælg den sammenhængende True-run i række y, der indeholder center_x (eller nærmest center_x).
    Bridger små huller (<= max_gap_px) for at undgå at tøjtekstur splitter run.
    Returnerer (x0, x1) eller (-1,-1).
    """
    h, w = mask.shape[:2]
    y = int(np.clip(y, 0, h-1))
    row = (mask[y, :] > 0).astype(np.uint8)

    if max_gap_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max_gap_px, 1))
        row = (cv2.morphologyEx(row[None, None, :], cv2.MORPH_CLOSE, k)[0,0,:] > 0).astype(np.uint8)

    edges = np.diff(np.concatenate(([0], row, [0])))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0] - 1
    if len(starts) == 0:
        return -1, -1

    center_x = int(np.clip(center_x, 0, w-1))
    idx = None
    for i,(a,b) in enumerate(zip(starts, ends)):
        if a <= center_x <= b:
            idx = i; break
    if idx is None:
        dists = [abs((a+b)//2 - center_x) for a,b in zip(starts, ends)]
        idx = int(np.argmin(dists))
    return int(starts[idx]), int(ends[idx])

def build_torso_mask(mask_front_raw: np.ndarray, center_x: int, y_top: int, y_bot: int, pad_px: int = 2) -> np.ndarray:
    """
    Byg en 'torso only'-maske mellem y_top..y_bot ved at vælge run omkring center_x pr. række.
    pad_px udvider let for ikke at 'bide' i torso-kanten.
    """
    h, w = mask_front_raw.shape[:2]
    m = np.zeros_like(mask_front_raw, dtype=np.uint8)
    y0 = int(np.clip(min(y_top, y_bot), 0, h-1))
    y1 = int(np.clip(max(y_top, y_bot), 0, h-1))
    for y in range(y0, y1+1):
        x0, x1 = main_interval_at_y(mask_front_raw, y, center_x, max_gap_px=8)
        if x0 >= 0:
            x0p = max(0, x0 - pad_px)
            x1p = min(w-1, x1 + pad_px)
            m[y, x0p:x1p+1] = 1
    # lille glatning
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 1)
    return keep_largest_component(m)

def ellipse_circumference_cm(width_cm: float, depth_cm: float) -> float:
    a = max(width_cm, 0.0)/2.0; b = max(depth_cm, 0.0)/2.0
    if a==0 or b==0: return 0.0
    return float(pi*(3*(a+b) - sqrt((3*a+b)*(a+3*b))))

def dist_cm(p1, p2, ppc: float) -> float:
    return float(np.linalg.norm(np.array(p1[:2], float) - np.array(p2[:2], float)) / max(ppc, 1e-6))

def map_y(front_y:int, top_f:int, bot_f:int, top_s:int, bot_s:int)->int:
    denom=max(bot_f-top_f,1); r=(front_y-top_f)/denom
    return int(round(top_s + r*(bot_s-top_s)))

def safe_pick(kp:Dict[str,Tuple[int,int,float]], ln:str, rn:str):
    lp=kp.get(ln); rp=kp.get(rn)
    if lp is None and rp is None: raise RuntimeError(f"Mangler {ln} og {rn}")
    if lp is None: return rp
    if rp is None: return lp
    return lp if lp[2] >= rp[2] else rp

def draw_point(img,p,color,r=3): cv2.circle(img,(int(p[0]),int(p[1])),r,color,-1,cv2.LINE_AA)

def put_label(img,text,org,color=(255,255,255),alpha=0.6,pad=4):
    font=cv2.FONT_HERSHEY_SIMPLEX; scale=0.55; thick=1
    (tw,th),_=cv2.getTextSize(text,font,scale,thick)
    x,y=org; x0,y0,x1,y1=x-pad, y-th-pad, x+tw+pad, y+pad
    x0=max(0,x0); y0=max(0,y0)
    overlay=img.copy()
    cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,0,0),-1)
    cv2.addWeighted(overlay,alpha,img,1-alpha,0,img)
    cv2.putText(img,text,(x,y),font,scale,color,thick,cv2.LINE_AA)

def draw_scanline(img,y,x0,x1,color,label="",label_y=None):
    cv2.line(img,(x0,y),(x1,y),color,2)
    if label:
        yy = y-10 if label_y is None else label_y
        put_label(img,label,(x0,yy),color)

# --------- Skala/kalibrering ----------
def pixels_per_cm_known_height(mask: np.ndarray, known_height_cm: float)->float:
    top,bot = top_bottom_from_mask(mask)
    return (bot-top)/max(known_height_cm,1e-6)

def pixels_per_cm_aruco(img_bgr: np.ndarray, marker_cm: float)->float:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco kræver opencv-contrib-python.")
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(adict, params)
    corners, ids, _ = detector.detectMarkers(img_bgr)
    if ids is None or len(ids)==0: raise RuntimeError("ArUco ikke fundet.")
    c = corners[0].reshape(-1,2)
    side_px = np.mean([np.linalg.norm(c[i]-c[(i+1)%4]) for i in range(4)])
    return side_px / marker_cm

def pixels_per_cm_chessboard(img_bgr: np.ndarray, cols:int, rows:int, square_mm: float)->float:
    gray=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (cols,rows), None)
    if not ret: raise RuntimeError("Chessboard ikke fundet.")
    corners = corners.squeeze(1)
    pts = []
    for r in range(rows):
        for c in range(cols-1):
            i=r*cols+c; j=r*cols+c+1
            pts.append(np.linalg.norm(corners[i]-corners[j]))
    for r in range(rows-1):
        for c in range(cols):
            i=r*cols+c; j=(r+1)*cols+c
            pts.append(np.linalg.norm(corners[i]-corners[j]))
    px_per_square = np.median(pts)
    return px_per_square / (square_mm/10.0)  # mm -> cm

# ----------- Side-dybde stabilisering ----------
def stabilized_depth_cm(mask_side: np.ndarray, y_center:int, window:int, ppc_side:float,
                        width_cm_at_front: float, min_depth_ratio:float, min_depth_cm:float):
    h=mask_side.shape[0]; half=max(int(window//2),1)
    ys=range(max(0,y_center-half), min(h-1,y_center+half)+1)
    cands=[]
    for y in ys:
        wpx,x0,x1 = silhouette_width_at_y(mask_side, y)
        cands.append((y,wpx,x0,x1))
    if not cands:
        wpx,x0,x1=silhouette_width_at_y(mask_side, y_center)
        return wpx/max(ppc_side,1e-6), (y_center,x0,x1)
    widths=[c[1] for c in cands]
    med_w=int(np.median(widths))
    y_used,wpx,x0,x1=min(cands, key=lambda c: abs(c[1]-med_w))
    depth_cm = wpx/max(ppc_side,1e-6)
    min_allowed = max(min_depth_ratio*max(width_cm_at_front,1e-6), min_depth_cm)
    if depth_cm < min_allowed: depth_cm = float(min_allowed)
    return depth_cm, (y_used,x0,x1)

# ----------------- Hoved-beregning ----------------
def compute(front_path:str, side_path:str,
            height_cm: Optional[float],
            calib: Optional[str],
            marker_cm: Optional[float],
            cb_cols: Optional[int], cb_rows: Optional[int], square_mm: Optional[float],
            save_debug: bool, ellipse_corr: float,
            depth_window:int, min_depth_ratio:float, min_depth_cm:float,
            label_spacing:int)->Dict[str,float]:

    # indlæs
    front=cv2.imread(front_path); side=cv2.imread(side_path)
    if front is None: raise FileNotFoundError(front_path)
    if side  is None: raise FileNotFoundError(side_path)
    front_vis=front.copy(); side_vis=side.copy()

    # keypoints
    kp_f,_=detect_pose_keypoints(front); kp_s,_=detect_pose_keypoints(side)
    if kp_f is None or kp_s is None: raise RuntimeError("Pose ikke fundet i begge billeder.")

    # masker (største komponent)
    mask_f_raw = keep_largest_component(person_mask(front))
    mask_s_raw = keep_largest_component(person_mask(side))

    # side: glat horisontalt (luk små hak)
    kx = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    mask_s = cv2.morphologyEx(mask_s_raw, cv2.MORPH_CLOSE, kx, 1)

    # skala
    if height_cm is not None:
        ppc_front = pixels_per_cm_known_height(mask_f_raw, height_cm)
        ppc_side  = pixels_per_cm_known_height(mask_s_raw, height_cm)
        total_height_cm = float(height_cm)
    else:
        if calib == "aruco":
            ppc_front = pixels_per_cm_aruco(front, marker_cm)
            ppc_side  = pixels_per_cm_aruco(side,  marker_cm)
            top_f_tmp,bot_f_tmp = top_bottom_from_mask(mask_f_raw)
            total_height_cm = (bot_f_tmp-top_f_tmp)/ppc_front
        elif calib == "chessboard":
            ppc_front = pixels_per_cm_chessboard(front, cb_cols, cb_rows, square_mm)
            ppc_side  = pixels_per_cm_chessboard(side,  cb_cols, cb_rows, square_mm)
            top_f_tmp,bot_f_tmp = top_bottom_from_mask(mask_f_raw)
            total_height_cm = (bot_f_tmp-top_f_tmp)/ppc_front
        else:
            raise ValueError("Angiv enten --height-cm eller --calib [aruco|chessboard].")

    # top/bund
    top_f,bot_f = top_bottom_from_mask(mask_f_raw)
    top_s,bot_s = top_bottom_from_mask(mask_s)

    # nøglepunkter (front)
    L_SH=kp_f["LEFT_SHOULDER"]; R_SH=kp_f["RIGHT_SHOULDER"]
    L_HIP=kp_f["LEFT_HIP"];     R_HIP=kp_f["RIGHT_HIP"]
    L_KNEE=kp_f["LEFT_KNEE"];   R_KNEE=kp_f["RIGHT_KNEE"]
    L_ELB=kp_f["LEFT_ELBOW"];   R_ELB=kp_f["RIGHT_ELBOW"]
    L_WR =kp_f["LEFT_WRIST"];   R_WR =kp_f["RIGHT_WRIST"]

    # torso-center i front
    center_x_front = int(round((L_SH[0] + R_SH[0]) / 2))

    # torso-only mask (front) for at fjerne arme
    torso_mask = build_torso_mask(mask_f_raw, center_x_front, top_f, bot_f, pad_px=2)

    # Anatomiske y-niveauer (front)
    y_shoulder = int(round((L_SH[1]+R_SH[1])/2))
    y_hip_base = int(round((L_HIP[1]+R_HIP[1])/2))

    y_chest = int(round(y_shoulder + 0.18*(y_hip_base - y_shoulder)))
    # hofte: lokalt maks ±20 px omkring landmark-bånd (brug torso_mask)
    band = 20
    cand = [(y, silhouette_width_at_y(torso_mask,y)[0]) for y in range(max(0,y_hip_base-band), min(torso_mask.shape[0]-1,y_hip_base+band))]
    y_hip = max(cand, key=lambda t:t[1])[0] if cand else y_hip_base
    # talje: lokalt minimum mellem bryst og hofte
    lo,hi = sorted([y_chest+6, y_hip-6])
    cand = [(y, silhouette_width_at_y(torso_mask,y)[0]) for y in range(lo, hi, 2)]
    y_waist = min(cand, key=lambda t:t[1])[0] if cand else int(round(0.85*y_hip + 0.15*y_shoulder))
    # belly: lokalt maksimum mellem bryst og talje
    lo,hi = sorted([y_chest+4, y_waist-4])
    cand = [(y, silhouette_width_at_y(torso_mask,y)[0]) for y in range(lo, hi, 2)]
    y_belly = max(cand, key=lambda t:t[1])[0] if cand else y_waist

    # FRONT bredder (målt på torso_mask via main_interval_at_y)
    def width_on_torso(y):
        x0,x1 = main_interval_at_y(torso_mask, y, center_x_front, max_gap_px=8)
        return max(0, x1-x0), x0, x1

    chest_w_px, cx0, cx1 = width_on_torso(y_chest)
    waist_w_px, wx0, wx1 = width_on_torso(y_waist)
    belly_w_px, bx0, bx1 = width_on_torso(y_belly)
    hip_w_px,   hx0, hx1 = width_on_torso(y_hip)

    shoulder_width_cm = dist_cm(L_SH,R_SH,ppc_front)
    chest_width_cm = chest_w_px/max(ppc_front,1e-6)
    waist_width_cm = waist_w_px/max(ppc_front,1e-6)
    belly_width_cm = belly_w_px/max(ppc_front,1e-6)
    hip_width_cm   = hip_w_px  /max(ppc_front,1e-6)

    # SIDE dybder (stabiliseret, på mask_s)
    y_chest_s=map_y(y_chest,top_f,bot_f,top_s,bot_s)
    y_waist_s=map_y(y_waist,top_f,bot_f,top_s,bot_s)
    y_belly_s=map_y(y_belly,top_f,bot_f,top_s,bot_s)
    y_hip_s  =map_y(y_hip,  top_f,bot_f,top_s,bot_s)

    chest_depth_cm,(yc_used,cx0s,cx1s) = stabilized_depth_cm(mask_s,y_chest_s,depth_window,ppc_side,chest_width_cm,min_depth_ratio,min_depth_cm)
    waist_depth_cm,(yw_used,wx0s,wx1s) = stabilized_depth_cm(mask_s,y_waist_s,depth_window,ppc_side,waist_width_cm,min_depth_ratio,min_depth_cm)
    belly_depth_cm,(yb_used,bx0s,bx1s) = stabilized_depth_cm(mask_s,y_belly_s,depth_window,ppc_side,belly_width_cm,min_depth_ratio,min_depth_cm)
    hip_depth_cm,  (yh_used,hx0s,hx1s) = stabilized_depth_cm(mask_s,y_hip_s,  depth_window,ppc_side,hip_width_cm,  min_depth_ratio,min_depth_cm)

    # Hovedlinje ved øjenhøjde; skub 2.5% ned for hår; mål på raw front/side
    if "LEFT_EYE" in kp_f and "RIGHT_EYE" in kp_f:
        y_head_f = int(round((kp_f["LEFT_EYE"][1]+kp_f["RIGHT_EYE"][1])/2))
    elif "LEFT_EAR" in kp_f and "RIGHT_EAR" in kp_f:
        y_head_f = int(round((kp_f["LEFT_EAR"][1]+kp_f["RIGHT_EAR"][1])/2))
    else:
        y_head_f = max(int(round(y_shoulder - 0.15*(y_hip - y_shoulder))), 0)
    y_head_f += int(0.025*(bot_f - top_f))
    y_head_s = map_y(y_head_f, top_f, bot_f, top_s, bot_s)
    head_w_px, hx0f, hx1f = silhouette_width_at_y(mask_f_raw, y_head_f)
    head_d_px, hx0s2,hx1s2 = silhouette_width_at_y(mask_s, y_head_s)
    head_width_cm = max(0.0, head_w_px/max(ppc_front,1e-6) - 2*1.0)  # -1 cm margin pr. side
    head_depth_cm = head_d_px/max(ppc_side,1e-6)

    # Omkredse (ellipse + korrektion)
    def circ(w,d): return ellipse_circumference_cm(w,d)*float(ellipse_corr)
    chest_circ_cm=circ(chest_width_cm, chest_depth_cm)
    waist_circ_cm=circ(waist_width_cm, waist_depth_cm)
    belly_circ_cm=circ(belly_width_cm, belly_depth_cm)
    hip_circ_cm  =circ(hip_width_cm,   hip_depth_cm)
    head_circ_cm =circ(head_width_cm,  head_depth_cm)

    # Længder
    sh_for_arm=safe_pick(kp_f,"LEFT_SHOULDER","RIGHT_SHOULDER")
    elb_for_arm=safe_pick(kp_f,"LEFT_ELBOW","RIGHT_ELBOW")
    wr_for_arm =safe_pick(kp_f,"LEFT_WRIST","RIGHT_WRIST")
    arm_len_cm=dist_cm(sh_for_arm,elb_for_arm,ppc_front)+dist_cm(elb_for_arm,wr_for_arm,ppc_front)

    shoulder_to_waist_cm = abs(y_waist - y_shoulder)/max(ppc_front,1e-6)
    knee_mid_y=int(round((kp_f["LEFT_KNEE"][1]+kp_f["RIGHT_KNEE"][1])/2))
    waist_to_knee_cm = abs(knee_mid_y - y_waist)/max(ppc_front,1e-6)

    # Benlængde via fodsål (front)
    ank = safe_pick(kp_f,"LEFT_ANKLE","RIGHT_ANKLE")
    x_col = int(np.clip(ank[0], 0, mask_f_raw.shape[1]-1))
    ys = np.where(mask_f_raw[:,x_col] > 0)[0]
    if ys.size:
        y_sole = int(ys.max())
        hip_for_leg = safe_pick(kp_f,"LEFT_HIP","RIGHT_HIP")
        leg_len_cm = abs(y_sole - hip_for_leg[1]) / max(ppc_front,1e-6)
    else:
        hip_for_leg = safe_pick(kp_f,"LEFT_HIP","RIGHT_HIP")
        leg_len_cm = dist_cm(hip_for_leg, ank, ppc_front)

    # ------------- Overlays -------------
    # keypoints
    for p in kp_f.values(): draw_point(front_vis,p,CLR_POINTS,2)
    for p in kp_s.values(): draw_point(side_vis,p,CLR_POINTS,2)

    # skulderbredde
    cv2.line(front_vis,(L_SH[0],L_SH[1]),(R_SH[0],R_SH[1]),CLR_SHOULDER,2)
    put_label(front_vis,f"Shoulder {shoulder_width_cm:.1f} cm",(min(L_SH[0],R_SH[0]), max(15,y_shoulder-14)), CLR_SHOULDER)

    # front scanlines (torso-maskens intervaller)
    def draw_front_line(y, w_cm, color, label):
        x0,x1 = main_interval_at_y(torso_mask, y, center_x_front, max_gap_px=8)
        x0p = x0 if x0>=0 else 10
        x1p = x1 if x1>=0 else front.shape[1]-10
        draw_scanline(front_vis,y,x0p,x1p,color,f"{label} {w_cm:.1f} cm", y-8)

    draw_front_line(y_chest, chest_width_cm, CLR_CHEST, "Chest")
    draw_front_line(y_belly, belly_width_cm, CLR_BELLY, "Belly")
    draw_front_line(y_waist, waist_width_cm, CLR_WAIST, "Waist")
    draw_front_line(y_hip,   hip_width_cm,   CLR_HIP,   "Hip")
    # head (målt på raw-masken)
    hx0p = hx0f if hx0f>=0 else 10; hx1p = hx1f if hx1f>=0 else front.shape[1]-10
    draw_scanline(front_vis, y_head_f, hx0p, hx1p, CLR_HEAD, f"Head {head_width_cm:.1f} cm", y_head_f-8)

    # side scanlines
    base_y = 20
    draw_scanline(side_vis,yc_used,cx0s if cx0s>=0 else 10,cx1s if cx1s>=0 else side.shape[1]-10, CLR_CHEST, f"Depth chest {chest_depth_cm:.1f} cm", base_y)
    draw_scanline(side_vis,yb_used,bx0s if bx0s>=0 else 10,bx1s if bx1s>=0 else side.shape[1]-10, CLR_BELLY, f"Depth belly {belly_depth_cm:.1f} cm", base_y+label_spacing)
    draw_scanline(side_vis,yw_used,wx0s if wx0s>=0 else 10,wx1s if wx1s>=0 else side.shape[1]-10, CLR_WAIST, f"Depth waist {waist_depth_cm:.1f} cm", base_y+2*label_spacing)
    draw_scanline(side_vis,yh_used,hx0s if hx0s>=0 else 10,hx1s if hx1s>=0 else side.shape[1]-10, CLR_HIP,   f"Depth hip {hip_depth_cm:.1f} cm", base_y+3*label_spacing)
    draw_scanline(side_vis,y_head_s,hx0s2 if hx0s2>=0 else 10,hx1s2 if hx1s2>=0 else side.shape[1]-10, CLR_HEAD, f"Depth head {head_depth_cm:.1f} cm", base_y+4*label_spacing)

    # top/bottom guides
    for img, t, b, W in [(front_vis, top_f, bot_f, front.shape[1]), (side_vis, top_s, bot_s, side.shape[1])]:
        cv2.line(img,(10,t),(W-10,t),CLR_GUIDE,1,cv2.LINE_AA)
        cv2.line(img,(10,b),(W-10,b),CLR_GUIDE,1,cv2.LINE_AA)

    # circumference labels
    ctexts = [
        (CLR_CHEST, f"Chest circ {chest_circ_cm:.1f} cm"),
        (CLR_WAIST, f"Waist circ {waist_circ_cm:.1f} cm"),
        (CLR_BELLY, f"Belly circ {belly_circ_cm:.1f} cm"),
        (CLR_HIP,   f"Hip circ {hip_circ_cm:.1f} cm"),
        (CLR_HEAD,  f"Head circ {head_circ_cm:.1f} cm"),
    ]
    x0txt, y0txt = 12, 20
    for i,(clr,txt) in enumerate(ctexts):
        put_label(front_vis, txt, (x0txt, y0txt + i*label_spacing), clr)

    results={
        "HeadCircumference_cm": round(head_circ_cm,1),
        "ShoulderWidth_cm": round(shoulder_width_cm,1),

        "ChestWidth_cm": round(chest_width_cm,1),
        "ChestDepth_cm": round(chest_depth_cm,1),
        "ChestCircumference_cm": round(chest_circ_cm,1),

        "BellyWidth_cm": round(belly_width_cm,1),
        "BellyDepth_cm": round(belly_depth_cm,1),
        "BellyCircumference_cm": round(belly_circ_cm,1),

        "WaistWidth_cm": round(waist_width_cm,1),
        "WaistDepth_cm": round(waist_depth_cm,1),
        "WaistCircumference_cm": round(waist_circ_cm,1),

        "HipWidth_cm": round(hip_width_cm,1),
        "HipDepth_cm": round(hip_depth_cm,1),
        "HipCircumference_cm": round(hip_circ_cm,1),

        "ArmLength_cm": round(arm_len_cm,1),
        "ShoulderToWaist_cm": round(shoulder_to_waist_cm,1),
        "WaistToKnee_cm": round(waist_to_knee_cm,1),
        "LegLength_cm": round(leg_len_cm,1),

        "TotalHeight_cm": round(total_height_cm,1),
    }

    if save_debug:
        cv2.imwrite("front_mask_raw.png", mask_f_raw*255)
        cv2.imwrite("front_mask_torso.png", torso_mask*255)
        cv2.imwrite("side_mask.png",  mask_s*255)
        cv2.imwrite("front_overlay.png", front_vis)
        cv2.imwrite("side_overlay.png",  side_vis)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results

# ---------------- CLI ----------------
def parse_args():
    ap=argparse.ArgumentParser(description="Body measurements from front+side with torso-only front mask.")
    ap.add_argument("--front", required=True)
    ap.add_argument("--side", required=True)

    # Skalering: enten kendt højde eller kalibrering
    ap.add_argument("--height-cm", type=float, default=None)
    ap.add_argument("--calib", choices=["aruco","chessboard"], default=None)
    ap.add_argument("--marker-cm", type=float, default=None, help="ArUco marker size (cm)")
    ap.add_argument("--cb-cols", type=int, default=7, help="Chessboard inner corners cols")
    ap.add_argument("--cb-rows", type=int, default=10, help="Chessboard inner corners rows")
    ap.add_argument("--square-mm", type=float, default=25.0, help="Chessboard square size (mm)")

    ap.add_argument("--save-debug", action="store_true")
    ap.add_argument("--circ-corr", type=float, default=1.00)
    ap.add_argument("--depth-window", type=int, default=45)
    ap.add_argument("--min-depth-ratio", type=float, default=0.72)
    ap.add_argument("--min-depth-cm", type=float, default=14.0)
    ap.add_argument("--label-spacing", type=int, default=28)
    return ap.parse_args()

def main():
    a=parse_args()
    if a.height_cm is None and a.calib is None:
        raise ValueError("Angiv enten --height-cm eller --calib [aruco|chessboard].")
    if a.calib=="aruco" and not a.marker_cm:
        raise ValueError("--calib aruco kræver --marker-cm.")
    # læs med absolutte stier for bedre fejlmeddelelser
    compute(a.front, a.side, a.height_cm, a.calib, a.marker_cm,
            a.cb_cols, a.cb_rows, a.square_mm,
            a.save_debug, a.circ_corr, a.depth_window, a.min_depth_ratio, a.min_depth_cm, a.label_spacing)

if __name__=="__main__":
    main()
