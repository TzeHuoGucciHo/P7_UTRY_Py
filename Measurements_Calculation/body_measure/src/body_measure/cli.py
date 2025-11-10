import argparse, json, os, sys, time
from body_measure import measure_v2 as M
import os
import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    """Robust billedindlæsning (understøtter også Windows-stier med mellemrum/unicode)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    # Brug imdecode for at undgå problemer med unicode-stier
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _to_worksheet_schema(res: dict) -> dict:
    """
    Mapper til ønskede kolonnenavne.
    Ignorerer bevidst: Gender, Age, HeadCircumference, Belly, ArmLength,
    ShoulderToWaist, WaistToKnee, LegLength.
    """
    return {
        "ShoulderWidth": res.get("shoulder_width_cm"),
        "ChestWidth":    res.get("chest_width_cm"),
        "Waist":         res.get("waist_cm"),
        "Hips":          res.get("hip_cm"),
        "TotalHeight":   res.get("input_height_cm"),  # direkte den indtastede højde
    }

def _read_img(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--front", required=True, help="Front image path")
    p.add_argument("--side",  required=True, help="Side image path")
    p.add_argument("--height-cm", type=float, default=None, help="Person height in cm (recommended)")
    p.add_argument("--backend", choices=["deeplabv3","opencv","auto"], default="deeplabv3")
    p.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    p.add_argument("--debug-dir", default=None)
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--setup-load", default=None)
    p.add_argument("--setup-save", default=None)
    p.add_argument("--calibrate-many", default=None)
    p.add_argument("--aruco-mm", type=float, default=None)
    p.add_argument("--target-height", type=int, default=2000)
    p.add_argument("--crop-margin", type=float, default=0.02)
    p.add_argument("--no-profile-scale", action="store_true")
    p.add_argument("--gender", choices=["male","female"], default="male",
                   help="Use gender-specific measurement tuning")

    p.add_argument("--print-runtime-info", action="store_true")
    args = p.parse_args()

    front = _read_img(args.front)
    side  = _read_img(args.side)
    if front is None: raise FileNotFoundError(args.front)
    if side  is None: raise FileNotFoundError(args.side)

    to_call = dict(
        front_bgr=front,
        side_bgr=side,
        height_cm=args.height_cm,
        debug_dir=args.debug_dir,
        save_masks=args.save_masks,
        prefer_backend=args.backend,
        device=args.device,
        setup_load=args.setup_load,
        setup_save=args.setup_save,
        calibrate_many=args.calibrate_many,
        aruco_mm=args.aruco_mm,
        target_height_px=args.target_height,
        crop_margin_ratio=args.crop_margin,
        no_profile_scale=args.no_profile_scale,
        gender=args.gender,
    )

    if args.print_runtime_info:
        import inspect, os as _os
        print("body_measure file:", _os.path.abspath(M.__file__))
        print("measure file     :", _os.path.abspath(M.__file__))
        sig = inspect.signature(M.compute)
        print("compute signature:", sig)
        print("passing keys     :", list(to_call.keys()))

    res = M.compute(**to_call)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
