from __future__ import annotations
import argparse, json, cv2
from .measure import compute

def main():
    ap = argparse.ArgumentParser(
        description="Marker-less two-view body measurement (DeepLabV3 default). "
                    "If --height-cm is provided it DEFINES the scale; any profile ppc is ignored."
    )
    ap.add_argument("--front", required=True, help="Path to front image")
    ap.add_argument("--side",  required=True, help="Path to side image")

    ap.add_argument("--height-cm", type=float, default=None,
                    help="User height in cm (RECOMMENDED). If set, it overrides any profile ppc.")
    ap.add_argument("--backend", choices=["deeplabv3","opencv","auto"], default="deeplabv3")
    ap.add_argument("--device",  choices=["cpu","cuda"], default="cpu")

    ap.add_argument("--debug-dir", type=str, default=None, help="Write overlays/masks here")
    ap.add_argument("--save-masks", action="store_true", help="Also write raw masks as PNG")

    ap.add_argument("--setup-load", type=str, default=None,
                    help="Load setup profile JSON (processing knobs; may contain ppc_ref as FALLBACK only)")
    ap.add_argument("--setup-save", type=str, default=None,
                    help="Save setup profile JSON with ppc_ref from this run (optional)")

    ap.add_argument("--calibrate-many", type=str, default=None,
                    help="Known measurements to refine scale, e.g. 'chest=116,shoulder=52,hip=115,inseam=87,thigh=69'")

    # Backward-compatible (ignored in marker-less mode)
    ap.add_argument("--aruco-mm", type=float, default=None, help="Ignored (marker-less)")

    # Normalization knobs
    ap.add_argument("--target-height", type=int, default=2000, help="Normalized bbox height in pixels (default: 2000)")
    ap.add_argument("--crop-margin", type=float, default=0.02, help="Fractional margin around bbox (default: 0.02)")
    args = ap.parse_args()

    front = cv2.imread(args.front, cv2.IMREAD_COLOR)
    side  = cv2.imread(args.side,  cv2.IMREAD_COLOR)
    if front is None: raise FileNotFoundError(args.front)
    if side  is None: raise FileNotFoundError(args.side)

    res = compute(
        front_bgr=front, side_bgr=side,
        height_cm=args.height_cm,
        debug_dir=args.debug_dir, save_masks=args.save_masks,
        prefer_backend=args.backend, device=args.device,
        setup_load=args.setup_load, setup_save=args.setup_save,
        calibrate_many=args.calibrate_many,
        aruco_mm=args.aruco_mm,
        target_height_px=args.target_height,
        crop_margin_ratio=args.crop_margin,
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
