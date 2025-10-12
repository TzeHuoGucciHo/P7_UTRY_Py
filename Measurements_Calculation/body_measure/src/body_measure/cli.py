from __future__ import annotations
import argparse, json, cv2, os, sys, inspect
from . import measure_v2 as M

def _resolve(p: str | None) -> str | None:
    if p is None: return None
    return p if os.path.isabs(p) else os.path.abspath(p)

def main():
    ap = argparse.ArgumentParser(
        description="Two-view body measurement (DeepLabV3). CLI er kompatibel på tværs af measure.py-versioner."
    )
    ap.add_argument("--front", required=True, help="Path to front image")
    ap.add_argument("--side",  required=True, help="Path to side image")

    ap.add_argument("--height-cm", type=float, default=None, help="User height in cm")
    ap.add_argument("--backend", choices=["deeplabv3","opencv","auto"], default="deeplabv3")
    ap.add_argument("--device",  choices=["cpu","cuda"], default="cuda")

    ap.add_argument("--debug-dir", type=str, default=None, help="Write overlays/masks here")
    ap.add_argument("--save-masks", action="store_true", help="Also write raw masks as PNG")

    ap.add_argument("--setup-load", type=str, default=None, help="Load setup profile JSON")
    ap.add_argument("--setup-save", type=str, default=None, help="Save setup profile JSON")

    # Normalisering
    ap.add_argument("--target-height", type=int, default=2000, help="Normalized bbox height in pixels")
    ap.add_argument("--crop-margin", type=float, default=0.02, help="Fractional margin around bbox")

    # Skala fallback-kontrol
    ap.add_argument("--no-profile-scale", action="store_true",
                    help="If height is missing, ignore profile ppc_ref (will error without scale).")

    # Nye/valgfri flag – sendes KUN hvis measure.compute accepterer dem:
    ap.add_argument("--row-mode", choices=["auto","pose","profile"], default="auto",
                    help="Only used if your measure.py supports it (keypoint-anchored rows). Ignored otherwise.")
    ap.add_argument("--aruco-mm", type=float, default=None, help="Ignored by marker-less versions (accepted if supported).")
    ap.add_argument("--waist-band", type=str, default=None, help="[compat] Ignored unless supported.")
    ap.add_argument("--hip-band", type=str, default=None, help="[compat] Ignored unless supported.")
    ap.add_argument("--waist-shift", type=float, default=None, help="[compat] Ignored unless supported.")
    ap.add_argument("--shoulder-shift", type=float, default=None, help="[compat] Ignored unless supported.")
    ap.add_argument("--calibrate-many", type=str, default=None, help="[compat] Ignored unless supported.")

    # Debug: print hvilke filer der bruges
    ap.add_argument("--print-runtime-info", action="store_true",
                    help="Print paths for body_measure and measure module + compute signature.")

    args = ap.parse_args()

    # Indlæs billeder
    front_path = _resolve(args.front)
    side_path  = _resolve(args.side)
    front = cv2.imread(front_path, cv2.IMREAD_COLOR)
    side  = cv2.imread(side_path,  cv2.IMREAD_COLOR)
    if front is None:
        print(f"ERROR: Cannot read front image: {front_path}", file=sys.stderr)
        raise FileNotFoundError(front_path)
    if side is None:
        print(f"ERROR: Cannot read side image: {side_path}", file=sys.stderr)
        raise FileNotFoundError(side_path)

    # Byg argumenter til compute
    want = {
        "front_bgr": front,
        "side_bgr": side,
        "height_cm": args.height_cm,
        "debug_dir": _resolve(args.debug_dir),
        "save_masks": args.save_masks,
        "prefer_backend": args.backend,
        "device": args.device,
        "setup_load": _resolve(args.setup_load),
        "setup_save": _resolve(args.setup_save),
        "target_height_px": args.target_height,
        "crop_margin_ratio": args.crop_margin,
        "no_profile_scale": args.no_profile_scale,
        # Valgfri/”nye” – sendes kun hvis supported:
        "row_mode": args.row_mode,
        "aruco_mm": args.aruco_mm,
        "waist_band": args.waist_band,
        "hip_band": args.hip_band,
        "waist_shift_rel": args.waist_shift,
        "shoulder_shift_rel": args.shoulder_shift,
        "calibrate_many": args.calibrate_many,
    }

    # Filtrér baseret på compute-signaturen
    sig = inspect.signature(M.compute)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in want.items() if k in allowed and v is not None}

    # Har compute **kwargs?
    has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    to_call = want if has_var_kwargs else filtered

    if args.print_runtime_info:
        import body_measure as BM
        print("body_measure file:", BM.__file__, file=sys.stderr)
        print("measure file     :", M.__file__, file=sys.stderr)
        print("compute signature:", sig, file=sys.stderr)
        print("passing keys     :", sorted(to_call.keys()), file=sys.stderr)

    res = M.compute(**to_call)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
