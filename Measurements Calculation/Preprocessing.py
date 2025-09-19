# preprocess_and_split_modcloth_parquet.py
# Læs ModCloth JSONL -> rens -> vælg kernefelter -> group-split -> gem som Parquet (snappy)

import os, re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ========= konfiguration =========
JSON_FILENAME = "modcloth_final_data.json"   # ligger i ./data/
OUT_DIRNAME   = "processed_modcloth_parquet" # output-mappe under ./data/
TEST_SIZE = 0.20
VAL_SIZE  = 0.20
RANDOM_SEED = 42

# kernefelter vi bevarer efter parsing
KEEP_COLS = ["item_id","user_id","category","size","fit","height_cm","hips_cm"]
# neutral pasform (bruges til at definere “sand” size-labels)
FIT_NEUTRAL = {"fit", "true to size", "just right", "perfect", "fits well"}
INCH_TO_CM = 2.54
PARQUET_ENGINE = "pyarrow"   # kræver pyarrow installeret
PARQUET_COMPRESSION = "snappy"
# =================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IN_PATH  = os.path.join(DATA_DIR, JSON_FILENAME)
OUT_DIR  = os.path.join(DATA_DIR, OUT_DIRNAME)
os.makedirs(OUT_DIR, exist_ok=True)

def parse_height_to_cm(x):
    if pd.isna(x): return np.nan
    s = str(x).lower().strip()
    s = s.replace('"','').replace("”","").replace("’","'")
    s = s.replace("feet","ft").replace("foot","ft").replace("inches","in").replace("inch","in")
    m = re.match(r"^\s*(\d+)\s*'\s*(\d+)\s*$", s)           # 5'6
    if m: return (int(m.group(1))*12 + int(m.group(2))) * INCH_TO_CM
    m = re.match(r"^\s*(\d+)\s*ft\s*(\d+)\s*in\s*$", s)     # 5 ft 6 in
    if m: return (int(m.group(1))*12 + int(m.group(2))) * INCH_TO_CM
    m = re.match(r"^\s*([\d.]+)\s*in\s*$", s)               # 66 in
    if m: return float(m.group(1)) * INCH_TO_CM
    m = re.match(r"^\s*([\d.]+)\s*cm\s*$", s)               # 167 cm
    if m: return float(m.group(1))
    m = re.match(r"^\s*([\d.]+)\s*$", s)                    # bare tal
    if m:
        v = float(m.group(1))
        if 20 <= v <= 90:   return v * INCH_TO_CM
        if 120 <= v <= 220: return v
    return np.nan

def parse_inches_str_to_cm(x):
    if pd.isna(x): return np.nan
    try:    return float(str(x).strip()) * INCH_TO_CM
    except: return np.nan

def load_and_clean(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Filen findes ikke: {path}")
    # JSON Lines
    df = pd.read_json(path, lines=True)

    # normaliser kolonnenavne
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # height -> cm
    df["height_cm"] = df["height"].apply(parse_height_to_cm) if "height" in df.columns else np.nan
    # hips -> cm (waist/bust ignoreres pga. massiv missing)
    df["hips_cm"]   = df["hips"].apply(parse_inches_str_to_cm) if "hips" in df.columns else np.nan

    # size (behold numerisk hvis muligt)
    if "size" in df.columns and pd.api.types.is_numeric_dtype(df["size"]):
        df["size"] = df["size"].astype("Int64")
    elif "size" in df.columns:
        df["size"] = df["size"].astype(str)
    else:
        df["size"] = pd.NA

    # fit → lower/trim
    df["fit"] = df["fit"].astype(str).str.lower().str.strip() if "fit" in df.columns else pd.NA

    # user_id / item_id / category sikres
    for col in ["user_id","item_id","category"]:
        if col not in df.columns: df[col] = pd.NA

    keep = df[KEEP_COLS].copy()

    # drop rækker uden height eller size
    keep = keep.dropna(subset=["height_cm","size"])

    # blød outlier-trim (1..99 pct) for height/hips
    for c in ["height_cm","hips_cm"]:
        if c in keep.columns:
            q1, q99 = keep[c].quantile([0.01,0.99])
            keep = keep[(keep[c].isna()) | ((keep[c] >= q1) & (keep[c] <= q99))]

    # håndtér typer for stabil parquet
    keep["item_id"] = keep["item_id"].astype(str)
    keep["user_id"] = keep["user_id"].astype(str)
    keep["category"] = keep["category"].astype(str)
    keep["fit"] = keep["fit"].astype(str)
    # size: int hvis muligt, ellers string (bevaring ovenfor)
    return keep.reset_index(drop=True)

def write_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
    print(f"Gemte: {path}  ({len(df)} rækker)")

def split_grouped(df: pd.DataFrame, test_size=0.2, val_size=0.2, seed=RANDOM_SEED):
    """GroupShuffleSplit på user_id: først (train+val) vs test, derefter train vs val."""
    groups = df["user_id"].astype(str).fillna("NA").values

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_trval, idx_te = next(gss1.split(df, groups=groups))
    df_trval = df.iloc[idx_trval].reset_index(drop=True)
    df_te    = df.iloc[idx_te].reset_index(drop=True)

    groups_trval = df_trval["user_id"].astype(str).values
    rel_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    idx_tr, idx_va = next(gss2.split(df_trval, groups=groups_trval))
    df_tr = df_trval.iloc[idx_tr].reset_index(drop=True)
    df_va = df_trval.iloc[idx_va].reset_index(drop=True)
    return df_tr, df_va, df_te

def main():
    df = load_and_clean(IN_PATH)

    # Gem “all clean” (uden fit-filter) for generel brug/EDA
    write_parquet(df, os.path.join(OUT_DIR, "modcloth_all_clean.parquet"))

    # Fit-only subset (neutral pasform) → sandt label for size
    df_fit = df[df["fit"].isin(FIT_NEUTRAL)].copy()
    write_parquet(df_fit, os.path.join(OUT_DIR, "modcloth_fit_only.parquet"))

    # Spor A: height-only
    cols_A = ["item_id","user_id","category","size","fit","height_cm"]
    df_A = df_fit[cols_A].dropna(subset=["height_cm","size"]).reset_index(drop=True)
    tr_A, va_A, te_A = split_grouped(df_A, TEST_SIZE, VAL_SIZE)
    write_parquet(tr_A, os.path.join(OUT_DIR, "train_height_only.parquet"))
    write_parquet(va_A, os.path.join(OUT_DIR, "val_height_only.parquet"))
    write_parquet(te_A, os.path.join(OUT_DIR, "test_height_only.parquet"))

    # Spor B: height + hips
    cols_B = ["item_id","user_id","category","size","fit","height_cm","hips_cm"]
    df_B = df_fit[cols_B].dropna(subset=["height_cm","hips_cm","size"]).reset_index(drop=True)
    tr_B, va_B, te_B = split_grouped(df_B, TEST_SIZE, VAL_SIZE)
    write_parquet(tr_B, os.path.join(OUT_DIR, "train_height_hips.parquet"))
    write_parquet(va_B, os.path.join(OUT_DIR, "val_height_hips.parquet"))
    write_parquet(te_B, os.path.join(OUT_DIR, "test_height_hips.parquet"))

    # lille status
    print("\n=== Overblik ===")
    print(f"All clean          : {len(df)}")
    print(f"Fit only (neutral) : {len(df_fit)}")
    print(f"Spor A (height)    : train={len(tr_A)}, val={len(va_A)}, test={len(te_A)}")
    print(f"Spor B (h+hips)    : train={len(tr_B)}, val={len(va_B)}, test={len(te_B)}")

if __name__ == "__main__":
    main()
