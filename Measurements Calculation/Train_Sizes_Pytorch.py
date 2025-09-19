# train_tabular_pytorch_parquet.py
# Træner 2 tabulære PyTorch-MLP'er på Parquet-splits:
#  - Spor A: height-only
#  - Spor B: height + hips
# Gemmer artefakter og udskriver sammenlignende metrikker.

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# ----------------- Utils -----------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TabularDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, n_classes)
        )
    def forward(self, x): return self.net(x)

def build_size_order_from_series(s: pd.Series) -> List[str]:
    """Byg ordnet liste af sizes. Hvis numerisk -> sortér stigende som str; ellers unikke str."""
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        uniq = np.sort(s.dropna().astype(int).unique())
        return [str(int(v)) for v in uniq.tolist()]
    # tekstlabels
    uniq = sorted(s.dropna().astype(str).unique().tolist())
    # Minimal ordning: som alfabetisk; kan erstattes af custom mapping hvis nødvendigt
    return uniq

def to_index_array(size_series: pd.Series, size_order: List[str]) -> np.ndarray:
    map_ = {s:i for i,s in enumerate(size_order)}
    # kast til str for sikkerhed, men bevar ints som str(int)
    if pd.api.types.is_integer_dtype(size_series) or pd.api.types.is_float_dtype(size_series):
        arr = size_series.fillna(-99999).astype(int).astype(str).map(map_)
    else:
        arr = size_series.fillna("NA").astype(str).map(map_)
    return arr.values.astype(np.int64)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    exact = (y_true == y_pred).mean()
    ord_mae = np.abs(y_true - y_pred).mean()
    plus1 = (np.abs(y_true - y_pred) <= 1).mean()
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"exact": float(exact), "ord_mae": float(ord_mae), "plus1": float(plus1), "macro_f1": float(f1)}

@dataclass
class TrainCfg:
    batch_size: int = 512
    epochs: int = 200
    patience: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

# ----------------- Training core -----------------

def train_one_track(df_tr: pd.DataFrame,
                    df_va: pd.DataFrame,
                    df_te: pd.DataFrame,
                    feature_cols: List[str],
                    cfg: TrainCfg,
                    outdir: str) -> Dict:
    set_seed(cfg.seed)
    os.makedirs(outdir, exist_ok=True)

    # --- Size-orden og labels ---
    size_order = build_size_order_from_series(pd.concat([df_tr["size"], df_va["size"], df_te["size"]], axis=0))
    y_tr = to_index_array(df_tr["size"], size_order)
    y_va = to_index_array(df_va["size"], size_order)
    y_te = to_index_array(df_te["size"], size_order)

    # --- Features & skalering ---
    X_tr = df_tr[feature_cols].to_numpy()
    X_va = df_va[feature_cols].to_numpy()
    X_te = df_te[feature_cols].to_numpy()
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr); X_va = scaler.transform(X_va); X_te = scaler.transform(X_te)

    # --- Class weights (mod skævhed) ---
    cls, cnt = np.unique(y_tr, return_counts=True)
    weights = np.ones(len(size_order), dtype=np.float32)
    total = cnt.sum()
    for c, n in zip(cls, cnt):
        weights[c] = total / (len(cls) * n)
    cw = torch.tensor(weights)

    # --- Dataloaders ---
    dl_tr = DataLoader(TabularDS(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(TabularDS(X_va, y_va), batch_size=1024, shuffle=False, num_workers=0)
    dl_te = DataLoader(TabularDS(X_te, y_te), batch_size=1024, shuffle=False, num_workers=0)

    # --- Model/opt/loss ---
    model = MLP(d_in=X_tr.shape[1], n_classes=len(size_order)).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=cw.to(cfg.device))

    # --- Early stopping (på val-ordMAE) ---
    best_val = math.inf
    best_state = None
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(cfg.device); yb = yb.to(cfg.device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pv = []
            for xb, _ in dl_va:
                xb = xb.to(cfg.device)
                pv.append(model(xb).argmax(1).cpu().numpy())
            pv = np.concatenate(pv)
        mv = compute_metrics(y_va, pv)
        score = mv["ord_mae"]

        improved = score < best_val - 1e-4
        if improved:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Test ---
    model.eval()
    with torch.no_grad():
        pt = []
        for xb, _ in dl_te:
            xb = xb.to(cfg.device)
            pt.append(model(xb).argmax(1).cpu().numpy())
        pt = np.concatenate(pt)
    mt = compute_metrics(y_te, pt)

    # --- Persist artifacts ---
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    with open(os.path.join(outdir, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump({"mean": scaler.mean_.tolist(),
                   "scale": scaler.scale_.tolist(),
                   "features": feature_cols,
                   "size_order": size_order}, f, indent=2)
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val_ordMAE": best_val, "test": mt}, f, indent=2)

    return {"size_order": size_order, "val_best_ordMAE": best_val, "test_metrics": mt, "artifacts": outdir}

# ----------------- IO helpers -----------------

def read_parquet_required(path: str, cols: List[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mangler fil: {path}")
    return pd.read_parquet(path, columns=cols)

def load_track_frames(data_dir: str, track: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    if track == "A":
        feat = ["height_cm"]                       # height-only
        tr = read_parquet_required(os.path.join(data_dir, "train_height_only.parquet"))
        va = read_parquet_required(os.path.join(data_dir, "val_height_only.parquet"))
        te = read_parquet_required(os.path.join(data_dir, "test_height_only.parquet"))
    elif track == "B":
        feat = ["height_cm", "hips_cm"]            # height + hips
        tr = read_parquet_required(os.path.join(data_dir, "train_height_hips.parquet"))
        va = read_parquet_required(os.path.join(data_dir, "val_height_hips.parquet"))
        te = read_parquet_required(os.path.join(data_dir, "test_height_hips.parquet"))
    else:
        raise ValueError("Ukendt track; brug 'A' eller 'B'")
    # sikre kolonner
    need = set(["size"] + feat)
    missing = [c for c in need if c not in tr.columns]
    if missing:
        raise KeyError(f"Træningsfil mangler kolonner: {missing}")
    return tr, va, te, feat

# ----------------- CLI / Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Train PyTorch tabular models on ModCloth Parquet splits.")
    ap.add_argument("--data-dir", type=str, default=os.path.join("data", "processed_modcloth_parquet"),
                    help="Mappe med parquet-splits.")
    ap.add_argument("--tracks", type=str, nargs="+", default=["A", "B"],
                    help="Tracks to train: A (height), B (height+hips).")
    ap.add_argument("--out-dir", type=str, default=os.path.join("artifacts_parquet"),
                    help="Hvor artefakter gemmes.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = TrainCfg(seed=args.seed)
    summary_rows = []

    for t in args.tracks:
        tr, va, te, feat = load_track_frames(args.data_dir, t)
        track_name = "trackA_height" if t == "A" else "trackB_height_hips"
        outdir = os.path.join(args.out_dir, track_name)
        res = train_one_track(tr, va, te, feat, cfg, outdir)

        mt = res["test_metrics"]
        print(f"\n=== {track_name} ===")
        print("Test:",
              f"exact={mt['exact']:.3f}",
              f"±1={mt['plus1']:.3f}",
              f"ordMAE={mt['ord_mae']:.3f}",
              f"macroF1={mt['macro_f1']:.3f}")

        summary_rows.append([track_name, ",".join(feat), mt["exact"], mt["plus1"], mt["ord_mae"], mt["macro_f1"], outdir])

    # Samlet oversigt
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows, columns=["track","features","exact","±1","ordMAE","macroF1","artifacts"])
        print("\n=== Sammenligning (TEST) ===")
        # sortér efter lav ordMAE og høj exact
        df_sum = df_sum.sort_values(["ordMAE","exact"], ascending=[True, False])
        # pæn udskrift
        with pd.option_context('display.max_colwidth', None):
            print(df_sum.to_string(index=False))

if __name__ == "__main__":
    main()
