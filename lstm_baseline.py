import sys, logging
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
Path("./results").mkdir(exist_ok=True)
Path("./checkpoints").mkdir(exist_ok=True)

from drought_model import (
    CFG, WeatherDataPipeline, DroughtIndexCalculator, DroughtGraph, DroughtDataset, DroughtEvaluator
)

CFG["epochs"] = 200
CFG["patience"] = 20
CFG["batch_size"] = 1
CFG["d_model"] = 64
CFG["n_heads"] = 4
CFG["n_layers"] = 2
CFG["gat_heads"] = 2
CFG["hidden_dim"] = 128
CFG["dropout"] = 0.15
CFG["lr"] = 3e-4
CFG["seq_len"] = 24
COARSEN_DEG = 1.0

class DroughtLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.15, n_horizons=6):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim, hidden_size = hidden_dim, num_layers = n_layers, batch_first = True, dropout = dropout if n_layers > 1 else 0.0, bidirectional = False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_horizons),
        )
        for name, param in self.lstm.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: nn.init.zeros_(param)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x_seq, edge_index=None):

        out, _ = self.lstm(x_seq)
        h = self.norm(out[:, -1, :])
        return self.head(h)

def drought_weighted_mse(pred, target, weight=3.0):

    loss = F.mse_loss(pred, target, reduction='none')
    weights = torch.where(
        target < -1.0, torch.tensor(weight, device=pred.device), torch.tensor(1.0, device=pred.device),
    )
    return (loss * weights).mean()

def run_epoch(model, loader, optimizer, device, grad_clip=1.0, train=True):
    model.train() if train else model.eval()
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_seq, y in loader:
            x_seq = x_seq.to(device)
            y = y.to(device)
            if train: optimizer.zero_grad()
            preds = [model(x_seq[b]) for b in range(x_seq.shape[0])]
            pred = torch.stack(preds, dim=0)
            loss = drought_weighted_mse(
                pred.view(-1, pred.shape[-1]),
                y.view(-1, y.shape[-1]),
            )
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item()
    return total / len(loader)

def compute_metrics(model, loader, device, pred_horizons):

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x_seq, y in loader:
            preds = [model(x_seq[b].to(device)) for b in range(x_seq.shape[0])]
            pred = torch.stack(preds, dim=0)
            all_pred.append(pred.cpu().numpy().reshape(-1, pred.shape[-1]))
            all_true.append(y.numpy().reshape(-1, y.shape[-1]))
    pred_arr = np.concatenate(all_pred)
    true_arr = np.concatenate(all_true)
    metrics = {}
    for i, h in enumerate(pred_horizons):
        rmse = np.sqrt(mean_squared_error(true_arr[:, i], pred_arr[:, i]))
        r2 = r2_score(true_arr[:, i], pred_arr[:, i])
        obs = (true_arr[:, i] < -1.0).astype(int)
        prd = (pred_arr[:, i] < -1.0).astype(int)
        hits = np.sum((obs == 1) & (prd == 1))
        miss = np.sum((obs == 1) & (prd == 0))
        fa = np.sum((obs == 0) & (prd == 1))
        metrics[f"lead{h}_rmse"] = rmse
        metrics[f"lead{h}_r2"] = r2
        metrics[f"lead{h}_pod"] = hits / (hits + miss + 1e-8)
        metrics[f"lead{h}_far"] = fa / (hits + fa   + 1e-8)
    return metrics

def train_model(model, train_loader, val_loader, cfg, device, ckpt_path):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    best_val = float("inf")
    patience = 0
    history = []
    model.to(device)

    for epoch in range(cfg["epochs"]):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_m = compute_metrics(model, val_loader, device, cfg["pred_horizons"])
        val_rmse6 = val_m.get("lead6_rmse", 999)
        val_r2_6 = val_m.get("lead6_r2", -99)
        scheduler.step()

        log.info(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | " f"Val RMSE@6mo: {val_rmse6:.4f} | Val R²@6mo: {val_r2_6:.4f}")
        history.append({"epoch": epoch+1, "train_loss": train_loss, **val_m})

        if val_rmse6 < best_val:
            best_val = val_rmse6
            patience = 0
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                log.info(f"Early stopping at epoch {epoch+1}")
                break
    return pd.DataFrame(history)

def coarsen(df, deg=1.0):
    df = df.copy()
    df["lat"] = (df["lat"] / deg).round() * deg
    df["lon"] = (df["lon"] / deg).round() * deg
    num_cols = [c for c in df.columns if c not in ["lat","lon"]]
    return df.groupby([df.index,"lat","lon"])[num_cols].mean()\
             .reset_index(level=["lat","lon"])

def run_lstm(region_name):
    out_path = f"./results/lstm_{region_name}.csv"
    ckpt_path = f"./checkpoints/lstm_{region_name}_best.pt"

    if Path(out_path).exists():
        log.info(f"Already done: {out_path}")
        return pd.read_csv(out_path)

    log.info("=" * 60)
    log.info(f"LSTM BASELINE: {region_name}")
    log.info("=" * 60)

    parq = f"./data/era5/{region_name}_monthly.parquet"
    if not Path(parq).exists():
        raise FileNotFoundError(
            f"ERA5 parquet not found: {parq}\n"
            f"Run run_region.py for '{region_name}' first.")

    climate_df = coarsen(pd.read_parquet(parq), COARSEN_DEG)
    teleconn_df = WeatherDataPipeline(CFG).fetch_teleconnection_indices()
    climate_df = DroughtIndexCalculator().compute_all(climate_df)

    dataset = DroughtDataset(
        climate_df, teleconn_df, CFG["seq_len"], CFG["pred_horizons"])
    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    log.info(f"Dataset: {n} samples | train={n_train} val={n_val} "
             f"test={n - n_train - n_val}")
    log.info(f"Features ({len(dataset.feature_cols)}): {dataset.feature_cols}")

    train_loader = DataLoader(Subset(dataset, range(0, n_train)), batch_size=1, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(n_train, n_train+n_val)), batch_size=1, shuffle=False)
    test_loader = DataLoader(Subset(dataset, range(n_train+n_val, n)), batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(dataset.feature_cols)

    model = DroughtLSTM(
        input_dim = input_dim, hidden_dim = CFG["hidden_dim"], n_layers = CFG["n_layers"], dropout = CFG["dropout"], n_horizons = len(CFG["pred_horizons"]),
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"LSTM parameters: {n_params:,}  (DroughtTFT: 176,518)")

    history = train_model(model, train_loader, val_loader, CFG, device, ckpt_path)
    history.to_csv(f"./results/lstm_{region_name}_history.csv", index=False)

    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        log.info(f"Loaded best checkpoint (epoch {ckpt['epoch']+1})")

    metrics = compute_metrics(model, test_loader, device, CFG["pred_horizons"])
    evaluator = DroughtEvaluator(CFG)
    table = evaluator.skill_summary_table(metrics)
    table["model"] = "LSTM"
    table["region"] = region_name
    table.to_csv(out_path, index=False)

    print(f"\n{'='*55}")
    print(f"LSTM TEST RESULTS: {region_name}")
    print(f"{'='*55}")
    print(table[["Lead (months)","RMSE","R²","POD"]].to_string(index=False))
    return table

CURRENT_REGION = "horn_of_africa"

if __name__ == "__main__":
    run_lstm(CURRENT_REGION)
    log.info(f"Saved: ./results/lstm_{CURRENT_REGION}.csv")
    log.info("Change CURRENT_REGION and re-run for the next region.")
