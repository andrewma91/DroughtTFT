import sys, os, logging
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
Path("./results").mkdir(exist_ok=True)

from drought_model import (
    CFG, WeatherDataPipeline, DroughtIndexCalculator, DroughtGraph,
    DroughtDataset, DroughtGAT, DroughtTrainer, DroughtEvaluator,
    TemporalEncoder, SpatioTemporalGAT
)
import torch.nn as nn

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

class TFTOnly(nn.Module):

    def __init__(self, cfg, input_dim):
        super().__init__()
        d = cfg["d_model"]
        self.temporal_enc = TemporalEncoder(input_dim, d, cfg["n_heads"], cfg["n_layers"], cfg["dropout"])

        self.proj = nn.Sequential(
            nn.Linear(d, cfg["hidden_dim"]), nn.GELU(), nn.Dropout(cfg["dropout"]),
        )
        self.multi_head = nn.Linear(cfg["hidden_dim"], len(cfg["pred_horizons"]))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq, edge_index=None):
        h = self.temporal_enc(x_seq)
        h = self.proj(h)
        return self.multi_head(h)

def compute_metrics(true_arr, pred_arr, horizons):

    rows = []
    for i, h in enumerate(horizons):
        t = true_arr[:, i]
        p = pred_arr[:, i]
        rmse = np.sqrt(mean_squared_error(t, p))
        r2 = r2_score(t, p)
        obs_dry = (t < -1.0).astype(int)
        pred_dry = (p < -1.0).astype(int)
        hits = np.sum((obs_dry == 1) & (pred_dry == 1))
        misses = np.sum((obs_dry == 1) & (pred_dry == 0))
        fa = np.sum((obs_dry == 0) & (pred_dry == 1))
        pod = hits / (hits + misses + 1e-8)
        far = fa  / (hits + fa + 1e-8)
        rows.append({
            "Lead (months)": h, "RMSE": round(rmse, 3), "R²": round(r2, 3), "POD": round(pod, 3), "FAR": round(far, 3),
        })
    return pd.DataFrame(rows)

def persistence_baseline(dataset, test_indices, horizons):

    spei_feat_idx = dataset.feature_cols.index("spei6")
    scaler = dataset.scaler
    mean = scaler.mean_[spei_feat_idx]
    std  = scaler.scale_[spei_feat_idx]

    all_true, all_pred = [], []
    for idx in test_indices:
        x_seq, y = dataset[idx]

        current_spei_scaled = x_seq[:, -1, spei_feat_idx]
        current_spei = current_spei_scaled.numpy() * std + mean

        pred = np.stack([current_spei] * len(horizons), axis=1)
        all_true.append(y.numpy())
        all_pred.append(pred)

    true_arr = np.concatenate(all_true, axis=0)
    pred_arr = np.concatenate(all_pred, axis=0)
    return compute_metrics(true_arr, pred_arr, horizons)

def climatology_baseline(dataset, train_indices, test_indices, horizons):

    monthly_means = {m: [] for m in range(1, 13)}
    for idx in train_indices:
        x_seq, y = dataset[idx]
        sample_month = (idx % 12) + 1
        for h_idx in range(len(horizons)):
            target_month = ((sample_month + horizons[h_idx] - 1) % 12) + 1

            monthly_means[target_month].extend(y[:, h_idx].numpy().tolist())

    clim = {m: np.mean(v) if v else 0.0 for m, v in monthly_means.items()}

    all_true, all_pred = [], []
    for idx in test_indices:
        x_seq, y = dataset[idx]
        sample_month = (idx % 12) + 1

        clim_pred = np.array([
            clim[((sample_month + h - 1) % 12) + 1]
            for h in horizons
        ])
        N = y.shape[0]
        pred = np.tile(clim_pred, (N, 1))
        all_true.append(y.numpy())
        all_pred.append(pred)

    true_arr = np.concatenate(all_true, axis=0)
    pred_arr = np.concatenate(all_pred, axis=0)
    return compute_metrics(true_arr, pred_arr, horizons)

def coarsen(df, deg=1.0):
    df = df.copy()
    df["lat"] = (df["lat"] / deg).round() * deg
    df["lon"] = (df["lon"] / deg).round() * deg
    num_cols = [c for c in df.columns if c not in ["lat","lon"]]
    return df.groupby([df.index, "lat","lon"])[num_cols].mean().reset_index(
        level=["lat","lon"])

def run_ablation(region_name):
    log.info("="*70)
    log.info(f"ABLATION: {region_name}")
    log.info("="*70)

    out_path = f"./results/ablation_{region_name}.csv"
    if Path(out_path).exists():
        log.info(f"Already done: {out_path}")
        return pd.read_csv(out_path)

    parq = f"./data/era5/{region_name}_monthly.parquet"
    if not Path(parq).exists():
        raise FileNotFoundError(
            f"No parquet at {parq}. Run run_region.py for {region_name} first.")

    climate_df = pd.read_parquet(parq)
    climate_df = coarsen(climate_df, COARSEN_DEG)

    pipe = WeatherDataPipeline(CFG)
    teleconn_df = pipe.fetch_teleconnection_indices()

    calc = DroughtIndexCalculator()
    climate_df = calc.compute_all(climate_df)

    dataset = DroughtDataset(climate_df, teleconn_df, CFG["seq_len"],
                              CFG["pred_horizons"])
    n       = len(dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    train_idx = list(range(0, n_train))
    val_idx   = list(range(n_train, n_train + n_val))
    test_idx  = list(range(n_train + n_val, n))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=1, shuffle=False)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=1, shuffle=False)

    coords = climate_df.groupby(["lat","lon"]).first().reset_index()[
        ["lat","lon"]].values
    edge_index = DroughtGraph(radius_km=300).build_spatial_edges(coords)
    input_dim  = len(dataset.feature_cols)
    horizons   = CFG["pred_horizons"]

    tft_ckpt = f"{CFG['checkpoint_dir']}/tftonly_{region_name}_best.pt"

    log.info("Training TFT-only (no graph)...")
    tft_model = TFTOnly(CFG, input_dim)
    tft_trainer = DroughtTrainer(tft_model, CFG, edge_index)

    tft_trainer.ckpt_path = tft_ckpt

    import types
    def tft_train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for x_seq, y in loader:
            x_seq = x_seq.to(self.device)
            y     = y.to(self.device)
            self.optimizer.zero_grad()
            preds = []
            for b in range(x_seq.shape[0]):
                preds.append(self.model(x_seq[b]))
            pred = torch.stack(preds, 0)
            loss = self.drought_weighted_mse(
                pred.view(-1, pred.shape[-1]),
                y.view(-1, y.shape[-1])
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    original_train = tft_trainer.train.__func__

    def tft_train(self, train_loader, val_loader):
        best_val_rmse  = float("inf")
        patience_count = 0
        history        = []
        for epoch in range(self.cfg["epochs"]):
            train_loss = tft_train_epoch(self, train_loader)
            val_metrics = self.evaluate(val_loader)
            val_rmse_6  = val_metrics.get("lead6_rmse", 999)
            self.scheduler.step()
            log.info(
                f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                f"Val RMSE@6mo: {val_rmse_6:.4f} | "
                f"Val R²@6mo: {val_metrics.get('lead6_r2', 0):.4f} | "
                f"POD@6mo: {val_metrics.get('lead6_pod', 0):.4f}"
            )
            history.append({"epoch": epoch+1, "train_loss": train_loss, **val_metrics})
            if val_rmse_6 < best_val_rmse:
                best_val_rmse  = val_rmse_6
                patience_count = 0
                torch.save(
                    {"epoch": epoch, "model": self.model.state_dict(),
                     "optimizer": self.optimizer.state_dict(), "metrics": val_metrics},
                    tft_ckpt
                )
            else:
                patience_count += 1
                if patience_count >= self.cfg["patience"]:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break
        return pd.DataFrame(history)

    tft_trainer.train = types.MethodType(tft_train, tft_trainer)

    if Path(tft_ckpt).exists():
        log.info(f"TFT-only checkpoint exists — skipping training, loading {tft_ckpt}")
    else:
        tft_history = tft_trainer.train(train_loader, val_loader)
        tft_history.to_csv(
            f"./results/ablation_tftonly_{region_name}_history.csv", index=False)

    original_evaluate = tft_trainer.evaluate.__func__
    def tft_evaluate(self, loader):
        self.model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x_seq, y in loader:
                x_seq = x_seq.to(self.device)
                preds = []
                for b in range(x_seq.shape[0]):
                    preds.append(self.model(x_seq[b]))
                pred = torch.stack(preds, 0)
                all_pred.append(pred.cpu().numpy().reshape(-1, pred.shape[-1]))
                all_true.append(y.numpy().reshape(-1, y.shape[-1]))
        pred_arr = np.concatenate(all_pred)
        true_arr = np.concatenate(all_true)
        metrics = {}
        for i, h in enumerate(self.cfg["pred_horizons"]):
            from sklearn.metrics import mean_squared_error, r2_score
            rmse = np.sqrt(mean_squared_error(true_arr[:, i], pred_arr[:, i]))
            r2 = r2_score(true_arr[:, i], pred_arr[:, i])
            obs_dry = (true_arr[:, i] < -1.0).astype(int)
            pred_dry = (pred_arr[:, i] < -1.0).astype(int)
            hits = np.sum((obs_dry == 1) & (pred_dry == 1))
            misses = np.sum((obs_dry == 1) & (pred_dry == 0))
            fa = np.sum((obs_dry == 0) & (pred_dry == 1))
            pod = hits / (hits + misses + 1e-8)
            far = fa  / (hits + fa + 1e-8)
            metrics[f"lead{h}_rmse"] = rmse
            metrics[f"lead{h}_r2"]   = r2
            metrics[f"lead{h}_pod"]  = pod
            metrics[f"lead{h}_far"]  = far
        return metrics
    tft_trainer.evaluate = types.MethodType(tft_evaluate, tft_trainer)

    if Path(tft_ckpt).exists():
        ckpt = torch.load(tft_ckpt, map_location=tft_trainer.device, weights_only=False)
        tft_model.load_state_dict(ckpt["model"])
        log.info(f"Loaded TFT-only best checkpoint")

    tft_metrics = tft_trainer.evaluate(test_loader)
    tft_df = DroughtEvaluator(CFG).skill_summary_table(tft_metrics)
    tft_df["model"] = "TFT-only (no graph)"

    log.info("Computing persistence baseline...")
    pers_df = persistence_baseline(dataset, test_idx, horizons)
    pers_df["model"] = "Persistence"

    log.info("Computing climatology baseline...")
    clim_df = climatology_baseline(dataset, train_idx, test_idx, horizons)
    clim_df["model"] = "Climatology"

    gat_path = f"./results/{region_name}_skill_table.csv"
    if Path(gat_path).exists():
        gat_df = pd.read_csv(gat_path)[["Lead (months)","RMSE","R²","POD","FAR"]]
        gat_df["model"] = "DroughtGAT (full)"
    else:
        log.warning(f"DroughtGAT results not found at {gat_path}. "
                     f"Run run_region.py first.")
        gat_df = pd.DataFrame()

    combined = pd.concat([gat_df, tft_df, pers_df, clim_df], ignore_index=True)
    combined["region"] = region_name
    combined.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"ABLATION RESULTS: {region_name}")
    print(f"{'='*70}")

    for model_name in ["DroughtGAT (full)", "TFT-only (no graph)", "Persistence", "Climatology"]:
        sub = combined[combined["model"] == model_name]
        if len(sub):
            print(f"\n{model_name}:")
            print(sub[["Lead (months)","RMSE","R²","POD"]].to_string(index=False))

    return combined

CURRENT_REGION = "horn_of_africa"

if __name__ == "__main__":
    result = run_ablation(CURRENT_REGION)
    log.info(f"Saved to ./results/ablation_{CURRENT_REGION}.csv")
    log.info("Change CURRENT_REGION and re-run for each region.")
