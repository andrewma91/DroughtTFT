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

from drought_model import (CFG, WeatherDataPipeline, DroughtIndexCalculator, DroughtGraph, DroughtDataset, DroughtTrainer, DroughtEvaluator)

CFG["epochs"] = 200
CFG["patience"] = 20
CFG["batch_size"] = 1
CFG["d_model"] = 64
CFG["n_heads"] = 4
CFG["n_layers"] = 2
CFG["hidden_dim"] = 128
CFG["dropout"] = 0.15
CFG["lr"] = 3e-4
CFG["seq_len"] = 24
COARSEN_DEG = 1.0

def coarsen(df, deg=1.0):
    df = df.copy()
    df["lat"] = (df["lat"] / deg).round() * deg
    df["lon"] = (df["lon"] / deg).round() * deg
    num_cols = [c for c in df.columns if c not in ["lat", "lon"]]
    return df.groupby([df.index, "lat", "lon"])[num_cols].mean()\
             .reset_index(level=["lat", "lon"])

def run_region(region_name):
    log.info("=" * 65)
    log.info(f"DroughtTFT: {region_name}  (ERA5 1950-2023)")
    log.info("=" * 65)

    out_skill = f"./results/{region_name}_skill_table.csv"
    out_hist = f"./results/{region_name}_history.csv"
    ckpt_path = f"./checkpoints/droughttft_{region_name}_best.pt"

    if Path(out_skill).exists():
        log.info(f"Already done — loading: {out_skill}")
        return pd.read_csv(out_skill)

    parq = f"./data/era5/{region_name}_monthly.parquet"
    if not Path(parq).exists():
        raise FileNotFoundError(
            f"No parquet at {parq}. Run download_era5.py first.")

    log.info(f"Loading {parq}...")
    climate_df  = coarsen(pd.read_parquet(parq), COARSEN_DEG)
    log.info(f"Grid: {climate_df.groupby(['lat','lon']).ngroups} points, "
             f"{climate_df.index.min()} to {climate_df.index.max()}")

    pipe = WeatherDataPipeline(CFG)
    teleconn_df = pipe.fetch_teleconnection_indices()

    calc = DroughtIndexCalculator()
    climate_df = calc.compute_all(climate_df)

    dataset = DroughtDataset(climate_df, teleconn_df, CFG["seq_len"], CFG["pred_horizons"])
    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    log.info(f"Dataset: {n} samples | train={n_train} val={n_val} test={n_test}")
    log.info(f"Features ({len(dataset.feature_cols)}): {dataset.feature_cols}")

    train_loader = DataLoader(Subset(dataset, range(0, n_train)), batch_size=1, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(n_train, n_train + n_val)), batch_size=1, shuffle=False)
    test_loader = DataLoader(Subset(dataset, range(n_train + n_val, n)), batch_size=1, shuffle=False)

    coords = climate_df.groupby(["lat","lon"]).first().reset_index()[["lat","lon"]].values
    edge_index = DroughtGraph(radius_km=300).build_spatial_edges(coords)
    input_dim = len(dataset.feature_cols)

    from drought_model import TemporalEncoder
    class DroughtTFT(nn.Module):
        def __init__(self, cfg, input_dim):
            super().__init__()
            d = cfg["d_model"]
            self.temporal_enc = TemporalEncoder(input_dim, d, cfg["n_heads"], cfg["n_layers"], cfg["dropout"])
            self.proj = nn.Sequential(nn.Linear(d, cfg["hidden_dim"]), nn.GELU(), nn.Dropout(cfg["dropout"]),)
            self.head = nn.Linear(cfg["hidden_dim"], len(cfg["pred_horizons"]))
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)

        def forward(self, x_seq, edge_index=None):
            h = self.temporal_enc(x_seq)
            h = self.proj(h)
            return self.head(h)

    model = DroughtTFT(CFG, input_dim)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"DroughtTFT parameters: {n_params:,}")

    trainer = DroughtTrainer(model, CFG, edge_index)
    history = trainer.train(train_loader, val_loader)
    history.to_csv(out_hist, index=False)

    best_ckpt = f"{CFG['checkpoint_dir']}/droughtgat_best.pt"
    if Path(best_ckpt).exists():
        ckpt = torch.load(best_ckpt, map_location=trainer.device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        torch.save(ckpt, ckpt_path)
        log.info(f"Best checkpoint loaded and saved to {ckpt_path}")

    metrics = trainer.evaluate(test_loader)
    evaluator = DroughtEvaluator(CFG)
    table = evaluator.skill_summary_table(metrics)
    table["region"] = region_name
    table.to_csv(out_skill, index=False)

    print(f"\n{'='*55}")
    print(f"DroughtTFT TEST RESULTS: {region_name}")
    print(f"{'='*55}")
    print(table[["Lead (months)", "RMSE", "R²", "POD"]].to_string(index=False))
    log.info(f"\nSaved: {out_skill}")
    return table

CURRENT_REGION = "horn_of_africa"

if __name__ == "__main__":
    run_region(CURRENT_REGION)
