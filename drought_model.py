import os
import warnings
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import xarray as xr
import requests
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

    pass

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CFG = {

    "regions": {
        "horn_of_africa": {"lat_range": (-5, 15), "lon_range": (33, 51), "driver": "ENSO/IOD"}, "ne_brazil": {"lat_range": (-15, -2), "lon_range": (-45, -34), "driver": "ENSO/Atlantic"}, "sahel": {"lat_range": (10, 18), "lon_range": (-17, 35), "driver": "AMO"}, "sw_us": {"lat_range": (28, 42), "lon_range": (-124, -103), "driver": "PDO/ENSO"}, "south_asia": {"lat_range": (8, 30), "lon_range": (68, 90), "driver": "ENSO/IOD monsoon"},
    },
    "resolution": 1.0, "era5_vars": ["tp", "t2m", "sp", "u10", "v10", "e"], "sst_regions": {"nino34": (-5, 5, -170, -120), "iod_west": (-10, 10, 50, 70), "iod_east": (-10, 0, 90, 110), "amo": (0, 70, -75, -10), "pdo": (20, 60, 120, 210),}, "seq_len": 24, "pred_horizons": [1, 2, 3, 4, 5, 6], "d_model": 128, "n_heads": 8, "n_layers": 4, "gat_heads": 4, "dropout": 0.1, "hidden_dim": 256, "batch_size": 32, "lr": 1e-4, "weight_decay": 1e-5, "epochs": 100, "patience": 15, "grad_clip": 1.0, "data_dir": "./data", "output_dir": "./outputs", "checkpoint_dir": "./checkpoints", "openmeteo_base": "https://archive-api.open-meteo.com/v1/archive", "commercial_api_key": os.getenv("WEATHER_API_KEY", ""),
}
class WeatherDataPipeline:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        Path(cfg["data_dir"]).mkdir(parents=True, exist_ok=True)

    def fetch_openmeteo_gridpoint(
        self, lat: float, lon: float, start_date: str = "1979-01-01", end_date: str   = "2024-12-31", ) -> pd.DataFrame:
        url = self.cfg["openmeteo_base"]
        params = {
            "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date, "daily": ["precipitation_sum", "temperature_2m_mean", "wind_speed_10m_max", "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm_mean",], "timezone": "UTC",}
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=90)
                r.raise_for_status()
                data = r.json()
                df = pd.DataFrame(data["daily"])
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)

                monthly = df.resample("MS").agg({
                    "precipitation_sum": "sum", "temperature_2m_mean": "mean", "wind_speed_10m_max": "mean", "et0_fao_evapotranspiration": "sum", "soil_moisture_0_to_7cm_mean": "mean",
                })
                monthly.columns = ["precip_mm", "temp_c", "wind_ms", "et0_mm", "soil_moist"]
                monthly["lat"] = lat
                monthly["lon"] = lon
                return monthly
            except Exception as e:
                if attempt < 2:
                    wait = 60 * (attempt + 1)
                    log.warning(f"Open-Meteo fetch attempt {attempt+1}/3 failed for " f"({lat},{lon}): {e}. Retrying in {wait}s...")
                    import time; time.sleep(wait)
                else:
                    log.warning(f"Open-Meteo fetch failed for ({lat},{lon}) after " f"3 attempts: {e}")
        return pd.DataFrame()

    def fetch_commercial_api(self, lat: float, lon: float, start_date: str, end_date: str, endpoint: str = "https://api.example-weather-provider.com/v1/historical",) -> pd.DataFrame:
        headers = {"Authorization": f"Bearer {self.cfg['commercial_api_key']}"}
        params  = {
            "lat": lat, "lon": lon, "start": start_date, "end": end_date, "vars": "precip,temp,wind,et0,soil_moisture", "freq": "monthly",
        }
        try:
            r = requests.get(endpoint, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            raw = r.json()

            df = pd.DataFrame(raw.get("data", raw))
            df.rename(columns={"date": "time", "precipitation": "precip_mm", "temperature": "temp_c", "wind_speed": "wind_ms", "evapotransp": "et0_mm", "soil_moisture": "soil_moist",}, inplace=True)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df["lat"] = lat
            df["lon"] = lon
            return df[["precip_mm","temp_c","wind_ms","et0_mm","soil_moist","lat","lon"]]
        except Exception as e:
            log.warning(f"Commercial API fetch failed for ({lat},{lon}): {e}. " f"Falling back to Open-Meteo.")
            return self.fetch_openmeteo_gridpoint(lat, lon, start_date, end_date)

    def fetch_teleconnection_indices(self) -> pd.DataFrame:

        teleconn_urls = {
            "nino34": "https://psl.noaa.gov/data/correlation/nina34.data", "pdo": "https://psl.noaa.gov/data/correlation/pdo.data", "amo":    "https://psl.noaa.gov/data/correlation/amo.data",}
        dfs = {}
        for name, url in teleconn_urls.items():
            try:
                r = requests.get(url, timeout=30)
                lines = [l.strip() for l in r.text.split("\n")
                         if l.strip() and not l.strip().startswith("-99")]
                records = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 13 and parts[0].isdigit():
                        year = int(parts[0])
                        for m, val in enumerate(parts[1:13], 1):
                            try:
                                v = float(val)
                                if abs(v) < 90:
                                    records.append({
                                        "date": pd.Timestamp(year=year, month=m, day=1),
                                        name: v,
                                    })
                            except ValueError:
                                pass
                dfs[name] = pd.DataFrame(records).set_index("date")
            except Exception as e:
                log.warning(f"Teleconnection index {name} fetch failed: {e}")

        if dfs:
            combined = pd.concat(dfs.values(), axis=1).sort_index()
            combined.index = pd.to_datetime(combined.index)
            return combined
        return pd.DataFrame()

    def build_regional_grid(self, regions: Optional[List[str]] = None, n_per_region: Optional[int] = 60, use_commercial_api: bool = False, start_date: str = "1981-01-01", end_date:   str = "2023-12-31",) -> Tuple[pd.DataFrame, pd.DataFrame]:
        region_names = regions or list(self.cfg["regions"].keys())
        rng = np.random.default_rng(42)
        all_dfs = []

        for region_name in region_names:
            rdef = self.cfg["regions"][region_name]
            lats = np.arange(rdef["lat_range"][0], rdef["lat_range"][1], self.cfg["resolution"])
            lons = np.arange(rdef["lon_range"][0], rdef["lon_range"][1], self.cfg["resolution"])
            grid_points = [(la, lo) for la in lats for lo in lons]

            if n_per_region and n_per_region < len(grid_points):
                idx = rng.choice(len(grid_points), n_per_region, replace=False)
                grid_points = [grid_points[i] for i in sorted(idx)]

            log.info(f"Region '{region_name}' ({rdef['driver']}): " f"{len(grid_points)} gridpoints")

            for i, (lat, lon) in enumerate(grid_points):
                if i % 20 == 0:
                    log.info(f"  [{region_name}] gridpoint {i}/{len(grid_points)} " f"({lat:.1f}, {lon:.1f})")
                if use_commercial_api:
                    df = self.fetch_commercial_api(lat, lon, start_date, end_date)
                else:
                    import time
                    time.sleep(3)
                    df = self.fetch_openmeteo_gridpoint(lat, lon, start_date, end_date)
                if not df.empty:
                    df["region"] = region_name
                    all_dfs.append(df)

        climate_df = pd.concat(all_dfs, axis=0) if all_dfs else pd.DataFrame()
        teleconn_df = self.fetch_teleconnection_indices()
        log.info(f"Multi-region grid built: {len(all_dfs)} gridpoints across " f"{len(region_names)} regions, teleconn shape: {teleconn_df.shape}")
        return climate_df, teleconn_df

class DroughtIndexCalculator:

    @staticmethod
    def compute_spi(precip_series: pd.Series, scale: int = 6) -> pd.Series:

        rolling = precip_series.rolling(scale).sum().dropna()

        spi = pd.Series(index=rolling.index, dtype=float)
        for month in range(1, 13):
            mask = rolling.index.month == month
            vals = rolling[mask].values
            vals = vals[vals > 0]
            if len(vals) < 10:
                continue
            a, loc, scale_g = stats.gamma.fit(vals, floc=0)
            prob = stats.gamma.cdf(rolling[mask].values, a, loc=loc, scale=scale_g)
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            spi[mask] = stats.norm.ppf(prob)
        return spi

    @staticmethod
    def compute_spei(
        precip_series: pd.Series, pet_series: pd.Series, scale: int = 6,
    ) -> pd.Series:

        D = precip_series - pet_series
        D_roll = D.rolling(scale).sum().dropna()

        spei = pd.Series(index=D_roll.index, dtype=float)
        for month in range(1, 13):
            mask = D_roll.index.month == month
            vals = D_roll[mask].values
            if len(vals) < 10:
                continue

            shift = vals.min() - 0.001 if vals.min() <= 0 else 0
            shifted = vals - shift
            lw, ls, ll = stats.fisk.fit(shifted, floc=0)
            prob = stats.fisk.cdf(shifted, lw, loc=ls, scale=ll)
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            spei[mask] = stats.norm.ppf(prob)
        return spei

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:

        results = []
        grouped = df.groupby(["lat", "lon"])
        for (lat, lon), grp in grouped:
            grp = grp.sort_index()
            grp["spi6"]  = self.compute_spi(grp["precip_mm"], scale=6)
            grp["spei6"] = self.compute_spei(grp["precip_mm"], grp["et0_mm"], scale=6)
            results.append(grp)
        return pd.concat(results).sort_index()

class DroughtGraph:

    def __init__(self, radius_km: float = 500, corr_threshold: float = 0.35):
        self.radius_km = radius_km
        self.corr_threshold = corr_threshold

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
             * np.sin(dlon / 2) ** 2)
        return 2 * R * np.arcsin(np.sqrt(a))

    def build_spatial_edges(
        self, coords: np.ndarray,
    ) -> torch.LongTensor:

        edges = []
        N = len(coords)
        for i in range(N):
            for j in range(i + 1, N):
                d = self.haversine(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
                if d <= self.radius_km:
                    edges += [[i, j], [j, i]]
        return torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2,0), dtype=torch.long)

    def build_teleconn_edges(
        self, node_spei: np.ndarray, teleconn:  np.ndarray, max_lag:   int = 6,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:

        N, T = node_spei.shape
        K = teleconn.shape[1]

        edges, weights = [], []
        for k in range(K):
            src_node = N + k
            for i in range(N):
                for lag in range(1, max_lag + 1):
                    if T - lag < 1:
                        continue
                    ts_node = node_spei[i, lag:]
                    ts_tele = teleconn[:-lag, k] if lag > 0 else teleconn[:, k]
                    min_len = min(len(ts_node), len(ts_tele))
                    if min_len < 10:
                        continue
                    r, _ = stats.pearsonr(ts_node[:min_len], ts_tele[:min_len])
                    if abs(r) >= self.corr_threshold:
                        edges.append([src_node, i])
                        weights.append(r)
                        break

        if edges:
            ei = torch.tensor(edges, dtype=torch.long).T
            ew = torch.tensor(weights, dtype=torch.float)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ew = torch.zeros(0)
        return ei, ew

class TemporalEncoder(nn.Module):

    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_enc = self._sinusoidal_encoding(512, d_model)

    @staticmethod
    def _sinusoidal_encoding(max_len: int, d_model: int) -> nn.Parameter:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, _ = x.shape
        x = self.input_proj(x) + self.pos_enc[:, :T, :]
        x = self.transformer(x)
        return x[:, -1, :]

class SpatioTemporalGAT(nn.Module):

    def __init__(self, d_model: int, gat_heads: int, dropout: float):
        super().__init__()
        self.gat1 = GATv2Conv(d_model, d_model // gat_heads, heads=gat_heads, dropout=dropout, concat=True)
        self.gat2 = GATv2Conv(d_model, d_model, heads=1, dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        x = self.norm1(x + self.gat1(x, edge_index))
        x = self.norm2(x + self.gat2(x, edge_index))
        x = x + self.ff(x)
        return x

class DroughtGAT(nn.Module):

    def __init__(self, cfg: dict, input_dim: int):
        super().__init__()
        d = cfg["d_model"]
        self.temporal_enc = TemporalEncoder(
            input_dim, d, cfg["n_heads"], cfg["n_layers"], cfg["dropout"]
        )
        self.st_gat = SpatioTemporalGAT(d, cfg["gat_heads"], cfg["dropout"])
        self.proj = nn.Sequential(
            nn.Linear(d, cfg["hidden_dim"]), nn.GELU(), nn.Dropout(cfg["dropout"]),)
        n_horizons = len(cfg["pred_horizons"])
        self.multi_head = nn.Linear(cfg["hidden_dim"], n_horizons)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_seq: torch.Tensor, edge_index: torch.LongTensor, )-> torch.Tensor:
        h = self.temporal_enc(x_seq)
        h = self.st_gat(h, edge_index)
        h = self.proj(h)
        return self.multi_head(h)

class DroughtDataset(Dataset):

    def __init__(
        self, climate_df: pd.DataFrame, teleconn_df: pd.DataFrame, seq_len: int = 24, pred_horizons: List[int] = None, scaler: Optional[StandardScaler] = None, fit_scaler: bool = True,
    ):
        self.seq_len = seq_len
        self.pred_horizons = pred_horizons or [1,2,3,4,5,6]
        self.samples = []

        teleconn_cols = teleconn_df.columns.tolist() if not teleconn_df.empty else []
        if teleconn_cols:
            df = climate_df.join(teleconn_df, how="left").ffill()
        else:
            df = climate_df.copy()

        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
        df["lat_norm"] = df["lat"] / 90.0
        df["lon_norm"] = df["lon"] / 180.0

        self.feature_cols = ["precip_mm","temp_c","et0_mm","soil_moist","wind_ms", "spi6","spei6",] + teleconn_cols + ["month_sin","month_cos","lat_norm","lon_norm"]

        self.feature_cols = [c for c in self.feature_cols if c in df.columns]

        features = df[self.feature_cols].fillna(0).values
        if fit_scaler:
            self.scaler = StandardScaler().fit(features)
        else:
            self.scaler = scaler
        df[self.feature_cols] = self.scaler.transform(
            df[self.feature_cols].fillna(0).values
        )

        node_keys = df.groupby(["lat","lon"]).groups.keys()
        self.node_coords = list(node_keys)
        N = len(self.node_coords)

        node_series = {}
        common_dates = None
        for (lat, lon) in self.node_coords:
            grp = df[(df["lat"] == lat) & (df["lon"] == lon)].sort_index()
            node_series[(lat, lon)] = grp
            if common_dates is None:
                common_dates = grp.index
            else:
                common_dates = common_dates.intersection(grp.index)

        common_dates = sorted(common_dates)
        T_total = len(common_dates)

        if T_total < seq_len + max(self.pred_horizons):
            raise ValueError(
                f"Not enough common timesteps ({T_total}) across all nodes for "
                f"seq_len={seq_len} + max_horizon={max(self.pred_horizons)}. "
                f"Try a longer date range or check that all gridpoints have data."
            )

        all_features = np.zeros((N, T_total, len(self.feature_cols)), dtype=np.float32)
        all_spei6 = np.zeros((N, T_total), dtype=np.float32)
        for i, (lat, lon) in enumerate(self.node_coords):
            grp = node_series[(lat, lon)].loc[common_dates, self.feature_cols].values
            spei = node_series[(lat, lon)].loc[common_dates, "spei6"].values
            all_features[i] = grp
            all_spei6[i] = spei

        max_h = max(self.pred_horizons)
        for t in range(seq_len, T_total - max_h + 1):
            x = all_features[:, t - seq_len : t, :]
            y = np.stack([
                all_spei6[:, t + h - 1] for h in self.pred_horizons
            ], axis=1)
            self.samples.append((
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class DroughtTrainer:

    def __init__(self, model: DroughtGAT, cfg: dict, edge_index: torch.LongTensor):
        self.model = model
        self.cfg = cfg
        self.edge_index = edge_index
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.edge_index = edge_index.to(self.device)
        Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    def drought_weighted_mse(
        self, pred: torch.Tensor, target: torch.Tensor, weight_extreme: float = 3.0
    ) -> torch.Tensor:

        loss = F.mse_loss(pred, target, reduction="none")
        weights = torch.where(target < -1.5, weight_extreme, 1.0)
        return (loss * weights).mean()

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x_seq, y in loader:

            x_seq = x_seq.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()

            preds = []
            for b in range(x_seq.shape[0]):
                preds.append(self.model(x_seq[b], self.edge_index))
            pred = torch.stack(preds, dim=0)

            loss = self.drought_weighted_mse(
                pred.view(-1, pred.shape[-1]), y.view(-1, y.shape[-1])
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_pred, all_true = [], []
        for x_seq, y in loader:
            x_seq = x_seq.to(self.device)
            preds = []
            for b in range(x_seq.shape[0]):
                preds.append(self.model(x_seq[b], self.edge_index))
            pred = torch.stack(preds, dim=0)
            all_pred.append(pred.cpu().numpy().reshape(-1, pred.shape[-1]))
            all_true.append(y.numpy().reshape(-1, y.shape[-1]))

        pred_arr = np.concatenate(all_pred)
        true_arr = np.concatenate(all_true)

        metrics = {}
        for i, h in enumerate(self.cfg["pred_horizons"]):
            rmse = np.sqrt(mean_squared_error(true_arr[:, i], pred_arr[:, i]))
            r2 = r2_score(true_arr[:, i], pred_arr[:, i])

            obs_dry = (true_arr[:, i] < -1.0).astype(int)
            pred_dry = (pred_arr[:, i] < -1.0).astype(int)
            hits = np.sum((obs_dry == 1) & (pred_dry == 1))
            misses = np.sum((obs_dry == 1) & (pred_dry == 0))
            false_alm = np.sum((obs_dry == 0) & (pred_dry == 1))
            pod = hits / (hits + misses + 1e-8)
            far = false_alm / (hits + false_alm + 1e-8)
            metrics[f"lead{h}_rmse"]  = rmse
            metrics[f"lead{h}_r2"] = r2
            metrics[f"lead{h}_pod"] = pod
            metrics[f"lead{h}_far"] = far
        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        best_val_rmse = float("inf")
        patience_count = 0
        history = []

        for epoch in range(self.cfg["epochs"]):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_rmse_6 = val_metrics.get("lead6_rmse", 999)
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
                    {"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "metrics": val_metrics}, f"{self.cfg['checkpoint_dir']}/droughtgat_best.pt"
                )
            else:
                patience_count += 1
                if patience_count >= self.cfg["patience"]:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break

        return pd.DataFrame(history)

class DroughtEvaluator:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    def skill_summary_table(
        self,
        metrics: Dict[str, float],
        baselines: Optional[Dict[str, Dict]] = None,
    ) -> pd.DataFrame:

        rows = []
        for h in self.cfg["pred_horizons"]:
            row = {
                "Lead (months)": h,
                "RMSE": f"{metrics.get(f'lead{h}_rmse', np.nan):.3f}",
                "R²": f"{metrics.get(f'lead{h}_r2',   np.nan):.3f}",
                "POD": f"{metrics.get(f'lead{h}_pod',  np.nan):.3f}",
                "FAR": f"{metrics.get(f'lead{h}_far',  np.nan):.3f}",
            }
            if baselines:
                for bname, bmet in baselines.items():
                    row[f"{bname} RMSE"] = f"{bmet.get(f'lead{h}_rmse', np.nan):.3f}"
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_skill_by_lead(self, history_df: pd.DataFrame, save_path: str = None):

        leads = self.cfg["pred_horizons"]
        rmse_v = [history_df[f"lead{h}_rmse"].iloc[-1] for h in leads]
        r2_v = [history_df[f"lead{h}_r2"].iloc[-1]   for h in leads]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(leads, rmse_v, "o-", color="#2563EB", lw=2, ms=8)
        axes[0].set(xlabel="Lead Time (months)", ylabel="RMSE", title="SPEI-6 Forecast RMSE vs. Lead Time")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(leads, r2_v, "s-", color="#16A34A", lw=2, ms=8)
        axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[1].set(xlabel="Lead Time (months)", ylabel="R²", title="Coefficient of Determination vs. Lead Time")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_global_skill_map(
        self,
        skill_grid: np.ndarray, lats: np.ndarray, lons: np.ndarray, lead: int = 6, save_path: str = None,
    ):

        if not CARTOPY_AVAILABLE:
            raise ImportError(
                "cartopy is required for plot_global_skill_map() but is not installed"
            )
        fig = plt.figure(figsize=(16, 8))
        ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", alpha=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        lon_g, lat_g = np.meshgrid(lons, lats)
        cmap = plt.cm.RdYlGn
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-0.5, vmax=1.0)
        scatter = ax.scatter(
            lon_g.ravel(), lat_g.ravel(),c=skill_grid.ravel(), cmap=cmap, norm=norm, s=4, transform=ccrs.PlateCarree(), alpha=0.85,
        )
        plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02, label=f"R² (SPEI-6, {lead}-month lead)")
        ax.set_title(f"DroughtGAT Global Skill Map — {lead}-Month Lead SPEI-6", fontsize=14, fontweight="bold")

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

class BaselineModels:

    @staticmethod
    def persistence(spei_series: np.ndarray, lead: int) -> np.ndarray:

        return spei_series[:-lead]

    @staticmethod
    def climatological_mean(spei_series: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:

        monthly_means = {m: spei_series[dates.month == m].mean() for m in range(1, 13)}
        return np.array([monthly_means[d.month] for d in dates])

    @staticmethod
    def lstm_baseline(
        x_seq:   torch.Tensor, input_dim: int, d_model: int = 64, n_horizons: int = 6,
    ) -> nn.Module:

        return nn.Sequential(
            nn.LSTM(input_dim, d_model, num_layers=2, batch_first=True, dropout=0.1),
        )

def run_full_pipeline(
    use_commercial_api: bool = False,
    regions: Optional[List[str]] = None,
    n_per_region: int = 60,
    start_date: str = "1981-01-01", end_date:   str = "2023-12-31",
):

    log.info("=" * 70)
    log.info("DroughtGAT - Multi-Region Teleconnection-Aware Drought Prediction")
    log.info("=" * 70)

    pipe = WeatherDataPipeline(CFG)
    log.info("Step 1/6: Fetching climate data across study regions...")
    climate_df, teleconn_df = pipe.build_regional_grid(
        regions=regions, n_per_region=n_per_region, use_commercial_api=use_commercial_api, start_date=start_date, end_date=end_date,
    )
    climate_df.to_parquet(f"{CFG['data_dir']}/climate_raw.parquet")
    teleconn_df.to_parquet(f"{CFG['data_dir']}/teleconn.parquet")

    log.info("Step 2/6: Computing SPEI-6...")
    calc       = DroughtIndexCalculator()
    climate_df = calc.compute_all(climate_df)
    climate_df.to_parquet(f"{CFG['data_dir']}/climate_spei.parquet")

    log.info("Step 3/6: Building dataset and graph...")
    dataset = DroughtDataset(climate_df, teleconn_df, CFG["seq_len"], CFG["pred_horizons"])
    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_ds, val_ds, test_ds = (
        torch.utils.data.Subset(dataset, range(0, n_train)), torch.utils.data.Subset(dataset, range(n_train, m_train + n_val)), torch.utils.data.Subset(dataset, range(n_train + n_val, n)),
    )
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=2)

    coords = climate_df.groupby(["lat","lon"]).first()[[]].reset_index()[["lat","lon"]].values
    graph_builder = DroughtGraph(radius_km=500, corr_threshold=0.30)
    log.info(f"  Building spatial graph for {len(coords)} nodes...")
    edge_index = graph_builder.build_spatial_edges(coords[:min(200, len(coords))])

    log.info("Step 4/6: Training DroughtGAT...")
    input_dim = len(dataset.feature_cols)
    model = DroughtGAT(CFG, input_dim)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  DroughtGAT parameters: {n_params:,}")

    trainer = DroughtTrainer(model, CFG, edge_index)
    history = trainer.train(train_loader, val_loader)
    history.to_csv(f"{CFG['output_dir']}/training_history.csv", index=False)

    log.info("Step 5/6: Evaluating on test set...")

    ckpt = torch.load(f"{CFG['checkpoint_dir']}/droughtgat_best.pt", map_location=trainer.device)
    model.load_state_dict(ckpt["model"])
    evaluator  = DroughtEvaluator(CFG)
    test_metrics = trainer.evaluate(test_loader)

    table = evaluator.skill_summary_table(test_metrics)
    table.to_csv(f"{CFG['output_dir']}/skill_table.csv", index=False)
    log.info("\n" + table.to_string(index=False))

    log.info("Step 6/6: Generating publication figures...")
    evaluator.plot_skill_by_lead(history, save_path=f"{CFG['output_dir']}/fig_skill_vs_lead.png")
    log.info("Pipeline complete. Outputs saved to ./outputs/")
    return model, history, test_metrics

if __name__ == "__main__":

    model, history, metrics = run_full_pipeline(
        use_commercial_api = False, regions = None, n_per_region = 60, start_date = "1981-01-01", end_date = "2023-12-31",
    )
