# DroughtTFT

**A Multi-Region Temporal Fusion Transformer for Sub-Seasonal to Seasonal SPEI-6 Drought Forecasting at 1–6 Month Lead Times**

ERA5 1950–2023 · Test period 2018–2023 · 5 regions · 3 continents

## Results

| Metric | Value |
|--------|-------|
| Mean R² (lead 1 month) | 0.559 |
| Mean R² (lead 6 months) | 0.041 |
| Sahel R² (lead 1 month) | 0.634 |
| MSSS (lead 1 → lead 6) | 0.148 → 0.487 |
| Regions positive R² at lead 6 | 3/5 |

## Study Regions

| Region | Teleconnection driver |
|--------|----------------------|
| Horn of Africa | ENSO/IOD |
| NE Brazil | ENSO/Atlantic |
| Sahel | AMO |
| SW United States | PDO/ENSO |
| South Asia | ENSO/IOD monsoon |

## Pipeline

```
1. download_era5.py       Download ERA5 monthly data (1950-2023) per region
2. run_region.py          Train DroughtTFT per region
3. ablation.py            Run TFT-only ablation per region
4. lstm_baseline.py       Train LSTM baseline per region
5. compile_results.py     Compile skill tables across regions
6. compile_ablation.py    Compile ablation results
7. compile_lstm.py        Compile LSTM comparison + figures
```

## Setup

```bash
pip install torch pandas numpy xarray cdsapi scikit-learn matplotlib seaborn
```

For ERA5 downloads, create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR-CDS-TOKEN
```

## Usage

```bash
# 1. Download data for each region (change REGION= at bottom of file)
python download_era5.py   # REGION = "horn_of_africa"
python download_era5.py   # REGION = "ne_brazil"
python download_era5.py   # REGION = "sahel"
python download_era5.py   # REGION = "sw_us"
python download_era5.py   # REGION = "south_asia"

# 2. Train DroughtTFT (change CURRENT_REGION= at bottom)
python run_region.py      # CURRENT_REGION = "horn_of_africa"
# ... repeat for all 5 regions

# 3. Run ablation
python ablation.py        # CURRENT_REGION = "horn_of_africa"
# ... repeat for all 5 regions
python compile_ablation.py

# 4. LSTM baseline
python lstm_baseline.py   # CURRENT_REGION = "horn_of_africa"
# ... repeat for all 5 regions
python compile_lstm.py

# 5. Compile all results
python compile_results.py
```

## Model Architecture

DroughtTFT applies an independent Temporal Fusion Transformer per gridpoint:
- 24-month lookback window
- 2-layer multi-head self-attention (d=64, 4 heads)
- 14 input features: ERA5 climate variables + teleconnection indices (Nino3.4, PDO, AMO, IOD) + cyclic month encoding + lat/lon
- Multi-horizon output: simultaneous 1–6 month lead forecasts
- Drought-weighted MSE loss (3× weight for SPEI-6 < -1.0)

## Files

| File | Description |
|------|-------------|
| `drought_model.py` | Core model, dataset, training classes |
| `download_era5.py` | ERA5 data download and preprocessing |
| `run_region.py` | Train DroughtTFT per region |
| `ablation.py` | TFT-only ablation study |
| `lstm_baseline.py` | LSTM baseline training |
| `compile_results.py` | Aggregate skill metrics across regions |
| `compile_ablation.py` | Aggregate ablation results |
| `compile_lstm.py` | Aggregate LSTM results and generate figures |

## Citation

If you use this code, please cite:

```bibtex
@article{droughttft2025,
  title={DroughtTFT: A Multi-Region Temporal Fusion Transformer for
         Sub-Seasonal to Seasonal SPEI-6 Drought Forecasting at
         1--6 Month Lead Times},
  author={[Author Name(s)]},
  journal={Environmental Research Letters},
  year={2025}
}
```

## License

MIT License
