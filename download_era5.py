import os, zipfile, logging
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
Path("./data/era5").mkdir(parents=True, exist_ok=True)

REGIONS = {
    "horn_of_africa": {"north": 15, "south": -5, "west": 33, "east": 51}, "ne_brazil": {"north": -2, "south": -15, "west": -45, "east": -34}, "sahel": {"north": 18, "south": 10, "west": -17, "east": 35}, "sw_us": {"north": 42, "south": 28, "west": -124, "east": -103}, "south_asia": {"north": 30, "south": 8, "west": 68, "east": 90},
}
YEARS  = list(range(1950, 2024))
VARS_A = ["total_precipitation", "evaporation"]
VARS_B = ["2m_temperature", "volumetric_soil_water_layer_1", "10m_u_component_of_wind", "10m_v_component_of_wind"]

def download_vars(vars_list, suffix, region_name, region, years):
    out = f"./data/era5/{region_name}_{suffix}.nc"
    if Path(out).exists():
        log.info(f"Already exists, skipping: {out}")
        return out
    c = cdsapi.Client()
    log.info(f"Requesting ERA5 [{suffix}] for {region_name} {years[0]}-{years[-1]}...")
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {"product_type": "monthly_averaged_reanalysis", "variable": vars_list,"year": [str(y) for y in years], "month": [f"{m:02d}" for m in range(1, 13)], "time": "00:00", "area": [region["north"], region["west"], region["south"], region["east"]], "format": "netcdf",},out,)
    log.info(f"Downloaded: {out}")
    return out

def open_nc(path):
    actual = path
    if zipfile.is_zipfile(path):
        ed = path + "_extracted"
        os.makedirs(ed, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            nc_files = [f for f in z.namelist() if f.endswith(".nc")]
            if not nc_files:
                raise ValueError(f"No .nc inside zip: {z.namelist()}")
            z.extractall(ed)
            actual = os.path.join(ed, nc_files[0])
        log.info(f"Extracted: {actual}")
    for engine in ["netcdf4", "scipy", "h5netcdf"]:
        try:
            ds = xr.open_dataset(actual, engine=engine)
            log.info(f"Opened '{engine}' — vars: {list(ds.data_vars)}")
            return ds
        except Exception:
            pass
    raise RuntimeError(f"Cannot open {actual} — try: pip install netcdf4")

def ds_to_df_chunked(ds, var_map, chunk_years=10):

    lat_dim  = "latitude"  if "latitude"  in ds.coords else "lat"
    lon_dim  = "longitude" if "longitude" in ds.coords else "lon"
    time_dim = next((c for c in ["valid_time", "time"] if c in ds.coords), None)
    if time_dim is None:
        raise ValueError(f"No time coord. Available: {list(ds.coords)}")

    avail = {v: col for v, col in var_map.items() if v in ds.data_vars}
    if not avail:
        raise ValueError(f"None of {list(var_map)} found. Got: {list(ds.data_vars)}")

    times = pd.to_datetime(ds[time_dim].values)
    years = sorted(set(times.year))
    chunks = [years[i:i+chunk_years] for i in range(0, len(years), chunk_years)]
    log.info(f"  Converting in {len(chunks)} chunks of ~{chunk_years} years each...")

    parts = []
    for chunk in chunks:
        year_min, year_max = chunk[0], chunk[-1]
        mask = (times.year >= year_min) & (times.year <= year_max)
        time_vals = times[mask]

        ds_chunk = ds[list(avail.keys())].isel(
            {time_dim: np.where(mask)[0]}
        ).rename(avail)

        rows = []
        lats = ds_chunk[lat_dim].values
        lons = ds_chunk[lon_dim].values
        for col_name in avail.values():
            pass

        n_times = len(time_vals)
        n_lats  = len(lats)
        n_lons  = len(lons)

        t_arr = np.repeat(time_vals, n_lats * n_lons)
        la_arr = np.tile(np.repeat(lats, n_lons), n_times)
        lo_arr = np.tile(lons, n_times * n_lats)

        df_chunk = pd.DataFrame({"date": t_arr, "lat": la_arr, "lon": lo_arr})
        df_chunk["date"] = pd.to_datetime(df_chunk["date"])\
                             .dt.to_period("M").dt.to_timestamp()
        df_chunk["lat"]  = df_chunk["lat"].round(4)
        df_chunk["lon"]  = df_chunk["lon"].round(4)

        for era5_name, col_name in avail.items():
            vals = ds_chunk[col_name].values
            df_chunk[col_name] = vals.reshape(-1)

        df_chunk = df_chunk.drop_duplicates(subset=["date","lat","lon"])
        parts.append(df_chunk)
        log.info(f"    Chunk {year_min}-{year_max}: {len(df_chunk):,} rows")

    df = pd.concat(parts, ignore_index=True)
    df = df.set_index(["date","lat","lon"]).dropna(how="all")
    return df

def merge_and_convert(ds_a, ds_b):
    log.info("Converting precip/ET file (chunked)...")
    df_a = ds_to_df_chunked(ds_a, {"tp": "precip_raw", "e": "et0_raw"})
    log.info(f"  precip/ET: {len(df_a):,} rows")

    log.info("Converting temp/wind/soil file (chunked)...")
    df_b = ds_to_df_chunked(ds_b, {"t2m": "temp_raw", "swvl1": "soil_moist", "u10": "u10", "v10": "v10"})
    log.info(f"  temp/wind: {len(df_b):,} rows")

    df = df_a.join(df_b, how="inner", lsuffix="", rsuffix="_r")
    df = df.drop(columns=[c for c in df.columns  if c.endswith("_r") or c in ["number","expver"]], errors="ignore")
    log.info(f"After inner join: {len(df):,} rows")

    if len(df) == 0:
        raise RuntimeError("Inner join produced 0 rows. Check coordinate alignment.")

    df = df.reset_index()
    df["precip_mm"] = df["precip_raw"] * 1000
    df["et0_mm"] = np.abs(df["et0_raw"]) * 1000
    df["temp_c"] = df["temp_raw"] - 273.15
    df["wind_ms"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
    df = df.set_index("date")
    df = df[["precip_mm","temp_c","et0_mm","soil_moist","wind_ms","lat","lon"]]
    df = df[df["temp_c"].notna() & df["precip_mm"].notna()]

    n_pts = df.groupby(["lat","lon"]).ngroups
    log.info(f"Final DataFrame: {len(df):,} rows, {n_pts} gridpoints")
    log.info(f"Date range: {df.index.min()} to {df.index.max()}")
    log.info(f"\nStats:\n{df[['precip_mm','temp_c','et0_mm']].describe().to_string()}")
    return df

REGION = "sahel"

if __name__ == "__main__":
    region = REGIONS[REGION]
    log.info(f"Region: {REGION} | Years: {YEARS[0]}-{YEARS[-1]} "
             f"({len(YEARS)} years, ~{len(YEARS)*12} months)")

    path_a = download_vars(VARS_A, "precip_et", REGION, region, YEARS)
    path_b = download_vars(VARS_B, "temp_wind",  REGION, region, YEARS)

    log.info("Opening and merging both files...")
    ds_a = open_nc(path_a)
    ds_b = open_nc(path_b)
    df   = merge_and_convert(ds_a, ds_b)

    parq = f"./data/era5/{REGION}_monthly.parquet"
    df.to_parquet(parq)
    log.info(f"\nSaved: {parq}")
    log.info(f"Rows: {len(df):,} | Gridpoints: {df.groupby(['lat','lon']).ngroups}")
    log.info(f"\nDone. Now run: python run_region.py  (CURRENT_REGION = '{REGION}')")
