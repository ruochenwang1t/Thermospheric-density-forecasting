import os
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime, timedelta
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

CONFIG = {
    'TEST_DIR'    : 'Test/Test7/raw_data.mat',
    'START_DATE'  : '2024/05/10 00:00:00',
    'END_DATE'    : '2024/05/13 23:59:00',
    'OUTPUT'      : 'WAM_IPE_Prediction.csv',
    'DATA_DIR'    : 'data_root/v1.2/',
    'SEARCH_HOURS': 6,
    'USE_INTERP'  : False,
    # columns of [alt(m), lat(deg), lon(deg in (-180,180])]
    # 'ALT_LAT_LON_COL_IDX': (6, 7, 8), # CHAMP/GRACE
    # 'Den_IDX': 10, # CHAMP/GRACE
    'ALT_LAT_LON_COL_IDX': (7, 8, 9), # SWARM
    'Den_IDX': 6, # SWARM
}

def find_match_traj(cfg):
    """Load raw MAT -> DataFrame; ONLY do: alt[m]->km, lat clip, lon->0..360."""
    data = sio.loadmat(cfg['TEST_DIR'])['save_data']

    year, month, day  = data[:,0].astype(int), data[:,1].astype(int), data[:,2].astype(int)
    hour, minute, sec = data[:,3].astype(int), data[:,4].astype(int), data[:,5].astype(int)
    rho_true = data[:,cfg['Den_IDX']].astype(float) if data.shape[1] > 6 else np.full(len(year), np.nan)

    ialt, ilat, ilon = cfg['ALT_LAT_LON_COL_IDX']
    alt_km = data[:, ialt].astype(float) / 1000.0                   # m -> km
    lat    = np.clip(data[:, ilat].astype(float), -90.0, 90.0)      # keep as-is
    lon    = np.mod(data[:, ilon].astype(float) + 360.0, 360.0)     # (-180,180] -> [0,360)

    df = pd.DataFrame({
        'year': year,'month': month,'day': day,
        'hour': hour,'minute': minute,'second': sec,
        'rho_true': rho_true, 'alt': alt_km, 'lat': lat, 'lon': lon
    })
    df.insert(0, 'datetime', pd.to_datetime(df[['year','month','day','hour','minute','second']],
                                            errors='coerce'))
    df = df.dropna(subset=['datetime']).reset_index(drop=True)

    s = datetime.strptime(cfg['START_DATE'], "%Y/%m/%d %H:%M:%S")
    e = datetime.strptime(cfg['END_DATE'],   "%Y/%m/%d %H:%M:%S")
    return df[(df['datetime'] >= s) & (df['datetime'] <= e)].reset_index(drop=True)

def match_wam_files(result, data_dir, search_hours=48):
    """Find matching WAM-IPE file for each timestamp (latest model init, within window)."""
    matched_files, matched_models = [], []
    for t in result['datetime']:
        date_str = t.strftime("%Y%m%d")
        base_dir = os.path.join(data_dir, f"wfs.{date_str}")
        cand = []
        if os.path.exists(base_dir):
            for rh in ['00','06','12','18']:
                d = os.path.join(base_dir, rh)
                if not os.path.exists(d): continue
                init_t = pd.to_datetime(f"{date_str} {rh}:00:00")
                if not ((init_t - timedelta(hours=3)) <= t <= (init_t + timedelta(hours=48))): 
                    continue
                if t < init_t or (t - init_t).total_seconds()/3600.0 > search_hours:
                    continue
                dtstr = t.strftime("%Y%m%d_%H%M%S")
                found = None
                for root, _, files in os.walk(d):
                    for f in files:
                        if f.endswith('.nc') and dtstr in f:
                            found = os.path.join(root, f); break
                    if found: break
                if found: cand.append((init_t, found))
        if cand:
            cand.sort(key=lambda x: x[0])
            matched_files.append([cand[-1][1]])
            matched_models.append([cand[-1][0].strftime("%Y%m%d_%H%M")])
        else:
            matched_files.append([None]); matched_models.append([None])
    result = result.copy()
    result['matched_files'], result['model_inits'] = matched_files, matched_models
    return result

def extract_density_from_nc(result, cfg):
    """Sample density at (alt[km], lat[deg], lon[deg]); interpolate if USE_INTERP."""
    out = result.copy()
    preds, use_interp = [], bool(cfg.get('USE_INTERP', False))

    for _, row in out.iterrows():
        mf = row['matched_files'][0] if isinstance(row['matched_files'], list) else None
        if not mf:
            preds.append(np.nan); continue
        try:
            with Dataset(mf, 'r') as nc:
                latg = np.array(nc.variables['lat'][:], dtype=float)
                long = np.array(nc.variables['lon'][:], dtype=float)
                hg   = np.array(nc.variables['hlevs'][:], dtype=float)    # km
                den  = nc.variables['den']
                rho  = np.ma.filled(den[:].astype(np.float64) *
                                    float(getattr(den, "scale_factor", 1.0)) +
                                    float(getattr(den, "add_offset", 0.0)), np.nan)
                rho3 = rho[0] if (rho.ndim == 4 and rho.shape[0] == 1) else rho

                # Map lon to grid convention
                lon0 = float(row['lon'])
                if np.min(long) < 0.0:   # grid is [-180,180]
                    lon0 = ((lon0 + 180.0) % 360.0) - 180.0
                    if lon0 == 180.0: lon0 = -180.0
                lat0 = float(np.clip(row['lat'], np.min(latg), np.max(latg)))
                alt0 = float(np.clip(row['alt'], np.min(hg),   np.max(hg)))

                if use_interp:
                    # ensure increasing for RGI (inline)
                    idx_h = np.argsort(hg);   h_i = hg[idx_h];   r_i = rho3[idx_h, ...]
                    idx_a = np.argsort(latg); la_i = latg[idx_a]; r_i = r_i[:, idx_a, :]
                    idx_o = np.argsort(long); lo_i = long[idx_o]; r_i = r_i[:, :, idx_o]
                    # lon to lo_i convention
                    if np.min(lo_i) < 0.0:
                        lon0 = ((lon0 + 180.0) % 360.0) - 180.0
                        if lon0 == 180.0: lon0 = -180.0
                    else:
                        lon0 = lon0 % 360.0
                    val = RegularGridInterpolator(
                        (h_i, la_i, lo_i), r_i, bounds_error=False, fill_value=np.nan
                    )([[alt0, lat0, lon0]])[0]
                    if not np.isfinite(val):
                        i_h = np.argmin(np.abs(h_i - alt0))
                        i_a = np.argmin(np.abs(la_i - lat0))
                        i_o = np.argmin(np.abs(lo_i - lon0))
                        val = r_i[i_h, i_a, i_o]
                    preds.append(float(val))
                else:
                    # nearest neighbor (inline)
                    i_h = int(np.argmin(np.abs(hg   - alt0)))
                    i_a = int(np.argmin(np.abs(latg - lat0)))
                    # circular nearest lon
                    d = np.abs(long - lon0); d = np.minimum(d, 360.0 - d)
                    i_o = int(np.argmin(d))
                    preds.append(float(rho3[i_h, i_a, i_o]))
        except Exception as e:
            print(f"[Warn] read {mf} failed: {e}")
            preds.append(np.nan)

    out['pred_density'] = preds
    return out

if __name__ == "__main__":
    df = find_match_traj(CONFIG)
    df = match_wam_files(df, CONFIG['DATA_DIR'], CONFIG['SEARCH_HOURS'])
    df = extract_density_from_nc(df, CONFIG)

    cols = ['datetime', 'lat', 'lon', 'alt', 'pred_density', 'rho_true']
    out_df = df.loc[:, cols].rename(columns={'rho_true': 'true_density'})

    out_dir = os.path.dirname(os.path.abspath(CONFIG['TEST_DIR']))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, CONFIG['OUTPUT'])
    out_df.to_csv(out_path, index=False)

    print(out_df.head())
    print("Saved to:", out_path)
