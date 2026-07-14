# -*- coding: utf-8 -*-
"""
Script 2: Feature engineering + LightGBM global + validasi time-based + submission.
Jalankan: python 2_train_lgbm.py
Output : submission_lgbm.csv, feature_importance.csv
Butuh  : pandas, numpy, lightgbm  (pip install lightgbm)

Desain:
- Regresi tabular dengan future covariates (data lingkungan tersedia s/d akhir test).
- TANPA lag target (horizon 8 bulan, nilai aktual tak tersedia saat inference).
- Target dinormalisasi per stasiun: (y - mean) / std.
- Validasi: train <= 2025-01-18, val 8 bulan berikutnya (meniru horizon test).
- Prediksi akhir: retrain full data, clip ke rentang historis per stasiun,
  blend dengan climatology (default 0.85 LGBM / 0.15 clim).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = "."
BLEND_W = 0.85          # bobot LGBM saat blend dengan climatology
VAL_CUT = pd.Timestamp("2025-01-18 23:59:59")

# ================= Load =================
print("Loading data...")
train = pd.read_csv(f"{BASE}/train.csv", parse_dates=["datetime"])
test = pd.read_csv(f"{BASE}/test.csv")
test["datetime"] = pd.to_datetime(test["id"].str[:19])
test["nama_pos"] = test["id"].str[22:]
env = pd.read_csv(f"{BASE}/data_pendukung/data_lingkungan.csv",
                  parse_dates=["datetime"])
coords = pd.read_csv(f"{BASE}/data_pendukung/koordinat_pos.csv")

# ================= Fitur dari data lingkungan (hourly -> rolling) =================
print("Building rolling features (agak lama, ~1-3 menit)...")
env = env.sort_values(["nama_pos", "datetime"]).reset_index(drop=True)
num_ffill = ["rainfall_mm", "rainfall_openmeteo_mm", "humidity_pct",
             "temperature_c", "surface_pressure_hpa", "solar_radiation_mj_m2",
             "soil_moisture_0_7cm", "soil_moisture_7_28cm",
             "soil_moisture_28_100cm", "soil_moisture_100_255cm",
             "dew_point_c", "wind_speed_kmh", "cloud_cover_pct"]
env[num_ffill] = env.groupby("nama_pos")[num_ffill].ffill()

g = env.groupby("nama_pos", sort=False)

def roll(col, window, agg):
    r = g[col].rolling(window, min_periods=1)
    return getattr(r, agg)().reset_index(level=0, drop=True)

# Hujan: akumulasi bertingkat (jam). Ini fitur terpenting.
for col, tag in [("rainfall_mm", "rain"), ("rainfall_openmeteo_mm", "rainom")]:
    for w in [24, 72, 168, 336, 720, 2160]:  # 1d,3d,7d,14d,30d,90d
        env[f"{tag}_sum_{w}h"] = roll(col, w, "sum")
    env[f"{tag}_max_168h"] = roll(col, 168, "max")

# Soil moisture: level saat ini + rolling mean
for col in ["soil_moisture_0_7cm", "soil_moisture_7_28cm",
            "soil_moisture_28_100cm", "soil_moisture_100_255cm"]:
    env[f"{col}_m168h"] = roll(col, 168, "mean")
    env[f"{col}_m720h"] = roll(col, 720, "mean")

# Cuaca lain: rata-rata harian & bulanan
for col in ["temperature_c", "humidity_pct", "surface_pressure_hpa",
            "solar_radiation_mj_m2"]:
    env[f"{col}_m24h"] = roll(col, 24, "mean")
    env[f"{col}_m720h"] = roll(col, 720, "mean")

roll_cols = [c for c in env.columns if any(
    s in c for s in ["_sum_", "_m24h", "_m168h", "_m720h", "_max_"])]
level_cols = ["rainfall_max_24h_mm", "soil_moisture_0_7cm",
              "soil_moisture_7_28cm", "soil_moisture_28_100cm",
              "soil_moisture_100_255cm", "temperature_c", "humidity_pct",
              "surface_pressure_hpa", "cloud_cover_pct", "wind_speed_kmh",
              "dew_point_c", "rmm1", "rmm2", "mjo_phase", "mjo_amplitude",
              "mjo_active", "nino_34", "built_surface_m2", "landcover_class"]
env_feat = env[["datetime", "nama_pos"] + level_cols + roll_cols]

# ================= Merge & fitur kalender =================
def add_features(df):
    df = df.merge(env_feat, on=["datetime", "nama_pos"], how="left")
    df = df.merge(coords, on="nama_pos", how="left")
    doy = df.datetime.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df.datetime.dt.month
    df["hour"] = df.datetime.dt.hour
    df["pos_cat"] = df.nama_pos.astype("category")
    return df

train = add_features(train)
test = add_features(test)

miss = train[env_feat.columns.drop(["datetime", "nama_pos"])].isna().mean()
print("Fitur dengan missing >1% setelah merge:")
print(miss[miss > 0.01].round(3).to_string() if (miss > 0.01).any() else "  (tidak ada)")

# Fitur climatology sebagai prior: median per (pos, month, hour) dari TRAIN saja
clim = train.groupby(["nama_pos", "month", "hour"]).tma_mdpl.median().rename("clim_med")
train = train.merge(clim, on=["nama_pos", "month", "hour"], how="left")
test = test.merge(clim, on=["nama_pos", "month", "hour"], how="left")
clim_m = train.groupby(["nama_pos", "month"]).tma_mdpl.median().rename("clim_med_m")
test = test.merge(clim_m, on=["nama_pos", "month"], how="left")
test["clim_med"] = test.clim_med.fillna(test.pop("clim_med_m"))

# ================= Normalisasi target per stasiun =================
stats = train.groupby("nama_pos").tma_mdpl.agg(["mean", "std", "min", "max"])
stats["std"] = stats["std"].replace(0, 1e-6)
for df in (train, test):
    df["pos_mean"] = df.nama_pos.map(stats["mean"])
    df["pos_std"] = df.nama_pos.map(stats["std"])
train["y_norm"] = (train.tma_mdpl - train.pos_mean) / train.pos_std
train["clim_norm"] = (train.clim_med - train.pos_mean) / train.pos_std
test["clim_norm"] = (test.clim_med - test.pos_mean) / test.pos_std

FEATURES = (level_cols + roll_cols +
            ["latitude", "longitude", "doy_sin", "doy_cos", "month", "hour",
             "clim_norm", "pos_cat"])
CAT = ["pos_cat", "landcover_class", "month", "hour", "mjo_phase"]
for df in (train, test):
    for c in CAT:
        df[c] = df[c].astype("category")

PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.03,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, seed=42)

# ================= Validasi time-based =================
tr = train[train.datetime <= VAL_CUT]
va = train[train.datetime > VAL_CUT]
print(f"\nTrain: {len(tr)} rows s/d {tr.datetime.max().date()} | "
      f"Val: {len(va)} rows ({va.datetime.min().date()} -> {va.datetime.max().date()})")

dtr = lgb.Dataset(tr[FEATURES], tr.y_norm)
dva = lgb.Dataset(va[FEATURES], va.y_norm, reference=dtr)
model = lgb.train(PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)])
best_iter = model.best_iteration
print(f"Best iteration: {best_iter}")

# RMSE di skala ASLI (ini yang dinilai Kaggle)
pred_va = model.predict(va[FEATURES]) * va.pos_std + va.pos_mean
# clip ke rentang historis train-split per stasiun (margin 10% dari range)
st = tr.groupby("nama_pos").tma_mdpl.agg(["min", "max"])
rng = (st["max"] - st["min"]).clip(lower=0.1)
lo = va.nama_pos.map(st["min"] - 0.10 * rng).to_numpy()
hi = va.nama_pos.map(st["max"] + 0.10 * rng).to_numpy()
pred_va = np.clip(pred_va, lo, hi)

rmse_lgbm = float(np.sqrt(np.mean((va.tma_mdpl - pred_va) ** 2)))
rmse_clim = float(np.sqrt(np.mean((va.tma_mdpl - va.clim_med) ** 2)))
pred_blend = BLEND_W * pred_va + (1 - BLEND_W) * va.clim_med
rmse_blend = float(np.sqrt(np.mean((va.tma_mdpl - pred_blend) ** 2)))
print(f"\n=== RMSE VALIDASI (skala asli) ===")
print(f"Climatology : {rmse_clim:.4f}")
print(f"LightGBM    : {rmse_lgbm:.4f}")
print(f"Blend {BLEND_W:.2f}  : {rmse_blend:.4f}")

va_ = va.copy(); va_["err2"] = (va.tma_mdpl - pred_va) ** 2
print("\nRMSE LGBM per stasiun (top 10 penyumbang — fokus perbaikan di sini):")
print(va_.groupby("nama_pos").err2.mean().pow(0.5)
      .sort_values(ascending=False).head(10).round(3).to_string())

imp = pd.DataFrame({"feature": FEATURES,
                    "gain": model.feature_importance("gain")}
                   ).sort_values("gain", ascending=False)
imp.to_csv(f"{BASE}/feature_importance.csv", index=False)
print("\nTop 15 fitur:")
print(imp.head(15).to_string(index=False))

# ================= Retrain full & submit =================
print("\nRetrain on full data...")
n_final = int(best_iter * 1.1)
dfull = lgb.Dataset(train[FEATURES], train.y_norm)
model_f = lgb.train(PARAMS, dfull, num_boost_round=n_final)

pred = model_f.predict(test[FEATURES]) * test.pos_std + test.pos_mean
rng = (stats["max"] - stats["min"]).clip(lower=0.1)
lo = test.nama_pos.map(stats["min"] - 0.10 * rng).to_numpy()
hi = test.nama_pos.map(stats["max"] + 0.10 * rng).to_numpy()
pred = np.clip(pred, lo, hi)
pred = BLEND_W * pred + (1 - BLEND_W) * test.clim_med.to_numpy()

sub = pd.DataFrame({"id": test.id, "tma_mdpl": pred})
sub.to_csv(f"{BASE}/submission_lgbm.csv", index=False)
print(f"\nSaved submission_lgbm.csv ({len(sub)} rows)")
print(sub.head(3).to_string(index=False))
