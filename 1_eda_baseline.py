# -*- coding: utf-8 -*-
"""
Script 1: EDA ringkas + baseline climatology submission.
Jalankan: python 1_eda_baseline.py
Output : eda_summary.txt, submission_climatology.csv
Butuh  : pandas, numpy
"""
import pandas as pd
import numpy as np

BASE = "."  # jalankan dari folder uns

# ---------- Load ----------
train = pd.read_csv(f"{BASE}/train.csv", parse_dates=["datetime"])
test = pd.read_csv(f"{BASE}/test.csv")

# id format: "YYYY-MM-DD HH:MM:SS - nama_pos" (nama pos bisa mengandung " - ")
test["datetime"] = pd.to_datetime(test["id"].str[:19])
test["nama_pos"] = test["id"].str[22:]

lines = []
def log(s=""):
    print(s)
    lines.append(str(s))

# ---------- EDA ----------
log(f"TRAIN: {train.shape}, {train.datetime.min()} -> {train.datetime.max()}")
log(f"TEST : {test.shape}, {test.datetime.min()} -> {test.datetime.max()}")
log(f"Stasiun train={train.nama_pos.nunique()}, test={test.nama_pos.nunique()}, "
    f"sama={set(train.nama_pos)==set(test.nama_pos)}")
log(f"NaN target: {train.tma_mdpl.isna().sum()}")
log(f"Duplikat (datetime,pos): {train.duplicated(['datetime','nama_pos']).sum()}")

# Kelengkapan observasi per stasiun (ekspektasi: 3x/hari)
n_days = (train.datetime.max() - train.datetime.min()).days + 1
per_pos = train.groupby("nama_pos").agg(
    n=("tma_mdpl", "size"), mean=("tma_mdpl", "mean"), std=("tma_mdpl", "std"),
    min=("tma_mdpl", "min"), max=("tma_mdpl", "max"),
    first=("datetime", "min"), last=("datetime", "max"))
per_pos["coverage_pct"] = (100 * per_pos["n"] / (n_days * 3)).round(1)
per_pos["range"] = per_pos["max"] - per_pos["min"]
per_pos = per_pos.sort_values("std", ascending=False)

log("\n=== Per stasiun (urut std desc — stasiun atas MENDOMINASI RMSE) ===")
log(per_pos.round(3).to_string())

log("\n>>> Fokus effort ke stasiun dengan std terbesar:")
log(", ".join(per_pos.head(8).index))

# Cek gap besar dalam deret waktu tiap stasiun
log("\n=== Gap > 3 hari per stasiun ===")
for pos, gdf in train.sort_values("datetime").groupby("nama_pos"):
    gaps = gdf.datetime.diff()
    big = gaps[gaps > pd.Timedelta(days=3)]
    if len(big):
        log(f"{pos}: {len(big)} gap, terbesar {gaps.max()}")

# Tren tahunan kasar (mean per stasiun per tahun) — deteksi drift/shift datum
log("\n=== Mean per tahun (deteksi drift) ===")
yr = train.assign(y=train.datetime.dt.year).pivot_table(
    index="nama_pos", columns="y", values="tma_mdpl", aggfunc="mean")
log(yr.round(2).to_string())

# ---------- Baseline climatology ----------
# Median per (stasiun, bulan, jam); fallback (stasiun, bulan) lalu (stasiun)
train["month"] = train.datetime.dt.month
train["hour"] = train.datetime.dt.hour
clim_mh = train.groupby(["nama_pos", "month", "hour"]).tma_mdpl.median()
clim_m = train.groupby(["nama_pos", "month"]).tma_mdpl.median()
clim_p = train.groupby("nama_pos").tma_mdpl.median()

def clim_predict(df):
    idx = pd.MultiIndex.from_arrays(
        [df.nama_pos, df.datetime.dt.month, df.datetime.dt.hour])
    pred = clim_mh.reindex(idx).to_numpy()
    idx2 = pd.MultiIndex.from_arrays([df.nama_pos, df.datetime.dt.month])
    pred = np.where(np.isnan(pred), clim_m.reindex(idx2).to_numpy(), pred)
    pred = np.where(np.isnan(pred), clim_p.reindex(df.nama_pos).to_numpy(), pred)
    return pred

# Validasi lokal: train <= 2025-01-18, val = 8 bulan terakhir (horizon = test)
cut = pd.Timestamp("2025-01-18 23:59:59")
tr_, va_ = train[train.datetime <= cut], train[train.datetime > cut]
cmh = tr_.groupby(["nama_pos", "month", "hour"]).tma_mdpl.median()
cm = tr_.groupby(["nama_pos", "month"]).tma_mdpl.median()
cp = tr_.groupby("nama_pos").tma_mdpl.median()
idx = pd.MultiIndex.from_arrays([va_.nama_pos, va_.month, va_.hour])
p = cmh.reindex(idx).to_numpy()
idx2 = pd.MultiIndex.from_arrays([va_.nama_pos, va_.month])
p = np.where(np.isnan(p), cm.reindex(idx2).to_numpy(), p)
p = np.where(np.isnan(p), cp.reindex(va_.nama_pos).to_numpy(), p)
rmse = float(np.sqrt(np.mean((va_.tma_mdpl.to_numpy() - p) ** 2)))
log(f"\n=== RMSE validasi climatology (val {va_.datetime.min().date()} -> "
    f"{va_.datetime.max().date()}): {rmse:.4f} ===")
log("Ini angka pembanding — model ML harus lebih kecil dari ini.")

# Kontribusi RMSE per stasiun di validasi
va2 = va_.copy(); va2["err2"] = (va_.tma_mdpl.to_numpy() - p) ** 2
contrib = va2.groupby("nama_pos").err2.mean().pow(0.5).sort_values(ascending=False)
log("\n=== RMSE climatology per stasiun (validasi) ===")
log(contrib.round(3).to_string())

# Submission
test["tma_mdpl"] = clim_predict(test)
test[["id", "tma_mdpl"]].to_csv(f"{BASE}/submission_climatology.csv", index=False)
log(f"\nSaved submission_climatology.csv ({len(test)} rows)")

with open(f"{BASE}/eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nSaved eda_summary.txt")
