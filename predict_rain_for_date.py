#!/usr/bin/env python3
"""
predict_rain_for_date.py (mejorado: fallback short_name + ventana temporal ±1 día)

Mejoras:
 - Si no hay entradas para short_name exacto, pruebo variantes (quita sufijos, reemplaza '.' -> '_').
 - Si no hay entradas para la fecha exacta, pruebo una ventana temporal ampliada (día-1 .. día+1).
 - Imprime la URL CMR usada (útil para debug).
 - Usa token (~/.edl_token) para CMR y descargas.
"""
import os
import argparse
import time
import tempfile
import datetime
from urllib.parse import urlencode
import requests
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

# ----------------------------
# Configuración por defecto
# ----------------------------
DATASETS = {
    "precip": {"short_name": "GPM_3IMERGDF.07", "preferred_host": "data.gesdisc.earthdata.nasa.gov"},
    "merra2": {"short_name": "MERRA2_400.tavg1_2d_slv_Nx", "preferred_host": "gmao.gsfc.nasa.gov"},
}
CMR_BASE = "https://cmr.earthdata.nasa.gov/search/granules.json"
EDL_TOKEN_PATH = os.path.expanduser("~/.edl_token")

# ----------------------------
# Token / headers
# ----------------------------
def read_token():
    if os.path.exists(EDL_TOKEN_PATH):
        return open(EDL_TOKEN_PATH).read().strip()
    return None

def auth_headers(token):
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

# ----------------------------
# CMR search con fallback
# ----------------------------
def _cmr_request(params, token=None):
    headers = auth_headers(token)
    url = CMR_BASE + "?" + urlencode(params)
    # print query url for debugging
    print(f"[CMR QUERY] {url}")
    r = requests.get(CMR_BASE, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json().get("feed", {}).get("entry", [])

def candidate_short_names(short_name):
    """Devuelve variantes del short_name para intentar búsquedas."""
    out = [short_name]
    # si contiene '.', añadir variante con '_' y sin el suffix
    if "." in short_name:
        out.append(short_name.replace(".", "_"))
        base = short_name.split(".")[0]
        out.append(base)
        out.append(base.replace(".", "_"))
    # también agregar variante con _07 suffix common pattern
    if short_name.endswith(".07"):
        out.append(short_name.replace(".07", "_07"))
    # deduplicate preserving order
    seen = set()
    res = []
    for s in out:
        if s not in seen:
            res.append(s); seen.add(s)
    return res

def cmr_search_granules_with_fallback(short_name, date_iso, bbox=None, page_size=10, token=None):
    """
    Intenta variantes de short_name y una ventana temporal si la búsqueda exacta falla.
    date_iso: YYYY-MM-DD
    """
    # First try exact date
    try_names = candidate_short_names(short_name)
    # temporal exact
    params_base = {"page_size": page_size}
    if bbox:
        params_base["bounding_box"] = ",".join(map(str, bbox))

    # helper to call with temporal string
    def try_with_temporal(sname, temporal_str):
        params = params_base.copy()
        params["short_name"] = sname
        params["temporal"] = temporal_str
        return _cmr_request(params, token=token)

    # 1) try exact date for each candidate short_name
    temporal_exact = f"{date_iso}T00:00:00Z,{date_iso}T23:59:59Z"
    for s in try_names:
        try:
            entries = try_with_temporal(s, temporal_exact)
            if entries:
                print(f"[CMR] Found {len(entries)} entries for short_name={s} exact date")
                return entries
        except Exception as e:
            print(f"[CMR] error searching {s} exact date: {e}")

    # 2) try expanded window ±1 day (useful si granule tiene timestamp distinto)
    dt = datetime.datetime.fromisoformat(date_iso)
    dt_minus = (dt - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    dt_plus = (dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    temporal_window = f"{dt_minus}T00:00:00Z,{dt_plus}T23:59:59Z"
    for s in try_names:
        try:
            entries = try_with_temporal(s, temporal_window)
            if entries:
                print(f"[CMR] Found {len(entries)} entries for short_name={s} window {dt_minus}..{dt_plus}")
                return entries
        except Exception as e:
            print(f"[CMR] error searching {s} window: {e}")

    # 3) last resort: search by keyword (short_name base) without temporal (may return many)
    base = short_name.split(".")[0]
    try:
        params = params_base.copy()
        params["keyword"] = base
        print(f"[CMR QUERY] fallback keyword search for {base}")
        entries = _cmr_request(params, token=token)
        if entries:
            print(f"[CMR] Found {len(entries)} entries by keyword={base}")
            return entries
    except Exception as e:
        print("[CMR] fallback keyword search error:", e)

    # nothing found
    return []

def pick_download_link(entry):
    for link in entry.get("links", []):
        href = link.get("href", "")
        if not href:
            continue
        if ("data.gesdisc.earthdata.nasa.gov" in href or "data.gesdisc" in href) and href.endswith((".nc4", ".nc", ".h5", ".hdf")):
            return href
    for link in entry.get("links", []):
        href = link.get("href", "")
        if href.endswith((".nc4", ".nc", ".h5", ".hdf")):
            return href
    for link in entry.get("links", []):
        if link.get("href"):
            return link.get("href")
    return None

def download_with_token(url, out_path, token):
    headers = auth_headers(token)
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out_path

# ----------------------------
# Extract point (download first if protected URL)
# ----------------------------
def extract_point_from_path_or_url(path_or_url, lat, lon, token=None, var_candidates=("precipitation","precipitationCal","precip")):
    """
    Si es URL pública, intenta abrir; si es URL protegida y token disponible -> descargar con token y abrir local.
    """
    # if url and token: download
    if isinstance(path_or_url, str) and path_or_url.startswith(("http://","https://")) and token:
        try:
            suffix = os.path.splitext(path_or_url)[1] or ".nc"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.close()
            print(f"[INFO] descargando con token: {path_or_url} -> {tmp.name}")
            download_with_token(path_or_url, tmp.name, token)
            try:
                val = extract_point_from_path_or_url(tmp.name, lat, lon, token=None, var_candidates=var_candidates)
            finally:
                try: os.remove(tmp.name)
                except: pass
            return val
        except Exception as e:
            print("[WARN] descarga con token falló:", e)
            # fall through to direct open (maybe public)

    # direct open (local file or public url)
    try:
        ds = xr.open_dataset(path_or_url, engine="netcdf4", decode_times=True)
    except Exception as e:
        print(f"[ERROR] no se pudo abrir {path_or_url} directamente: {e}")
        return None

    try:
        var = None
        for c in var_candidates:
            if c in ds.variables or c in ds.data_vars:
                var = c; break
        if var is None:
            ds.close(); return None
        da = ds[var]
        # coordinate names heuristics
        if ("lat" in da.coords and "lon" in da.coords):
            sel = da.sel(lat=lat, lon=lon, method="nearest")
        elif ("latitude" in da.coords and "longitude" in da.coords):
            sel = da.sel(latitude=lat, longitude=lon, method="nearest")
        else:
            sel = da
        other_dims = [d for d in sel.dims if d not in ("time",)]
        if other_dims:
            sel = sel.mean(dim=other_dims)
        valarr = sel.values
        try:
            v = float(np.asarray(valarr).squeeze().item())
        except Exception:
            v = None
        ds.close()
        return v
    except Exception as e:
        try: ds.close()
        except: pass
        print("[ERROR] extraer variable:", e)
        return None

# ----------------------------
# Builder histórico
# ----------------------------
def build_history_for_day(target_date, lat, lon, years_back=10, bbox=None, verbose=True, precip_short_name=None):
    token = read_token()
    if token:
        if verbose: print("[INFO] Usando token desde", EDL_TOKEN_PATH)
    else:
        if verbose: print("[WARN] No se encontró ~/.edl_token. Intentaré usar enlaces públicos si existen.")

    T = datetime.datetime.fromisoformat(target_date)
    month, day = T.month, T.day
    start_year = T.year - years_back
    end_year = T.year - 1
    rows = []
    for y in range(start_year, end_year + 1):
        date_iso = f"{y:04d}-{month:02d}-{day:02d}"
        if verbose: print("Buscando granules para", date_iso)
        precip_val = None
        try:
            entries = cmr_search_granules_with_fallback(precip_short_name or DATASETS["precip"]["short_name"],
                                                       date_iso, bbox=bbox, page_size=5, token=token)
            if entries:
                link = pick_download_link(entries[0])
                if link:
                    if verbose: print("  link seleccionado:", link)
                    precip_val = extract_point_from_path_or_url(link, lat, lon, token=token)
                else:
                    if verbose: print("  No hay link descargable en entry.")
            else:
                if verbose: print("  No hay entries precip para", date_iso)
        except Exception as e:
            if verbose: print("  Error precip CMR/descarga:", e)

        # variables auxiliares (MERRA2) - best-effort
        t2m = qv2m = pres = wind_speed = rad = None
        try:
            entries2 = cmr_search_granules_with_fallback(DATASETS["merra2"]["short_name"], date_iso, bbox=bbox, page_size=3, token=token)
            if entries2:
                link2 = pick_download_link(entries2[0])
                if link2:
                    if verbose: print("  merra2 link:", link2)
                    tval = extract_point_from_path_or_url(link2, lat, lon, token=token, var_candidates=("T2M","t2m","T"))
                    if tval is not None:
                        t2m = tval - 273.15 if tval > 100 else tval
                    qval = extract_point_from_path_or_url(link2, lat, lon, token=token, var_candidates=("QV2M","qv2m","q"))
                    qv2m = qval
                    pval = extract_point_from_path_or_url(link2, lat, lon, token=token, var_candidates=("PRES","pres","pressure"))
                    pres = pval
                    u = extract_point_from_path_or_url(link2, lat, lon, token=token, var_candidates=("U10M","U10","U"))
                    v = extract_point_from_path_or_url(link2, lat, lon, token=token, var_candidates=("V10M","V10","V"))
                    if u is not None and v is not None:
                        wind_speed = float((float(u)**2 + float(v)**2)**0.5)
        except Exception as e:
            if verbose: print("  Error merra2 lookup:", e)

        rows.append({"date": date_iso, "year": y, "precip": precip_val, "t2m": t2m, "qv2m": qv2m, "pres": pres, "wind_speed": wind_speed, "rad": rad})
        time.sleep(0.6)
    return pd.DataFrame(rows)

# ----------------------------
# Entrenamiento / predicción (misma lógica)
# ----------------------------
def build_features_and_train(df, umbral_lluvia=0.5, verbose=True):
    df = df.copy()
    df["llovio"] = df["precip"].apply(lambda x: 1 if (x is not None and not pd.isna(x) and x > umbral_lluvia) else 0)
    features = ["t2m","qv2m","pres","wind_speed"]
    df_clean = df.dropna(subset=features, how="all")
    for f in features:
        if f in df_clean.columns:
            med = df_clean[f].median(skipna=True)
            df_clean[f] = df_clean[f].fillna(med)
    y = df_clean["llovio"].astype(int).values if "llovio" in df_clean.columns else np.array([])
    X = df_clean[features].fillna(0).values if not df_clean.empty else np.empty((0,len(features)))
    if len(y) < 3:
        if verbose: print("Pocos datos para entrenar (>3 required). Usaremos heurística (frecuencia).")
        return None, df_clean, None
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X, y)
    if verbose:
        preds = pipe.predict(X); probs = pipe.predict_proba(X)[:,1]
        print("Train acc:", accuracy_score(y, preds), "AUC:", (roc_auc_score(y, probs) if len(np.unique(y))>1 else "NA"))
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X,y)
    return {"logreg": pipe, "rf": rf}, df_clean, features

def predict_for_target(models, features_list, row_features):
    x = np.array([row_features.get(f, 0) for f in features_list], dtype=float).reshape(1,-1)
    out = {}
    if models is None:
        return None
    try:
        out["logreg_prob"] = float(models["logreg"].predict_proba(x)[:,1][0])
        out["rf_prob"] = float(models["rf"].predict_proba(x)[:,1][0])
    except Exception:
        return None
    return out

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="Fecha objetivo YYYY-MM-DD")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--years_back", type=int, default=15)
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W","S","E","N"), default=None)
    p.add_argument("--umbral", type=float, default=0.5)
    p.add_argument("--precip_short_name", default=DATASETS["precip"]["short_name"])
    args = p.parse_args()

    print("Construyendo histórico para", args.date, "en", args.lat, args.lon)
    df_hist = build_history_for_day(args.date, args.lat, args.lon, years_back=args.years_back, bbox=args.bbox, verbose=True, precip_short_name=args.precip_short_name)
    if df_hist.empty:
        print("No se obtuvieron años históricos. Revisa fecha/years_back/bbox/token/short_name.")
        return
    print("Histórico obtenido (primeras filas):")
    print(df_hist.head(20))

    models, df_clean, feature_names = build_features_and_train(df_hist, umbral_lluvia=args.umbral, verbose=True)

    row = {}
    if feature_names:
        for f in feature_names:
            row[f] = float(df_clean[f].median(skipna=True)) if f in df_clean.columns and not df_clean[f].isna().all() else 0.0

    if models is None:
        hist_freq = df_hist["precip"].apply(lambda x: 1 if (x is not None and not pd.isna(x) and x > args.umbral) else 0).mean()
        print(f"Predicción por frecuencia histórica: P(llover) ≈ {hist_freq:.2f} (umbral {args.umbral} mm)")
        df_hist.to_csv("prediction_history_used.csv", index=False)
        print("Histórico guardado en: prediction_history_used.csv")
        return

    probs = predict_for_target(models, feature_names, row)
    if probs is None:
        print("No se pudo predecir con modelos entrenados.")
        return
    print("Predicción (probabilidades):", probs)
    avg_prob = (probs["logreg_prob"] + probs["rf_prob"]) / 2.0
    print(f"Probabilidad promedio de lluvia: {avg_prob:.2%}")
    print("Decisión (umbral 0.5):", "LLoverá" if avg_prob >= 0.5 else "No lloverá")
    df_hist.to_csv("prediction_history_used.csv", index=False)
    print("Histórico guardado en: prediction_history_used.csv")

if __name__ == "__main__":
    main()
