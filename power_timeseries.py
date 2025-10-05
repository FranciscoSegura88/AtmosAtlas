#!/usr/bin/env python3
"""
power_timeseries.py

Descarga series diarias de NASA POWER para un punto (lat,lon) entre start/end.
Guarda automáticamente CSV: power_timeseries_{lat}_{lon}_{start}_{end}.csv

Ejemplo:
  python power_timeseries.py --lat 20.67 --lon -103.35 --start 1984-01-01 --end 2024-12-31
"""
import argparse
import time
import requests
import pandas as pd
import os
from datetime import datetime

# Cache HTTP: acelera y reduce rate-limits
try:
    import requests_cache
    requests_cache.install_cache("power_cache", expire_after=6 * 3600)
except Exception:
    pass

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

# parámetros por defecto que solicitamos (puedes ajustar)
DEFAULT_PARAMS = [
    "PRECTOTCORR",       # precipitación total corregida (mm/día)
    "T2M",               # temperatura 2m (°C)
    "RH2M",              # humedad relativa 2m (%)
    "WS10M",             # velocidad del viento 10m (m/s)
    "ALLSKY_SFC_SW_DWN", # radiación solar (kWh/m²/día) en daily/AG
    "PS"                 # presión en superficie (kPa)
]

def fetch_power_point(lat, lon, start, end, parameters=DEFAULT_PARAMS, community="AG", retries=5, backoff=2):
    """Consulta la API POWER para un punto. Reintentos en errores HTTP/timeout con backoff exponencial."""
    params = {
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "latitude": float(lat),
        "longitude": float(lon),
        "community": community,
        "parameters": ",".join(parameters),
        "format": "JSON"
    }
    attempt = 0
    while attempt < retries:
        try:
            r = requests.get(POWER_BASE, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            print(f"  error: {status} {e} for url: {r.url if 'r' in locals() else POWER_BASE}")
            attempt += 1
            sleep_for = backoff ** attempt
            print(f"   retrying in {sleep_for}s (attempt {attempt}/{retries})")
            time.sleep(sleep_for)
        except Exception as e:
            print("  error:", e)
            attempt += 1
            sleep_for = backoff ** attempt
            print(f"   retrying in {sleep_for}s (attempt {attempt}/{retries})")
            time.sleep(sleep_for)
    raise RuntimeError("Failed to fetch POWER after retries.")

def power_json_to_dataframe(power_json, parameters=DEFAULT_PARAMS):
    """Convierte JSON POWER en DataFrame (date + parameters)."""
    props = power_json.get("properties", {})
    param_block = props.get("parameter", {})
    dates = set()
    for p in param_block:
        dates.update(param_block[p].keys())
    if not dates:
        return pd.DataFrame()
    dates = sorted(dates)
    rows = []
    for d in dates:
        dt = datetime.strptime(d, "%Y%m%d").date()
        row = {"date": dt}
        for p in parameters:
            val = param_block.get(p, {}).get(d)
            row[p] = None if val is None else val
        rows.append(row)
    df = pd.DataFrame(rows)
    rename_map = {
        "PRECTOTCORR": "precip",
        "T2M": "t2m",
        "RH2M": "rh2m",
        "WS10M": "wind",
        "ALLSKY_SFC_SW_DWN": "rad",
        "PS": "ps"
    }
    df = df.rename(columns=rename_map)
    numeric_cols = [rename_map.get(p, p) for p in parameters]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", default=None, help="Nombre de archivo de salida (opcional)")
    p.add_argument("--params", default=",".join(DEFAULT_PARAMS), help="Parámetros POWER separados por coma (códigos API)")
    args = p.parse_args()

    lat = args.lat
    lon = args.lon
    start = args.start
    end = args.end
    params = [p.strip() for p in args.params.split(",") if p.strip()]

    print(f"Fetching POWER for {start} -> {end} at {lat},{lon}")
    try:
        pj = fetch_power_point(lat, lon, start, end, parameters=params, retries=6, backoff=2)
    except Exception as e:
        print("ERROR: no se pudo obtener datos de POWER:", e)
        return

    df = power_json_to_dataframe(pj, parameters=params)
    if df.empty:
        print("No se encontraron datos en la respuesta POWER.")
        return

    if args.out:
        out_fn = args.out
    else:
        safe_lat = str(lat).replace(".", "p")
        safe_lon = str(lon).replace(".", "p").replace("-", "m")
        out_fn = f"power_timeseries_{safe_lat}_{safe_lon}_{start.replace('-','')}_{end.replace('-','')}.csv"
    df.to_csv(out_fn, index=False)
    print("Datos guardados en:", os.path.abspath(out_fn))
    print(df.head())

if __name__ == "__main__":
    main()
