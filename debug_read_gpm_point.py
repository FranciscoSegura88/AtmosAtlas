#!/usr/bin/env python3
"""
Debug reader para GPM IMERG: intenta OPeNDAP y fallback a descarga HTTPS.
Imprime trazas completas y escribe debug_output.txt en la carpeta actual.
"""
import os, sys, traceback, argparse, tempfile, time
import xarray as xr
import pandas as pd
import requests
from urllib.parse import urlparse

EDL_TOKEN_PATH = os.path.expanduser("~/.edl_token")

def log(msg):
    print(msg)
    # flush so -u shows it immediately
    sys.stdout.flush()

def read_opendap(url):
    log(f"[OPeNDAP] intentando abrir: {url}")
    try:
        ds = xr.open_dataset(url, decode_times=True, engine="netcdf4")
        log(f"[OPeNDAP] opened dataset, variables: {list(ds.variables)[:10]} ...")
        return ds
    except Exception:
        log("[OPeNDAP] excepción al abrir OPeNDAP:")
        traceback.print_exc()
        return None

def download_with_token(url, out_path):
    log(f"[HTTP] descargando (token) {url} -> {out_path}")
    if not os.path.exists(EDL_TOKEN_PATH):
        raise FileNotFoundError("No .edl_token en home; crea token primero.")
    token = open(EDL_TOKEN_PATH, "r").read().strip()
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        log(f"[HTTP] status_code: {r.status_code}, headers: {dict(r.headers)}")
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out_path

def get_precip_series(ds, lat, lon, var_candidates=("precipitationCal","precipitation","precip")):
    try:
        log(f"[DS] variables disponibles: {list(ds.data_vars)[:20]}")
        var = None
        for v in var_candidates:
            if v in ds.variables or v in ds.data_vars:
                var = v
                break
        if var is None:
            raise KeyError("No se encontró variable de precipitación en el dataset.")
        log(f"[DS] usando variable: {var}")
        da = ds[var]
        # intentar lat/lon selection
        if ("lat" in da.coords) and ("lon" in da.coords):
            sel = da.sel(lat=lat, lon=lon, method="nearest")
        elif ("latitude" in da.coords) and ("longitude" in da.coords):
            sel = da.sel(latitude=lat, longitude=lon, method="nearest")
        else:
            # intentar variables lat/lon en dataset global
            if "lat" in ds.coords and "lon" in ds.coords:
                sel = da.sel(lat=lat, lon=lon, method="nearest")
            else:
                log("[DS] No hay coords lat/lon -> devolver promedio global")
                sel = da.mean()
        # reducir dimensiones espaciales
        dims_to_keep = [d for d in sel.dims if d == "time"]
        if dims_to_keep:
            s = sel.to_series()
            s.index = pd.to_datetime(s.index)
            return s
        else:
            return pd.Series([float(sel.values)], index=[pd.to_datetime("1970-01-01")])
    except Exception:
        log("[DS] excepción al extraer serie de precipitación:")
        traceback.print_exc()
        return None

def process_url_try(url, lat, lon):
    log("------------------------------------------------------------")
    log(f"Procesando URL: {url}")
    # 1) try OPeNDAP as-is
    ds = read_opendap(url)
    if ds is not None:
        try:
            s = get_precip_series(ds, lat, lon)
            ds.close()
            if s is not None:
                return s
        except Exception:
            log("[OPeNDAP] error al procesar dataset OPeNDAP:")
            traceback.print_exc()

    # 2) fallback: if url contains /opendap/ replace by /data/ (GESDISC pattern)
    if "/opendap/" in url:
        alt = url.replace("/opendap/", "/data/")
        log(f"[Fallback] intentando alternativa de descarga: {alt}")
    else:
        alt = url
    # try downloading alt
    try:
        parsed = urlparse(alt)
        if parsed.scheme not in ("http","https"):
            log(f"[Fallback] URL no descargable con HTTP: {alt}")
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.basename(parsed.path))
        tmp.close()
        dlpath = tmp.name
        download_with_token(alt, dlpath)
        log(f"[Fallback] archivo descargado: {dlpath} (size={os.path.getsize(dlpath)} bytes)")
        # open local file
        try:
            ds2 = xr.open_dataset(dlpath, decode_times=True, engine="netcdf4")
            s2 = get_precip_series(ds2, lat, lon)
            ds2.close()
            # borrar temporal
            try:
                os.remove(dlpath)
            except Exception:
                log("[Fallback] no se pudo borrar temporal")
            return s2
        except Exception:
            log("[Fallback] error al abrir archivo descargado con xarray:")
            traceback.print_exc()
            return None
    except Exception:
        log("[Fallback] error en descarga alternativa:")
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    # redirect stdout/stderr to debug file as well
    fh = open("debug_output.txt", "w", encoding="utf-8")
    # duplicate prints to file by reassigning sys.stdout? we'll manually write key messages
    try:
        all_series = []
        for u in args.urls:
            log(f"START {time.asctime()}")
            s = process_url_try(u, args.lat, args.lon)
            if s is not None and not s.empty:
                all_series.append(s)
                log(f"[OK] serie extraída con {len(s)} valores, índices: {list(s.index)[:5]}")
            else:
                log("[WARN] no se extrajo serie o quedó vacía para URL: " + u)
            log(f"END {time.asctime()}")
            fh.write(f"Processed URL: {u}\\n")
        # combine
        if all_series:
            df = pd.concat(all_series, axis=0)
            df = df.groupby(df.index).mean().sort_index()
            log("=== Resultado agregado ===")
            log(str(df.head(20)))
            out = "gpm_point_precip_series_debug.csv"
            df.to_csv(out, header=["precipitation"])
            log("Guardado en: " + out)
            fh.write("Saved CSV: " + out + "\\n")
        else:
            log("No se extrajo ninguna serie.")
            fh.write("No series extracted\\n")
    except Exception:
        log("Error crítico en main:")
        traceback.print_exc()
        fh.write("Fatal exception\\n")
    finally:
        fh.close()

if __name__ == '__main__':
    main()
