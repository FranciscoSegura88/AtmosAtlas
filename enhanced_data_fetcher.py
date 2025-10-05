# enhanced_data_fetcher.py
# ------------------------------------------------------------
# Ingesta multi-fuente robusta para AtmosAtlas:
#  - NASA POWER (REST)           -> precip, t2m, rh2m, wind, rad, ps (diario, 1981-presente)
#  - GPM IMERG V07 (OPeNDAP)     -> precip (diario, 2000-06-presente) [opcional]
#  - GPCP 1DD (OPeNDAP, NOAA PSL)-> precip (diario, 1996-10-presente) [opcional]
#  - GPCP Monthly (OPeNDAP, NOAA)-> precip (mensual, 1979-01-presente) [opcional, reamostrado a diario]
#
# Fusión con pesos por variable y estimación de incertidumbre inter-fuentes.
# Devuelve además un "provenance" completo por fuente y variable.
# Degrada elegantemente a POWER-only si no hay credenciales/red/paquetes.
# ------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import os

import numpy as np
import pandas as pd
import requests

# Cache HTTP para POWER (mejora robustez y evita rate limiting)
try:
    import requests_cache  # type: ignore
    requests_cache.install_cache("power_cache", expire_after=6 * 3600)
except Exception:
    pass

# OPeNDAP/Earthdata (opcional)
try:
    import xarray as xr  # type: ignore
    HAS_XARRAY = True
except Exception:
    HAS_XARRAY = False

try:
    import earthaccess  # type: ignore
    HAS_EARTHACCESS = True
except Exception:
    HAS_EARTHACCESS = False


# ----------------- Cobertura temporal por fuente -----------------
EARLIEST = {
    "power": "1981-01-01",      # POWER daily API (doc oficial)
    "imerg": "2000-06-01",      # IMERG V07 (TRMM/GPM)
    "gpcp_daily": "1996-10-01", # GPCP 1DD
    "gpcp_monthly": "1979-01-01" # GPCP Monthly
}

TODAY_STR = date.today().strftime("%Y-%m-%d")


# ----------------- Fuentes -----------------
class PowerDataSource:
    """NASA POWER: REST JSON diario (comunidad AG)."""
    BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
    # Variables núcleo (nombres POWER -> nombres normalizados)
    PARAMS = "PRECTOTCORR,T2M,RH2M,WS10M,ALLSKY_SFC_SW_DWN,PS"
    RENAME = {
        "PRECTOTCORR": "precip",            # mm/day
        "T2M": "t2m",                        # °C
        "RH2M": "rh2m",                      # %
        "WS10M": "wind",                     # m/s
        "ALLSKY_SFC_SW_DWN": "rad",         # kWh/m^2/day
        "PS": "ps"                           # kPa
    }

    def fetch(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        params = {
            "start": start.replace("-", ""),
            "end": end.replace("-", ""),
            "latitude": float(lat),
            "longitude": float(lon),
            "community": "AG",
            "parameters": self.PARAMS,
            "format": "JSON",
        }
        r = requests.get(self.BASE, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        block = data.get("properties", {}).get("parameter", {})
        if not block:
            return pd.DataFrame(columns=["date"] + list(self.RENAME.values()))

        # fechas presentes en la respuesta
        dates = sorted(set().union(*(set(v.keys()) for v in block.values())))
        rows = []
        for dstr in dates:
            dt = datetime.strptime(dstr, "%Y%m%d")
            row = {"date": dt}
            for k_power, k_std in self.RENAME.items():
                row[k_std] = block.get(k_power, {}).get(dstr, None)
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        for c in ["precip", "t2m", "rh2m", "wind", "rad", "ps"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df


class IMERGDataSource:
    """
    IMERG V07 Daily (Final/Late). Acceso vía Earthdata + OPeNDAP con xarray.
    Si no hay dependencias/credenciales, retorna DF vacío.
    """
    # Nota: para un solo punto, accedemos a cada "granule" y extraemos la grilla más próxima.
    # Este método es seguro pero puede ser lento; se usa sólo si la instalación lo permite.
    SHORT_NAME = "GPM_3IMERGDF"  # L3 Final Daily

    def fetch(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        if not (HAS_XARRAY and HAS_EARTHACCESS):
            return pd.DataFrame(columns=["date", "precip"])
        try:
            earthaccess.login()  # .netrc / variables de entorno
        except Exception:
            # sin login, degradar en silencio
            return pd.DataFrame(columns=["date", "precip"])

        try:
            results = earthaccess.search_data(short_name=self.SHORT_NAME, temporal=(start, end))
            if not results:
                return pd.DataFrame(columns=["date", "precip"])

            rows: List[Dict] = []
            for gran in results:
                urls = gran.data_links(access="opendap")
                if not urls:
                    continue
                url = urls[0]
                ds = xr.open_dataset(url)
                # nombres típicos
                var = "precipitation" if "precipitation" in ds.data_vars else list(ds.data_vars)[0]
                # índice del píxel más cercano
                i_lat = int(np.abs(ds["lat"].values - lat).argmin())
                i_lon = int(np.abs(ds["lon"].values - lon).argmin())
                ts = ds[var][:, i_lat, i_lon].to_series()  # mm/day
                ts.index = pd.to_datetime(ts.index)
                for t, v in ts.items():
                    rows.append({"date": pd.to_datetime(t).to_pydatetime(), "precip": float(v)})
                ds.close()

            if not rows:
                return pd.DataFrame(columns=["date", "precip"])
            out = pd.DataFrame(rows).groupby("date", as_index=False)["precip"].mean()
            return out.sort_values("date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["date", "precip"])


class GPCPDailyDataSource:
    """
    GPCP 1DD diario (1996-10 en adelante). OPeNDAP NOAA PSL.
    Este endpoint suele estar disponible sin credenciales.
    """
    # Dataset agregado típico (puede cambiar; por eso hay try/except y fallback silencioso):
    # https://psl.noaa.gov/thredds/dodsC/Datasets/gpcp/1DD/gpcp_v01dd_199610-present.nc
    # Como nombre "var": 'precip'
    CANDIDATE_URLS = [
        "https://psl.noaa.gov/thredds/dodsC/Datasets/gpcp/1DD/precip.1996-present.nc",
        "https://psl.noaa.gov/thredds/dodsC/Datasets/gpcp/1DD/gpcp_v01dd_199610-present.nc",
    ]
    VAR = "precip"

    def fetch(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        if not HAS_XARRAY:
            return pd.DataFrame(columns=["date", "precip"])
        for url in self.CANDIDATE_URLS:
            try:
                ds = xr.open_dataset(url)
                # recorte temporal
                ds_sel = ds.sel(time=slice(start, end))
                # punto más cercano
                i_lat = int(np.abs(ds_sel["lat"].values - lat).argmin())
                i_lon = int(np.abs(ds_sel["lon"].values - lon).argmin())
                ts = ds_sel[self.VAR][:, i_lat, i_lon].to_series()
                ts.index = pd.to_datetime(ts.index)
                df = ts.reset_index()
                df.columns = ["date", "precip"]  # mm/day
                ds.close()
                return df.sort_values("date").reset_index(drop=True)
            except Exception:
                continue
        return pd.DataFrame(columns=["date", "precip"])


class GPCPMonthlyDataSource:
    """
    GPCP Mensual (1979-01 en adelante). NOAA PSL OPeNDAP.
    Reamostramos a diario rellenando con el promedio diario del mes (mm/day).
    """
    URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc"
    VAR = "precip"

    def fetch(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        if not HAS_XARRAY:
            return pd.DataFrame(columns=["date", "precip"])
        try:
            ds = xr.open_dataset(self.URL)
            ds_sel = ds.sel(time=slice(start, end))
            i_lat = int(np.abs(ds_sel["lat"].values - lat).argmin())
            i_lon = int(np.abs(ds_sel["lon"].values - lon).argmin())
            ts = ds_sel[self.VAR][:, i_lat, i_lon].to_series()  # unidades: mm/day (promedio mensual)
            ts.index = pd.to_datetime(ts.index).to_period("M").to_timestamp("M")  # fin de mes
            # expandir a diario: repetimos el valor medio diario del mes en todos los días de ese mes
            daily_rows: List[Dict] = []
            for month_end, v in ts.items():
                if pd.isna(v):
                    continue
                # rango de ese mes
                m_start = (pd.to_datetime(month_end) - pd.offsets.MonthEnd(1)) + pd.offsets.Day(1)
                m_end = pd.to_datetime(month_end)
                dates = pd.date_range(m_start, m_end, freq="D")
                for d in dates:
                    daily_rows.append({"date": d.to_pydatetime(), "precip": float(v)})
            ds.close()
            if not daily_rows:
                return pd.DataFrame(columns=["date", "precip"])
            return pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["date", "precip"])


# ----------------- Pesos por variable -----------------
@dataclass
class EnsembleWeights:
    precip: Dict[str, float] = None
    t2m: Dict[str, float] = None
    rh2m: Dict[str, float] = None
    wind: Dict[str, float] = None
    rad: Dict[str, float] = None
    ps: Dict[str, float] = None

    def __post_init__(self):
        # Precisión típica: IMERG > GPCP > POWER para precipitación
        # Nota: cuando no exista IMERG/GPCP, el peso efectivo recae en POWER
        self.precip = self.precip or {
            "imerg": 0.55,        # si disponible
            "gpcp_daily": 0.30,   # si disponible
            "gpcp_monthly": 0.15, # si disponible (proxy mensual)
            "power": 0.35         # respaldo continuo
        }
        # El resto proviene esencialmente de POWER en este flujo
        self.t2m  = self.t2m  or {"power": 1.0}
        self.rh2m = self.rh2m or {"power": 1.0}
        self.wind = self.wind or {"power": 1.0}
        self.rad  = self.rad  or {"power": 1.0}
        self.ps   = self.ps   or {"power": 1.0}


# ----------------- FUSIONADOR PRINCIPAL -----------------
class EnhancedDataFetcher:
    """
    Ingesta multi-fuente con:
      - Auto-ajuste al inicio MÁS ANTIGUO posible por fuente (puede ignorar "start" del usuario).
      - Fusión ponderada por variable + incertidumbre inter-fuentes.
      - "Provenance" por fuente/variable (rango temporal usado, nº de registros, fracción de aporte).
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.sources = {
            "power": PowerDataSource(),
            "imerg": IMERGDataSource(),
            "gpcp_daily": GPCPDailyDataSource(),
            "gpcp_monthly": GPCPMonthlyDataSource(),
        }
        self.weights = EnsembleWeights()
        self.last_provenance: Dict = {}

    # ---------- UTILIDADES ----------
    @staticmethod
    def _clip_to_source_range(source: str, start: str, end: str) -> Tuple[str, str]:
        s0 = pd.to_datetime(EARLIEST.get(source, start))
        s1 = pd.to_datetime(start)
        e1 = pd.to_datetime(end)
        s_final = min(s0, s1)  # usar lo más antiguo disponible (puede ignorar el start del usuario)
        return s_final.strftime("%Y-%m-%d"), e1.strftime("%Y-%m-%d")

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
        out = out.sort_values("date").reset_index(drop=True)
        # evitar infinitos
        for c in [x for x in out.columns if x != "date"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out.loc[~np.isfinite(out[c]), c] = np.nan
        return out

    # ---------- DESCARGA EN CONJUNTO ----------
    def fetch_ensemble_data(
        self,
        lat: float, lon: float,
        start: str, end: str,
        sources: Optional[List[str]] = None,
        auto_earliest: bool = True
    ) -> pd.DataFrame:
        srcs = sources or list(self.sources.keys())
        results: Dict[str, pd.DataFrame] = {}
        self.last_provenance = {
            "request": {"lat": lat, "lon": lon, "target_start": start, "target_end": end, "auto_earliest": auto_earliest},
            "sources": {},
            "variables": {}
        }

        for name in srcs:
            try:
                s_use, e_use = (self._clip_to_source_range(name, start, end) if auto_earliest
                                else (start, end))
                df = self.sources[name].fetch(lat, lon, s_use, e_use)
                df = self._normalize_df(df)
                results[name] = df

                nrec = int(len(df)) if not df.empty else 0
                smin = df["date"].min().strftime("%Y-%m-%d") if nrec else None
                smax = df["date"].max().strftime("%Y-%m-%d") if nrec else None
                self.last_provenance["sources"][name] = {
                    "attempted_range": [s_use, e_use],
                    "actual_range": [smin, smax],
                    "records": nrec
                }
                if self.verbose:
                    print(f"✓ {name}: registros={nrec} rango={smin}..{smax}")
            except Exception as e:
                if self.verbose:
                    print(f"✗ {name}: {e}")
                results[name] = pd.DataFrame()
                self.last_provenance["sources"][name] = {
                    "attempted_range": [start, end],
                    "actual_range": [None, None],
                    "records": 0,
                    "error": str(e)
                }

        merged = self._merge_sources(results)
        # completar metadatos de variables
        self._finalize_variable_contributions(merged, results)
        return merged

    # ---------- FUSIÓN ----------
    def _merge_sources(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # calendario maestro = unión de todas las fechas disponibles
        all_dates = pd.Index([])
        for df in results.values():
            if df is not None and not df.empty and "date" in df.columns:
                all_dates = all_dates.union(pd.to_datetime(df["date"]).dt.normalize().unique())
        if len(all_dates) == 0:
            return pd.DataFrame()

        merged = pd.DataFrame({"date": pd.to_datetime(sorted(all_dates))})

        # Variables a fusionar (precip y meteo)
        variable_maps = {
            "precip": self.weights.precip,
            "t2m": self.weights.t2m,
            "rh2m": self.weights.rh2m,
            "wind": self.weights.wind,
            "rad": self.weights.rad,
            "ps": self.weights.ps
        }

        for var, wmap in variable_maps.items():
            series_list, src_names, weights = [], [], []
            for src, df in results.items():
                if df is not None and not df.empty and var in df.columns:
                    aligned = pd.merge(merged[["date"]], df[["date", var]], on="date", how="left")[var].values
                    series_list.append(aligned)
                    src_names.append(src)
                    weights.append(wmap.get(src, 0.0))
            if not series_list:
                continue

            arr = np.array(series_list, dtype="float64")
            w = np.array(weights, dtype="float64")
            # si todos los pesos son 0 (p.ej. var no soportada por esa fuente), usar promedio simple
            if np.nansum(w) > 0:
                w = w / np.nansum(w)
                fused = np.nansum(arr * w[:, None], axis=0)
            else:
                fused = np.nanmean(arr, axis=0)

            merged[var] = fused
            merged[f"{var}_uncertainty"] = np.nanstd(arr, axis=0)

        return merged

    # ---------- PROVENANCE / CONTRIBUCIONES ----------
    def _finalize_variable_contributions(self, merged: pd.DataFrame, results: Dict[str, pd.DataFrame]) -> None:
        # Por variable: fracción de puntos aportados por cada fuente (no nulos)
        variables = ["precip", "t2m", "rh2m", "wind", "rad", "ps"]
        contrib: Dict[str, Dict[str, float]] = {}
        for var in variables:
            total_non_nan = 0
            src_non_nan: Dict[str, int] = {}
            for src, df in results.items():
                if df is not None and not df.empty and var in df.columns:
                    aligned = pd.merge(merged[["date"]], df[["date", var]], on="date", how="left")[var]
                    nnz = int(aligned.notna().sum())
                    total_non_nan += nnz
                    src_non_nan[src] = nnz
            if total_non_nan == 0:
                continue
            contrib[var] = {src: round(n / total_non_nan, 4) for src, n in src_non_nan.items()}

        # Rango global del dataset resultante
        if merged is not None and not merged.empty:
            gmin = merged["date"].min().strftime("%Y-%m-%d")
            gmax = merged["date"].max().strftime("%Y-%m-%d")
        else:
            gmin = gmax = None

        self.last_provenance["variables"] = {
            "contributions_fraction": contrib,
            "global_range": [gmin, gmax]
        }

    # ---------- Resumen legible ----------
    def provenance_summary(self) -> str:
        """Crea una frase compacta del tipo:
        POWER (1981–2025), IMERG (2000–2025), GPCP(1979–2025)
        Sólo cita fuentes que realmente aportaron datos (records>0)."""
        srcs = self.last_provenance.get("sources", {})
        parts = []
        name_map = {
            "power": "POWER",
            "imerg": "IMERG",
            "gpcp_daily": "GPCP-1DD",
            "gpcp_monthly": "GPCP-Monthly"
        }
        for k in ["power", "imerg", "gpcp_daily", "gpcp_monthly"]:
            info = srcs.get(k, {})
            if info and info.get("records", 0) > 0:
                r = info.get("actual_range") or [None, None]
                s = (r[0] or "").split("-")[0]
                e = (r[1] or "").split("-")[0]
                if s and e:
                    parts.append(f"{name_map[k]} ({s}–{e})")
        # Rango global
        gr = self.last_provenance.get("variables", {}).get("global_range", [None, None])
        gtxt = ""
        if gr[0] and gr[1]:
            gtxt = f" | Rango combinado: {gr[0]} → {gr[1]}"
        return ("; ".join(parts) + gtxt).strip()


# ----------------- Calidad de datos -----------------
class DataQualityAnalyzer:
    """Métricas básicas de calidad/consistencia + provenance integrado."""
    @staticmethod
    def assess(df: pd.DataFrame, provenance: Optional[Dict] = None) -> Dict:
        metrics = {"completeness": {}, "consistency": {}, "reliability": {}, "provenance": provenance or {}}
        if df is None or df.empty:
            return metrics

        # completitud por columna (sin incertidumbres)
        for c in [c for c in df.columns if c != "date" and not c.endswith("_uncertainty")]:
            metrics["completeness"][c] = float(df[c].notna().mean())

        # incertidumbre media inter-fuente
        ucols = [c for c in df.columns if c.endswith("_uncertainty")]
        if ucols:
            metrics["consistency"]["avg_uncertainty"] = float(df[ucols].mean().mean())

        # fiabilidad simple: combina completitud + consistencia
        avg_comp = np.mean(list(metrics["completeness"].values())) if metrics["completeness"] else 0.0
        avg_cons = 1.0 - (metrics["consistency"].get("avg_uncertainty", 0.0) / 10.0)
        metrics["reliability"]["overall_score"] = float(max(0.0, min(1.0, 0.5 * (avg_comp + avg_cons))))

        # rango temporal global
        metrics["time_coverage"] = {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
            "n_days": int(len(df))
        }

        # resumen legible
        if provenance and "sources" in provenance:
            parts = []
            for src, info in provenance["sources"].items():
                if info.get("records", 0) > 0:
                    r = info.get("actual_range") or [None, None]
                    parts.append(f"{src}: {r[0]}→{r[1]} ({info.get('records')} registros)")
            metrics["provenance_summary"] = " | ".join(parts)
        else:
            metrics["provenance_summary"] = ""

        return metrics
