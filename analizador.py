# analizador.py (Versión Refactorizada para API)

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Importar funciones de los otros módulos
try:
    from power_timeseries import fetch_power_point, power_json_to_dataframe, DEFAULT_PARAMS
    from advanced_ml_predictor import AdvancedClimatePredictor, integrate_with_analyzer
    CAN_DOWNLOAD = True
except ImportError:
    CAN_DOWNLOAD = False
    logging.warning("No se pudo importar 'power_timeseries' o 'advanced_ml_predictor'. Funcionalidad limitada.")

class ClimateAnalyzer:
    """Analizador de probabilidades climáticas desde datos históricos de NASA POWER."""

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas para inicializar.")
        self.df = df
        self._prepare_dataframe()

    def _prepare_dataframe(self):
        """Prepara y valida el DataFrame interno."""
        self.df['date'] = pd.to_datetime(self.df['date'])
        required = ['date', 'precip', 't2m', 'rh2m', 'wind', 'ps', 'rad']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes en el DataFrame: {missing}")
        logging.info(f"✓ Analizador inicializado con {len(self.df)} registros.")

    def get_analysis_window(self, target_date_str: str, window_days: int) -> pd.DataFrame:
        """Obtiene la ventana de datos históricos para el análisis."""
        target_date = pd.to_datetime(target_date_str)
        day_of_year = target_date.dayofyear

        start_day = day_of_year - window_days
        end_day = day_of_year + window_days

        # Manejo de cruce de año
        if start_day < 1:
            # Días del año anterior y actual
            doy_filter = (self.df['date'].dt.dayofyear >= 365 + start_day) | (self.df['date'].dt.dayofyear <= end_day)
        elif end_day > 365:
            # Días del año actual y siguiente
            doy_filter = (self.df['date'].dt.dayofyear >= start_day) | (self.df['date'].dt.dayofyear <= end_day - 365)
        else:
            doy_filter = (self.df['date'].dt.dayofyear >= start_day) & (self.df['date'].dt.dayofyear <= end_day)

        window_df = self.df[doy_filter].copy()

        if window_df.empty:
            raise ValueError("No se encontraron datos en la ventana temporal para el análisis.")

        return window_df

    @staticmethod
    def calculate_statistics(df: pd.DataFrame, rain_threshold: float, hot_threshold: float, cold_threshold: float) -> Dict:
        """Calcula estadísticas descriptivas sobre la ventana de datos."""
        rainy_days = df[df['precip'] > rain_threshold]

        # Asegurar que la desviación estándar no sea cero para evitar división por cero
        t2m_std = df['t2m'].std() if df['t2m'].std() > 0 else 1.0

        stats = {
            "historical_summary": {
                "sample_size_days": len(df),
                "total_years": df['date'].dt.year.nunique(),
                "date_range_start": df['date'].min().strftime('%Y-%m-%d'),
                "date_range_end": df['date'].max().strftime('%Y-%m-%d'),
            },
            "precipitation": {
                "rain_probability_percent": (len(rainy_days) / len(df)) * 100 if len(df) > 0 else 0,
                "average_mm_when_rains": rainy_days['precip'].mean() if not rainy_days.empty else 0,
                "max_mm_recorded": df['precip'].max(),
                "rain_threshold_mm": rain_threshold,
            },
            "temperature": {
                "average_celsius": df['t2m'].mean(),
                "std_dev_celsius": t2m_std,
                "range_min_celsius": df['t2m'].min(),
                "range_max_celsius": df['t2m'].max(),
                "hot_day_probability_percent": (df['t2m'] > hot_threshold).mean() * 100,
                "cold_day_probability_percent": (df['t2m'] < cold_threshold).mean() * 100,
                "hot_threshold_celsius": hot_threshold,
                "cold_threshold_celsius": cold_threshold,
            },
            "humidity": {
                "average_percent": df['rh2m'].mean(),
                "range_min_percent": df['rh2m'].min(),
                "range_max_percent": df['rh2m'].max(),
            },
            "wind": {
                "average_mps": df['wind'].mean(),
                "max_mps_recorded": df['wind'].max(),
            },
            "pressure": {
                "average_kpa": df['ps'].mean(),
            },
            "solar_radiation": {
                "average_kw_m2": df['rad'].mean(),
            }
        }

        # Convertir todos los np.nan a None para compatibilidad con JSON
        for category in stats.values():
            for key, value in category.items():
                if pd.isna(value):
                    category[key] = None

        return stats

def run_full_analysis(
    lat: float, lon: float, start: str, end: str, target_date: str,
    window_days: int = 7, rain_threshold: float = 0.5,
    hot_threshold: float = 30.0, cold_threshold: float = 10.0,
    use_ml: bool = True
) -> Dict:
    """Función principal que encapsula todo el análisis."""
    if not CAN_DOWNLOAD:
        return {"error": "El módulo de descarga de datos (power_timeseries.py) no está disponible."}

    # 1. Descargar datos
    logging.info(f"Descargando datos para ({lat}, {lon}) desde {start} hasta {end}")
    raw_data = fetch_power_point(lat, lon, start, end)
    df_hist = power_json_to_dataframe(raw_data)

    # 2. Inicializar analizador y obtener ventana
    analyzer = ClimateAnalyzer(df_hist)
    logging.info(f"Calculando ventana de análisis para {target_date} con ±{window_days} días.")
    window_df = analyzer.get_analysis_window(target_date, window_days)

    # 3. Calcular estadísticas base
    logging.info("Calculando estadísticas históricas...")
    results = analyzer.calculate_statistics(window_df, rain_threshold, hot_threshold, cold_threshold)

    # 4. (Opcional) Ejecutar ML avanzado
    if use_ml:
        logging.info("Ejecutando predictores de Machine Learning avanzados...")
        predictor = AdvancedClimatePredictor()

        # Necesitamos el DataFrame histórico completo para crear features con lags
        hist_precip_std = window_df["precip"].std()

        ml_results = integrate_with_analyzer(
            analyzer, predictor, df_hist, target_date, rain_threshold, hist_precip_std
        )

        # Fusionar resultados
        results.update(ml_results)

    return results
