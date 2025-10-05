#!/usr/bin/env python3
"""
analizador.py

Analiza CSVs de power_timeseries.py y calcula probabilidades climáticas
para fechas específicas usando ventanas temporales inteligentes.

Uso:
    python analizador.py --csv datos.csv --target_date 2025-08-15
    python analizador.py --lat 20.67 --lon -103.35 --target_date 2025-08-15 --window_days 7
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Importar función de descarga si existe
try:
    from power_timeseries import fetch_power_point, power_json_to_dataframe, DEFAULT_PARAMS
    CAN_DOWNLOAD = True
except ImportError:
    CAN_DOWNLOAD = False
    print("[WARN] No se puede importar power_timeseries.py - solo modo CSV")


class ClimateAnalyzer:
    """Analizador de probabilidades climáticas desde datos históricos"""
    
    def __init__(self, csv_path: Optional[str] = None):
        self.df = None
        self.csv_path = csv_path
        if csv_path:
            self.load_csv(csv_path)
    
    def load_csv(self, csv_path: str) -> None:
        """Carga datos desde CSV validando estructura"""
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Validar columnas requeridas
        required = ['date', 'precip', 't2m']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes en CSV: {missing}")
        
        print(f"✓ CSV cargado: {len(self.df)} registros desde {self.df['date'].min()} hasta {self.df['date'].max()}")
    
    def download_data(self, lat: float, lon: float, start: str, end: str) -> None:
        """Descarga datos desde NASA POWER si no hay CSV"""
        if not CAN_DOWNLOAD:
            raise RuntimeError("No se puede descargar - power_timeseries.py no disponible")
        
        print(f"Descargando datos para ({lat}, {lon}) desde {start} hasta {end}...")
        pj = fetch_power_point(lat, lon, start, end, parameters=DEFAULT_PARAMS, retries=5)
        self.df = power_json_to_dataframe(pj, parameters=DEFAULT_PARAMS)
        
        if self.df.empty:
            raise ValueError("No se obtuvieron datos de NASA POWER")
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"✓ Descarga completa: {len(self.df)} registros")
    
    def get_historical_window(self, target_date: str, window_days: int = 7) -> pd.DataFrame:
        """Extrae ventana histórica: mismo período (±window_days) de años anteriores"""
        target = pd.to_datetime(target_date)
        target_year = target.year
        
        target_doy = target.timetuple().tm_yday
        min_doy = max(1, target_doy - window_days)
        max_doy = min(366, target_doy + window_days)
        
        hist_df = self.df[self.df['date'].dt.year < target_year].copy()
        if hist_df.empty:
            raise ValueError(f"No hay datos históricos antes de {target_year}")
        
        hist_df['doy'] = hist_df['date'].dt.dayofyear
        window_df = hist_df[(hist_df['doy'] >= min_doy) & (hist_df['doy'] <= max_doy)].copy()
        
        print(f"✓ Ventana histórica: {len(window_df)} registros de {window_df['date'].dt.year.nunique()} años")
        return window_df
    
    def calculate_probabilities(self, window_df: pd.DataFrame, 
                                 rain_threshold: float = 0.5,
                                 hot_threshold: float = 30.0,
                                 cold_threshold: float = 10.0) -> Dict:
        """Calcula probabilidades basadas en frecuencias históricas"""
        results = {
            'n_samples': len(window_df),
            'n_years': window_df['date'].dt.year.nunique(),
            'date_range': (window_df['date'].min().strftime('%Y-%m-%d'), 
                          window_df['date'].max().strftime('%Y-%m-%d'))
        }
        
        # Precipitación
        valid_precip = window_df['precip'].dropna()
        if len(valid_precip) > 0:
            results['rain_probability'] = (valid_precip > rain_threshold).mean()
            results['avg_precip'] = valid_precip.mean()
            results['max_precip'] = valid_precip.max()
            results['precip_std'] = valid_precip.std()
        
        # Temperatura
        valid_temp = window_df['t2m'].dropna()
        if len(valid_temp) > 0:
            results['hot_day_probability'] = (valid_temp > hot_threshold).mean()
            results['cold_day_probability'] = (valid_temp < cold_threshold).mean()
            results['avg_temp'] = valid_temp.mean()
            results['temp_std'] = valid_temp.std()
            results['temp_range'] = (valid_temp.min(), valid_temp.max())
        
        # Humedad (si existe)
        if 'rh2m' in window_df.columns:
            valid_rh = window_df['rh2m'].dropna()
            if len(valid_rh) > 0:
                results['avg_humidity'] = valid_rh.mean()
                results['high_humidity_prob'] = (valid_rh > 80).mean()
        
        # Viento (si existe)
        if 'wind' in window_df.columns:
            valid_wind = window_df['wind'].dropna()
            if len(valid_wind) > 0:
                results['avg_wind'] = valid_wind.mean()
                results['strong_wind_prob'] = (valid_wind > 7).mean()  # >25 km/h aprox
        
        return results
    
    def train_ml_models(self, window_df: pd.DataFrame, 
                        rain_threshold: float = 0.5) -> Optional[Dict]:
        """Entrena modelos ML si hay suficientes datos"""
        df_clean = window_df.dropna(subset=['precip']).copy()
        
        if len(df_clean) < 10:
            print("[WARN] Datos insuficientes para ML (<10 muestras)")
            return None
        
        df_clean['rained'] = (df_clean['precip'] > rain_threshold).astype(int)
        
        feature_cols = [c for c in ['t2m', 'rh2m', 'wind', 'rad', 'ps'] if c in df_clean.columns]
        if not feature_cols:
            print("[WARN] No hay features para entrenar modelos")
            return None
        
        X = df_clean[feature_cols].fillna(df_clean[feature_cols].median()).values
        y = df_clean['rained'].values
        
        if len(np.unique(y)) < 2:
            print("[WARN] Target sin variedad - todos días con/sin lluvia")
            return None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logreg = LogisticRegression(max_iter=1000, random_state=42)
        logreg.fit(X_scaled, y)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X_scaled, y)
        
        lr_pred = logreg.predict(X_scaled); lr_prob = logreg.predict_proba(X_scaled)[:, 1]
        rf_pred = rf.predict(X_scaled);     rf_prob = rf.predict_proba(X_scaled)[:, 1]
        
        return {
            'models': {'logreg': logreg, 'rf': rf, 'scaler': scaler},
            'features': feature_cols,
            'metrics': {
                'logreg_acc': accuracy_score(y, lr_pred),
                'logreg_auc': roc_auc_score(y, lr_prob) if len(np.unique(y)) > 1 else None,
                'rf_acc': accuracy_score(y, rf_pred),
                'rf_auc': roc_auc_score(y, rf_prob) if len(np.unique(y)) > 1 else None
            }
        }
    
    def predict_for_date(self, target_date: str, window_df: pd.DataFrame,
                        ml_results: Optional[Dict] = None) -> Dict:
        """Genera predicción para fecha objetivo usando estadísticas de la ventana"""
        prediction = {'target_date': target_date, 'method': 'statistical'}
        
        if ml_results is not None:
            models = ml_results['models']; features = ml_results['features']
            feature_values = [window_df[f].median() if f in window_df.columns else 0.0 for f in features]
            X_pred = np.array(feature_values).reshape(1, -1)
            X_pred_scaled = models['scaler'].transform(X_pred)
            
            lr_prob = models['logreg'].predict_proba(X_pred_scaled)[0, 1]
            rf_prob = models['rf'].predict_proba(X_pred_scaled)[0, 1]
            
            prediction['method'] = 'ml_ensemble'
            prediction['logreg_prob'] = float(lr_prob)
            prediction['rf_prob'] = float(rf_prob)
            prediction['ensemble_prob'] = float((lr_prob + rf_prob) / 2)
        
        return prediction


def print_results(probs: Dict, prediction: Dict, ml_results: Optional[Dict] = None) -> None:
    """Imprime resultados en formato legible"""
    print("\n" + "="*60)
    print(f"ANALISIS CLIMATICO PARA {prediction['target_date']}")
    print("="*60)
    
    print(f"\nDatos Historicos:")
    print(f"   * Muestras: {probs['n_samples']} dias de {probs['n_years']} años diferentes")
    print(f"   * Rango: {probs['date_range'][0]} a {probs['date_range'][1]}")
    
    if 'rain_probability' in probs:
        print(f"\nPrecipitacion:")
        print(f"   * Probabilidad de lluvia: {probs['rain_probability']*100:.1f}%")
        print(f"   * Promedio cuando llueve: {probs['avg_precip']:.1f} mm")
        print(f"   * Maximo historico: {probs['max_precip']:.1f} mm")
    
    if 'avg_temp' in probs:
        print(f"\nTemperatura:")
        print(f"   * Promedio: {probs['avg_temp']:.1f}°C (±{probs['temp_std']:.1f}°C)")
        print(f"   * Rango: {probs['temp_range'][0]:.1f}°C a {probs['temp_range'][1]:.1f}°C")
        print(f"   * Probabilidad dia caluroso (>30°C): {probs['hot_day_probability']*100:.1f}%")
        print(f"   * Probabilidad dia frio (<10°C): {probs['cold_day_probability']*100:.1f}%")
    
    if 'avg_humidity' in probs:
        print(f"\nHumedad:")
        print(f"   * Promedio: {probs['avg_humidity']:.1f}%")
        print(f"   * Probabilidad alta humedad (>80%): {probs['high_humidity_prob']*100:.1f}%")
    
    if 'avg_wind' in probs:
        print(f"\nViento:")
        print(f"   * Velocidad promedio: {probs['avg_wind']:.1f} m/s")
        print(f"   * Probabilidad viento fuerte (>7 m/s): {probs['strong_wind_prob']*100:.1f}%")
    
    if ml_results:
        print(f"\nModelos Machine Learning:")
        metrics = ml_results['metrics']
        print(f"   * Logistic Regression: {metrics['logreg_acc']*100:.1f}% acc, AUC={metrics['logreg_auc']:.3f}")
        print(f"   * Random Forest: {metrics['rf_acc']*100:.1f}% acc, AUC={metrics['rf_auc']:.3f}")
        
        if prediction['method'] == 'ml_ensemble':
            print(f"\nPrediccion Ensamble (ML):")
            print(f"   * Probabilidad de lluvia: {prediction['ensemble_prob']*100:.1f}%")
            print(f"   * Decision: {'LLOVERA' if prediction['ensemble_prob'] > 0.5 else 'NO LLOVERA'}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Analiza probabilidades climáticas desde datos históricos',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv', help='Archivo CSV con datos históricos')
    source.add_argument('--lat', type=float, help='Latitud (requiere --lon, --start, --end)')
    
    parser.add_argument('--lon', type=float, help='Longitud')
    parser.add_argument('--start', help='Fecha inicio (YYYY-MM-DD) para descarga')
    parser.add_argument('--end', help='Fecha fin (YYYY-MM-DD) para descarga')
    parser.add_argument('--target_date', required=True, help='Fecha objetivo (YYYY-MM-DD)')
    parser.add_argument('--window_days', type=int, default=7, help='Días ±alrededor de la fecha (default: 7)')
    parser.add_argument('--rain_threshold', type=float, default=0.5, help='Umbral lluvia mm (default: 0.5)')
    parser.add_argument('--hot_threshold', type=float, default=30.0, help='Umbral calor °C (default: 30)')
    parser.add_argument('--cold_threshold', type=float, default=10.0, help='Umbral frío °C (default: 10)')
    parser.add_argument('--no_ml', action='store_true', help='No entrenar modelos ML (solo estadísticas)')
    parser.add_argument('--save_window', help='Guardar ventana histórica a CSV')
    
    args = parser.parse_args()
    
    if args.lat is not None:
        if not all([args.lon, args.start, args.end]):
            parser.error('--lat requiere --lon, --start y --end')
    
    try:
        analyzer = ClimateAnalyzer()
        
        if args.csv:
            analyzer.load_csv(args.csv)
        else:
            analyzer.download_data(args.lat, args.lon, args.start, args.end)
        
        window_df = analyzer.get_historical_window(args.target_date, args.window_days)
        
        if args.save_window:
            window_df.to_csv(args.save_window, index=False)
            print(f"✓ Ventana histórica guardada en: {args.save_window}")
        
        probs = analyzer.calculate_probabilities(
            window_df, 
            rain_threshold=args.rain_threshold,
            hot_threshold=args.hot_threshold,
            cold_threshold=args.cold_threshold
        )
        
        ml_results = None
        if not args.no_ml:
            ml_results = analyzer.train_ml_models(window_df, args.rain_threshold)
        
        prediction = analyzer.predict_for_date(args.target_date, window_df, ml_results)
        print_results(probs, prediction, ml_results)
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
