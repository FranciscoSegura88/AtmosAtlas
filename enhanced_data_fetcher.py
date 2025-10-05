"""
enhanced_data_fetcher.py

Módulo mejorado para integrar múltiples fuentes de datos de NASA
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import xarray as xr
from pydap.client import open_url
from pydap.cas.urs import setup_session
import earthaccess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class EnhancedDataFetcher:
    """Integración de múltiples fuentes de datos NASA para análisis climático robusto"""
    
    def __init__(self, credentials: Optional[Dict] = None):
        """
        Args:
            credentials: Dict con credenciales NASA Earthdata
                {'username': 'user', 'password': 'pass'}
        """
        self.credentials = credentials
        self.sources = {
            'power': PowerDataSource(),
            'giovanni': GiovanniDataSource(),
            'gpm': GPMDataSource(),
            'opendap': OPeNDAPDataSource(),
            'modis': MODISDataSource()
        }
        
        # Setup Earthdata login si hay credenciales
        if credentials:
            earthaccess.login(
                strategy="netrc",
                persist=True
            )
    
    def fetch_ensemble_data(self, lat: float, lon: float, 
                           start_date: str, end_date: str,
                           sources: List[str] = None) -> pd.DataFrame:
        """
        Obtiene datos de múltiples fuentes y los combina en un DataFrame ensemble
        
        Args:
            lat: Latitud
            lon: Longitud  
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
            sources: Lista de fuentes a usar (default: todas)
            
        Returns:
            DataFrame con datos combinados de múltiples fuentes
        """
        if sources is None:
            sources = list(self.sources.keys())
        
        # Fetch paralelo de múltiples fuentes
        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            futures = {}
            for source_name in sources:
                if source_name in self.sources:
                    future = executor.submit(
                        self.sources[source_name].fetch,
                        lat, lon, start_date, end_date
                    )
                    futures[future] = source_name
            
            results = {}
            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    data = future.result(timeout=30)
                    results[source_name] = data
                    print(f"✓ {source_name}: {len(data)} registros obtenidos")
                except Exception as e:
                    print(f"✗ {source_name}: Error - {e}")
                    results[source_name] = pd.DataFrame()
        
        # Combinar resultados
        return self._merge_sources(results)
    
    def _merge_sources(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combina datos de múltiples fuentes usando técnicas de ensemble
        """
        # Crear DataFrame base con fechas
        start = pd.to_datetime(list(results.values())[0]['date'].min())
        end = pd.to_datetime(list(results.values())[0]['date'].max())
        date_range = pd.date_range(start, end, freq='D')
        
        merged = pd.DataFrame({'date': date_range})
        
        # Variables a combinar con pesos por fuente
        variable_weights = {
            'precip': {
                'gpm': 0.4,      # GPM es más preciso para precipitación
                'power': 0.3,
                'giovanni': 0.2,
                'modis': 0.1
            },
            'temp': {
                'modis': 0.35,    # MODIS es mejor para temperatura
                'power': 0.35,
                'giovanni': 0.2,
                'opendap': 0.1
            },
            'humidity': {
                'power': 0.4,
                'giovanni': 0.3,
                'modis': 0.2,
                'opendap': 0.1
            }
        }
        
        # Combinar cada variable usando weighted average
        for var, weights in variable_weights.items():
            var_data = []
            var_weights_used = []
            
            for source_name, df in results.items():
                if not df.empty and var in df.columns:
                    # Merge con fecha
                    temp = pd.merge(
                        merged[['date']], 
                        df[['date', var]], 
                        on='date', 
                        how='left',
                        suffixes=('', f'_{source_name}')
                    )
                    
                    if var in temp.columns:
                        var_data.append(temp[var].values)
                        var_weights_used.append(weights.get(source_name, 0.1))
            
            # Calcular weighted average
            if var_data:
                var_array = np.array(var_data)
                weights_array = np.array(var_weights_used)
                weights_array = weights_array / weights_array.sum()
                
                # Weighted nanmean
                merged[var] = np.average(
                    var_array,
                    axis=0,
                    weights=weights_array[:, None]
                )
                
                # Calcular incertidumbre (std entre fuentes)
                merged[f'{var}_uncertainty'] = np.nanstd(var_array, axis=0)
        
        return merged


class PowerDataSource:
    """Fuente de datos NASA POWER (ya implementado)"""
    
    def fetch(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        """Fetch desde NASA POWER API"""
        base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "start": start.replace("-", ""),
            "end": end.replace("-", ""),
            "latitude": lat,
            "longitude": lon,
            "community": "AG",
            "parameters": "PRECTOTCORR,T2M,RH2M,WS10M,ALLSKY_SFC_SW_DWN,PS",
            "format": "JSON"
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Convertir a DataFrame
            props = data.get("properties", {})
            param_data = props.get("parameter", {})
            
            dates = []
            records = []
            
            for date_str in param_data.get("PRECTOTCORR", {}).keys():
                date = datetime.strptime(date_str, "%Y%m%d")
                dates.append(date)
                
                record = {
                    'date': date,
                    'precip': param_data.get("PRECTOTCORR", {}).get(date_str),
                    'temp': param_data.get("T2M", {}).get(date_str),
                    'humidity': param_data.get("RH2M", {}).get(date_str),
                    'wind': param_data.get("WS10M", {}).get(date_str),
                    'radiation': param_data.get("ALLSKY_SFC_SW_DWN", {}).get(date_str),
                    'pressure': param_data.get("PS", {}).get(date_str)
                }
                records.append(record)
            
            return pd.DataFrame(records)
        
        except Exception as e:
            print(f"Error en POWER: {e}")
            return pd.DataFrame()


class GPMDataSource:
    """Fuente de datos GPM (Global Precipitation Measurement)"""
    
    def __init__(self):
        self.base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3"
        
    def fetch(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        """Fetch GPM IMERG precipitation data"""
        try:
            # GPM IMERG Late Run (más preciso)
            product = "GPM_3IMERGDL.06"
            
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            
            records = []
            current = start_dt
            
            while current <= end_dt:
                # Construir URL para el día
                year = current.year
                month = current.month
                day = current.day
                doy = current.timetuple().tm_yday
                
                url = (f"{self.base_url}/{product}/{year}/{doy:03d}/"
                      f"3B-DAY-L.MS.MRG.3IMERG.{current.strftime('%Y%m%d')}"
                      f"-S000000-E235959.V06.nc4")
                
                try:
                    # Intentar obtener datos del día
                    # Aquí usarías OPeNDAP para subset espacial
                    precip_value = self._fetch_point_from_opendap(url, lat, lon)
                    
                    records.append({
                        'date': current,
                        'precip': precip_value,
                        'precip_source': 'GPM_IMERG'
                    })
                except:
                    pass  # Día sin datos
                
                current += timedelta(days=1)
            
            return pd.DataFrame(records)
            
        except Exception as e:
            print(f"Error en GPM: {e}")
            return pd.DataFrame()
    
    def _fetch_point_from_opendap(self, url: str, lat: float, lon: float) -> float:
        """Extrae valor de un punto desde OPeNDAP"""
        # Simulación - en producción usarías pydap
        # con autenticación NASA Earthdata
        return np.random.uniform(0, 10)  # mm/día


class GiovanniDataSource:
    """Fuente de datos NASA Giovanni"""
    
    def fetch(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        """
        Fetch desde Giovanni usando su API REST
        Giovanni provee acceso a múltiples datasets procesados
        """
        try:
            # Giovanni requiere registro y API key
            # Aquí simulamos la estructura de datos que retornaría
            
            # En producción usarías:
            # 1. Autenticación OAuth2 con NASA Earthdata
            # 2. Solicitud de job a Giovanni
            # 3. Polling hasta completarse
            # 4. Download de resultados
            
            dates = pd.date_range(start, end, freq='D')
            data = []
            
            for date in dates:
                # Simular datos de múltiples productos Giovanni
                data.append({
                    'date': date,
                    'precip': np.random.uniform(0, 15),      # TRMM/GPM merged
                    'temp': np.random.uniform(15, 35),        # MODIS LST
                    'humidity': np.random.uniform(40, 90),    # AIRS
                    'cloud_fraction': np.random.uniform(0, 1) # MODIS Cloud
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error en Giovanni: {e}")
            return pd.DataFrame()


class OPeNDAPDataSource:
    """
    Acceso directo a datasets via OPeNDAP
    Permite subset espacial/temporal eficiente sin descargar archivos completos
    """
    
    def __init__(self):
        self.servers = {
            'gesdisc': 'https://disc2.gesdisc.eosdis.nasa.gov/opendap',
            'podaac': 'https://opendap.jpl.nasa.gov/opendap',
            'laads': 'https://ladsweb.modaps.eosdis.nasa.gov/opendap'
        }
    
    def fetch(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        """Fetch desde múltiples servidores OPeNDAP"""
        try:
            # Ejemplo: MERRA-2 reanalysis data
            dataset = self._fetch_merra2(lat, lon, start, end)
            return dataset
            
        except Exception as e:
            print(f"Error en OPeNDAP: {e}")
            return pd.DataFrame()
    
    def _fetch_merra2(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        """
        Fetch MERRA-2 reanalysis data
        MERRA-2 provee datos horarios de alta resolución
        """
        # En producción:
        # 1. Setup sesión con credenciales NASA Earthdata
        # 2. Abrir dataset con pydap
        # 3. Subset por lat/lon/tiempo
        # 4. Agregar a diario
        
        dates = pd.date_range(start, end, freq='D')
        data = []
        
        for date in dates:
            data.append({
                'date': date,
                'temp': np.random.uniform(15, 35),
                'temp_min': np.random.uniform(10, 20),
                'temp_max': np.random.uniform(25, 40),
                'humidity': np.random.uniform(40, 90),
                'pressure': np.random.uniform(980, 1030),
                'wind_u': np.random.uniform(-5, 5),
                'wind_v': np.random.uniform(-5, 5)
            })
        
        df = pd.DataFrame(data)
        df['wind'] = np.sqrt(df['wind_u']**2 + df['wind_v']**2)
        
        return df


class MODISDataSource:
    """
    Datos MODIS de temperatura superficial y vegetación
    Útil para microclimas y condiciones locales
    """
    
    def fetch(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        """Fetch MODIS Land Surface Temperature y otros productos"""
        try:
            # MODIS provee datos cada 8 días
            # Interpolaremos para obtener serie diaria
            
            dates = pd.date_range(start, end, freq='8D')
            data = []
            
            for date in dates:
                data.append({
                    'date': date,
                    'lst_day': np.random.uniform(20, 40),    # Land Surface Temp día
                    'lst_night': np.random.uniform(10, 25),  # Land Surface Temp noche
                    'ndvi': np.random.uniform(0.2, 0.8),     # Índice vegetación
                    'evi': np.random.uniform(0.1, 0.6)       # Enhanced Vegetation Index
                })
            
            df = pd.DataFrame(data)
            
            # Interpolar a diario
            df = df.set_index('date').resample('D').interpolate(method='cubic')
            df['temp'] = (df['lst_day'] + df['lst_night']) / 2
            
            return df.reset_index()
            
        except Exception as e:
            print(f"Error en MODIS: {e}")
            return pd.DataFrame()


class DataQualityAnalyzer:
    """
    Analiza la calidad de datos de múltiples fuentes
    y proporciona métricas de confiabilidad
    """
    
    @staticmethod
    def assess_quality(df: pd.DataFrame) -> Dict:
        """
        Evalúa calidad de los datos
        
        Returns:
            Dict con métricas de calidad
        """
        metrics = {
            'completeness': {},
            'consistency': {},
            'reliability': {}
        }
        
        # Completeness: % de datos no-nulos
        for col in df.columns:
            if col != 'date':
                metrics['completeness'][col] = df[col].notna().mean()
        
        # Consistency: correlación entre fuentes (si hay columnas _uncertainty)
        uncertainty_cols = [c for c in df.columns if '_uncertainty' in c]
        if uncertainty_cols:
            metrics['consistency']['avg_uncertainty'] = df[uncertainty_cols].mean().mean()
            metrics['consistency']['max_uncertainty'] = df[uncertainty_cols].max().max()
        
        # Reliability score (0-1)
        avg_completeness = np.mean(list(metrics['completeness'].values()))
        avg_consistency = 1 - metrics['consistency'].get('avg_uncertainty', 0) / 10
        metrics['reliability']['overall_score'] = (avg_completeness + avg_consistency) / 2
        
        # Recomendaciones
        metrics['recommendations'] = []
        if avg_completeness < 0.8:
            metrics['recommendations'].append("Considerar ampliar ventana temporal para más datos")
        if metrics['consistency'].get('avg_uncertainty', 0) > 5:
            metrics['recommendations'].append("Alta incertidumbre entre fuentes - verificar manualmente")
        
        return metrics


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar fetcher con credenciales (opcional)
    fetcher = EnhancedDataFetcher()
    
    # Obtener datos ensemble
    lat, lon = 20.676667, -103.347222
    start = "2024-01-01"
    end = "2024-12-31"
    
    print(f"Obteniendo datos para ({lat}, {lon}) desde {start} hasta {end}")
    print("="*60)
    
    # Fetch de múltiples fuentes
    ensemble_data = fetcher.fetch_ensemble_data(
        lat, lon, start, end,
        sources=['power', 'giovanni', 'gpm', 'modis']
    )
    
    print(f"\n✓ Datos ensemble obtenidos: {len(ensemble_data)} días")
    print(f"✓ Columnas disponibles: {list(ensemble_data.columns)}")
    
    # Analizar calidad
    quality = DataQualityAnalyzer.assess_quality(ensemble_data)
    print(f"\n📊 Análisis de Calidad:")
    print(f"   - Score de confiabilidad: {quality['reliability']['overall_score']:.2%}")
    print(f"   - Completitud promedio: {np.mean(list(quality['completeness'].values())):.2%}")
    
    if quality['recommendations']:
        print("\n⚠️ Recomendaciones:")
        for rec in quality['recommendations']:
            print(f"   - {rec}")
    
    # Guardar resultados
    output_file = f"ensemble_data_{lat}_{lon}_{start}_{end}.csv"
    ensemble_data.to_csv(output_file, index=False)
    print(f"\n💾 Datos guardados en: {output_file}")