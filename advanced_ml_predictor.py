"""
advanced_ml_predictor.py

Modelos de Machine Learning avanzados para predicción climática
Incluye XGBoost, LightGBM, LSTM y técnicas de ensemble
Versión completa integrada con AtmosAtlas
"""
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Advanced ML
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost no disponible - instalar con: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[WARN] LightGBM no disponible - instalar con: pip install lightgbm")

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, 
                                         Concatenate, BatchNormalization, 
                                         Bidirectional, Attention)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("[WARN] TensorFlow no disponible - instalar con: pip install tensorflow")


class AdvancedClimatePredictor:
    """
    Predictor avanzado usando múltiples algoritmos de ML y deep learning
    Optimizado para el NASA Space Apps Challenge
    """
    
    def __init__(self, config: dict = None, model_dir: str = "models"):
        """
        Args:
            config: Configuración de modelos y parámetros
            model_dir: Directorio para guardar modelos
        """
        self.config = config or self.get_default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.ensemble_weights = None
        self.feature_names = None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Historia de entrenamiento
        self.training_history = {
            'models_trained': [],
            'metrics': {},
            'timestamp': None
        }
        
    @staticmethod
    def get_default_config():
        """Configuración por defecto optimizada para clima"""
        config = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'min_samples_split': 5
            },
            'ridge': {
                'alpha': 1.0
            },
            'elastic': {
                'alpha': 0.1,
                'l1_ratio': 0.5
            }
        }
        
        if HAS_XGBOOST:
            config['xgboost'] = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.01,
                'reg_alpha': 0.01,
                'reg_lambda': 1,
                'min_child_weight': 3
            }
        
        if HAS_LIGHTGBM:
            config['lightgbm'] = {
                'n_estimators': 500,
                'max_depth': -1,
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.01,
                'lambda_l2': 0.01,
                'min_child_samples': 20,
                'verbose': -1
            }
        
        if HAS_TENSORFLOW:
            config['lstm'] = {
                'units': [128, 64, 32],
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 32,
                'patience': 15,
                'sequence_length': 30
            }
        
        return config
    
    def create_features(self, df: pd.DataFrame, target_col: str = 'precip') -> pd.DataFrame:
        """
        Crea features avanzadas para predicción
        
        Args:
            df: DataFrame con datos históricos
            target_col: Columna objetivo
            
        Returns:
            DataFrame con features engineered
        """
        df = df.copy()
        
        # Asegurar que date es datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Features temporales
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
            df['quarter'] = df['date'].dt.quarter
            df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
            
            # Features cíclicas (sin, cos para capturar periodicidad)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Features de lag (valores pasados) - optimizado para clima
        lag_cols = ['precip', 't2m', 'rh2m', 'ps', 'wind'] if target_col == 'precip' else ['t2m', 'rh2m', 'ps']
        for col in lag_cols:
            if col in df.columns:
                # Lags estratégicos
                for lag in [1, 3, 7, 14, 30]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Rolling statistics
                for window in [7, 14, 30, 60]:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window, min_periods=1).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window, min_periods=1).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window, min_periods=1).max()
                    df[f'{col}_rolling_median_{window}'] = df[col].rolling(window, min_periods=1).median()
        
        # Features de cambio (deltas)
        change_cols = ['t2m', 'ps', 'rh2m']
        for col in change_cols:
            if col in df.columns:
                df[f'{col}_change_1d'] = df[col].diff(1)
                df[f'{col}_change_7d'] = df[col].diff(7)
                df[f'{col}_change_30d'] = df[col].diff(30)
        
        # Índices climáticos
        if 't2m' in df.columns and 'rh2m' in df.columns:
            # Heat Index
            df['heat_index'] = self._calculate_heat_index(df['t2m'], df['rh2m'])
            # Humidex
            df['humidex'] = self._calculate_humidex(df['t2m'], df['rh2m'])
            # Temperatura sensación
            df['feels_like'] = df['t2m'] * 0.7 + df['rh2m'] * 0.3
        
        if 'wind' in df.columns and 't2m' in df.columns:
            # Wind Chill
            df['wind_chill'] = self._calculate_wind_chill(df['t2m'], df['wind'])
        
        # Interacciones entre variables
        if 't2m' in df.columns and 'rh2m' in df.columns:
            df['temp_humidity_interaction'] = df['t2m'] * df['rh2m'] / 100
        
        # Features de tendencia
        trend_cols = [col for col in ['precip', 't2m'] if col in df.columns]
        for col in trend_cols:
            for window in [7, 14, 30]:
                df[f'{col}_trend_{window}'] = df[col].rolling(window, min_periods=2).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    raw=True
                )
        
        # Features de anomalías
        for col in ['t2m', 'precip', 'rh2m']:
            if col in df.columns and 'month' in df.columns:
                # Anomalía respecto a la media histórica del mes
                monthly_mean = df.groupby('month')[col].transform('mean')
                monthly_std = df.groupby('month')[col].transform('std')
                df[f'{col}_anomaly'] = (df[col] - monthly_mean) / (monthly_std + 1e-5)
        
        # Features de eventos extremos
        if 'precip' in df.columns:
            df['heavy_rain_last_week'] = (df['precip'].rolling(7, min_periods=1).max() > 10).astype(int)
            df['dry_spell'] = (df['precip'].rolling(14, min_periods=1).sum() < 1).astype(int)
            df['wet_spell'] = (df['precip'].rolling(7, min_periods=1).sum() > 50).astype(int)
        
        if 't2m' in df.columns:
            df['heat_wave'] = (df['t2m'].rolling(3, min_periods=1).min() > 35).astype(int)
            df['cold_snap'] = (df['t2m'].rolling(3, min_periods=1).max() < 5).astype(int)
        
        # ENSO indicators (simulado - en producción usar índices reales)
        if 'year' in df.columns:
            df['enso_indicator'] = np.sin(2 * np.pi * df['year'] / 3.5)
            df['solar_cycle'] = np.sin(2 * np.pi * df['year'] / 11)
        
        return df
    
    def _calculate_heat_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calcula el índice de calor (Heat Index)"""
        c1, c2, c3 = -8.78469475556, 1.61139411, 2.33854883889
        c4, c5, c6 = -0.14611605, -0.012308094, -0.0164248277778
        c7, c8, c9 = 0.002211732, 0.00072546, -0.000003582
        
        T = temp * 9/5 + 32  # Celsius a Fahrenheit
        RH = humidity
        
        HI = (c1 + c2*T + c3*RH + c4*T*RH + c5*T**2 + 
              c6*RH**2 + c7*T**2*RH + c8*T*RH**2 + c9*T**2*RH**2)
        
        return (HI - 32) * 5/9  # Fahrenheit a Celsius
    
    def _calculate_humidex(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calcula el Humidex canadiense"""
        e = 6.112 * 10**(7.5 * temp / (237.3 + temp)) * humidity / 100
        return temp + 0.5555 * (e - 10)
    
    def _calculate_wind_chill(self, temp: pd.Series, wind: pd.Series) -> pd.Series:
        """Calcula el Wind Chill"""
        wind_kmh = wind * 3.6  # m/s a km/h
        return 13.12 + 0.6215 * temp - 11.37 * wind_kmh**0.16 + 0.3965 * temp * wind_kmh**0.16
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame = None, y_val: pd.Series = None,
                      verbose: bool = True) -> Dict:
        """
        Entrena ensemble de modelos
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación (opcional)
            y_val: Target de validación (opcional)
            verbose: Mostrar progreso
            
        Returns:
            Dict con métricas de entrenamiento
        """
        if verbose:
            print("\n🚀 Entrenando ensemble de modelos...")
        
        # Guardar nombres de features
        self.feature_names = X_train.columns.tolist()
        
        # Escalar features
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_train_robust = self.scalers['robust'].fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scalers['standard'].transform(X_val)
            X_val_robust = self.scalers['robust'].transform(X_val)
        
        metrics = {}
        
        # 1. Random Forest
        if verbose:
            print("   📊 Training Random Forest...")
        self.models['random_forest'] = RandomForestRegressor(
            **self.config['random_forest'],
            random_state=42
        )
        self.models['random_forest'].fit(X_train_robust, y_train)
        
        if X_val is not None:
            pred = self.models['random_forest'].predict(X_val_robust)
            metrics['random_forest'] = self._calculate_metrics(y_val, pred)
        
        # 2. Gradient Boosting
        if verbose:
            print("   📊 Training Gradient Boosting...")
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            **self.config['gradient_boosting'],
            random_state=42
        )
        self.models['gradient_boosting'].fit(X_train_robust, y_train)
        
        if X_val is not None:
            pred = self.models['gradient_boosting'].predict(X_val_robust)
            metrics['gradient_boosting'] = self._calculate_metrics(y_val, pred)
        
        # 3. XGBoost
        if HAS_XGBOOST and 'xgboost' in self.config:
            if verbose:
                print("   📊 Training XGBoost...")
            self.models['xgboost'] = xgb.XGBRegressor(
                **self.config['xgboost'],
                random_state=42,
                n_jobs=-1
            )
            
            eval_set = [(X_val_scaled, y_val)] if X_val is not None else None
            self.models['xgboost'].fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Feature importance
            self.feature_importance['xgboost'] = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models['xgboost'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            if X_val is not None:
                pred = self.models['xgboost'].predict(X_val_scaled)
                metrics['xgboost'] = self._calculate_metrics(y_val, pred)
        
        # 4. LightGBM
        if HAS_LIGHTGBM and 'lightgbm' in self.config:
            if verbose:
                print("   📊 Training LightGBM...")
            self.models['lightgbm'] = lgb.LGBMRegressor(
                **self.config['lightgbm'],
                random_state=42,
                n_jobs=-1
            )
            
            eval_set = [(X_val_scaled, y_val)] if X_val is not None else None
            self.models['lightgbm'].fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            if X_val is not None:
                pred = self.models['lightgbm'].predict(X_val_scaled)
                metrics['lightgbm'] = self._calculate_metrics(y_val, pred)
        
        # 5. Ridge
        if verbose:
            print("   📊 Training Ridge...")
        self.models['ridge'] = Ridge(**self.config['ridge'], random_state=42)
        self.models['ridge'].fit(X_train_scaled, y_train)
        
        if X_val is not None:
            pred = self.models['ridge'].predict(X_val_scaled)
            metrics['ridge'] = self._calculate_metrics(y_val, pred)
        
        # 6. ElasticNet
        if verbose:
            print("   📊 Training ElasticNet...")
        self.models['elastic'] = ElasticNet(**self.config['elastic'], random_state=42)
        self.models['elastic'].fit(X_train_scaled, y_train)
        
        if X_val is not None:
            pred = self.models['elastic'].predict(X_val_scaled)
            metrics['elastic'] = self._calculate_metrics(y_val, pred)
        
        # Calcular pesos del ensemble
        if X_val is not None:
            self._calculate_ensemble_weights(X_val_scaled, X_val_robust, y_val)
            if verbose:
                print("\n   ⚖️  Pesos del ensemble:")
                for name, weight in self.ensemble_weights.items():
                    print(f"      {name}: {weight:.3f}")
        else:
            # Pesos por defecto
            n_models = len(self.models)
            self.ensemble_weights = {name: 1.0/n_models for name in self.models.keys()}
        
        # Guardar historia
        self.training_history['models_trained'] = list(self.models.keys())
        self.training_history['metrics'] = metrics
        self.training_history['timestamp'] = pd.Timestamp.now().isoformat()
        
        if verbose:
            print("\n✅ Ensemble entrenado exitosamente")
            if metrics:
                print("\n📈 Métricas de validación:")
                for model_name, model_metrics in metrics.items():
                    print(f"   {model_name}:")
                    print(f"      RMSE: {model_metrics['rmse']:.4f}")
                    print(f"      MAE: {model_metrics['mae']:.4f}")
                    print(f"      R²: {model_metrics['r2']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calcula métricas de evaluación"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _calculate_ensemble_weights(self, X_val_scaled: np.ndarray, 
                                   X_val_robust: np.ndarray, 
                                   y_val: pd.Series) -> None:
        """Calcula pesos óptimos del ensemble usando validación"""
        predictions = {}
        errors = {}
        
        for name, model in self.models.items():
            if name in ['xgboost', 'lightgbm', 'ridge', 'elastic']:
                pred = model.predict(X_val_scaled)
            else:
                pred = model.predict(X_val_robust)
            
            predictions[name] = pred
            errors[name] = mean_squared_error(y_val, pred)
        
        # Pesos inversamente proporcionales al error
        total_inv_error = sum(1/(e + 1e-10) for e in errors.values())
        self.ensemble_weights = {
            name: (1/(error + 1e-10)) / total_inv_error 
            for name, error in errors.items()
        }
    
    def predict(self, X: pd.DataFrame, include_uncertainty: bool = True) -> Dict:
        """
        Realiza predicción usando el ensemble
        
        Args:
            X: Features para predicción
            include_uncertainty: Si incluir estimación de incertidumbre
            
        Returns:
            Dict con predicción, intervalo de confianza y detalles
        """
        X_scaled = self.scalers['standard'].transform(X)
        X_robust = self.scalers['robust'].transform(X)
        
        predictions = {}
        
        # Predicciones individuales
        for name, model in self.models.items():
            if name in ['xgboost', 'lightgbm', 'ridge', 'elastic']:
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X_robust)
            
            predictions[name] = float(pred[0] if hasattr(pred, '__len__') else pred)
        
        # Predicción del ensemble (weighted average)
        ensemble_pred = sum(
            predictions[name] * self.ensemble_weights.get(name, 0)
            for name in predictions
        )
        
        result = {
            'prediction': ensemble_pred,
            'individual_predictions': predictions,
            'ensemble_weights': self.ensemble_weights
        }
        
        if include_uncertainty:
            # Estimar incertidumbre
            pred_values = list(predictions.values())
            std_dev = np.std(pred_values)
            
            # Intervalo de confianza 95%
            result['confidence_interval'] = {
                'lower': max(0, ensemble_pred - 1.96 * std_dev),  # No negativo para precip
                'upper': ensemble_pred + 1.96 * std_dev
            }
            result['uncertainty'] = std_dev
            
            # Nivel de confianza
            cv = std_dev / (abs(ensemble_pred) + 1e-5)
            if cv < 0.1:
                result['confidence_level'] = 'high'
            elif cv < 0.25:
                result['confidence_level'] = 'medium'
            else:
                result['confidence_level'] = 'low'
        
        return result
    
    def save_models(self, prefix: str = "climate_model"):
        """Guarda todos los modelos entrenados"""
        save_path = self.model_dir / f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        save_path.mkdir(exist_ok=True)
        
        # Guardar cada modelo
        for name, model in self.models.items():
            model_file = save_path / f"{name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        # Guardar scalers
        for name, scaler in self.scalers.items():
            scaler_file = save_path / f"scaler_{name}.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Guardar metadata
        metadata = {
            'feature_names': self.feature_names,
            'ensemble_weights': self.ensemble_weights,
            'config': self.config,
            'training_history': self.training_history
        }
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Modelos guardados en: {save_path}")
        return str(save_path)
    
    def load_models(self, model_path: str):
        """Carga modelos previamente guardados"""
        model_path = Path(model_path)
        
        # Cargar metadata
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.ensemble_weights = metadata['ensemble_weights']
        self.config = metadata['config']
        self.training_history = metadata['training_history']
        
        # Cargar modelos
        for model_file in model_path.glob("*.pkl"):
            if model_file.stem.startswith("scaler_"):
                scaler_name = model_file.stem.replace("scaler_", "")
                with open(model_file, 'rb') as f:
                    self.scalers[scaler_name] = pickle.load(f)
            else:
                model_name = model_file.stem
                with open(model_file, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
        
        print(f"✅ Modelos cargados desde: {model_path}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Obtiene las features más importantes"""
        if 'xgboost' in self.feature_importance:
            return self.feature_importance['xgboost'].head(top_n)
        elif 'random_forest' in self.models:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models['random_forest'].feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df.head(top_n)
        return pd.DataFrame()


class ClimateDataPreprocessor:
    """Preprocesador especializado para datos climáticos"""
    
    @staticmethod
    def prepare_training_data(df: pd.DataFrame, 
                             target_col: str = 'precip',
                             test_size: float = 0.2,
                             val_size: float = 0.1) -> Dict:
        """
        Prepara datos para entrenamiento con split temporal
        
        Args:
            df: DataFrame con datos
            target_col: Columna objetivo
            test_size: Proporción para test
            val_size: Proporción para validación
            
        Returns:
            Dict con train/val/test splits
        """
        # Ordenar por fecha
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remover target de features
        feature_cols = [c for c in df.columns if c not in ['date', target_col]]
        
        # Split temporal (importante para series temporales)
        n = len(df)
        train_size = int(n * (1 - test_size - val_size))
        val_size_abs = int(n * val_size)
        
        train_data = df.iloc[:train_size]
        val_data = df.iloc[train_size:train_size + val_size_abs]
        test_data = df.iloc[train_size + val_size_abs:]
        
        return {
            'X_train': train_data[feature_cols],
            'y_train': train_data[target_col],
            'X_val': val_data[feature_cols],
            'y_val': val_data[target_col],
            'X_test': test_data[feature_cols],
            'y_test': test_data[target_col],
            'train_dates': train_data['date'],
            'val_dates': val_data['date'],
            'test_dates': test_data['date']
        }


# Integración con el analizador existente
def integrate_with_analyzer(analyzer, predictor, window_df: pd.DataFrame, 
                           target_date: str, rain_threshold: float = 0.5) -> Dict:
    """
    Integra el predictor avanzado con el ClimateAnalyzer existente
    
    Args:
        analyzer: Instancia de ClimateAnalyzer
        predictor: Instancia de AdvancedClimatePredictor
        window_df: DataFrame con ventana histórica
        target_date: Fecha objetivo
        rain_threshold: Umbral de lluvia
        
    Returns:
        Dict con predicciones combinadas
    """
    # Crear features avanzadas
    df_features = predictor.create_features(window_df, target_col='precip')
    df_clean = df_features.dropna()
    
    if len(df_clean) < 10:
        return {'error': 'Datos insuficientes para predicción avanzada'}
    
    # Preparar datos
    data = ClimateDataPreprocessor.prepare_training_data(
        df_clean, 
        target_col='precip',
        test_size=0.15,
        val_size=0.15
    )
    
    # Entrenar ensemble
    predictor.train_ensemble(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        verbose=False
    )
    
    # Predecir para fecha objetivo usando características históricas
    # Usar estadísticas de la ventana como proxy
    feature_values = {}
    for col in data['X_train'].columns:
        if col in df_clean.columns:
            feature_values[col] = df_clean[col].iloc[-1]  # Último valor disponible
        else:
            feature_values[col] = 0
    
    X_pred = pd.DataFrame([feature_values])
    
    # Realizar predicción
    prediction = predictor.predict(X_pred, include_uncertainty=True)
    
    # Clasificar
    will_rain = prediction['prediction'] > rain_threshold
    
    return {
        'advanced_ml': {
            'precipitation_mm': prediction['prediction'],
            'will_rain': will_rain,
            'confidence_interval': prediction['confidence_interval'],
            'confidence_level': prediction['confidence_level'],
            'uncertainty': prediction['uncertainty'],
            'individual_models': prediction['individual_predictions']
        },
        'model_weights': prediction['ensemble_weights'],
        'training_metrics': predictor.training_history.get('metrics', {})
    }