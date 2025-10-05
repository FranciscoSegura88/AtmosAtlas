"""
models.py

Modelos Pydantic para requests y responses de la API de AtmosAtlas.
Compatible con FastAPI.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Tuple
from datetime import date, datetime


# ==================== REQUEST MODELS ====================

class PredictionRequest(BaseModel):
    """Request para análisis climático de una ubicación y fecha"""
    
    latitude: float = Field(
        ..., 
        ge=-90, 
        le=90,
        description="Latitud en grados decimales (-90 a 90)",
        examples=[20.676667]
    )
    
    longitude: float = Field(
        ..., 
        ge=-180, 
        le=180,
        description="Longitud en grados decimales (-180 a 180)",
        examples=[-103.347222]
    )
    
    target_date: str = Field(
        ...,
        description="Fecha objetivo en formato YYYY-MM-DD",
        examples=["2025-08-15"]
    )
    
    # Parámetros opcionales
    start_date: Optional[str] = Field(
        default="1990-01-01",
        description="Fecha inicio datos históricos (YYYY-MM-DD)",
        examples=["1990-01-01"]
    )
    
    end_date: Optional[str] = Field(
        default="2024-12-31",
        description="Fecha fin datos históricos (YYYY-MM-DD)",
        examples=["2024-12-31"]
    )
    
    window_days: Optional[int] = Field(
        default=7,
        ge=1,
        le=30,
        description="Días ±alrededor de la fecha objetivo (1-30)",
        examples=[7]
    )
    
    rain_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=100.0,
        description="Umbral de precipitación en mm para considerar 'lluvia'",
        examples=[0.5]
    )
    
    hot_threshold: Optional[float] = Field(
        default=30.0,
        ge=-50.0,
        le=60.0,
        description="Umbral de temperatura en °C para considerar 'día caluroso'",
        examples=[30.0]
    )
    
    cold_threshold: Optional[float] = Field(
        default=10.0,
        ge=-50.0,
        le=60.0,
        description="Umbral de temperatura en °C para considerar 'día frío'",
        examples=[10.0]
    )
    
    use_ml: Optional[bool] = Field(
        default=True,
        description="Usar modelos de Machine Learning para predicción",
        examples=[True]
    )
    
    @field_validator('target_date', 'start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Valida formato de fecha YYYY-MM-DD"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError(f"Fecha debe estar en formato YYYY-MM-DD, recibido: {v}")
    
    @field_validator('target_date')
    @classmethod
    def validate_target_after_end(cls, v: str, info) -> str:
        """Valida que target_date sea posterior a end_date"""
        if 'end_date' in info.data:
            target = datetime.strptime(v, '%Y-%m-%d')
            end = datetime.strptime(info.data['end_date'], '%Y-%m-%d')
            if target <= end:
                raise ValueError(
                    f"target_date ({v}) debe ser posterior a end_date ({info.data['end_date']})"
                )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "latitude": 20.676667,
                    "longitude": -103.347222,
                    "target_date": "2025-08-15",
                    "start_date": "1990-01-01",
                    "end_date": "2024-12-31",
                    "window_days": 7,
                    "rain_threshold": 0.5,
                    "hot_threshold": 30.0,
                    "cold_threshold": 10.0,
                    "use_ml": True
                }
            ]
        }
    }


# ==================== RESPONSE MODELS ====================

class LocationInfo(BaseModel):
    """Información de la ubicación analizada"""
    latitude: float
    longitude: float
    coordinates_formatted: str = Field(
        description="Coordenadas en formato legible",
        examples=["20.68°N, 103.35°W"]
    )


class HistoricalDataStats(BaseModel):
    """Estadísticas de los datos históricos utilizados"""
    n_samples: int = Field(description="Número total de días analizados")
    n_years: int = Field(description="Número de años únicos en el análisis")
    date_range_start: str = Field(description="Primera fecha del rango histórico")
    date_range_end: str = Field(description="Última fecha del rango histórico")
    window_days: int = Field(description="Ventana temporal usada (±días)")


class PrecipitationStats(BaseModel):
    """Estadísticas de precipitación"""
    rain_probability: float = Field(
        ge=0.0, 
        le=1.0,
        description="Probabilidad de lluvia (0-1)"
    )
    rain_probability_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Probabilidad de lluvia en porcentaje (0-100)"
    )
    avg_precip_mm: float = Field(description="Precipitación promedio en mm")
    max_precip_mm: float = Field(description="Precipitación máxima histórica en mm")
    std_precip_mm: float = Field(description="Desviación estándar de precipitación en mm")


class TemperatureStats(BaseModel):
    """Estadísticas de temperatura"""
    avg_temp_celsius: float = Field(description="Temperatura promedio en °C")
    std_temp_celsius: float = Field(description="Desviación estándar de temperatura en °C")
    min_temp_celsius: float = Field(description="Temperatura mínima histórica en °C")
    max_temp_celsius: float = Field(description="Temperatura máxima histórica en °C")
    hot_day_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidad de día caluroso (0-1)"
    )
    cold_day_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidad de día frío (0-1)"
    )


class HumidityStats(BaseModel):
    """Estadísticas de humedad relativa"""
    avg_humidity_percent: float = Field(description="Humedad relativa promedio (%)")
    high_humidity_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidad de humedad alta >80% (0-1)"
    )


class WindStats(BaseModel):
    """Estadísticas de viento"""
    avg_wind_speed_ms: float = Field(description="Velocidad promedio del viento en m/s")
    strong_wind_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidad de viento fuerte >7 m/s (0-1)"
    )


class MLModelMetrics(BaseModel):
    """Métricas de los modelos de Machine Learning"""
    logistic_regression_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Precisión del modelo Logistic Regression"
    )
    logistic_regression_auc: float = Field(
        ge=0.0,
        le=1.0,
        description="AUC-ROC del modelo Logistic Regression"
    )
    random_forest_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Precisión del modelo Random Forest"
    )
    random_forest_auc: float = Field(
        ge=0.0,
        le=1.0,
        description="AUC-ROC del modelo Random Forest"
    )


class MLPrediction(BaseModel):
    """Predicción mediante Machine Learning"""
    method: str = Field(description="Método usado: 'ml_ensemble' o 'statistical'")
    logistic_regression_prob: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probabilidad de lluvia según Logistic Regression"
    )
    random_forest_prob: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probabilidad de lluvia según Random Forest"
    )
    ensemble_prob: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidad de lluvia promediada (ensamble)"
    )
    ensemble_prob_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Probabilidad ensamble en porcentaje"
    )
    will_rain: bool = Field(description="Predicción final: True=lloverá, False=no lloverá")
    confidence: str = Field(
        description="Nivel de confianza: 'high', 'medium', 'low'",
        examples=["high"]
    )


class PredictionResponse(BaseModel):
    """Response completa del análisis climático"""
    
    # Metadata de la solicitud
    target_date: str = Field(description="Fecha objetivo analizada (YYYY-MM-DD)")
    location: LocationInfo
    
    # Datos históricos
    historical_data: HistoricalDataStats
    
    # Estadísticas climáticas
    precipitation: PrecipitationStats
    temperature: TemperatureStats
    humidity: Optional[HumidityStats] = None
    wind: Optional[WindStats] = None
    
    # Predicción ML (si aplica)
    ml_prediction: Optional[MLPrediction] = None
    ml_model_metrics: Optional[MLModelMetrics] = None
    
    # Metadata del procesamiento
    processing_time_seconds: float = Field(description="Tiempo de procesamiento en segundos")
    data_source: str = Field(
        default="NASA POWER",
        description="Fuente de los datos",
        examples=["NASA POWER"]
    )
    api_version: str = Field(
        default="1.0.0",
        description="Versión de la API",
        examples=["1.0.0"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "target_date": "2025-08-15",
                    "location": {
                        "latitude": 20.676667,
                        "longitude": -103.347222,
                        "coordinates_formatted": "20.68°N, 103.35°W"
                    },
                    "historical_data": {
                        "n_samples": 525,
                        "n_years": 35,
                        "date_range_start": "1990-08-08",
                        "date_range_end": "2024-08-22",
                        "window_days": 7
                    },
                    "precipitation": {
                        "rain_probability": 0.68,
                        "rain_probability_percent": 68.0,
                        "avg_precip_mm": 8.2,
                        "max_precip_mm": 45.3,
                        "std_precip_mm": 12.1
                    },
                    "temperature": {
                        "avg_temp_celsius": 24.5,
                        "std_temp_celsius": 2.3,
                        "min_temp_celsius": 18.2,
                        "max_temp_celsius": 30.1,
                        "hot_day_probability": 0.12,
                        "cold_day_probability": 0.0
                    },
                    "humidity": {
                        "avg_humidity_percent": 78.5,
                        "high_humidity_probability": 0.45
                    },
                    "wind": {
                        "avg_wind_speed_ms": 3.2,
                        "strong_wind_probability": 0.08
                    },
                    "ml_prediction": {
                        "method": "ml_ensemble",
                        "logistic_regression_prob": 0.71,
                        "random_forest_prob": 0.65,
                        "ensemble_prob": 0.68,
                        "ensemble_prob_percent": 68.0,
                        "will_rain": True,
                        "confidence": "high"
                    },
                    "ml_model_metrics": {
                        "logistic_regression_accuracy": 0.84,
                        "logistic_regression_auc": 0.91,
                        "random_forest_accuracy": 0.86,
                        "random_forest_auc": 0.93
                    },
                    "processing_time_seconds": 2.35,
                    "data_source": "NASA POWER",
                    "api_version": "1.0.0"
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Response para errores de la API"""
    error: str = Field(description="Tipo de error")
    message: str = Field(description="Mensaje de error detallado")
    details: Optional[Dict] = Field(
        default=None,
        description="Detalles adicionales del error"
    )
    timestamp: str = Field(description="Timestamp del error en ISO 8601")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "ValidationError",
                    "message": "target_date debe ser posterior a end_date",
                    "details": {
                        "target_date": "2023-08-15",
                        "end_date": "2024-12-31"
                    },
                    "timestamp": "2025-01-15T10:30:45Z"
                }
            ]
        }
    }