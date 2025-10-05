"""
main_api_actualizado.py

API REST con FastAPI para AtmosAtlas - Versión con ML Avanzado
Integra los modelos avanzados de XGBoost, LightGBM y ensemble learning
"""
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime
from typing import Optional, Dict, List
import traceback
import asyncio
from pathlib import Path

# Importar modelos Pydantic
from models import (
    PredictionRequest, 
    PredictionResponse, 
    ErrorResponse,
    LocationInfo,
    HistoricalDataStats,
    PrecipitationStats,
    TemperatureStats,
    HumidityStats,
    WindStats,
    MLPrediction,
    MLModelMetrics
)

# Importar analizadores
from analizador import ClimateAnalyzer
from advanced_ml_predictor import (
    AdvancedClimatePredictor,
    ClimateDataPreprocessor,
    integrate_with_analyzer
)

# Inicializar FastAPI
app = FastAPI(
    title="AtmosAtlas API - ML Avanzado",
    description="API de análisis climático con Machine Learning avanzado usando datos de NASA",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache global para modelos (evitar reentrenamiento)
model_cache = {}
MODEL_CACHE_DIR = Path("models/cache")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==================== UTILIDADES ====================

def format_coordinates(lat: float, lon: float) -> str:
    """Formatea coordenadas en formato legible"""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{lat_dir}, {abs(lon):.2f}°{lon_dir}"


def calculate_confidence(prob: float) -> str:
    """Calcula nivel de confianza basado en probabilidad"""
    if prob >= 0.7 or prob <= 0.3:
        return "high"
    elif prob >= 0.55 or prob <= 0.45:
        return "medium"
    else:
        return "low"


def get_cache_key(lat: float, lon: float, start: str, end: str) -> str:
    """Genera clave de cache para modelos"""
    return f"{lat:.4f}_{lon:.4f}_{start}_{end}"


# ==================== EXCEPTION HANDLERS ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Manejador global de excepciones"""
    error_response = ErrorResponse(
        error=exc.__class__.__name__,
        message=str(exc),
        details={"traceback": traceback.format_exc()},
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )


# ==================== ENDPOINTS ====================

@app.get("/", tags=["Health"])
async def root():
    """Endpoint raíz"""
    return {
        "message": "AtmosAtlas API v2.0.0 - ML Avanzado",
        "status": "online",
        "features": {
            "basic_ml": True,
            "advanced_ml": True,
            "ensemble_learning": True,
            "feature_engineering": True
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "predict_advanced": "/predict/advanced",
            "models_info": "/models/info"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "2.0.0",
        "cache_size": len(model_cache)
    }


@app.get("/models/info", tags=["Models"])
async def models_info():
    """Información sobre modelos disponibles"""
    try:
        import xgboost
        import lightgbm
        has_advanced = True
        versions = {
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__
        }
    except ImportError:
        has_advanced = False
        versions = {}
    
    return {
        "basic_models": ["LogisticRegression", "RandomForest"],
        "advanced_models": ["XGBoost", "LightGBM", "GradientBoosting", "Ridge", "ElasticNet"] if has_advanced else [],
        "ensemble": has_advanced,
        "versions": versions,
        "cached_models": len(model_cache)
    }


@app.post("/predict", 
          response_model=PredictionResponse,
          tags=["Predictions"],
          summary="Predicción climática básica")
async def predict_basic(request: PredictionRequest):
    """
    Endpoint básico (compatible con versión anterior)
    Usa solo ML básico
    """
    start_time = time.time()
    
    try:
        # Inicializar analizador básico
        analyzer = ClimateAnalyzer()
        
        # Descargar datos
        analyzer.download_data(
            request.latitude, 
            request.longitude, 
            request.start_date, 
            request.end_date
        )
        
        # Extraer ventana
        window_df = analyzer.get_historical_window(
            request.target_date, 
            request.window_days
        )
        
        # Calcular probabilidades
        probs = analyzer.calculate_probabilities(
            window_df,
            rain_threshold=request.rain_threshold,
            hot_threshold=request.hot_threshold,
            cold_threshold=request.cold_threshold
        )
        
        # ML básico
        ml_results = None
        ml_prediction_data = None
        ml_metrics_data = None
        
        if request.use_ml:
            ml_results = analyzer.train_ml_models(window_df, request.rain_threshold)
            
            if ml_results:
                prediction = analyzer.predict_for_date(
                    request.target_date, 
                    window_df, 
                    ml_results
                )
                
                ml_prediction_data = MLPrediction(
                    method=prediction['method'],
                    logistic_regression_prob=prediction.get('logreg_prob'),
                    random_forest_prob=prediction.get('rf_prob'),
                    ensemble_prob=prediction.get('ensemble_prob', 0),
                    ensemble_prob_percent=prediction.get('ensemble_prob', 0) * 100,
                    will_rain=prediction.get('ensemble_prob', 0) > 0.5,
                    confidence=calculate_confidence(prediction.get('ensemble_prob', 0))
                )
                
                metrics = ml_results['metrics']
                ml_metrics_data = MLModelMetrics(
                    logistic_regression_accuracy=metrics['logreg_acc'],
                    logistic_regression_auc=metrics.get('logreg_auc', 0.0),
                    random_forest_accuracy=metrics['rf_acc'],
                    random_forest_auc=metrics.get('rf_auc', 0.0)
                )
        
        # Construir response
        response = PredictionResponse(
            target_date=request.target_date,
            location=LocationInfo(
                latitude=request.latitude,
                longitude=request.longitude,
                coordinates_formatted=format_coordinates(request.latitude, request.longitude)
            ),
            historical_data=HistoricalDataStats(
                n_samples=probs['n_samples'],
                n_years=probs['n_years'],
                date_range_start=probs['date_range'][0],
                date_range_end=probs['date_range'][1],
                window_days=request.window_days
            ),
            precipitation=PrecipitationStats(
                rain_probability=probs['rain_probability'],
                rain_probability_percent=probs['rain_probability'] * 100,
                avg_precip_mm=probs['avg_precip'],
                max_precip_mm=probs['max_precip'],
                std_precip_mm=probs['precip_std']
            ),
            temperature=TemperatureStats(
                avg_temp_celsius=probs['avg_temp'],
                std_temp_celsius=probs['temp_std'],
                min_temp_celsius=probs['temp_range'][0],
                max_temp_celsius=probs['temp_range'][1],
                hot_day_probability=probs['hot_day_probability'],
                cold_day_probability=probs['cold_day_probability']
            ),
            humidity=HumidityStats(
                avg_humidity_percent=probs.get('avg_humidity', 0),
                high_humidity_probability=probs.get('high_humidity_prob', 0)
            ) if 'avg_humidity' in probs else None,
            wind=WindStats(
                avg_wind_speed_ms=probs.get('avg_wind', 0),
                strong_wind_probability=probs.get('strong_wind_prob', 0)
            ) if 'avg_wind' in probs else None,
            ml_prediction=ml_prediction_data,
            ml_model_metrics=ml_metrics_data,
            processing_time_seconds=round(time.time() - start_time, 2),
            data_source="NASA POWER",
            api_version="2.0.0"
        )
        
        return response
        
    except Exception as e:
        error = ErrorResponse(
            error=e.__class__.__name__,
            message=str(e),
            details={"traceback": traceback.format_exc()},
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        raise HTTPException(status_code=500, detail=error.model_dump())


@app.post("/predict/advanced",
          tags=["Predictions"],
          summary="Predicción con ML avanzado (XGBoost, LightGBM, Ensemble)")
async def predict_advanced(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Endpoint avanzado con ensemble de modelos ML
    Incluye: XGBoost, LightGBM, Gradient Boosting, Ridge, ElasticNet
    """
    start_time = time.time()
    
    try:
        # Verificar si tenemos modelo en cache
        cache_key = get_cache_key(
            request.latitude, 
            request.longitude,
            request.start_date,
            request.end_date
        )
        
        # Inicializar analizadores
        basic_analyzer = ClimateAnalyzer()
        
        # Descargar datos
        print(f"[API] Descargando datos para ({request.latitude}, {request.longitude})")
        basic_analyzer.download_data(
            request.latitude,
            request.longitude,
            request.start_date,
            request.end_date
        )
        
        # Extraer ventana
        window_df = basic_analyzer.get_historical_window(
            request.target_date,
            request.window_days
        )
        
        # Estadísticas básicas
        probs = basic_analyzer.calculate_probabilities(
            window_df,
            rain_threshold=request.rain_threshold,
            hot_threshold=request.hot_threshold,
            cold_threshold=request.cold_threshold
        )
        
        # ML básico
        ml_basic = basic_analyzer.train_ml_models(window_df, request.rain_threshold)
        basic_pred = None
        if ml_basic:
            basic_pred = basic_analyzer.predict_for_date(
                request.target_date,
                window_df,
                ml_basic
            )
        
        # ML avanzado
        advanced_predictor = AdvancedClimatePredictor()
        
        print("[API] Entrenando modelos avanzados...")
        advanced_results = integrate_with_analyzer(
            basic_analyzer,
            advanced_predictor,
            window_df,
            request.target_date,
            request.rain_threshold
        )
        
        # Guardar modelo en cache (background)
        if 'error' not in advanced_results:
            model_cache[cache_key] = {
                'predictor': advanced_predictor,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # Construir response extendida
        response_data = {
            "target_date": request.target_date,
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude,
                "coordinates_formatted": format_coordinates(request.latitude, request.longitude)
            },
            "historical_data": {
                "n_samples": probs['n_samples'],
                "n_years": probs['n_years'],
                "date_range_start": probs['date_range'][0],
                "date_range_end": probs['date_range'][1],
                "window_days": request.window_days
            },
            "statistics": {
                "precipitation": {
                    "probability": probs['rain_probability'],
                    "probability_percent": probs['rain_probability'] * 100,
                    "avg_mm": probs['avg_precip'],
                    "max_mm": probs['max_precip'],
                    "std_mm": probs['precip_std']
                },
                "temperature": {
                    "avg_celsius": probs['avg_temp'],
                    "std_celsius": probs['temp_std'],
                    "min_celsius": probs['temp_range'][0],
                    "max_celsius": probs['temp_range'][1],
                    "hot_day_prob": probs['hot_day_probability'],
                    "cold_day_prob": probs['cold_day_probability']
                }
            },
            "ml_basic": {
                "prediction": basic_pred if basic_pred else {},
                "metrics": ml_basic['metrics'] if ml_basic else {}
            },
            "ml_advanced": advanced_results,
            "final_recommendation": self._generate_consensus(
                probs['rain_probability'],
                basic_pred.get('ensemble_prob', 0) if basic_pred else 0,
                advanced_results.get('advanced_ml', {}).get('precipitation_mm', 0) / 10
            ),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "api_version": "2.0.0",
            "cache_hit": False
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        error = ErrorResponse(
            error=e.__class__.__name__,
            message=str(e),
            details={"traceback": traceback.format_exc()},
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        raise HTTPException(status_code=500, detail=error.model_dump())


def _generate_consensus(stat_prob: float, basic_prob: float, advanced_prob: float) -> Dict:
    """Genera recomendación final por consenso"""
    probabilities = [stat_prob, basic_prob, advanced_prob]
    weights = [0.3, 0.3, 0.4]  # Mayor peso al ML avanzado
    
    weighted_prob = sum(p * w for p, w in zip(probabilities, weights))
    
    votes_rain = sum(1 for p in probabilities if p > 0.5)
    consensus = votes_rain / len(probabilities)
    
    return {
        "will_rain": weighted_prob > 0.5,
        "probability": weighted_prob,
        "consensus": consensus,
        "confidence": "high" if consensus >= 0.8 or consensus <= 0.2 else "medium",
        "individual_predictions": {
            "statistical": stat_prob,
            "ml_basic": basic_prob,
            "ml_advanced": advanced_prob
        }
    }


@app.post("/predict/batch",
          tags=["Predictions"],
          summary="Predicción para múltiples fechas")
async def predict_batch(
    latitude: float,
    longitude: float,
    target_dates: List[str],
    start_date: str = "1990-01-01",
    end_date: str = "2024-12-31"
):
    """
    Predice para múltiples fechas en una sola llamada
    Optimizado para comparar fechas alternativas
    """
    try:
        results = []
        
        # Descargar datos una sola vez
        analyzer = ClimateAnalyzer()
        analyzer.download_data(latitude, longitude, start_date, end_date)
        
        for target_date in target_dates:
            try:
                window_df = analyzer.get_historical_window(target_date, window_days=7)
                probs = analyzer.calculate_probabilities(window_df, rain_threshold=0.5)
                
                results.append({
                    "target_date": target_date,
                    "rain_probability": probs['rain_probability'],
                    "avg_temp": probs['avg_temp'],
                    "success": True
                })
            except Exception as e:
                results.append({
                    "target_date": target_date,
                    "error": str(e),
                    "success": False
                })
        
        # Ordenar por probabilidad de lluvia (menor primero)
        results_success = [r for r in results if r['success']]
        results_success.sort(key=lambda x: x['rain_probability'])
        
        return {
            "location": {"latitude": latitude, "longitude": longitude},
            "results": results,
            "recommendation": results_success[0] if results_success else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/clear", tags=["Cache"])
async def clear_cache():
    """Limpia el cache de modelos"""
    model_cache.clear()
    return {"message": "Cache limpiado", "timestamp": datetime.utcnow().isoformat()}


# ==================== EJECUTAR SERVIDOR ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("AtmosAtlas API v2.0.0 - ML Avanzado")
    print("NASA Space Apps Challenge 2025")
    print("="*70)
    print("\nCaracterísticas:")
    print("  - ML Básico: Logistic Regression + Random Forest")
    print("  - ML Avanzado: XGBoost + LightGBM + Ensemble")
    print("  - Feature Engineering: 50+ features")
    print("  - Cache de modelos para optimización")
    print("\nEndpoints principales:")
    print("  - POST /predict - Predicción básica")
    print("  - POST /predict/advanced - Predicción avanzada")
    print("  - POST /predict/batch - Múltiples fechas")
    print(f"\nDocumentación: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(
        "main_api_actualizado:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )