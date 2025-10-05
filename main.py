"""
main.py

API REST con FastAPI para AtmosAtlas - Análisis Climático Histórico
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime
from typing import Optional
import traceback

# Importar modelos
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

# Importar analizador
from analizador import ClimateAnalyzer


# Inicializar FastAPI
app = FastAPI(
    title="AtmosAtlas API",
    description="API de análisis climático histórico usando datos de la NASA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS (permitir requests desde cualquier origen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios: ["https://tudominio.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """Endpoint raíz - Verificar que la API está funcionando"""
    return {
        "message": "AtmosAtlas API v1.0.0",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "saludo": "/saludo"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Error de validación"},
        500: {"model": ErrorResponse, "description": "Error del servidor"}
    },
    tags=["Predictions"],
    summary="Analizar probabilidades climáticas",
    description="""
    Analiza datos históricos de la NASA para calcular probabilidades climáticas
    en una ubicación y fecha específicas.
    
    **Ejemplo de uso:**
    ```json
    {
        "latitude": 20.676667,
        "longitude": -103.347222,
        "target_date": "2025-08-15",
        "start_date": "1990-01-01",
        "end_date": "2024-12-31",
        "window_days": 7,
        "rain_threshold": 0.5,
        "use_ml": true
    }
    ```
    """
)
async def predict_climate(request: PredictionRequest):
    """
    Endpoint principal: Análisis climático histórico
    """
    start_time = time.time()
    
    try:
        # Inicializar analizador
        analyzer = ClimateAnalyzer()
        
        # Descargar datos
        print(f"[API] Descargando datos para ({request.latitude}, {request.longitude})")
        analyzer.download_data(
            request.latitude, 
            request.longitude, 
            request.start_date, 
            request.end_date
        )
        
        # Extraer ventana histórica
        print(f"[API] Extrayendo ventana histórica para {request.target_date}")
        window_df = analyzer.get_historical_window(
            request.target_date, 
            request.window_days
        )
        
        # Calcular probabilidades
        print("[API] Calculando probabilidades")
        probs = analyzer.calculate_probabilities(
            window_df,
            rain_threshold=request.rain_threshold,
            hot_threshold=request.hot_threshold,
            cold_threshold=request.cold_threshold
        )
        
        # Entrenar modelos ML (si aplica)
        ml_results = None
        ml_prediction_data = None
        ml_metrics_data = None
        
        if request.use_ml:
            print("[API] Entrenando modelos ML")
            ml_results = analyzer.train_ml_models(window_df, request.rain_threshold)
            
            if ml_results:
                # Predicción ML
                prediction = analyzer.predict_for_date(
                    request.target_date, 
                    window_df, 
                    ml_results
                )
                
                ml_prediction_data = MLPrediction(
                    method=prediction['method'],
                    logistic_regression_prob=prediction.get('logreg_prob'),
                    random_forest_prob=prediction.get('rf_prob'),
                    ensemble_prob=prediction['prob'],
                    ensemble_prob_percent=prediction['prob'] * 100,
                    will_rain=prediction['prob'] > 0.5,
                    confidence=calculate_confidence(prediction['prob'])
                )
                
                # Métricas ML
                metrics = ml_results['metrics']
                ml_metrics_data = MLModelMetrics(
                    logistic_regression_accuracy=metrics['logreg_acc'],
                    logistic_regression_auc=metrics['logreg_auc'],
                    random_forest_accuracy=metrics['rf_acc'],
                    random_forest_auc=metrics['rf_auc']
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
                avg_humidity_percent=probs['avg_humidity'],
                high_humidity_probability=probs['high_humidity_prob']
            ) if 'avg_humidity' in probs else None,
            wind=WindStats(
                avg_wind_speed_ms=probs['avg_wind'],
                strong_wind_probability=probs['strong_wind_prob']
            ) if 'avg_wind' in probs else None,
            ml_prediction=ml_prediction_data,
            ml_model_metrics=ml_metrics_data,
            processing_time_seconds=round(time.time() - start_time, 2),
            data_source="NASA POWER",
            api_version="1.0.0"
        )
        
        print(f"[API] Análisis completado en {response.processing_time_seconds}s")
        return response
        
    except ValueError as e:
        # Errores de validación
        error = ErrorResponse(
            error="ValidationError",
            message=str(e),
            details={"request": request.model_dump()},
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        raise HTTPException(status_code=400, detail=error.model_dump())
    
    except Exception as e:
        # Otros errores
        error = ErrorResponse(
            error=e.__class__.__name__,
            message=str(e),
            details={"traceback": traceback.format_exc()},
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        raise HTTPException(status_code=500, detail=error.model_dump())


# ==================== ENDPOINT ADICIONAL: QUICK ANALYSIS ====================

@app.get(
    "/quick-predict",
    response_model=dict,
    tags=["Predictions"],
    summary="Análisis rápido con parámetros por URL",
    description="Versión simplificada del endpoint /predict usando query parameters"
)
async def quick_predict(
    lat: float,
    lon: float,
    date: str,
    window: int = 7
):
    """
    Análisis rápido con parámetros GET
    
    Ejemplo: /quick-predict?lat=20.67&lon=-103.35&date=2025-08-15&window=7
    """
    request = PredictionRequest(
        latitude=lat,
        longitude=lon,
        target_date=date,
        window_days=window
    )
    return await predict_climate(request)


# ==================== EJECUTAR SERVIDOR ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🌍 AtmosAtlas API v1.0.0")
    print("="*60)
    print("\nIniciando servidor...")
    print("📖 Documentación: http://localhost:8000/docs")
    print("🔄 ReDoc: http://localhost:8000/redoc")
    print("💚 Health Check: http://localhost:8000/health")
    print("\nPresiona CTRL+C para detener\n")
    print("="*60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )