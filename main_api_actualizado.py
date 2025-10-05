# main_api_actualizado.py
# ------------------------------------------------------------
# API v2 robusta con ML avanzado + consenso calibrado + EXPLICABILIDAD.
# - /predict           -> respuesta compatible (PredictionResponse)
# - /predict/advanced  -> respuesta extendida con multi-fuente, detalle ML y explicabilidad
# Arreglos:
#   * Clave correcta 'ensemble_prob' (no 'prob').
#   * Consenso sin 'self' (función libre).
#   * Agregado bloque 'explainability'.
# ------------------------------------------------------------
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import traceback
import time
from typing import Dict

from models import (
    PredictionRequest, PredictionResponse, ErrorResponse,
    LocationInfo, HistoricalDataStats, PrecipitationStats,
    TemperatureStats, HumidityStats, WindStats, MLPrediction, MLModelMetrics
)
from analizador import ClimateAnalyzer  # v1 baseline
from analizador_integrado import IntegratedClimateAnalyzer  # v2 robusto



app = FastAPI(
    title="AtmosAtlas API - Robust v2",
    description="Análisis climático histórico con ML avanzado (NASA POWER + IMERG opcional) y explicabilidad",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------- UTILES ----------------

def fmt_coords(lat: float, lon: float) -> str:
    return f"{abs(lat):.2f}°{'N' if lat>=0 else 'S'}, {abs(lon):.2f}°{'E' if lon>=0 else 'W'}"

def conf_bucket(p: float) -> str:
    if p >= 0.7 or p <= 0.3: return "high"
    if p >= 0.55 or p <= 0.45: return "medium"
    return "low"

def _generate_consensus(stat_prob: float, basic_prob: float, advanced_prob: float) -> Dict:
    probs = [stat_prob, basic_prob, advanced_prob]
    weights = [0.25, 0.25, 0.50]
    wp = sum(p*w for p, w in zip(probs, weights))
    votes = sum(1 for p in probs if p > 0.5)
    cons = votes / len(probs)
    return {
        "will_rain": bool(wp > 0.5),
        "probability": float(wp),
        "consensus": float(cons),
        "confidence": "high" if cons >= 0.8 or cons <= 0.2 else "medium",
        "individual_predictions": {"statistical": probs[0], "ml_basic": probs[1], "ml_advanced": probs[2]},
    }

# ---------------- ENDPOINTS ----------------

@app.get("/", tags=["Health"])
async def root():
    return {"message": "AtmosAtlas Robust API v2.2.0", "status": "online", "docs": "/docs"}

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z", "version": "2.2.0"}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_basic(request: PredictionRequest):
    start = time.time()
    try:
        analyzer = ClimateAnalyzer()
        analyzer.download_data(request.latitude, request.longitude, request.start_date, request.end_date)
        win = analyzer.get_historical_window(request.target_date, request.window_days)
        probs = analyzer.calculate_probabilities(win,
                                                 rain_threshold=request.rain_threshold,
                                                 hot_threshold=request.hot_threshold,
                                                 cold_threshold=request.cold_threshold)
        ml_pred = None
        ml_metrics = None
        if request.use_ml:
            ml = analyzer.train_ml_models(win, request.rain_threshold)
            if ml:
                pred = analyzer.predict_for_date(request.target_date, win, ml)
                # 🔧 clave correcta: 'ensemble_prob'
                p = float(pred.get("ensemble_prob", 0.0))
                ml_pred = MLPrediction(
                    method=pred["method"],
                    logistic_regression_prob=pred.get("logreg_prob"),
                    random_forest_prob=pred.get("rf_prob"),
                    ensemble_prob=p,
                    ensemble_prob_percent=p*100.0,
                    will_rain=bool(p > 0.5),
                    confidence=conf_bucket(p)
                )
                mm = ml["metrics"]
                ml_metrics = MLModelMetrics(
                    logistic_regression_accuracy=mm["logreg_acc"],
                    logistic_regression_auc=mm.get("logreg_auc") or 0.0,
                    random_forest_accuracy=mm["rf_acc"],
                    random_forest_auc=mm.get("rf_auc") or 0.0
                )

        resp = PredictionResponse(
            target_date=request.target_date,
            location=LocationInfo(latitude=request.latitude, longitude=request.longitude,
                                  coordinates_formatted=fmt_coords(request.latitude, request.longitude)),
            historical_data=HistoricalDataStats(
                n_samples=probs["n_samples"], n_years=probs["n_years"],
                date_range_start=probs["date_range"][0], date_range_end=probs["date_range"][1],
                window_days=request.window_days
            ),
            precipitation=PrecipitationStats(
                rain_probability=probs["rain_probability"],
                rain_probability_percent=probs["rain_probability"]*100.0,
                avg_precip_mm=probs["avg_precip"], max_precip_mm=probs["max_precip"],
                std_precip_mm=probs["precip_std"]
            ),
            temperature=TemperatureStats(
                avg_temp_celsius=probs["avg_temp"], std_temp_celsius=probs["temp_std"],
                min_temp_celsius=probs["temp_range"][0], max_temp_celsius=probs["temp_range"][1],
                hot_day_probability=probs["hot_day_probability"], cold_day_probability=probs["cold_day_probability"]
            ),
            humidity=HumidityStats(
                avg_humidity_percent=probs.get("avg_humidity", 0.0),
                high_humidity_probability=probs.get("high_humidity_prob", 0.0)
            ) if "avg_humidity" in probs else None,
            wind=WindStats(
                avg_wind_speed_ms=probs.get("avg_wind", 0.0),
                strong_wind_probability=probs.get("strong_wind_prob", 0.0)
            ) if "avg_wind" in probs else None,
            ml_prediction=ml_pred,
            ml_model_metrics=ml_metrics,
            processing_time_seconds=round(time.time() - start, 2),
            data_source="NASA POWER",
            api_version="2.2.0"
        )
        return resp
    except Exception as e:
        err = ErrorResponse(error=e.__class__.__name__, message=str(e),
                            details={"traceback": traceback.format_exc()},
                            timestamp=datetime.utcnow().isoformat() + "Z")
        raise HTTPException(status_code=500, detail=err.model_dump())


@app.post("/predict/advanced", tags=["Predictions"])
async def predict_advanced(request: PredictionRequest,
                           use_multisource: bool = False,
                           explain_recent_days: int = 14):
    """
    Predicción avanzada con:
      - Estadística histórica
      - ML básico (LR + RF)
      - ML avanzado (regresión mm + clasificación, calibración)
      - Consenso final + EXPLICABILIDAD
    Si `use_multisource=true`, intenta fusionar POWER+IMERG(+MODIS).
    """
    start = time.time()
    try:
        analyzer = IntegratedClimateAnalyzer()
        res = analyzer.analyze(
            lat=request.latitude, lon=request.longitude,
            target_date=request.target_date,
            start_date=request.start_date, end_date=request.end_date,
            window_days=request.window_days, rain_threshold=request.rain_threshold,
            use_multisource=use_multisource,
            explain_recent_days=explain_recent_days
        )

        stat = res["statistics"]
        basic_prob = res.get("ml_basic", {}).get("prediction", {}).get("ensemble_prob", 0.0)
        adv_prob = res.get("ml_advanced", {}).get("advanced_ml", {}).get("rain_probability_fused", 0.0)
        final = _generate_consensus(stat.get("rain_probability", 0.0), float(basic_prob), float(adv_prob))

        content = {
            "target_date": request.target_date,
            "location": {
                "latitude": request.latitude, "longitude": request.longitude,
                "coordinates_formatted": fmt_coords(request.latitude, request.longitude)
            },
            "historical_data": {
                "n_samples": stat["n_samples"], "n_years": stat["n_years"],
                "date_range_start": stat["date_range"][0], "date_range_end": stat["date_range"][1],
                "window_days": request.window_days
            },
            "statistics": {
                "precipitation": {
                    "probability": stat.get("rain_probability", 0.0),
                    "avg_mm": stat.get("avg_precip", None),
                    "max_mm": stat.get("max_precip", None),
                    "std_mm": stat.get("precip_std", None)
                },
                "temperature": {
                    "avg_celsius": stat.get("avg_temp", None),
                    "std_celsius": stat.get("temp_std", None),
                    "min_celsius": stat.get("temp_range", (None, None))[0],
                    "max_celsius": stat.get("temp_range", (None, None))[1],
                    "hot_day_prob": stat.get("hot_day_probability", None),
                    "cold_day_prob": stat.get("cold_day_probability", None),
                }
            },
            "ml_basic": res.get("ml_basic", {}),
            "ml_advanced": res.get("ml_advanced", {}),
            "final_recommendation": {**final, "reasoning": res.get("final_recommendation", {}).get("reasoning")},
            "explainability": res.get("explainability", {}),
            "processing_time_seconds": round(time.time() - start, 2),
            "api_version": "2.2.0",
            "data_sources": "NASA POWER (+IMERG/MODIS opcional)"
        }
        if "data_quality" in res:
            content["data_quality"] = res["data_quality"]
        return JSONResponse(content=content)
    except Exception as e:
        err = ErrorResponse(error=e.__class__.__name__, message=str(e),
                            details={"traceback": traceback.format_exc()},
                            timestamp=datetime.utcnow().isoformat() + "Z")
        raise HTTPException(status_code=500, detail=err.model_dump())


if __name__ == "__main__":
    import uvicorn
    print("AtmosAtlas Robust API v2.2.0")
    uvicorn.run("main_api_actualizado:app", host="0.0.0.0", port=8000, reload=True)
