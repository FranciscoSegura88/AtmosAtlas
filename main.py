# main.py (Versión final con Lifespan)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Optional
from pathlib import Path
import glob
import os
from datetime import datetime, timedelta

from analizador import power_json_to_dataframe, fetch_power_point
from advanced_ml_predictor import AdvancedClimatePredictor

# --- CONFIGURACIÓN ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_PREFIX = "atmosatlas_predictor_gdl"
predictor: Optional[AdvancedClimatePredictor] = None

# --- MANEJADOR DE CICLO DE VIDA (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Esta función se ejecuta al iniciar y apagar la API.
    """
    # --- CÓDIGO DE INICIO ---
    global predictor
    logging.info("Iniciando la API... Cargando predictor pre-entrenado.")

    model_dir = "models"
    list_of_model_folders = glob.glob(f"{model_dir}/{MODEL_PREFIX}_*")

    if not list_of_model_folders:
        logging.warning("!!! ADVERTENCIA: No se encontró ningún predictor entrenado. !!!")
        logging.warning(f"Ejecuta 'python train_predictor.py' primero. La API no podrá responder.")
    else:
        latest_model_folder = max(list_of_model_folders, key=os.path.getmtime)
        logging.info(f"Cargando modelo desde: {latest_model_folder}")
        try:
            predictor = AdvancedClimatePredictor()
            predictor.load(latest_model_folder)
            logging.info("✅ Predictor cargado exitosamente en memoria.")
        except Exception as e:
            logging.error(f"No se pudo cargar el modelo: {e}", exc_info=True)
            predictor = None

    yield  # La API está lista para recibir peticiones

    # --- CÓDIGO DE APAGADO (si fuera necesario) ---
    logging.info("Apagando la API...")


# --- INICIALIZACIÓN DE LA APP ---
# Le pasamos nuestra función de lifespan a FastAPI
app = FastAPI(
    title="AtmosAtlas API",
    description="Una API para obtener análisis de probabilidad climática.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las fuentes, para desarrollo. En producción, pon aquí la URL de tu front-end.
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todas las cabeceras
)

# --- ENDPOINT DE PREDICCIÓN ---
@app.get("/analyze")
async def get_climate_analysis(
    lat: float = Query(..., description="Latitud"),
    lon: float = Query(..., description="Longitud"),
    target_date: str = Query(..., description="Fecha a analizar (YYYY-MM-DD)")
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible: El modelo de ML no está cargado. Ejecuta 'train_predictor.py' para generarlo.")

    try:
        # Descargamos solo los últimos 90 días para generar las features
        end_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=92)).strftime('%Y-%m-%d')

        logging.info(f"Descargando datos recientes para features ({start_date} a {end_date})...")
        raw_data = fetch_power_point(lat, lon, start_date, end_date)
        df_recent = power_json_to_dataframe(raw_data)

        feats = predictor.create_features(df_recent)
        # Nos aseguramos de tomar la última fila con datos válidos
        last_valid_index = feats[predictor.feature_names].last_valid_index()
        last_row = feats.loc[[last_valid_index]][predictor.feature_names].copy()

        # Realizar predicciones rápidas
        logging.info("Realizando predicciones...")
        reg_pred = predictor.predict_regression(last_row)
        cls_pred = predictor.predict_classification(last_row)

        return {
            "prediction_for": target_date,
            "location": {"lat": lat, "lon": lon},
            "ml_precipitation_mm": reg_pred,
            "ml_rain_probability": cls_pred
        }
    except Exception as e:
        logging.error(f"Error durante la predicción: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
