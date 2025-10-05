# train_predictor.py

import logging
import os
# Asegúrate de que los otros archivos necesarios estén en la misma carpeta
from analizador import power_json_to_dataframe, fetch_power_point
from advanced_ml_predictor import AdvancedClimatePredictor

# --- CONFIGURACIÓN ---
LAT = 20.67
LON = -103.35
START_DATE = "1990-01-01"
END_DATE = "2024-12-31"
RAIN_THRESHOLD = 0.5
MODEL_PREFIX = "atmosatlas_predictor_gdl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Script para entrenar y guardar el predictor avanzado.
    Se ejecuta una sola vez para preparar el modelo para producción.
    """
    logging.info("--- INICIANDO ENTRENAMIENTO OFFLINE DEL PREDICTOR ---")

    # 1. Descargar el dataset histórico completo
    logging.info(f"Descargando datos históricos desde {START_DATE} hasta {END_DATE}...")
    raw_data = fetch_power_point(LAT, LON, START_DATE, END_DATE)
    df_hist = power_json_to_dataframe(raw_data)
    logging.info(f"Datos descargados: {len(df_hist)} registros.")

    # 2. Inicializar el predictor y crear features
    predictor = AdvancedClimatePredictor()
    logging.info("Creando features de Machine Learning...")
    feats = predictor.create_features(df_hist, target_col="precip")
    feats["rained"] = (feats["precip"] > RAIN_THRESHOLD).astype(int)

    feature_cols = [c for c in feats.columns if c not in ["date", "precip", "rained"]]

    feats_clean_reg = feats.dropna(subset=["precip"] + feature_cols)
    feats_clean_cls = feats.dropna(subset=["rained"] + feature_cols)

    # 3. Preparar datos para entrenamiento
    logging.info("Dividiendo datos en conjuntos de entrenamiento y validación...")
    Xtr_reg, ytr_reg, Xva_reg, yva_reg = predictor._time_split(feats_clean_reg, "precip", feature_cols)
    Xtr_cls, ytr_cls, Xva_cls, yva_cls = predictor._time_split(feats_clean_cls, "rained", feature_cols)

    predictor.feature_names = feature_cols

    # 4. Entrenar los modelos
    logging.info("Entrenando modelos de regresión (esto puede tardar varios minutos)...")
    predictor.train_regression(Xtr_reg, ytr_reg, Xva_reg, yva_reg)

    logging.info("Entrenando modelos de clasificación (esto puede tardar varios minutos)...")
    predictor.train_classification(Xtr_cls, ytr_cls, Xva_cls, yva_cls)

    # 5. Guardar el predictor entrenado
    logging.info("Guardando predictor, scalers y metadatos...")
    saved_path = predictor.save(prefix=MODEL_PREFIX)
    logging.info(f"✅ Predictor entrenado y guardado exitosamente en la carpeta: {saved_path}")
    logging.info("--- ENTRENAMIENTO OFFLINE COMPLETADO ---")

if __name__ == "__main__":
    main()
