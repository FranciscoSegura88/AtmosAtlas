# main.py

from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import logging

# Importamos la función principal de nuestro analizador
from analizador import run_full_analysis

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializamos la aplicación FastAPI
app = FastAPI(
    title="AtmosAtlas API",
    description="Una API para obtener análisis de probabilidad climática basados en datos históricos de NASA POWER.",
    version="1.0.0",
)

@app.get("/analyze", summary="Realiza un análisis climático completo para una ubicación y fecha")
async def get_climate_analysis(
    lat: float = Query(..., description="Latitud del lugar. Ejemplo: 20.67"),
    lon: float = Query(..., description="Longitud del lugar. Ejemplo: -103.35"),
    target_date: str = Query(..., description="Fecha futura a analizar en formato YYYY-MM-DD. Ejemplo: 2025-08-15"),
    start: str = Query("1990-01-01", description="Fecha de inicio para los datos históricos."),
    end: str = Query("2024-12-31", description="Fecha de fin para los datos históricos."),
    window_days: int = Query(7, description="Ventana de días (±) alrededor de la fecha objetivo para el análisis."),
    rain_threshold: float = Query(0.5, description="Umbral en mm para considerar un día como lluvioso."),
    hot_threshold: float = Query(30.0, description="Umbral en °C para considerar un día como caluroso."),
    cold_threshold: float = Query(10.0, description="Umbral en °C para considerar un día como frío."),
    use_ml: bool = Query(True, description="Activar o desactivar las predicciones de Machine Learning.")
):
    """
    Este endpoint ejecuta un análisis climático completo.

    - **Descarga** datos históricos de NASA POWER para el rango de fechas y ubicación especificados.
    - **Calcula** estadísticas históricas (probabilidad de lluvia, rangos de temperatura, etc.) para una ventana temporal alrededor de la `target_date`.
    - **Entrena (opcionalmente)** un ensamble de modelos de Machine Learning para refinar las predicciones.
    - **Devuelve** un objeto JSON con todos los resultados.
    """
    try:
        logging.info(f"Petición de análisis recibida para {target_date} en ({lat}, {lon})")

        # Llamamos a nuestra lógica de análisis, que ahora está encapsulada en una sola función
        analysis_results = run_full_analysis(
            lat=lat,
            lon=lon,
            start=start,
            end=end,
            target_date=target_date,
            window_days=window_days,
            rain_threshold=rain_threshold,
            hot_threshold=hot_threshold,
            cold_threshold=cold_threshold,
            use_ml=use_ml
        )

        if "error" in analysis_results:
            raise HTTPException(status_code=400, detail=analysis_results["error"])

        return analysis_results

    except ValueError as e:
        logging.error(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logging.error(f"Error de archivo: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Error inesperado en el servidor: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno en el servidor: {e}")

@app.get("/", summary="Endpoint de bienvenida")
def read_root():
    return {"message": "Bienvenido a la API de AtmosAtlas. Usa el endpoint /docs para ver la documentación interactiva."}
