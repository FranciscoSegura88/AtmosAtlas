🌦️ AtmosAtlas
AtmosAtlas es una herramienta de análisis climático que utiliza datos históricos de la NASA para calcular probabilidades de condiciones meteorológicas en fechas futuras. No predice el tiempo, analiza patrones del pasado para tomar decisiones más informadas.
🎯 ¿Qué hace?
Dado un lugar y una fecha, AtmosAtlas te dice:

¿Cuál es la probabilidad de que llueva?
¿Qué tan caluroso o frío suele ser ese día?
Estadísticas de humedad, viento y presión atmosférica
Predicciones con Machine Learning basadas en 30+ años de datos de la NASA


📋 Requisitos Previos

Python 3.10 o superior (recomendado 3.11+)
Conexión a Internet (para descargar datos de NASA POWER)
Windows, macOS o Linux


🚀 Instalación (Paso a Paso)
1️⃣ Descargar el Proyecto
Opción A: Clonar con Git
bashgit clone https://github.com/tuusuario/franciscosegura88-atmosatlas.git
cd franciscosegura88-atmosatlas
Opción B: Descargar ZIP

Haz clic en el botón verde "Code" → "Download ZIP"
Extrae el archivo
Abre una terminal/CMD en esa carpeta


2️⃣ Crear Entorno Virtual
En Windows (PowerShell):
powershellpython -m venv .venv
.venv\Scripts\Activate.ps1
En Windows (CMD tradicional):
cmdpython -m venv .venv
.venv\Scripts\activate.bat
En macOS/Linux:
bashpython3 -m venv .venv
source .venv/bin/activate
✅ Verificar que está activo:
Tu prompt debería verse así:
(.venv) C:\Users\TuNombre\AtmosAtlas>
Nota para PowerShell: Si obtienes un error de permisos, ejecuta primero:
powershellSet-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

3️⃣ Instalar Dependencias
bashpip install -r requirements.txt
Esto instalará:

requests - Para descargar datos de la NASA
pandas - Procesamiento de datos
numpy - Cálculos numéricos
scikit-learn - Modelos de Machine Learning


📖 Cómo Usar
Comando Básico
bashpython climate_probability_analyzer.py \
  --lat 20.67 \
  --lon -103.35 \
  --start 1990-01-01 \
  --end 2024-12-31 \
  --target_date 2025-08-15
Parámetros Explicados
ParámetroDescripciónEjemplo--latLatitud del lugar20.67 (Guadalajara)--lonLongitud del lugar-103.35--startFecha inicio datos históricos1990-01-01--endFecha fin datos históricos2024-12-31--target_dateFecha a analizar2025-08-15
Opciones Avanzadas
bash# Cambiar ventana temporal (±10 días en lugar de ±7)
--window_days 10

# Cambiar umbral de lluvia (default: 0.5mm)
--rain_threshold 2.0

# Cambiar umbral de calor (default: 30°C)
--hot_threshold 35

# Guardar datos históricos usados
--save_window historico.csv

# Desactivar Machine Learning (solo estadísticas)
--no_ml

📝 Ejemplos Completos
Ejemplo 1: Boda en Guadalajara
bashpython climate_probability_analyzer.py \
  --lat 20.676667 \
  --lon -103.347222 \
  --start 1990-01-01 \
  --end 2024-12-31 \
  --target_date 2025-12-15
Resultado esperado:
============================================================
ANALISIS CLIMATICO PARA 2025-12-15
============================================================

Datos Historicos:
   * Muestras: 525 dias de 35 años diferentes
   * Rango: 1990-12-08 a 2024-12-22

Precipitacion:
   * Probabilidad de lluvia: 2.3%
   * Promedio cuando llueve: 3.2 mm
   * Maximo historico: 18.5 mm

Temperatura:
   * Promedio: 17.8°C (±2.1°C)
   * Rango: 11.2°C a 23.4°C
   * Probabilidad dia caluroso (>30°C): 0.0%
   * Probabilidad dia frio (<10°C): 0.2%

Prediccion Ensamble (ML):
   * Probabilidad de lluvia: 3.1%
   * Decision: NO LLOVERA
============================================================

Ejemplo 2: Festival de Música en CDMX
bashpython climate_probability_analyzer.py \
  --lat 19.432608 \
  --lon -99.133209 \
  --start 1985-01-01 \
  --end 2024-12-31 \
  --target_date 2025-06-20 \
  --rain_threshold 5.0 \
  --save_window festival_historico.csv

Ejemplo 3: Usar CSV Previamente Descargado
Paso 1: Descargar datos una vez
bashpython power_timeseries.py \
  --lat 20.67 \
  --lon -103.35 \
  --start 1990-01-01 \
  --end 2024-12-31
Esto genera: power_timeseries_20p67_m103p35_19900101_20241231.csv
Paso 2: Analizar múltiples fechas sin re-descargar
bashpython climate_probability_analyzer.py \
  --csv power_timeseries_20p67_m103p35_19900101_20241231.csv \
  --target_date 2025-08-15

python climate_probability_analyzer.py \
  --csv power_timeseries_20p67_m103p35_19900101_20241231.csv \
  --target_date 2025-12-25

🗺️ Cómo Obtener Coordenadas
Método 1: Google Maps

Abre Google Maps
Haz clic derecho en el lugar deseado
Selecciona "¿Qué hay aquí?"
Las coordenadas aparecen abajo (ej: 20.676667, -103.347222)

Método 2: Sitio Web
Usa latlong.net - escribe el nombre del lugar y obtendrás las coordenadas.

🔧 Solución de Problemas
Error: "No module named 'sklearn'"
bashpip install scikit-learn
Error: "Can only use .dt accessor with datetimelike values"
Asegúrate de usar la versión más reciente de climate_probability_analyzer.py.
Error: "No hay datos históricos antes de XXXX"
Tu --target_date debe ser posterior a --end. Ejemplo:

✅ Correcto: --end 2024-12-31 --target_date 2025-08-15
❌ Incorrecto: --end 2024-12-31 --target_date 2023-08-15

La descarga de NASA POWER falla

Verifica tu conexión a Internet
Reduce el rango de años (ej: últimos 20 años en lugar de 40)
Intenta nuevamente - la API de NASA puede tener caídas temporales


🧠 Cómo Funciona
Ventanas Temporales Inteligentes
En lugar de solo mirar el día exacto, AtmosAtlas analiza una ventana de ±7 días alrededor de tu fecha objetivo en todos los años históricos.
Ejemplo: Para agosto 15, analiza agosto 8-22 de 1990-2024.
Machine Learning
Entrena dos modelos:

Logistic Regression - Modelo estadístico clásico
Random Forest - Modelo de ensamble de árboles de decisión

Ambos predicciones se promedian para mayor robustez.

📊 Fuente de Datos
NASA POWER (Prediction Of Worldwide Energy Resources)

URL: https://power.larc.nasa.gov
Cobertura: Global, desde 1981 hasta presente
Resolución: 0.5° x 0.5° (~50km)
Variables:

PRECTOTCORR - Precipitación (mm/día)
T2M - Temperatura a 2m (°C)
RH2M - Humedad relativa (%)
WS10M - Velocidad del viento (m/s)
ALLSKY_SFC_SW_DWN - Radiación solar (W/m²)
PS - Presión atmosférica (kPa)




🤝 Contribuir
¿Encontraste un bug? ¿Tienes una idea?

Abre un Issue
Crea un Pull Request


📜 Licencia
MIT License - Úsalo libremente para tu proyecto, comercial o no.

👨‍💻 Créditos
Desarrollado para el NASA Space Apps Challenge 2025
Reto: "Will It Rain On My Parade?"
Equipo:

@xcesarg
@Raledro
jazmin.diaz4616@alumnos.udg.mx
eduardo.martinez5436@alumnos.udg.mx
maria.gomez6796@alumnos.udg.mx


🆘 Soporte
¿Necesitas ayuda? Abre un Issue o contacta al equipo.

"No es un pronóstico del tiempo, es un pronóstico del pasado." 🌍📈