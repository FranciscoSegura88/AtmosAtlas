AtmosAtlas — README (completo, paso a paso)

Proyecto mínimo para consultar variables climáticas (NASA POWER y colecciones CMR/GPM), extraer series temporales puntuales y estimar la probabilidad de lluvia para una fecha objetivo usando estadísticas y modelos sencillos.
Todo explicado para alguien sin experiencia previa: desde crear y activar el venv, instalar dependencias, hasta ejemplos de uso en Windows (cmd / PowerShell), Git Bash y Linux.

Contenido del repositorio (resumen)
franciscosegura88-atmosatlas/
├── README.md
├── requirements.txt
├── .dodsrc
├── .venv/                     # virtualenv (opcionalmente creado localmente)
├── power_timeseries.py        # descarga series diarias desde NASA POWER y guarda CSV
├── predict_rain_for_date_power.py  # predicción usando series POWER (batch fetch)
├── get_edl_token.py           # (opcional) generar token Earthdata y guardarlo en ~/.edl_token
├── cmr_granules_and_links.py  # buscar granules CMR y listar enlaces (GPM, MERRA2, etc.)
├── debug_read_gpm_point.py    # utilitario para depurar lectura de archivos GPM (opendap/descarga)
├── predict_rain_for_date.py   # predicción usando GPM/MERRA2 (requiere Earthdata token)
├── analizador.py              # analizador general: usa CSV o descarga y calcula probabilidades
└── power_timeseries_...csv    # ejemplo de CSV grande (si ya lo descargaste)

Requisitos (software)

Python 3.10 — 3.13 (recomendado 3.11+)

git (opcional)

Conexión a internet para descargar datos

(Opcional) cuenta Earthdata (para datasets protegidos como GPM IMERG)

El proyecto incluye un requirements.txt con las librerías necesarias:

Ejemplo de contenido requirements.txt

numpy
pandas
requests
scikit-learn
xarray
netCDF4
pydap

1 — Crear y activar el entorno virtual (venv)

Hazlo desde la carpeta del proyecto.

Windows (cmd.exe)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt


Si PowerShell bloquea la ejecución: ejecuta Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser como administrador o usa cmd.

Git Bash / Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


Verifica que el prompt muestre (.venv).

2 — Descargar series diarias desde NASA POWER (fácil y recomendado)

Script: power_timeseries.py
Genera un CSV con las variables solicitadas (precipitación, T2M, humedad, viento, radiación, presión superficial ps…).

Uso básico (una línea, recomendable)
python power_timeseries.py --lat 20.67 --lon -103.35 --start 1984-01-01 --end 2024-12-31


Esto:

Llamará a la API NASA POWER para el punto y rango pedidos.

Guardará un CSV con nombre informativo: power_timeseries_20p67_m103p35_19840101_20241231.csv (puntos convertidos para filename).

Imprimirá las primeras filas en pantalla.

Parámetros

--lat (float): latitud

--lon (float): longitud (negativa para oeste)

--start --end: rango en YYYY-MM-DD

--out (opcional): nombre de archivo de salida

--params (opcional): parámetros POWER (códigos) separados por comas

Qué columnas tiene el CSV

date (YYYY-MM-DD)

precip (PRECTOTCORR) — mm

t2m (T2M) — °C

rh2m (RH2M) — %

wind (WS10M) — m/s

rad (ALLSKY_SFC_SW_DWN) — W/m²

ps (PS) — kPa (si está en parámetros solicitados)

3 — Analizar CSV y obtener probabilidades/clasificación (analizador.py)

Script: analizador.py (si lo encuentras en tu carpeta lo usamos; es equivalente a climate_probability_analyzer.py en la doc)

Usos típicos
Desde CSV (recomendado si ya descargaste con power_timeseries.py):
python analizador.py --csv power_timeseries_20p67_m103p35_19840101_20241231.csv --target_date 2025-12-25

Descargar y analizar (sin CSV previo):
python analizador.py --lat 20.67 --lon -103.35 --start 1990-01-01 --end 2024-12-31 --target_date 2025-12-25


En ese caso el script usa la función fetch_power_point del power_timeseries.py.

Opciones relevantes

--csv <archivo.csv> — usar CSV existente (mutuamente excluyente con --lat)

--lat --lon --start --end — descargar datos antes de analizar (requiere --lat)

--target_date (requerido): fecha objetivo YYYY-MM-DD

--window_days (por defecto 7): ventana ± días alrededor del día objetivo (en años previos)

--rain_threshold (por defecto 0.5 mm): umbral para contar como "llovió"

--no_ml — no entrenar modelos ML; devuelve solo estadísticas

--save_window <archivo.csv> — guarda la ventana histórica usada

Qué hace analizador.py

Carga datos (CSV o descarga).

Extrae una ventana histórica: mismo día/mes ± window_days en años anteriores.

Calcula estadísticas: probabilidad de lluvia (frecuencia), medias, máximos, índice de temperatura, humedad, viento.

(Opcional) Entrena modelos ML (LogisticRegression, RandomForest) si hay suficientes muestras y variedad en etiqueta.

Presenta resultados legibles y guarda archivos opcionales.

4 — Predicción específica: predict_rain_for_date_power.py

Este script toma una --date objetivo (p. ej. 2036-08-11), construye un historial llamando a POWER en bloque, extrae el mismo día/mes en años anteriores y entrena modelos si hay suficientes datos.
Uso:

python predict_rain_for_date_power.py --date 2036-08-11 --lat 20.67 --lon -103.35 --years_back 15


Salida típica:

history_used_power.csv guardado.

Predicción: probabilidad por logistic regression, random forest y promedio.

5 — Usar GPM / CMR (datasets protegidos, requiere Earthdata token)

Si quieres leer GPM IMERG y otros granules desde GES DISC (a menudo protegidos), necesitas generar y guardar un token Earthdata en ~/.edl_token.

Generar token (script incluido)
python get_edl_token.py


Te pedirá username y password Earthdata; guardará token en ~/.edl_token.

Alternativa: seguir instrucciones del portal Earthdata para crear un Personal Access Token. NO compartas tu token.

Consultas CMR y descarga de granules

cmr_granules_and_links.py permite listar granules y enlaces para un short_name.
Ejemplo (Windows cmd / PowerShell — bbox separada por espacios):

python cmr_granules_and_links.py --short_name GPM_3IMERGDF.07 --start 2010-08-10 --end 2010-08-12 --bbox -103.6 20.3 -103.1 20.8 --page_size 5


En Git Bash / Linux puedes usar quotes:

python cmr_granules_and_links.py --short_name GPM_3IMERGDF.07 --start 2010-08-10 --end 2010-08-12 --bbox "-103.6 20.3 -103.1 20.8" --page_size 5


debug_read_gpm_point.py permite probar lectura OPeNDAP / descarga con token y extraer series puntuales (útil para depurar acceso y variables).

predict_rain_for_date.py usa CMR para buscar GPM/MERRA2, descarga con token cuando sea necesario y entrena modelos. Ejemplo:

python predict_rain_for_date.py --date 2024-08-11 --lat 20.67 --lon -103.35 --years_back 10 --bbox -103.6 20.3 -103.1 20.8 --precip_short_name GPM_3IMERGDF.07


Nota: GESDISC/CMR a veces requiere cookies/.netrc o .dodsrc en Windows con ubicación de cookie. El repo contiene .dodsrc con rutas esperadas; ajústalo si necesitas.

6 — Interpretación de resultados (cómo saber si va a llover)

Probabilidad histórica (rain_probability): fracción de años dentro de la ventana históricos donde precip > rain_threshold.

Ej.: 0.80 → 80% probabilidad de lluvia (según frecuencia histórica para esa ventana).

Modelos (si se entrenan):

logreg_prob y rf_prob: probabilidades devueltas por cada modelo.

ensemble_prob: promedio (decisión: si ≥ 0.5 → "LLOVERÁ").

Umbral de lluvia: por defecto 0.5 mm. Puedes cambiar usando --rain_threshold o --umbral en scripts.

Cuidado: si hay pocos años / datos faltantes, los modelos ML no son confiables. En ese caso el script usa la heurística de frecuencia.

7 — Problemas frecuentes y soluciones

ModuleNotFoundError: No module named 'requests'
-> Activa .venv y ejecuta pip install -r requirements.txt.

argument --bbox: expected one argument
-> En Windows cmd o PowerShell pasa el bbox sin comillas y separado en 4 valores: --bbox -103.6 20.3 -103.1 20.8.
En Git Bash / Linux puedes usar --bbox "-103.6 20.3 -103.1 20.8" o --bbox -103.6 20.3 -103.1 20.8.

Power API devuelve 500 para años futuros
-> predict_rain_for_date_power.py evita pedir años en el futuro. Si haces llamadas individuales (por año) la API puede ser más estable.

OPeNDAP access denied al abrir opendap URL con xarray
-> Ocurre si la URL requiere autorización. Usa get_edl_token.py para obtener ~/.edl_token. Los scripts hacen fallback: descargar con token y abrir localmente.

Problemas con rutas Windows y \ al pegar comandos multi-línea

En cmd.exe usa ^ para continuaciones.

En PowerShell usa backtick `.

En Git Bash / Linux usa \.

Si analizador.py no encuentra columnas en CSV
-> Asegúrate que el CSV generada por power_timeseries.py tiene columnas mínimas: date, precip, t2m. Otros nombres permitidos: rh2m, wind, rad, ps.

8 — Ejemplos prácticos (copiar y pegar)
1) Crear venv, activarlo e instalar deps (Windows PowerShell)
cd C:\Users\Cesar\Desktop\back\AtmosAtlas
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

2) Descargar POWER (rango amplio)
python power_timeseries.py --lat 20.67 --lon -103.35 --start 1984-01-01 --end 2024-12-31
# -> genera power_timeseries_20p67_m103p35_19840101_20241231.csv

3) Analizar CSV para target date con ventana ±7 días
python analizador.py --csv power_timeseries_20p67_m103p35_19840101_20241231.csv --target_date 2025-12-25 --window_days 7 --save_window ventana_2025-12-25.csv

4) Predicción usando POWER batch
python predict_rain_for_date_power.py --date 2036-08-11 --lat 20.67 --lon -103.35 --years_back 15

5) Usar GPM/CMR (token requerido)
python get_edl_token.py
# guarda token en ~/.edl_token
python predict_rain_for_date.py --date 2024-08-11 --lat 20.67 --lon -103.35 --years_back 10 --bbox -103.6 20.3 -103.1 20.8 --precip_short_name GPM_3IMERGDF.07

9 — ¿Qué archivos son indispensables y cuáles prescindibles?
Indispensables (para flujos básicos)

power_timeseries.py — necesario para descargar series POWER

analizador.py — análisis y predicción desde CSV o descarga

requirements.txt — dependencias

.venv (opcional local) — entorno virtual

Útiles (pero opcionales)

predict_rain_for_date_power.py — flujo especializado que usa batch POWER

predict_rain_for_date.py + debug_read_gpm_point.py + cmr_granules_and_links.py — necesarios si quieres usar GPM/CMR (requieren token Earthdata)

get_edl_token.py — hace conveniente crear token (opcional; también puedes generarlo en Earthdata web)

10 — Seguridad / buenas prácticas

No compartas ~/.edl_token ni tu contraseña Earthdata.

Si trabajas en una laptop pública, elimina temporales con datos descargados si tienen información sensible.

Para trabajos más serios en producción, considera almacenar datos en S3 o base de datos y usar colas/CRON para descargas programadas.

11 — Limitaciones y recomendaciones de uso

Estos scripts son herramientas exploratorias y baselines. No reemplazan modelos meteorológicos profesionales.

Para predicciones fiables necesitas:

Más datos y limpieza (imputación avanzada).

Features temporales (lag, ENSO index, etc.).

Validación temporal robusta (cross-validation por años).

Uso práctico: obtener probabilidades históricas (climatología puntual) y un primer modelo simple (baseline).

12 — Soporte / ¿qué hago si algo falla?

Copia el mensaje de error completo.

Verifica que estés usando el venv correcto ((.venv) en prompt).

Asegúrate de que el CSV existe (dir / ls).

Si es relacionado con token: comprueba ~/.edl_token y que puedes hacer curl -I https://data.gesdisc.earthdata.nasa.gov/... con cabecera Authorization: Bearer <token> (o usa debug_read_gpm_point.py para trazas).

Pega aquí el error (output) y yo te ayudo a depurarlo.

13 — Licencia y créditos

Código: libre para uso y adaptaciones (indica si deseas licencia específica, p. ej. MIT).

Datos: suministrados por NASA (POWER, GPM, MERRA2). Revisa condiciones de uso de cada dataset.

14 — Resumen/cheat-sheet (comandos más usados)

Activar entorno:

.venv\Scripts\Activate.ps1   # PowerShell
.venv\Scripts\activate       # cmd
source .venv/bin/activate   # bash


Descargar POWER:

python power_timeseries.py --lat 20.67 --lon -103.35 --start 1984-01-01 --end 2024-12-31


Analizar CSV y predecir:

python analizador.py --csv power_timeseries_20p67_m103p35_19840101_20241231.csv --target_date 2025-12-25 --window_days 7


Usar GPM/CMR (requiere token):

python get_edl_token.py
python predict_rain_for_date.py --date 2024-08-11 --lat 20.67 --lon -103.35 --years_back 