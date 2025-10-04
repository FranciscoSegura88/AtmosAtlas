# 🛰️ AtmosAtlas

**AtmosAtlas** es un conjunto de scripts en Python diseñados para consultar datos climáticos históricos de la NASA (principalmente del servicio **POWER**) y estimar la **probabilidad de lluvia** en una fecha y ubicación determinadas, basándose en los registros de años anteriores.

Incluye herramientas para:
- Descargar series históricas (temperatura, precipitación, presión, radiación, viento, humedad, etc.)  
- Entrenar modelos estadísticos simples (Logistic Regression y Random Forest) para estimar lluvia.  
- Consultar datos satelitales (GPM / MERRA2) con credenciales de **Earthdata** (opcional).

---

## ⚙️ Requisitos previos

### 🧩 Software necesario

- **Python 3.10 o superior** (probado en 3.13)
- **Virtualenv** o el módulo estándar `venv`
- Conexión a Internet (para consultar datos de NASA)

---

## 🧱 Instalación paso a paso

### 1️⃣ Clonar o descargar el proyecto

```bash
git clone https://github.com/tuusuario/franciscosegura88-atmosatlas.git
cd franciscosegura88-atmosatlas
python -m venv .venv
.venv\Scripts\Activate.ps1
Verifica que el entorno está activo:
el prompt debería verse como
(.venv) C:\Users\Cesar\Desktop\back\AtmosAtlas>

Crea (si no existe) el archivo requirements.txt con este contenido:

requests>=2.28
pandas>=2.0
numpy>=1.23
xarray>=2024.0
netCDF4>=1.6
scikit-learn>=1.2
matplotlib>=3.6
pydap>=3.5
earthaccess>=0.15


Luego instala todo:

pip install -r requirements.txt


(O actualiza primero pip si lo deseas)

python -m pip install --upgrade pip
1. Descargar series históricas desde NASA POWER

El script power_timeseries.py descarga variables diarias del servicio NASA POWER para una latitud/longitud dada entre dos fechas.

📘 Ejemplo
python power_timeseries.py --lat 20.67 --lon -103.35 --start 1984-01-01 --end 2024-12-31


Genera un archivo CSV con nombre tipo:

power_timeseries_20p67_m103p35_19840101_20241231.csv

🧾 Contenido del CSV
date	precip	t2m	rh2m	wind	rad	ps
1984-01-01	2.3	22.1	75.2	1.8	20.5	89.7
1984-01-02	0.0	21.7	70.1	2.0	19.8	89.5
...	...	...	...	...	...	...

Donde:

precip → Precipitación total (mm)

t2m → Temperatura a 2 m (°C)

rh2m → Humedad relativa (%)

wind → Velocidad del viento (m/s)

rad → Radiación solar (W/m²)

ps → Presión atmosférica (kPa)

🌧️ 2. Predecir probabilidad de lluvia con POWER

El script predict_rain_for_date_power.py usa los datos históricos de POWER para el mismo día y mes en años pasados y estima la probabilidad de lluvia mediante modelos simples.

📘 Ejemplo
python predict_rain_for_date_power.py --date 2036-08-11 --lat 20.67 --lon -103.35 --years_back 15

📤 Salida esperada
Construyendo histórico para 2036-08-11 en 20.67 -103.35
[INFO] años con precip disponible: 14 de 15
Train acc (logreg): 0.86
Train AUC (logreg): 0.93
Predicción (logreg, rf, avg): 0.72 0.68 0.70
Probabilidad promedio de lluvia: 70.0%
Decisión (umbral 0.5): Lloverá
Histórico guardado en history_used_power.csv

💡 Interpretación
Campo	Descripción
Train acc	Precisión del modelo con los datos históricos
Train AUC	Capacidad del modelo para distinguir lluvia/no lluvia
logreg_prob / rf_prob	Probabilidad de lluvia según cada modelo
avg	Promedio entre ambos modelos
P(llover)	Probabilidad final (si >0.5 → “Lloverá”)
⚡ 3. Predicción rápida desde un CSV existente

Si ya descargaste los datos con power_timeseries.py, puedes usar quick_predict_from_power_csv.py para hacer una predicción directa, sin volver a consultar NASA.

📘 Ejemplo
python quick_predict_from_power_csv.py --csv power_timeseries_20p67_m103p35_19840101_20241231.csv --target_date 2025-08-11

📤 Salida esperada
Datos cargados: 14976 registros
Usando 40 años de históricos (mismo día y mes)
Entrenando modelos...
Train acc (logreg): 0.84  |  AUC: 0.91
Predicción promedio: 0.63
Probabilidad de lluvia ≈ 63.0%

💡 Interpretación

El script toma todas las fechas con el mismo día/mes (por ejemplo, 11 de agosto de 1984–2024).

Entrena modelos en función de las variables meteorológicas.

Devuelve la probabilidad esperada de lluvia para la fecha solicitada.

(Ideal cuando ya tienes los datos descargados localmente.)