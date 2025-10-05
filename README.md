# 🌦️ AtmosAtlas

**AtmosAtlas** analiza clima histórico (NASA POWER y opcionalmente IMERG/GPCP) para estimar la **probabilidad de lluvia** y otras variables en una **fecha futura** (climatología + ML básico + ML avanzado calibrado).  
> **No** es pronóstico meteorológico en tiempo real; es **riesgo basado en historia** para decisiones informadas.

---

## 🧭 Arquitectura (resumen)
- **Datos**:  
  - `POWER` diario (precip, t2m, rh2m, wind, rad, ps).  
  - `IMERG` V07 diario (precip) *opcional* vía **Earthdata** (OPeNDAP ó descarga HTTPS) — requiere token.  
  - `GPCP Daily/Monthly` *opcional* como respaldo.
- **Ingesta y fusión**: `enhanced_data_fetcher.py` con “provenance” y pesos por variable + incertidumbre inter‑fuente.
- **Análisis**:  
  - Estadístico (frecuencias históricas).  
  - ML básico (LR + RF).  
  - **ML avanzado** (regresión mm + clasificación) con anti‑fragmentación y calibración mm→prob.
- **Orquestación + explicabilidad**: `analizador_integrado.py`.
- **API**: FastAPI (`main_api_actualizado.py`) con respuesta extendida y narrativa de explicabilidad.

---

## ⚙️ Requisitos
- Python 3.10+ (recomendado 3.11+)
- Windows/macOS/Linux
- Internet (POWER). Para IMERG/GPCP se recomienda: `xarray`, `netCDF4`, `h5netcdf`, `pydap`, `earthaccess`.

```bash
pip install -r requirements.txt
