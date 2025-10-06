# 🌦️ AtmosAtlas

**AtmosAtlas** analyzes historical weather (NASA POWER and optionally IMERG/GPCP) to estimate the **probability of rain** and other variables on a **future date** (climatology + basic ML + calibrated advanced ML).  
> It is **not** a real-time forecast; it provides **history-based risk** to support informed decisions.

---

## 🧭 Architecture (summary)
- **Data**
  - Daily `POWER` (precipitation, T2M, RH2M, wind, radiation, surface pressure).
  - Optional `IMERG` V07 daily (precip) via **Earthdata** (OPeNDAP/HTTPS) — token required.
  - Optional `GPCP` Daily/Monthly as fallback.
- **Ingestion & fusion:** `enhanced_data_fetcher.py` with data provenance, per-variable weighting, and inter-source uncertainty.
- **Analysis:**
  - Statistical (historical frequencies).
  - Basic ML (Logistic Regression + Random Forest).
  - **Advanced ML** (mm regression + classification) with anti-fragmentation and mm→probability calibration.
- **Orchestration & explainability:** `analizador_integrado.py`.
- **API:** FastAPI (`main_api_actualizado.py`) with extended response and an explainability narrative.

---

## ⚙️ Requirements
- Python 3.10+ (3.11+ recommended)
- Windows/macOS/Linux
- Internet access (POWER). For IMERG/GPCP it’s recommended to install: `xarray`, `netCDF4`, `h5netcdf`, `pydap`, `earthaccess`.

```bash
pip install -r requirements.txt

