from dotenv import load_dotenv
load_dotenv()  # <-- carga .env

from enhanced_data_fetcher import EnhancedDataFetcher
import pprint

# Punto y fechas pequeñas para prueba
LAT, LON = 20.6767, -103.3472
START, END = "2020-01-01", "2020-01-07"
from helper_login_earthdata import login_via_token_file
_ = login_via_token_file(token_path=r"C:\Users\Cesar\.edl_token", username="xcesarg")

fetcher = EnhancedDataFetcher(verbose=True)
df = fetcher.fetch_ensemble_data(LAT, LON, START, END)

print("\n=== HEAD DEL ENSEMBLE ===")
print(df.head())

print("\n=== PROVENANCE COMPLETO ===")
pprint.pp(fetcher.last_provenance)

print("\n=== RESUMEN CORTO PROVENANCE ===")
print(fetcher.provenance_summary())

# Verificación “automática”: ¿IMERG aportó registros?
imerg_info = fetcher.last_provenance.get("sources", {}).get("imerg", {})
print("\nIMERG records ->", imerg_info.get("records", 0))
if imerg_info.get("records", 0) > 0:
    print("\n✅ Éxito: IMERG está autenticado y aportó datos.")
else:
    print("\n⚠️ IMERG no aportó (xarray/earthaccess sin instalar, sin login, o red bloqueada).")
