# quick_imerg_check.py  (v2)
from datetime import datetime, timedelta
import os
import earthaccess as ea
import xarray as xr

# --- Configura aquí una fecha de prueba razonable ---
DATE_TO_TEST = datetime(2020, 1, 1)  # cambia si quieres

# Colecciones IMERG "Daily" (Final/Late/Early) sin sufijo de versión
COLLECTIONS = [
    ("GPM_3IMERGDF", "07"),  # Daily Final v07
    ("GPM_3IMERGDL", "07"),  # Daily Late v07
    ("GPM_3IMERGDE", "07"),  # Daily Early v07
    ("GPM_3IMERGDF", "06"),  # v06 como respaldo
    ("GPM_3IMERGDL", "06"),
    ("GPM_3IMERGDE", "06"),
]

# Si quieres intentar también productos de 30 min, descomenta:
# COLLECTIONS_30MIN = [
#     ("GPM_3IMERGHHE", "07"),  # Half-hourly Early
#     ("GPM_3IMERGHHL", "07"),  # Half-hourly Late
#     ("GPM_3IMERGHH",  "07"),  # Half-hourly Final
# ]

def main():
    # 1) Login por entorno (ya tienes EARTHDATA_TOKEN en tu PS)
    s = ea.login(strategy="environment")
    print("EA session ok?", bool(s))

    # 2) Ventana [start, end) de varios días para asegurar resultados
    start = DATE_TO_TEST.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = start + timedelta(days=7)   # 1 semana para evitar huecos
    temporal = (start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"))

    results = []
    used = None

    # 3) Buscar en GES_DISC, no cloud, con short_name + version
    for short_name, ver in COLLECTIONS:
        print(f"Buscando {short_name} v{ver} en {temporal} ...")
        try:
            r = ea.search_data(
                daac="GES_DISC",
                short_name=short_name,
                version=ver,
                temporal=temporal,
                cloud_hosted=False,  # clave para GES_DISC
            )
        except Exception as e:
            print(f"  Error al buscar {short_name} v{ver}: {e}")
            continue

        n = len(r) if r else 0
        print(f"  -> {n} resultados")
        if n > 0:
            results = r
            used = (short_name, ver)
            break

    if not results:
        print("\nNo hubo resultados con las colecciones Daily probadas.")
        print("Prueba cambiar la fecha (p.ej. 2020-01-15) o ampliar más el rango.")
        print("Si persiste, descomenta las colecciones de 30 min y vuelve a correr.")
        return

    # 4) Descargar 1 archivo por HTTPS y abrir localmente con xarray
    os.makedirs("_imerg_test", exist_ok=True)
    files = ea.download(results[:1], "_imerg_test")
    if not files:
        print("Descarga no realizada (¿bloqueo de red/firewall?).")
        return

    local = files[0]
    print("Descargado:", local)

    # 5) Abrir con xarray (HDF5/netCDF)
    try:
        ds = xr.open_dataset(local, engine="h5netcdf", decode_times=False)
    except Exception as e:
        print("h5netcdf falló, probando netCDF4:", e)
        ds = xr.open_dataset(local, engine="netcdf4", decode_times=False)

    print("\n=== DATASET ===")
    print(ds)
    print("\nVariables:", list(ds.data_vars))
    print("\nColección usada:", f"{used[0]} v{used[1]}")

if __name__ == "__main__":
    main()
