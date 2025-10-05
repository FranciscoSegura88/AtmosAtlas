# helper_login_earthdata.py
import os
from pathlib import Path
import earthaccess

def login_via_token_file(token_path: str, username: str | None = None, persist: bool = False):
    """
    Lee un token EDL desde un archivo y hace login con earthaccess usando la
    estrategia 'environment'. Usa la variable CORRECTA: EARTHDATA_TOKEN.

    Args:
        token_path: ruta al archivo que contiene SOLO el token (texto plano, sin comillas).
        username: opcional, no es necesario para token; se ignora salvo que quieras forzarlo.
        persist: si True, earthaccess puede cachear sesión.

    Returns:
        La sesión devuelta por earthaccess.login(), o levanta excepción con mensaje claro.
    """
    p = Path(token_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de token: {p}")

    token = p.read_text(encoding="utf-8").strip()
    if not token or any(c.isspace() for c in token):
        # Espacios internos suelen indicar token pegado con saltos de línea, JSON, etc.
        # Se permite \n final por strip(), pero no espacios dentro.
        # Aun así, no bloqueamos: solo avisamos.
        pass

    # LIMPIA variables que puedan interferir (por si se puso mal antes)
    for var in ("EARTHACCESS_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        os.environ.pop(var, None)

    # Establece LA correcta
    os.environ["EARTHDATA_TOKEN"] = token

    # Opcional: si quieres dejar constancia del usuario, no hace falta para token
    if username:
        os.environ["EARTHDATA_USERNAME"] = username  # no requerido para token

    try:
        session = earthaccess.login(strategy="environment", persist=persist)
        if not session:
            raise RuntimeError("earthaccess.login devolvió una sesión vacía.")
        return session
    except Exception as e:
        # Mensaje más claro para diagnóstico
        raise RuntimeError(
            "Fallo al autenticar con Earthdata usando EARTHDATA_TOKEN. "
            "Verifica que tu token sea válido y no esté caducado. "
            f"Detalle: {e}"
        ) from e


if __name__ == "__main__":
    # Prueba rápida local:
    token_path = r"C:\Users\Cesar\.edl_token"
    s = login_via_token_file(token_path, username="xcesarg")
    print("Login ok?", bool(s))
