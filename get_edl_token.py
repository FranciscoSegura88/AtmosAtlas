# get_edl_token.py
import requests
from base64 import b64encode
from getpass import getpass
import os
import json

url = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
username = input("Earthdata username: ")
password = getpass("Earthdata password: ")

creds = b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")
headers = {"Authorization": f"Basic {creds}"}
resp = requests.post(url, headers=headers)

if resp.status_code == 200:
    info = resp.json()
    token = info.get("access_token")
    if token:
        token_path = os.path.join(os.path.expanduser("~"), ".edl_token")
        with open(token_path, "w") as f:
            f.write(token)
        print("Token obtenido y guardado en:", token_path)
        # opcional: imprimir (cuidado)
        # print("Token:", token)
    else:
        print("Respuesta OK pero no se encontró access_token en JSON:")
        print(json.dumps(info, indent=2))
else:
    print("Error al obtener token. Código:", resp.status_code)
    print(resp.text)
