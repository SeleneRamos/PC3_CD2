"""
recoleccion.py
Descarga reseñas públicas de Google Play de las apps bancarias peruanas.

Uso desde la terminal o Colab:
    python -m src.recoleccion --n 3000
"""
import argparse
import time

import pandas as pd

# Apps analizadas: nombre corto -> id en Google Play
APPS = {
    "Yape": "com.bcp.innovacxion.yapeapp",
    "BCP": "com.bcp.bank.bcp",
    "Interbank": "pe.com.interbank.mobilebanking",
    "BBVA": "com.bbva.nxt_peru",
    "Scotiabank": "pe.com.scotiabank.blpm.android.client",
}

# Columnas que se guardan. Se descartan userName y userImage para no
# guardar datos personales de los usuarios.
COLUMNAS = [
    "reviewId", "app", "content", "score", "thumbsUpCount",
    "reviewCreatedVersion", "at", "replyContent", "repliedAt",
]


def descargar_resenas(app_id, n=3000, lote=200, pausa=1.0):
    """Descarga hasta n reseñas (las más recientes) de una app en español de Perú."""
    from google_play_scraper import Sort, reviews

    resultado, token = [], None
    while len(resultado) < n:
        cantidad = min(lote, n - len(resultado))
        datos, token = reviews(
            app_id, lang="es", country="pe", sort=Sort.NEWEST,
            count=cantidad, continuation_token=token,
        )
        if not datos:
            break
        resultado.extend(datos)
        if token is None or token.token is None:
            break
        time.sleep(pausa)  # pausa corta para no saturar el servidor
    return resultado


def recolectar(apps=APPS, n=3000):
    """Recolecta las reseñas de todas las apps y devuelve un DataFrame anonimizado."""
    tablas = []
    for nombre, app_id in apps.items():
        datos = descargar_resenas(app_id, n=n)
        df = pd.DataFrame(datos)
        if df.empty:
            print(f"{nombre}: 0 reseñas (revisa el id {app_id})")
            continue
        df["app"] = nombre
        tablas.append(df.reindex(columns=COLUMNAS))
        print(f"{nombre}: {len(df):,} reseñas")
    if not tablas:
        raise RuntimeError("No se descargó ninguna reseña. Revisa tu conexión.")
    return pd.concat(tablas, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000, help="reseñas por app")
    parser.add_argument("--salida", default="data/raw/resenas_raw.csv")
    args = parser.parse_args()
    df = recolectar(n=args.n)
    df.to_csv(args.salida, index=False, encoding="utf-8")
    print(f"Guardado en {args.salida}: {len(df):,} filas")
