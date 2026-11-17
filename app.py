"""
Analizador de reseñas de apps bancarias peruanas
App de Streamlit del Proyecto PC3 de Procesamiento de Lenguaje Natural (UNALM).

Ejecutar localmente:  streamlit run app.py
Los modelos se generan con el notebook PC3_PLN_resenas_banca.ipynb.
"""
import html
import io
import json
import os
import re
import unicodedata
from collections import Counter

import emoji
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.pln_utils import (CATEGORIAS, RE_FECHA, RE_FECHA_PALABRA, RE_HACE, RE_MONTO,
                           RE_URL, RE_VERSION, RUIDO_NER, SIN_CATEGORIA, STOP_DOMINIO, STOPWORDS,
                           DISPOSITIVOS, ENTIDADES, OPERADORES, buscar_similares, cargar_spacy,
                           clasificar_reclamo, extraer_informacion, limpiar_texto,
                           normalizar_jerga, preprocesar_lote, quitar_stopwords, quitar_tildes,
                           stemizar, tokenizar, _lemas_doc)
from src.estilo import COLORES_APP as _COLORES_APP_BASE, ORDEN_SENT

st.set_page_config(page_title="Reseñas de banca móvil", page_icon="💬", layout="wide")

# ---------------------------------------------------------------- paletas oficiales de bancos y tema vanguardista
TINTA = "#F1F5F9"
TINTA_SEC = "#94A3B8"
GRILLA = "rgba(148, 163, 184, 0.12)"

COLORES_SENT = {
    "negativo": "#F43F5E",
    "neutro": "#64748B",
    "positivo": "#10B981",
}

PALETAS_BANCOS = {
    "bcp": {
        "primario": "#FF7800",      # Naranja oficial BCP
        "secundario": "#002A8D",    # Azul marino oficial BCP
        "acento": "#FF9E44",
        "gradiente": "linear-gradient(135deg, #002A8D 0%, #0041C4 52%, #FF7800 100%)",
        "superficie": "linear-gradient(165deg, rgba(0, 42, 141, 0.34) 0%, rgba(255, 120, 0, 0.12) 100%)",
        "borde": "rgba(255, 120, 0, 0.48)",
        "glow": "rgba(255, 120, 0, 0.22)",
        "sigla": "BCP",
        "tag": "Banco de Crédito",
    },
    "yape": {
        "primario": "#A838C7",      # Morado Yape luminoso (basado en #742384)
        "secundario": "#00CBB2",    # Turquesa oficial Yape
        "acento": "#00CBB2",
        "gradiente": "linear-gradient(135deg, #742384 0%, #9C27B0 55%, #00CBB2 100%)",
        "superficie": "linear-gradient(165deg, rgba(116, 35, 132, 0.38) 0%, rgba(0, 203, 178, 0.14) 100%)",
        "borde": "rgba(0, 203, 178, 0.50)",
        "glow": "rgba(0, 203, 178, 0.22)",
        "sigla": "YP",
        "tag": "Billetera BCP",
    },
    "bbva": {
        "primario": "#2DCCCD",      # Aqua Digital oficial BBVA
        "secundario": "#004481",    # Azul Core oficial BBVA (#072146 / #004481)
        "acento": "#5CE1E6",
        "gradiente": "linear-gradient(135deg, #072146 0%, #004481 52%, #2DCCCD 100%)",
        "superficie": "linear-gradient(165deg, rgba(0, 68, 129, 0.38) 0%, rgba(45, 204, 205, 0.13) 100%)",
        "borde": "rgba(45, 204, 205, 0.48)",
        "glow": "rgba(45, 204, 205, 0.22)",
        "sigla": "BBVA",
        "tag": "BBVA Perú",
    },
    "interbank": {
        "primario": "#05BE50",      # Verde oficial Interbank
        "secundario": "#0039A6",    # Azul oficial Interbank
        "acento": "#34D399",
        "gradiente": "linear-gradient(135deg, #05BE50 0%, #009B3A 55%, #0039A6 100%)",
        "superficie": "linear-gradient(165deg, rgba(5, 190, 80, 0.26) 0%, rgba(0, 57, 166, 0.22) 100%)",
        "borde": "rgba(5, 190, 80, 0.50)",
        "glow": "rgba(5, 190, 80, 0.22)",
        "sigla": "IBK",
        "tag": "Interbank",
    },
    "scotiabank": {
        "primario": "#EC111A",      # Rojo Scotia oficial
        "secundario": "#8B0000",    # Carmesí profundo Scotia
        "acento": "#FF5A5F",
        "gradiente": "linear-gradient(135deg, #EC111A 0%, #C80D15 60%, #FF6B6B 100%)",
        "superficie": "linear-gradient(165deg, rgba(236, 17, 26, 0.30) 0%, rgba(139, 0, 0, 0.16) 100%)",
        "borde": "rgba(236, 17, 26, 0.52)",
        "glow": "rgba(236, 17, 26, 0.24)",
        "sigla": "SCT",
        "tag": "Scotiabank Perú",
    },
    "plin": {
        "primario": "#00D1C1",
        "secundario": "#0055FF",
        "acento": "#5EEAD4",
        "gradiente": "linear-gradient(135deg, #0055FF 0%, #00D1C1 100%)",
        "superficie": "linear-gradient(165deg, rgba(0, 85, 255, 0.28) 0%, rgba(0, 209, 193, 0.14) 100%)",
        "borde": "rgba(0, 209, 193, 0.48)",
        "glow": "rgba(0, 209, 193, 0.22)",
        "sigla": "PLIN",
        "tag": "Billetera Plin",
    },
    "nacion": {
        "primario": "#E11D48",
        "secundario": "#881337",
        "acento": "#FB7185",
        "gradiente": "linear-gradient(135deg, #881337 0%, #E11D48 100%)",
        "superficie": "linear-gradient(165deg, rgba(136, 19, 55, 0.32) 0%, rgba(225, 29, 72, 0.14) 100%)",
        "borde": "rgba(225, 29, 72, 0.48)",
        "glow": "rgba(225, 29, 72, 0.22)",
        "sigla": "BN",
        "tag": "Banco de la Nación",
    },
}


def obtener_paleta_banco(nombre_app):
    """Devuelve la paleta corporativa oficial según el nombre del banco o app."""
    clave = quitar_tildes(str(nombre_app).lower())
    for k, paleta in PALETAS_BANCOS.items():
        if k in clave:
            return paleta
    color_fallback = _COLORES_APP_BASE.get(nombre_app, "#6366F1")
    return {
        "primario": color_fallback,
        "secundario": "#1E293B",
        "acento": color_fallback,
        "gradiente": f"linear-gradient(135deg, {color_fallback} 0%, #4F46E5 100%)",
        "superficie": "linear-gradient(165deg, rgba(99, 102, 241, 0.22) 0%, rgba(15, 23, 42, 0.4) 100%)",
        "borde": "rgba(99, 102, 241, 0.45)",
        "glow": "rgba(99, 102, 241, 0.20)",
        "sigla": str(nombre_app)[:3].upper(),
        "tag": str(nombre_app),
    }


def svg_emblema_banco(nombre_app):
    """Genera un emblema SVG vectorial con los colores originales de cada entidad financiera."""
    clave = quitar_tildes(str(nombre_app).lower())
    uid = re.sub(r"[^a-z0-9]", "", clave) or "app"
    if "bcp" in clave:
        return (
            f'<svg viewBox="0 0 44 44" width="42" height="42" fill="none" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><linearGradient id="g_{uid}" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0%" stop-color="#002A8D"/><stop offset="100%" stop-color="#0041C4"/></linearGradient></defs>'
            f'<rect width="44" height="44" rx="12" fill="url(#g_{uid})"/>'
            f'<path d="M25 8L15 24H23L18 36L31 19H22.5L25 8Z" fill="#FF7800"/>'
            f'<circle cx="34" cy="10" r="3" fill="#FF9E44"/></svg>'
        )
    if "yape" in clave:
        return (
            f'<svg viewBox="0 0 44 44" width="42" height="42" fill="none" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><linearGradient id="g_{uid}" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0%" stop-color="#742384"/><stop offset="100%" stop-color="#561466"/></linearGradient></defs>'
            f'<rect width="44" height="44" rx="12" fill="url(#g_{uid})"/>'
            f'<path d="M13 13L21 24V32H25V24L31 13H26.2L22.8 19.8L19.4 13H13Z" fill="#FFFFFF"/>'
            f'<circle cx="32" cy="31" r="4.5" fill="#00CBB2"/></svg>'
        )
    if "bbva" in clave:
        return (
            f'<svg viewBox="0 0 44 44" width="42" height="42" fill="none" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><linearGradient id="g_{uid}" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0%" stop-color="#072146"/><stop offset="100%" stop-color="#004481"/></linearGradient></defs>'
            f'<rect width="44" height="44" rx="12" fill="url(#g_{uid})"/>'
            f'<path d="M11 30L19.5 14H24.5L33 30H28L22 18.2L16 30H11Z" fill="#FFFFFF"/>'
            f'<polygon points="22,22 26.5,30 17.5,30" fill="#2DCCCD"/></svg>'
        )
    if "interbank" in clave:
        return (
            f'<svg viewBox="0 0 44 44" width="42" height="42" fill="none" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="44" height="44" rx="12" fill="#0039A6"/>'
            f'<path d="M8 12C8 9.8 9.8 8 12 8H36L29 36H12C9.8 36 8 34.2 8 32V12Z" fill="#05BE50"/>'
            f'<rect x="17" y="14" width="5" height="16" rx="2" transform="skewX(-12)" fill="#FFFFFF"/></svg>'
        )
    if "scotia" in clave:
        return (
            f'<svg viewBox="0 0 44 44" width="42" height="42" fill="none" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><linearGradient id="g_{uid}" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0%" stop-color="#EC111A"/><stop offset="100%" stop-color="#B30910"/></linearGradient></defs>'
            f'<rect width="44" height="44" rx="12" fill="url(#g_{uid})"/>'
            f'<circle cx="22" cy="22" r="5.2" fill="#FFFFFF"/>'
            f'<path d="M14 16C17 11.5 26 11.5 29.5 15.5M30 28C27 32.5 18 32.5 14.5 28.5" '
            f'stroke="#FFFFFF" stroke-width="3.2" stroke-linecap="round"/></svg>'
        )
    pal = obtener_paleta_banco(nombre_app)
    return (
        f'<svg viewBox="0 0 44 44" width="42" height="42" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="44" height="44" rx="12" fill="{pal["primario"]}"/>'
        f'<text x="50%" y="55%" dominant-baseline="middle" text-anchor="middle" '
        f'fill="#FFFFFF" font-family="sans-serif" font-weight="800" font-size="13">{pal["sigla"]}</text></svg>'
    )


COLORES_APP = {a: obtener_paleta_banco(a)["primario"] for a in _COLORES_APP_BASE}

RUTAS = {
    "datos": "data/processed/resenas_procesadas.csv",
    "sentimiento": "models/modelo_sentimiento.joblib",
    "temas": "models/modelo_temas.joblib",
    "lsa": "models/modelo_lsa.joblib",
    "matriz": "models/lsa_matriz.npy",
    "metricas": "models/metricas.json",
}
EJEMPLOS = {
    "App que se cuelga": "Pésima app!!! no me deja yapear desde ayer 😡 xq se cuelga a cada rato",
    "Robo de cuenta": "Me robaron el celular y me vaciaron la cuenta, 500 soles sin mi autorización. "
                      "El call center no me ayuda",
    "Validación con selfie": "La validación con selfie nunca funciona, no puedo ingresar desde mi Samsung "
                             "hace 3 días",
    "Comisión": "Me cobraron S/ 8 de comisión por una transferencia interbancaria, deberían avisar",
    "Reseña positiva": "Excelente aplicación, rápida y fácil de usar 👍 nunca me ha fallado",
}
NOMBRE_CORTO = {
    "Seguridad y fraude": "fraude",
    "Transferencias y pagos": "pagos",
    "Acceso y validación": "acceso",
    "Comisiones y límites": "comisiones",
    "Actualizaciones y compatibilidad": "actualización",
    "Fallas técnicas": "falla técnica",
    "Atención al cliente": "atención",
}
ETIQUETAS = {
    "reclamo": ("palabra de reclamo", "#FF6B35"),
    "monto": ("monto", "#10B981"),
    "fecha": ("fecha o tiempo", "#38BDF8"),
    "version": ("versión", "#A3E635"),
    "celular": ("celular o sistema", "#A855F7"),
    "entidad": ("entidad financiera", "#F43F5E"),
    "operador": ("operador móvil", "#06B6D4"),
    "ner": ("NER: persona, organización o lugar", "#94A3B8"),
}
NER_NOMBRE = {"PER": "persona", "ORG": "organización", "LOC": "lugar"}
TEXTO_SENT = {"negativo": "Negativo", "neutro": "Neutro", "positivo": "Positivo"}

# ---------------------------------------------------------------- estilos vanguardistas (UI/UX integral)
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=JetBrains+Mono:wght@500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg-obsidian: #070B12;
  --bg-surface: #0F1624;
  --bg-card: rgba(15, 23, 42, 0.82);
  --bg-elevated: rgba(30, 41, 59, 0.65);
  --linea: rgba(148, 163, 184, 0.16);
  --linea-fuerte: rgba(148, 163, 184, 0.28);
  --suave: rgba(30, 41, 59, 0.52);
  --texto: #F8FAFC;
  --sec: #94A3B8;
  --primario: #38BDF8;
  --acento: #6366F1;
  --estrella: #FBBF24;
  --titulo: "Bricolage Grotesque", "Plus Jakarta Sans", sans-serif;
  --cuerpo: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, sans-serif;
  --mono: "JetBrains Mono", monospace;
}

/* Lienzo general Obsidiana Profunda */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background-color: var(--bg-obsidian) !important;
  background-image:
    radial-gradient(at 12% 8%, rgba(56, 189, 248, 0.08) 0px, transparent 42%),
    radial-gradient(at 88% 12%, rgba(99, 102, 241, 0.08) 0px, transparent 45%),
    radial-gradient(at 50% 95%, rgba(16, 185, 129, 0.05) 0px, transparent 50%) !important;
  color: var(--texto) !important;
  font-family: var(--cuerpo) !important;
}

.block-container { padding-top: 3.4rem; padding-bottom: 4rem; max-width: 1280px; }
h1, h2, h3, h4, h5 { font-family: var(--titulo) !important; color: var(--texto) !important; letter-spacing: -0.02em; }
p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--texto); }

/* Hero vanguardista */
.hero {
  position: relative;
  padding: 1.65rem 1.85rem 1.75rem;
  border-radius: 24px;
  background: linear-gradient(145deg, rgba(19, 28, 46, 0.94) 0%, rgba(9, 13, 22, 0.97) 100%);
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: 0 24px 48px -12px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  margin-bottom: 1.45rem;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em;
  color: #38BDF8; background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.30);
  padding: 5px 13px; border-radius: 999px; margin-bottom: .85rem;
}
.hero h1 {
  font-family: var(--titulo); font-size: clamp(2.05rem, 4vw, 3.15rem); font-weight: 800;
  line-height: 1.05; margin: 0 0 .6rem; padding: 0; max-width: 22ch;
  background: linear-gradient(95deg, #FFFFFF 0%, #E2E8F0 65%, #38BDF8 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: var(--sec); font-size: 1.03rem; line-height: 1.55; margin: 0; max-width: 68ch; }

/* Cuadros de bancos con sus paletas originales resaltando sobre fondo obsidiana */
.apps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(205px, 1fr));
  gap: 14px;
  margin: 1.55rem 0 .2rem;
}
.app {
  position: relative;
  border-radius: 18px;
  padding: 16px 16px 15px;
  background: var(--bg-surface);
  background-image: var(--surf);
  border: 1px solid var(--borde);
  box-shadow: 0 14px 32px -10px rgba(0, 0, 0, 0.7), 0 0 24px -6px var(--glow);
  overflow: hidden;
  transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.app:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 38px -8px rgba(0, 0, 0, 0.82), 0 0 30px -2px var(--glow);
}
.app::before {
  content: "";
  position: absolute;
  inset: 0 0 auto 0;
  height: 5px;
  background: var(--grad);
}
.app::after {
  content: "";
  position: absolute;
  width: 120px; height: 120px;
  right: -35px; top: -35px;
  background: radial-gradient(circle, var(--c) 0%, transparent 70%);
  opacity: 0.22;
  pointer-events: none;
}
.app-top {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  margin-bottom: 10px;
}
.app-id { display: flex; align-items: center; gap: 10px; }
.app-logo {
  display: inline-flex; align-items: center; justify-content: center;
  border-radius: 12px;
  box-shadow: 0 6px 14px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.25);
  flex-shrink: 0;
}
.app .nom { font-weight: 800; font-size: 1.06rem; color: #FFFFFF; line-height: 1.15; letter-spacing: -0.01em; }
.app .subtag {
  font-size: .71rem; font-weight: 600; color: rgba(241, 245, 249, 0.72);
  text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;
}
.app .pill-n {
  font-family: var(--mono); font-size: .73rem; font-weight: 600; padding: 3px 9px; border-radius: 999px;
  background: rgba(15, 23, 42, 0.75); color: #E2E8F0;
  border: 1px solid rgba(255, 255, 255, 0.14);
}
.app .prom {
  font-family: var(--titulo); font-size: 1.85rem; font-weight: 800; line-height: 1.1;
  color: #FFFFFF; margin-top: 4px; display: flex; align-items: baseline; gap: 8px;
}
.app .det {
  font-size: .8rem; color: #CBD5E1; margin-top: 8px;
  display: flex; justify-content: space-between; align-items: center;
}
.app .det b { font-family: var(--mono); color: #FFFFFF; font-weight: 700; }
.negbar {
  height: 7px; border-radius: 999px;
  background: rgba(15, 23, 42, 0.78);
  border: 1px solid rgba(255, 255, 255, 0.08);
  margin-top: 6px; overflow: hidden;
}
.negbar > span {
  display: block; height: 100%;
  background: var(--grad);
  border-radius: 999px;
}

/* Cabeceras de sección y contenedores de gráficos UI/UX */
.sec-hdr {
  display: flex; flex-direction: column; gap: 3px;
  margin: .65rem 0 .55rem;
  padding-left: 12px;
  border-left: 3px solid #38BDF8;
}
.sec-hdr .sec-t {
  font-family: var(--titulo); font-size: 1.08rem; font-weight: 700;
  color: #F8FAFC; letter-spacing: -0.015em; line-height: 1.2;
}
.sec-hdr .sec-sub {
  font-size: .82rem; color: var(--sec); line-height: 1.35;
}

div[data-testid="stPlotlyChart"] {
  background: linear-gradient(160deg, rgba(15, 23, 42, 0.84) 0%, rgba(10, 15, 28, 0.90) 100%);
  border: 1px solid var(--linea);
  border-radius: 18px;
  padding: 12px 14px 6px;
  box-shadow: 0 14px 32px -12px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.05);
  margin-bottom: .6rem;
}

/* Tablas UI/UX Dark Glass */
.ui-table-wrap {
  border: 1px solid var(--linea);
  border-radius: 18px;
  background: linear-gradient(160deg, rgba(15, 23, 42, 0.88) 0%, rgba(9, 13, 24, 0.94) 100%);
  box-shadow: 0 16px 36px -12px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.06);
  overflow: hidden;
  margin: .35rem 0 1rem;
}
.ui-table-scroll {
  max-height: 480px;
  overflow-y: auto;
  overflow-x: auto;
}
.ui-table-scroll::-webkit-scrollbar { width: 7px; height: 7px; }
.ui-table-scroll::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.6); }
.ui-table-scroll::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.3); border-radius: 999px; }

.ui-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: .87rem;
  color: #E2E8F0;
}
.ui-table thead th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: rgba(15, 23, 42, 0.96);
  backdrop-filter: blur(10px);
  color: #94A3B8;
  font-size: .73rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  text-align: left;
  padding: 12px 15px;
  border-bottom: 1px solid var(--linea-fuerte);
  white-space: nowrap;
}
.ui-table tbody tr {
  transition: background 0.16s ease;
}
.ui-table tbody tr:nth-child(even) {
  background: rgba(30, 41, 59, 0.22);
}
.ui-table tbody tr:hover {
  background: rgba(56, 189, 248, 0.08);
}
.ui-table td {
  padding: 11px 15px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.09);
  vertical-align: middle;
  line-height: 1.45;
}
.ui-table tbody tr:last-child td {
  border-bottom: none;
}
.ui-table .num-cell {
  font-family: var(--mono);
  font-size: .82rem;
  font-weight: 600;
  color: #F8FAFC;
  white-space: nowrap;
}
.ui-table .date-pill {
  font-family: var(--mono);
  font-size: .76rem;
  padding: 2px 8px;
  border-radius: 6px;
  background: rgba(30, 41, 59, 0.75);
  border: 1px solid rgba(148, 163, 184, 0.18);
  color: #CBD5E1;
  white-space: nowrap;
}
.ui-table .bank-pill {
  display: inline-flex; align-items: center; gap: 7px;
  font-weight: 700; font-size: .83rem; color: #FFFFFF;
  padding: 3px 10px 3px 7px; border-radius: 999px;
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid var(--borde, rgba(255,255,255,0.18));
  white-space: nowrap;
}
.ui-table .bank-pill .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--c); box-shadow: 0 0 8px var(--c);
}
.vol-cell {
  display: flex; align-items: center; gap: 10px; min-width: 135px;
}
.vol-bar {
  flex: 1; height: 6px; border-radius: 999px;
  background: rgba(148, 163, 184, 0.15); overflow: hidden;
}
.vol-bar > span {
  display: block; height: 100%; border-radius: 999px;
  background: var(--grad);
}
.conf-wrap {
  display: inline-flex; align-items: center; gap: 8px; min-width: 105px;
}
.conf-bar {
  width: 54px; height: 6px; border-radius: 999px;
  background: rgba(148, 163, 184, 0.18); overflow: hidden;
}
.conf-bar > span {
  display: block; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #38BDF8, #818CF8);
}
.tech-badge {
  display: inline-block; font-weight: 700; font-size: .79rem;
  color: #7DD3FC; background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.28);
  padding: 3px 10px; border-radius: 8px; white-space: nowrap;
}
.topic-badge {
  display: inline-flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-weight: 700; font-size: .78rem;
  color: #C7D2FE; background: rgba(99, 102, 241, 0.18);
  border: 1px solid rgba(99, 102, 241, 0.40);
  padding: 2px 9px; border-radius: 7px;
}
.empty-val { color: rgba(148, 163, 184, 0.45); font-style: italic; font-size: .8rem; }

/* Estrellas */
.stars {
  position: relative; display: inline-block; font-size: .95rem; letter-spacing: 1.5px;
  color: rgba(255, 255, 255, 0.18); line-height: 1; vertical-align: 1px;
}
.stars::before { content: "★★★★★"; }
.stars > span {
  position: absolute; inset: 0 auto 0 0; width: var(--p); overflow: hidden;
  white-space: nowrap; color: var(--estrella);
  text-shadow: 0 0 10px rgba(251, 191, 36, 0.45);
}

/* Lectura anotada */
.anotado {
  font-size: 1.16rem; line-height: 2.3; padding: 16px 22px 18px;
  border-left: 4px solid var(--primario);
  background: linear-gradient(160deg, rgba(15, 23, 42, 0.88) 0%, rgba(10, 15, 28, 0.94) 100%);
  border-top: 1px solid var(--linea); border-right: 1px solid var(--linea); border-bottom: 1px solid var(--linea);
  border-radius: 0 16px 16px 0; margin: .35rem 0 .7rem; color: #F8FAFC;
  box-shadow: 0 12px 28px -10px rgba(0, 0, 0, 0.55);
}
.anotado mark {
  background: color-mix(in srgb, var(--c) 24%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 55%, transparent);
  color: #FFFFFF; padding: 3px 6px; border-radius: 7px;
  box-decoration-break: clone; -webkit-box-decoration-break: clone;
}
.anotado mark .lbl {
  font-size: .65rem; font-weight: 700; margin-left: 6px; padding: 2px 6px;
  border-radius: 4px; background: var(--c); color: #070B12; vertical-align: 2px;
  white-space: nowrap; text-transform: uppercase; letter-spacing: 0.03em;
}
.leyenda { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: .82rem; color: var(--sec); }
.leyenda span::before {
  content: ""; display: inline-block; width: 10px; height: 10px;
  border-radius: 3px; background: var(--c); margin-right: 6px; vertical-align: -1px;
}

/* Resultados */
.res-titulo { font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--sec); margin-bottom: 6px; }
.res-valor { font-family: var(--titulo); font-size: 1.65rem; font-weight: 800; line-height: 1.15; color: #FFFFFF; }
.res-nota { font-size: .84rem; color: var(--sec); margin-top: 6px; line-height: 1.4; }
.pfila {
  display: grid; grid-template-columns: 76px 1fr 44px; align-items: center; gap: 8px;
  font-size: .83rem; margin-top: 8px; color: #E2E8F0;
}
.pfila .track { height: 8px; border-radius: 999px; background: rgba(148, 163, 184, 0.16); overflow: hidden; }
.pfila .track > span { display: block; height: 100%; border-radius: 999px; }
.pfila b { font-family: var(--mono); text-align: right; font-weight: 700; color: #FFFFFF; font-size: .8rem; }

/* Chips */
.chip {
  display: inline-block; font-size: .75rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;
  border: 1px solid var(--linea); background: rgba(30, 41, 59, 0.78); color: #E2E8F0; margin: 3px 5px 3px 0;
  white-space: nowrap;
}
.chip.neg { background: rgba(244, 63, 94, 0.16); border-color: rgba(244, 63, 94, 0.45); color: #FDA4AF; }
.chip.neu { background: rgba(148, 163, 184, 0.16); border-color: rgba(148, 163, 184, 0.38); color: #CBD5E1; }
.chip.pos { background: rgba(16, 185, 129, 0.16); border-color: rgba(16, 185, 129, 0.45); color: #6EE7B7; }

/* Pasos del preprocesamiento */
.pasos-card {
  border: 1px solid var(--linea);
  border-radius: 18px;
  background: linear-gradient(160deg, rgba(15, 23, 42, 0.84) 0%, rgba(9, 13, 24, 0.92) 100%);
  padding: 6px 18px;
  box-shadow: 0 14px 32px -12px rgba(0, 0, 0, 0.6);
}
.paso {
  display: grid; grid-template-columns: 32px 1fr; gap: 12px; padding: 12px 0;
  border-bottom: 1px dashed var(--linea);
}
.paso:last-child { border-bottom: none; }
.paso .n {
  width: 27px; height: 27px; border-radius: 8px;
  background: linear-gradient(135deg, #38BDF8, #6366F1); color: #070B12;
  font-weight: 800; font-size: .82rem; display: grid; place-items: center; margin-top: 1px;
}
.paso .t { font-weight: 700; font-size: .92rem; color: #F8FAFC; }
.paso .d { font-size: .8rem; color: var(--sec); margin-bottom: 4px; }
.paso .orig { font-size: .91rem; color: #E2E8F0; }
.tok {
  display: inline-block; padding: 2px 9px; margin: 4px 4px 0 0; border-radius: 6px;
  background: rgba(30, 41, 59, 0.82); border: 1px solid var(--linea); font-size: .82rem; color: #F1F5F9;
}
.tok.fuera { text-decoration: line-through; opacity: .40; }
.tok.cambio {
  border-color: #38BDF8;
  background: rgba(56, 189, 248, 0.16);
  color: #7DD3FC; font-weight: 600;
}

/* Tarjetas de reseña con identidad de banco */
.rcard {
  position: relative;
  border: 1px solid var(--borde, var(--linea));
  border-left: 5px solid var(--c);
  border-radius: 8px 16px 16px 8px;
  padding: 13px 16px 13px;
  margin-bottom: 11px;
  background: linear-gradient(160deg, rgba(17, 25, 40, 0.92) 0%, rgba(11, 16, 28, 0.95) 100%);
  box-shadow: 0 8px 20px -6px rgba(0, 0, 0, 0.55);
  transition: transform 0.16s ease, border-color 0.16s ease;
}
.rcard:hover { transform: translateX(3px); }
.rcard .cab {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px;
  font-size: .82rem; color: var(--sec);
}
.rcard .cab .app-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-weight: 800; color: #FFFFFF; font-size: .84rem;
  padding: 2px 10px 2px 7px; border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--borde, rgba(255,255,255,0.15));
}
.rcard .cab .dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--c);
  box-shadow: 0 0 8px var(--c);
}
.rcard .txt { margin: 8px 0 5px; line-height: 1.55; font-size: .94rem; color: #F1F5F9; }
.sim { display: inline-flex; align-items: center; gap: 6px; margin-left: auto; color: #CBD5E1; font-weight: 600; font-family: var(--mono); font-size: .78rem; }
.sim .track { width: 72px; height: 6px; border-radius: 999px; background: rgba(148, 163, 184, 0.2); overflow: hidden; }
.sim .track > span { display: block; height: 100%; background: linear-gradient(90deg, #38BDF8, #6366F1); }

/* Indicadores KPI */
.kpis {
  display: flex; flex-wrap: wrap;
  border: 1px solid var(--linea); border-radius: 18px;
  background: linear-gradient(160deg, rgba(15, 23, 42, 0.86) 0%, rgba(10, 15, 28, 0.94) 100%);
  box-shadow: 0 14px 32px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.06);
  margin: .5rem 0 1.25rem; overflow: hidden;
}
.kpi {
  flex: 1 1 180px; padding: 16px 22px 17px;
  border-right: 1px solid var(--linea);
  position: relative;
}
.kpi::before {
  content: ""; position: absolute; top: 0; left: 22px; right: 22px; height: 2px;
  background: linear-gradient(90deg, #38BDF8, transparent); opacity: 0.65;
}
.kpi:last-child { border-right: none; }
.kpi .v { font-family: var(--titulo); font-size: 1.92rem; font-weight: 800; line-height: 1.12; color: #FFFFFF; }
.kpi .l { font-size: .82rem; color: var(--sec); margin-top: 4px; font-weight: 500; }

/* Recorrido del proyecto */
.flujo { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: .6rem 0 1.3rem; }
.flujo div {
  padding: 15px 16px; border-radius: 16px;
  background: linear-gradient(160deg, rgba(15, 23, 42, 0.84) 0%, rgba(10, 15, 28, 0.92) 100%);
  border: 1px solid var(--linea);
  border-top: 3px solid var(--primario);
  box-shadow: 0 10px 24px -10px rgba(0, 0, 0, 0.55);
}
.flujo b { display: block; font-family: var(--titulo); font-size: 1.05rem; margin: 6px 0 4px; color: #FFFFFF; }
.flujo small { color: var(--sec); font-size: .83rem; line-height: 1.42; display: block; }
.flujo .num { font-family: var(--mono); font-size: .74rem; color: var(--primario); font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }

.nota { font-size: .84rem; color: var(--sec); margin: -.15rem 0 .65rem; }

/* Controles e inputs de Streamlit integrados al panel obsidiana */
div[data-testid="stTabs"] button {
  border-radius: 10px 10px 0 0 !important;
  padding: 10px 16px !important;
}
div[data-testid="stTabs"] button p { font-size: .95rem; font-weight: 600; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(160deg, rgba(15, 23, 42, 0.82) 0%, rgba(10, 15, 28, 0.90) 100%) !important;
  border-color: var(--linea) !important;
  border-radius: 18px !important;
  box-shadow: 0 14px 32px -12px rgba(0, 0, 0, 0.6) !important;
}
div[data-testid="stExpander"] {
  background: rgba(15, 23, 42, 0.72) !important;
  border: 1px solid var(--linea) !important;
  border-radius: 16px !important;
  overflow: hidden;
}
.stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div {
  background-color: rgba(15, 23, 42, 0.88) !important;
  border-color: rgba(148, 163, 184, 0.24) !important;
  color: #F8FAFC !important;
  border-radius: 12px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: #38BDF8 !important;
  box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}
[data-testid="stFileUploader"] section {
  background-color: rgba(15, 23, 42, 0.78) !important;
  border: 1px dashed rgba(56, 189, 248, 0.38) !important;
  border-radius: 16px !important;
}
</style>
"""

# Plantilla común para todos los gráficos de Plotly
pio.templates["resenas"] = go.layout.Template(layout=dict(
    font=dict(family="Plus Jakarta Sans, sans-serif", size=13, color=TINTA),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=list(COLORES_APP.values()),
    xaxis=dict(automargin=True, gridcolor=GRILLA, griddash="dot", zeroline=False, linecolor=GRILLA, ticks="",
               title_font=dict(color=TINTA_SEC), tickfont=dict(color=TINTA_SEC, size=12)),
    yaxis=dict(automargin=True, ticklabelstandoff=8, gridcolor=GRILLA, griddash="dot", zeroline=False, linecolor=GRILLA, ticks="",
               title_font=dict(color=TINTA_SEC), tickfont=dict(color=TINTA, size=12)),
    hoverlabel=dict(bgcolor="#0B1120", bordercolor="rgba(56,189,248,0.45)",
                    font=dict(family="Plus Jakarta Sans, sans-serif", color="#F8FAFC", size=13)),
    legend=dict(orientation="h", y=1.14, x=0, title=None, font=dict(color=TINTA, size=12),
                 bgcolor="rgba(15,23,42,0.55)", bordercolor="rgba(148,163,184,0.14)", borderwidth=1),
    margin=dict(l=8, r=14, t=16, b=12),
    bargap=0.36, barcornerradius=6,
))
pio.templates.default = "resenas"


# ---------------------------------------------------------------- carga
faltan = [r for r in RUTAS.values() if not os.path.exists(r)]
if faltan:
    st.title("Faltan los modelos")
    st.error("Ejecuten primero el notebook `PC3_PLN_resenas_banca.ipynb` y suban al repositorio "
             "las carpetas `models/` y `data/processed/`.")
    st.code("\n".join(faltan))
    st.stop()


@st.cache_resource(show_spinner="Cargando los modelos de PLN...")
def cargar_recursos():
    return {
        "nlp": cargar_spacy(),
        "sent": joblib.load(RUTAS["sentimiento"]),
        "temas": joblib.load(RUTAS["temas"]),
        "lsa": joblib.load(RUTAS["lsa"]),
        "Z": np.load(RUTAS["matriz"]),
    }


@st.cache_data(show_spinner="Cargando las reseñas...")
def cargar_datos():
    df = pd.read_csv(RUTAS["datos"], parse_dates=["fecha"])
    df["categorias"] = df["categorias"].fillna("")
    df["texto_tema"] = df["texto_tema"].fillna("")
    with open(RUTAS["metricas"], encoding="utf-8") as f:
        metricas = json.load(f)
    return df, metricas


R = cargar_recursos()
df, metricas = cargar_datos()
nlp = R["nlp"]
apps = [a for a in _COLORES_APP_BASE if a in df["app"].unique()]
for a in df["app"].unique():
    if a not in apps:
        apps.append(a)
    COLORES_APP[a] = obtener_paleta_banco(a)["primario"]

etiquetas_temas = R["temas"]["etiquetas"]
st.html(CSS)


# ---------------------------------------------------------------- funciones de PLN
def predecir_sentimiento(textos):
    """Devuelve (etiquetas, probabilidades) para una lista de textos."""
    procesados = preprocesar_lote(textos, modo=R["sent"]["modo"], nlp=nlp)
    pipe = R["sent"]["pipeline"]
    proba = pipe.predict_proba(procesados)
    clases = list(pipe.classes_)
    proba = proba[:, [clases.index(c) for c in ORDEN_SENT]]
    return [ORDEN_SENT[i] for i in proba.argmax(axis=1)], proba


def predecir_tema(texto):
    """Tema LDA más probable de un texto (o None si no tiene palabras del vocabulario)."""
    lema = preprocesar_lote([texto], modo="lema", quitar_dominio=True, nlp=nlp)
    x = R["temas"]["vectorizador"].transform(lema)
    if x.nnz == 0:
        return None, None
    theta = R["temas"]["lda"].transform(x)[0]
    k = int(theta.argmax())
    return k, float(theta[k])


def similares(texto, k, apps_ok=None):
    """Reseñas más parecidas (sin textos repetidos)."""
    idx, sims = buscar_similares(texto, R["lsa"]["vectorizador"], R["lsa"]["svd"], R["Z"],
                                 k=len(df) if apps_ok else k * 4, nlp=nlp)
    if len(idx) == 0:
        return df.iloc[[]].assign(similitud=[])
    res = df.iloc[idx].assign(similitud=sims)
    if apps_ok:
        res = res[res["app"].isin(apps_ok)]
    return res.drop_duplicates("content").head(k)


# ---------------------------------------------------------------- lectura anotada
_CAT_RE = {c: [re.compile(p) for p in ps] for c, ps in CATEGORIAS.items()}


def _plano(texto):
    """Minúsculas y sin tildes, con el mismo largo que el texto (para ubicar las palabras)."""
    bajo = texto.lower()
    if len(bajo) != len(texto):
        return None
    plano = "".join(c if c == "ñ" else unicodedata.normalize("NFD", c)[0] for c in bajo)
    return plano if len(plano) == len(texto) else None


def _para_ner(texto):
    """Quita URLs, emojis y signos repetidos sin cambiar el largo del texto."""
    t = RE_URL.sub(lambda m: " " * len(m.group()), texto)
    t = emoji.replace_emoji(t, replace=lambda ch, datos: " " * len(ch))
    return re.sub(r"([!?.,])\1+", lambda m: m.group(1) + " " * (len(m.group()) - 1), t)


def anotar(texto):
    """Ubica en el texto lo que detectan las reglas, las regex de extracción y el NER."""
    plano = _plano(texto)
    if plano is None:
        return []
    marcas = []
    for m in RE_MONTO.finditer(texto):
        marcas.append((m.start(), m.end(), "monto", "monto"))
    for patron in (RE_FECHA, RE_HACE):
        for m in patron.finditer(texto):
            marcas.append((m.start(), m.end(), "fecha", "fecha"))
    for m in RE_FECHA_PALABRA.finditer(plano):
        marcas.append((m.start(), m.end(), "fecha", "fecha"))
    for m in RE_VERSION.finditer(texto):
        marcas.append((m.start(), m.end(), "version", "versión"))
    for lista, tipo, nombre in ((DISPOSITIVOS, "celular", "celular"), (ENTIDADES, "entidad", "entidad"),
                                (OPERADORES, "operador", "operador")):
        for palabra in lista:
            for m in re.finditer(rf"\b{re.escape(palabra)}\b", plano):
                marcas.append((m.start(), m.end(), tipo, nombre))
    doc = nlp(_para_ner(texto))
    for e in doc.ents:
        clave = quitar_tildes(e.text.lower().strip(" .,!?"))
        if e.label_ not in NER_NOMBRE or len(clave) < 3 or clave in RUIDO_NER or clave in STOPWORDS:
            continue
        if len(e) == 1 and e.start == e.sent.start:
            continue
        marcas.append((e.start_char, e.end_char, "ner", NER_NOMBRE[e.label_]))
    for cat, patrones in _CAT_RE.items():
        for p in patrones:
            for m in p.finditer(plano):
                if m.end() > m.start():
                    marcas.append((m.start(), m.end(), "reclamo", NOMBRE_CORTO[cat]))

    def completar(s, e):
        while s > 0 and texto[s - 1].isalpha() and texto[s].isalpha():
            s -= 1
        while e < len(texto) and texto[e].isalpha() and texto[e - 1].isalpha():
            e += 1
        return s, e
    marcas = [(*completar(s, e), t, l) for s, e, t, l in marcas]
    orden_pri = ["monto", "fecha", "version", "celular", "entidad", "operador", "ner", "reclamo"]
    marcas.sort(key=lambda x: (orden_pri.index(x[2]), -(x[1] - x[0])))
    elegidas = []
    for s, e, t, l in marcas:
        if all(e <= s2 or s >= e2 for s2, e2, _, _ in elegidas):
            elegidas.append((s, e, t, l))
    return sorted(elegidas)


def html_anotado(texto, marcas):
    partes, pos = [], 0
    for s, e, tipo, etiqueta in marcas:
        partes.append(html.escape(texto[pos:s]))
        color = ETIQUETAS[tipo][1]
        partes.append(f'<mark style="--c:{color}">{html.escape(texto[s:e])}'
                      f'<span class="lbl">{html.escape(etiqueta)}</span></mark>')
        pos = e
    partes.append(html.escape(texto[pos:]))
    return '<div class="anotado">' + "".join(partes) + "</div>"


def html_leyenda(marcas):
    tipos = [t for t in ETIQUETAS if any(m[2] == t for m in marcas)]
    if not tipos:
        return '<div class="leyenda">No se encontraron montos, fechas, entidades ni palabras de reclamo.</div>'
    return '<div class="leyenda">' + "".join(
        f'<span style="--c:{ETIQUETAS[t][1]}">{ETIQUETAS[t][0]}</span>' for t in tipos) + "</div>"


# ---------------------------------------------------------------- piezas visuales y tablas UI/UX
def estrellas(valor):
    return f'<span class="stars" style="--p:{max(0, min(valor, 5)) / 5 * 100:.0f}%"><span>★★★★★</span></span>'


def chip_sent(s):
    clase = {"negativo": "neg", "neutro": "neu", "positivo": "pos"}.get(s, "")
    return f'<span class="chip {clase}">{html.escape(str(s))}</span>'


def pill_banco(nombre_app):
    pal = obtener_paleta_banco(nombre_app)
    return (f'<span class="bank-pill" style="--c:{pal["primario"]};--borde:{pal["borde"]}">'
            f'<span class="dot"></span>{html.escape(str(nombre_app))}</span>')


def tarjeta_resena(fila, mostrar_sim=False):
    pal = obtener_paleta_banco(fila["app"])
    color = pal["primario"]
    fecha = pd.to_datetime(fila["fecha"]).strftime("%d/%m/%Y") if pd.notna(fila.get("fecha")) else ""
    sim = ""
    if mostrar_sim:
        sim = (f'<span class="sim">sim {fila["similitud"]:.2f}'
               f'<span class="track"><span style="width:{max(fila["similitud"], 0) * 100:.0f}%"></span></span></span>')
    tipo = fila.get("categoria_principal", "")
    chip_tipo = f'<span class="chip">{html.escape(tipo)}</span>' if tipo and tipo != SIN_CATEGORIA else ""
    return (f'<div class="rcard" style="--c:{color};--borde:{pal["borde"]}">'
            f'<div class="cab"><span class="app-badge"><span class="dot"></span>{html.escape(str(fila["app"]))}</span>'
            f'{estrellas(fila["score"])}<span class="date-pill">{fecha}</span>{sim}</div>'
            f'<div class="txt">{html.escape(str(fila["content"]))}</div>'
            f'{chip_sent(fila["sentimiento"])}{chip_tipo}</div>')


def kpis(items):
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items) + "</div>"


def titulo(texto, nota=None, color="#38BDF8"):
    sub = f'<div class="sec-sub">{nota}</div>' if nota else ""
    st.html(f'<div class="sec-hdr" style="border-left-color:{color}">'
            f'<div class="sec-t">{html.escape(texto)}</div>{sub}</div>')


def mostrar(fig, alto, key):
    fig.update_layout(height=alto)
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False})


def tabla_info_ui(info_dict):
    """Tabla Dark Glass para la información extraída en la pestaña 1."""
    filas_html = []
    for k, valores in info_dict.items():
        if valores:
            chips = "".join(f'<span class="tok cambio">{html.escape(str(v))}</span>' for v in valores)
        else:
            chips = '<span class="empty-val">Sin coincidencias</span>'
        filas_html.append(
            f'<tr><td style="width:42%"><span class="tech-badge">{html.escape(str(k))}</span></td>'
            f'<td>{chips}</td></tr>'
        )
    return (
        '<div class="ui-table-wrap"><table class="ui-table">'
        '<thead><tr><th>Campo extraído</th><th>Valores detectados</th></tr></thead>'
        f'<tbody>{"".join(filas_html)}</tbody></table></div>'
    )


def tabla_lote_ui(res_df, col_texto, limite_vista=150):
    """Tabla Dark Glass con barras de confianza y chips para los resultados de archivo en la pestaña 2."""
    sub = res_df.head(limite_vista)
    filas_html = []
    for _, r in sub.iterrows():
        txt = html.escape(str(r.get(col_texto, "")))
        if len(txt) > 180:
            txt = txt[:180] + "…"
        conf = float(r.get("confianza", 0))
        tipo = str(r.get("tipo_reclamo", ""))
        tipo_html = (f'<span class="chip">{html.escape(tipo)}</span>'
                     if tipo and tipo != SIN_CATEGORIA else '<span class="empty-val">Sin categoría</span>')
        montos = html.escape(str(r.get("montos", ""))) or "—"
        disp = html.escape(str(r.get("dispositivos", ""))) or "—"
        filas_html.append(
            f'<tr>'
            f'<td style="min-width:260px;max-width:440px">{txt}</td>'
            f'<td>{chip_sent(r.get("sentimiento", ""))}</td>'
            f'<td><div class="conf-wrap"><span class="conf-bar"><span style="width:{conf * 100:.0f}%"></span></span>'
            f'<span class="num-cell">{conf:.0%}</span></div></td>'
            f'<td>{tipo_html}</td>'
            f'<td class="num-cell">{montos}</td>'
            f'<td>{disp}</td>'
            f'</tr>'
        )
    return (
        '<div class="ui-table-wrap"><div class="ui-table-scroll">'
        '<table class="ui-table"><thead><tr>'
        '<th>Texto analizado</th><th>Sentimiento</th><th>Confianza</th>'
        '<th>Reclamo principal</th><th>Montos</th><th>Dispositivo</th>'
        f'</tr></thead><tbody>{"".join(filas_html)}</tbody></table></div></div>'
    )


def tabla_temas_ui(etiquetas_t, palabras_t):
    """Tabla Dark Glass para los tópicos LDA y sus palabras principales."""
    filas_html = []
    for k, (nom, pals) in enumerate(zip(etiquetas_t, palabras_t)):
        chips = "".join(f'<span class="tok">{html.escape(str(p))}</span>' for p in pals)
        filas_html.append(
            f'<tr>'
            f'<td style="width:80px"><span class="topic-badge">T{k}</span></td>'
            f'<td style="font-weight:700;color:#F8FAFC;white-space:nowrap">{html.escape(str(nom))}</td>'
            f'<td>{chips}</td>'
            f'</tr>'
        )
    return (
        '<div class="ui-table-wrap"><table class="ui-table">'
        '<thead><tr><th>Código</th><th>Nombre del tema</th><th>Palabras principales del modelo LDA</th></tr></thead>'
        f'<tbody>{"".join(filas_html)}</tbody></table></div>'
    )


def tabla_resenas_ui(vista_df, limite=250):
    """Tabla interactiva Dark Glass con scroll para explorar reseñas en la pestaña 3."""
    sub = vista_df.head(limite)
    filas_html = []
    for _, r in sub.iterrows():
        fecha = pd.to_datetime(r["fecha"]).strftime("%d/%m/%Y") if pd.notna(r.get("fecha")) else ""
        tipo = str(r.get("categoria_principal", ""))
        chip_t = (f'<span class="chip">{html.escape(tipo)}</span>'
                  if tipo and tipo != SIN_CATEGORIA else '<span class="empty-val">Sin categoría</span>')
        filas_html.append(
            f'<tr>'
            f'<td><span class="date-pill">{fecha}</span></td>'
            f'<td>{pill_banco(r["app"])}</td>'
            f'<td style="white-space:nowrap">{estrellas(r["score"])}</td>'
            f'<td>{chip_sent(r["sentimiento"])}</td>'
            f'<td>{chip_t}</td>'
            f'<td style="min-width:280px">{html.escape(str(r["content"]))}</td>'
            f'</tr>'
        )
    return (
        '<div class="ui-table-wrap"><div class="ui-table-scroll">'
        '<table class="ui-table"><thead><tr>'
        '<th>Fecha</th><th>Banco / App</th><th>Calificación</th>'
        '<th>Sentimiento</th><th>Tipo de reclamo</th><th>Reseña del usuario</th>'
        f'</tr></thead><tbody>{"".join(filas_html)}</tbody></table></div></div>'
    )


def tabla_datos_bancos_ui(lista_apps, metricas_dict):
    """Tabla Dark Glass para el resumen de datos por banco en la pestaña 5."""
    max_n = max((metricas_dict["por_app"].get(a, 0) for a in lista_apps), default=1) or 1
    filas_html = []
    for a in lista_apps:
        pal = obtener_paleta_banco(a)
        n = metricas_dict["por_app"].get(a, 0)
        pct = n / max_n * 100
        desde, hasta = metricas_dict["periodo"].get(a, ["—", "—"])
        filas_html.append(
            f'<tr>'
            f'<td>{pill_banco(a)}</td>'
            f'<td><div class="vol-cell"><span class="num-cell">{n:,}</span>'
            f'<span class="vol-bar" style="--grad:{pal["gradiente"]}"><span style="width:{pct:.0f}%"></span></span></div></td>'
            f'<td><span class="date-pill">{html.escape(str(desde))}</span></td>'
            f'<td><span class="date-pill">{html.escape(str(hasta))}</span></td>'
            f'</tr>'
        )
    return (
        '<div class="ui-table-wrap"><table class="ui-table">'
        '<thead><tr><th>App financiera</th><th>Volumen de reseñas</th><th>Desde</th><th>Hasta</th></tr></thead>'
        f'<tbody>{"".join(filas_html)}</tbody></table></div>'
    )


def tabla_tecnicas_ui(filas_tec):
    """Tabla Dark Glass para el resumen de técnicas de PLN en la pestaña 5."""
    filas_html = "".join(
        f'<tr><td style="width:34%"><span class="tech-badge">{html.escape(t)}</span></td>'
        f'<td style="color:#E2E8F0">{html.escape(d)}</td></tr>'
        for t, d in filas_tec
    )
    return (
        '<div class="ui-table-wrap"><table class="ui-table">'
        '<thead><tr><th>Técnica de PLN</th><th>Aplicación en el pipeline</th></tr></thead>'
        f'<tbody>{filas_html}</tbody></table></div>'
    )


def tabla_generica_ui(df_in):
    """Convierte cualquier DataFrame pequeño (como pysentimiento) en una tabla Dark Glass."""
    cols = [f'<th>{html.escape(str(c))}</th>' for c in df_in.columns]
    filas_html = []
    for _, row in df_in.iterrows():
        celdas = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                celdas.append(f'<td class="num-cell">{val:.3f}</td>')
            elif isinstance(val, (int, np.integer)):
                celdas.append(f'<td class="num-cell">{val:,}</td>')
            else:
                celdas.append(f'<td>{html.escape(str(val))}</td>')
        filas_html.append(f'<tr>{"".join(celdas)}</tr>')
    return (
        f'<div class="ui-table-wrap"><table class="ui-table">'
        f'<thead><tr>{"".join(cols)}</tr></thead>'
        f'<tbody>{"".join(filas_html)}</tbody></table></div>'
    )


# ---------------------------------------------------------------- gráficos con diseño UI/UX y paletas bancarias
def g_sentimiento(datos, grupo):
    prop = (pd.crosstab(datos[grupo], datos["sentimiento"], normalize="index")
              .reindex(columns=ORDEN_SENT).fillna(0) * 100)
    if grupo == "app":
        prop = prop.reindex([a for a in apps if a in prop.index])
    fig = go.Figure()
    for s in ORDEN_SENT:
        fig.add_bar(
            y=prop.index, x=prop[s], name=TEXTO_SENT.get(s, s), orientation="h",
            marker=dict(color=COLORES_SENT[s], line=dict(color="#0B1120", width=2.2)),
            text=[f"<b>{v:.0f}%</b>" if v >= 8 else "" for v in prop[s]],
            textposition="inside",
            insidetextfont=dict(color="#FFFFFF", size=12),
            hovertemplate="<b>%{y}</b><br>" + TEXTO_SENT.get(s, s) + ": <b>%{x:.1f}%</b><extra></extra>",
        )
    fig.update_layout(
        barmode="stack", barcornerradius=7, bargap=0.30,
        legend=dict(traceorder="normal", y=1.15, x=0),
        xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=13, color="#F8FAFC")),
        margin=dict(t=36, b=10, l=8, r=12),
    )
    return fig, 100 + 48 * len(prop)


def g_barras(serie, color="#38BDF8", sufijo="%"):
    serie = serie.sort_values()
    max_val = float(serie.max()) if len(serie) and serie.max() > 0 else 1.0
    etiqueta = [f"<b>{v:.1f}{sufijo}</b>" if sufijo else f"<b>{v:,.0f}</b>" for v in serie.values]
    colores_barra = [COLORES_APP.get(idx, color) for idx in serie.index]

    fig = go.Figure()
    # Pista de fondo (track) al 100% para dar acabado de barra de progreso UI
    fig.add_bar(
        x=[max_val * 1.18] * len(serie),
        y=serie.index,
        orientation="h",
        marker=dict(color="rgba(148, 163, 184, 0.08)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_bar(
        x=serie.values,
        y=serie.index,
        orientation="h",
        marker=dict(color=colores_barra, line=dict(color="rgba(255,255,255,0.18)", width=1)),
        text=etiqueta,
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color="#F8FAFC", size=12),
        hovertemplate="<b>%{y}</b>: %{x:.1f}" + sufijo + "<extra></extra>",
        showlegend=False,
    )
    fig.update_layout(
        barmode="overlay",
        bargap=0.34,
        barcornerradius=6,
        xaxis=dict(range=[0, max_val * 1.24], showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#E2E8F0", size=12)),
        margin=dict(r=54, t=10, b=10, l=8),
    )
    return fig, 46 + 37 * len(serie)


def g_calor(tabla, formato="{:.0f}%"):
    z_vals = tabla.values.astype(float)
    z_min = float(np.nanmin(z_vals)) if z_vals.size else 0.0
    z_max = float(np.nanmax(z_vals)) if z_vals.size else 1.0
    umbral = z_min + 0.68 * (z_max - z_min)

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=list(tabla.columns),
        y=list(tabla.index),
        colorscale=[
            [0.0, "#0B132B"],
            [0.30, "#1E293B"],
            [0.62, "#1D4ED8"],
            [0.85, "#0284C7"],
            [1.0, "#38BDF8"],
        ],
        hovertemplate="<b>%{y}</b> · %{x}<br>Valor: <b>%{z:.1f}</b><extra></extra>",
        showscale=False,
        xgap=5,
        ygap=5,
    ))
    # Anotaciones con contraste adaptativo por celda
    for i, fila_idx in enumerate(tabla.index):
        for j, col_idx in enumerate(tabla.columns):
            val = z_vals[i, j]
            color_txt = "#070B12" if val >= umbral else "#F8FAFC"
            fig.add_annotation(
                x=col_idx,
                y=fila_idx,
                text=f"<b>{formato.format(val)}</b>",
                showarrow=False,
                font=dict(family="JetBrains Mono, monospace", size=12, color=color_txt),
            )
    fig.update_layout(
        yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=12, color="#F8FAFC")),
        xaxis=dict(showgrid=False, side="top", tickfont=dict(size=12, color="#CBD5E1")),
        margin=dict(t=42, b=10, l=10, r=10),
    )
    return fig, 95 + 52 * len(tabla)


def g_mensual(datos, apps_sel):
    mensual = (datos.assign(mes=datos["fecha"].dt.to_period("M").dt.to_timestamp(),
                            es_neg=datos["sentimiento"] == "negativo")
                    .groupby(["app", "mes"]).agg(n=("es_neg", "size"), pct=("es_neg", "mean")).reset_index())
    mensual = mensual[mensual["n"] >= 30]
    fig = go.Figure()
    finales = []
    for app in [a for a in apps if a in (apps_sel or [])]:
        sub = mensual[mensual["app"] == app]
        if len(sub):
            pal = obtener_paleta_banco(app)
            fig.add_scatter(
                x=sub["mes"],
                y=sub["pct"] * 100,
                name=app,
                mode="lines+markers",
                line=dict(color=pal["primario"], width=3.2, shape="spline", smoothing=0.8),
                marker=dict(size=8, color="#070B12", line=dict(color=pal["primario"], width=2.6)),
                customdata=sub["n"],
                hovertemplate="<b>" + app + "</b>: %{y:.1f}% <span style='color:#94A3B8'>(n=%{customdata})</span><extra></extra>",
            )
            finales.append([app, sub["mes"].iloc[-1], float(sub["pct"].iloc[-1] * 100), pal["primario"]])
    # nombre de cada app al final de su línea, en cápsula con el color del banco
    finales.sort(key=lambda f: f[2])
    if finales:
        finales[0].append(finales[0][2])
    for i in range(1, len(finales)):
        finales[i].append(max(finales[i][2], finales[i - 1][4] + 4.2))
    for app, x, y, col_app, y_lbl in finales:
        fig.add_annotation(
            x=x, y=y_lbl, text=f"<b>{app}</b>", showarrow=False, xanchor="left", xshift=10,
            font=dict(color="#FFFFFF", size=11),
            bgcolor="rgba(15, 23, 42, 0.90)",
            bordercolor=col_app,
            borderwidth=1.4,
            borderpad=3,
        )
    fig.update_layout(
        yaxis=dict(ticksuffix="%", gridcolor=GRILLA, griddash="dot"),
        xaxis=dict(showgrid=False),
        hovermode="x unified",
        showlegend=True,
        legend=dict(y=1.15, x=0),
        margin=dict(r=108, t=36, b=12, l=8),
    )
    return fig


def tabla_tipos(categorias, n_total):
    conteo = categorias.str.split("; ").explode()
    conteo = conteo[conteo.ne("")].value_counts()
    return (conteo / max(n_total, 1) * 100).reindex(CATEGORIAS).fillna(0)


# ================================================================ ENCABEZADO VANGUARDISTA
resumen = df.groupby("app").agg(n=("app", "size"), prom=("score", "mean"),
                                neg=("sentimiento", lambda s: (s == "negativo").mean() * 100))

fichas = []
for a in apps:
    pal = obtener_paleta_banco(a)
    svg_icono = svg_emblema_banco(a)
    fichas.append(
        f'<div class="app" style="--c:{pal["primario"]};--grad:{pal["gradiente"]};'
        f'--surf:{pal["superficie"]};--borde:{pal["borde"]};--glow:{pal["glow"]}">'
        f'<div class="app-top">'
        f'<div class="app-id"><span class="app-logo">{svg_icono}</span>'
        f'<div><div class="nom">{html.escape(a)}</div><div class="subtag">{html.escape(pal["tag"])}</div></div></div>'
        f'<span class="pill-n">{resumen.loc[a, "n"]:,}</span>'
        f'</div>'
        f'<div class="prom">{resumen.loc[a, "prom"]:.1f} {estrellas(resumen.loc[a, "prom"])}</div>'
        f'<div class="det"><span>Reseñas negativas</span><b>{resumen.loc[a, "neg"]:.0f}%</b></div>'
        f'<div class="negbar"><span style="width:{resumen.loc[a, "neg"]:.0f}%"></span></div>'
        f'</div>'
    )

st.html(
    f'<div class="hero">'
    f'<div class="hero-badge">PLN · Monitor de Banca Móvil en Perú</div>'
    f'<h1>¿Qué dicen los usuarios de su banca móvil?</h1>'
    f'<p>{metricas["n_resenas"]:,} reseñas de Google Play de las principales apps financieras peruanas, '
    f'analizadas con procesamiento de lenguaje natural: sentimiento, tipo de reclamo, tópicos LDA y búsqueda semántica LSA.</p>'
    f'<div class="apps">{"".join(fichas)}</div>'
    f'</div>'
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":material/edit_note: Analizar una reseña",
    ":material/upload_file: Analizar un archivo",
    ":material/insights: Panorama",
    ":material/manage_search: Buscar reclamos parecidos",
    ":material/school: Sobre el proyecto",
])

# ================================================================ TAB 1: una reseña
with tab1:
    def usar_ejemplo():
        elegido = st.session_state.get("ejemplo")
        if elegido:
            st.session_state["texto"] = EJEMPLOS[elegido]
            st.session_state["analizado"] = EJEMPLOS[elegido]

    st.pills("Prueba con un ejemplo", list(EJEMPLOS), key="ejemplo", on_change=usar_ejemplo)
    texto = st.text_area("Escribe o pega una reseña", key="texto", height=100,
                         placeholder="Ej.: la app no carga desde ayer y no puedo transferir")
    if st.button("Analizar reseña", type="primary", icon=":material/play_arrow:",
                 disabled=not texto.strip()):
        st.session_state["analizado"] = texto

    analizado = st.session_state.get("analizado", "")
    if analizado.strip():
        marcas = anotar(analizado)
        titulo("Lectura anotada", "Detección automática de montos, fechas, entidades, dispositivos y palabras de reclamo.")
        st.html(html_anotado(analizado, marcas))
        st.html(html_leyenda(marcas))
        st.write("")

        etiqueta, proba = predecir_sentimiento([analizado])
        etiqueta, proba = etiqueta[0], proba[0]
        reclamo = clasificar_reclamo(analizado)

        with st.container(border=True):
            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                filas = "".join(
                    f'<div class="pfila"><span>{s}</span><span class="track"><span style="width:{p * 100:.0f}%;'
                    f'background:{COLORES_SENT[s]}"></span></span><b>{p:.0%}</b></div>'
                    for s, p in zip(ORDEN_SENT, proba))
                st.html(f'<div class="res-titulo">Sentimiento (modelo {R["sent"]["modelo"]})</div>'
                        f'<div class="res-valor" style="color:{COLORES_SENT[etiqueta]}">{TEXTO_SENT[etiqueta]}</div>'
                        f'{filas}')
            with c2:
                if reclamo["categorias"]:
                    otros = [c for c in reclamo["categorias"] if c != reclamo["principal"]]
                    nota = ("También menciona: " + ", ".join(otros).lower()) if otros else "Un solo tipo detectado."
                    st.html(f'<div class="res-titulo">Tipo de reclamo (reglas con regex)</div>'
                            f'<div class="res-valor">{reclamo["principal"]}</div><div class="res-nota">{nota}</div>')
                else:
                    st.html('<div class="res-titulo">Tipo de reclamo (reglas con regex)</div>'
                            '<div class="res-valor">Sin categoría</div>'
                            '<div class="res-nota">Ninguna regla coincide con el texto.</div>')
            with c3:
                if etiqueta == "positivo":
                    st.html('<div class="res-titulo">Tema (LDA)</div><div class="res-valor">No aplica</div>'
                            '<div class="res-nota">Los temas se aprendieron solo con reseñas negativas.</div>')
                else:
                    k, p = predecir_tema(analizado)
                    if k is None:
                        st.html('<div class="res-titulo">Tema (LDA)</div><div class="res-valor">Sin tema</div>'
                                '<div class="res-nota">El texto no tiene palabras del vocabulario del modelo.</div>')
                    else:
                        palabras = ", ".join(R["temas"]["palabras"][k][:6])
                        st.html(f'<div class="res-titulo">Tema (LDA)</div>'
                                f'<div class="res-valor">T{k}: {etiquetas_temas[k]}</div>'
                                f'<div class="res-nota">Peso en la reseña: {p:.0%}. Palabras del tema: {palabras}.</div>')

        st.write("")
        izq, der = st.columns([1.05, 1], gap="large")
        with izq:
            titulo("Cómo se procesó el texto",
                   f"Pipeline paso a paso · El modelo usa {R['sent']['preprocesamiento'].lower()}.",
                   color="#6366F1")
            limpio = limpiar_texto(analizado)
            tokens = tokenizar(limpio)
            normal = normalizar_jerga(tokens)
            sin_stop = quitar_stopwords(normal)
            with nlp.select_pipes(disable=[p for p in ("parser", "ner") if p in nlp.pipe_names]):
                lemas = _lemas_doc(nlp(" ".join(normal)))
            stems = [quitar_tildes(t) for t in stemizar(sin_stop)]

            def toks(lista, clase=lambda t: ""):
                return "".join(f'<span class="tok {clase(t)}">{html.escape(t)}</span>' for t in lista)

            fuera = lambda t: "fuera" if (t in STOPWORDS or len(t) <= 1) else ""
            pasos = [
                ("Texto original", "La reseña tal como la escribió el usuario.",
                 f'<div class="orig">{html.escape(analizado)}</div>'),
                ("Limpieza con regex", "Minúsculas, emojis a texto, sin URLs, números ni signos.",
                 f'<div class="orig">{html.escape(limpio)}</div>'),
                ("Tokenización", "El texto se separa en palabras.", toks(tokens)),
                ("Normalización de jerga", "Las abreviaturas se cambian por la palabra completa (resaltadas).",
                 toks(normal, lambda t: "cambio" if t not in tokens else "")),
                ("Stopwords", "Se tachan las palabras sin contenido. Se conservan no, nunca, bien, mal...",
                 toks(normal, fuera)),
                ("Lematización (spaCy)", "Cada palabra pasa a su forma base y se quitan las tildes.", toks(lemas)),
                ("Stemming (Snowball)", "Otra opción: cortar cada palabra a su raíz.", toks(stems)),
            ]
            st.html('<div class="pasos-card">' + "".join(
                f'<div class="paso"><span class="n">{i}</span><div><div class="t">{t}</div>'
                f'<div class="d">{d}</div>{c}</div></div>' for i, (t, d, c) in enumerate(pasos, 1)
            ) + '</div>')

        with der:
            titulo("Reseñas parecidas en la base",
                   "Búsqueda con LSA y similitud coseno: encuentra reclamos equivalentes aunque usen otras palabras.")
            parecidas = similares(analizado, 5)
            if len(parecidas):
                st.html("".join(tarjeta_resena(f, mostrar_sim=True) for _, f in parecidas.iterrows()))
            else:
                st.info("No hay reseñas parecidas en la base.")

            titulo("Información extraída", "Entidades nombradas (NER) y patrones estructurados.", color="#10B981")
            info = extraer_informacion(analizado, nlp=nlp)
            st.html(tabla_info_ui(info))
    else:
        st.info("Elige un ejemplo o escribe una reseña y presiona **Analizar reseña**.", icon=":material/arrow_upward:")

# ================================================================ TAB 2: archivo
with tab2:
    st.markdown("Sube un archivo con reseñas o comentarios para analizarlos en lote. "
                "Acepta **CSV**, **Excel** o **TXT** (en TXT, una reseña por línea).")
    archivo = st.file_uploader("Archivo", type=["csv", "xlsx", "txt"], label_visibility="collapsed")
    if archivo is not None:
        try:
            if archivo.name.endswith(".txt"):
                lineas = io.StringIO(archivo.getvalue().decode("utf-8", errors="ignore")).read().splitlines()
                datos = pd.DataFrame({"texto": [l.strip() for l in lineas if l.strip()]})
            elif archivo.name.endswith(".xlsx"):
                datos = pd.read_excel(archivo)
            else:
                datos = pd.read_csv(archivo, encoding_errors="ignore", sep=None, engine="python")
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            datos = None

        if datos is not None and len(datos):
            columnas = list(datos.columns)
            sugerida = next((c for c in columnas if str(c).lower() in {"content", "texto", "reseña", "resena",
                                                                        "comentario", "review"}), columnas[0])
            c1, c2 = st.columns(2)
            col_texto = c1.selectbox("Columna con el texto", columnas, index=columnas.index(sugerida))
            col_grupo = c2.selectbox("Columna para agrupar (opcional)", ["(ninguna)"] + columnas)
            LIMITE = 5000
            if len(datos) > LIMITE:
                st.warning(f"El archivo tiene {len(datos):,} filas; se analizarán las primeras {LIMITE:,}.")
                datos = datos.head(LIMITE)

            if st.button("Analizar archivo", type="primary", icon=":material/play_arrow:"):
                textos = datos[col_texto].fillna("").astype(str).tolist()
                with st.spinner(f"Analizando {len(textos):,} textos..."):
                    etiquetas, proba = predecir_sentimiento(textos)
                    reclamos = [clasificar_reclamo(t) for t in textos]
                    infos = [extraer_informacion(t, usar_ner=False) for t in textos]
                res = datos.copy()
                res["sentimiento"] = etiquetas
                res["confianza"] = proba.max(axis=1).round(3)
                res["tipo_reclamo"] = [r["principal"] for r in reclamos]
                res["categorias"] = ["; ".join(r["categorias"]) for r in reclamos]
                res["montos"] = [", ".join(i["Montos"]) for i in infos]
                res["dispositivos"] = [", ".join(i["Dispositivos / sistema"]) for i in infos]
                st.session_state["resultado_archivo"] = (res, col_grupo, col_texto)

    if "resultado_archivo" in st.session_state:
        guardado = st.session_state["resultado_archivo"]
        res, col_grupo = guardado[0], guardado[1]
        col_txt_usada = guardado[2] if len(guardado) > 2 else res.columns[0]
        st.html(kpis([
            (f"{len(res):,}", "textos analizados"),
            (f"{(res['sentimiento'] == 'negativo').mean():.0%}", "negativos"),
            (f"{(res['sentimiento'] == 'positivo').mean():.0%}", "positivos"),
            (f"{(res['tipo_reclamo'] != SIN_CATEGORIA).mean():.0%}", "con algún tipo de reclamo"),
        ]))
        g1, g2 = st.columns(2, gap="large")
        with g1:
            titulo("Sentimiento predicho", "Distribución porcentual del lote procesado.")
            dist = res["sentimiento"].value_counts(normalize=True).reindex(ORDEN_SENT).fillna(0) * 100
            fig = go.Figure(go.Bar(
                x=[TEXTO_SENT.get(s, s) for s in ORDEN_SENT],
                y=dist.values,
                marker=dict(color=[COLORES_SENT[s] for s in ORDEN_SENT],
                            line=dict(color="rgba(255,255,255,0.2)", width=1.2)),
                text=[f"<b>{v:.0f}%</b>" for v in dist.values],
                textposition="outside",
                textfont=dict(color=TINTA, size=12),
                cliponaxis=False,
                hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>",
            ))
            fig.update_layout(yaxis=dict(ticksuffix="%", range=[0, dist.max() * 1.25 + 2]),
                              xaxis=dict(showgrid=False), bargap=0.46)
            mostrar(fig, 310, "dist_archivo")
        with g2:
            titulo("Tipos de reclamo", "Porcentaje de los textos negativos que mencionan cada tipo.", color="#F43F5E")
            neg_arch = res[res["sentimiento"] == "negativo"]
            fig, alto = g_barras(tabla_tipos(neg_arch["categorias"], len(neg_arch)), color="#F43F5E")
            mostrar(fig, max(alto, 310), "tipos_archivo")
        if col_grupo != "(ninguna)" and col_grupo in res.columns:
            titulo(f"Sentimiento por {col_grupo}", "Comparativa entre los principales grupos del archivo.")
            top = res[col_grupo].astype(str).value_counts().head(12).index
            sub = res[res[col_grupo].astype(str).isin(top)].assign(_g=lambda d: d[col_grupo].astype(str))
            fig, alto = g_sentimiento(sub, "_g")
            mostrar(fig, alto, "grupo_archivo")
        titulo("Resultados por texto", f"Vista de tabla estilizada (primeros {min(len(res), 150)} registros).")
        st.html(tabla_lote_ui(res, col_txt_usada))
        st.download_button("Descargar resultados completos (CSV)", res.to_csv(index=False).encode("utf-8"),
                           file_name="resultados_pln.csv", mime="text/csv", icon=":material/download:")

# ================================================================ TAB 3: panorama
with tab3:
    f1, f2 = st.columns([3, 2], gap="large")
    sel_apps = f1.pills("Apps", apps, selection_mode="multi", default=apps, key="apps_dash")
    fmin, fmax = df["fecha"].min().date(), df["fecha"].max().date()
    rango = f2.slider("Periodo", min_value=fmin, max_value=fmax, value=(fmin, fmax), format="DD/MM/YYYY")
    d = df[df["app"].isin(sel_apps or []) & df["fecha"].dt.date.between(*rango)]
    d_neg = d[d["sentimiento"] == "negativo"]

    if d.empty:
        st.info("No hay reseñas con esos filtros. Elige al menos una app o amplía el periodo.")
    else:
        st.html(kpis([
            (f"{len(d):,}", "reseñas en el periodo"),
            (f"{d['score'].mean():.2f} {estrellas(d['score'].mean())}", "estrellas en promedio"),
            (f"{len(d_neg) / len(d):.0%}", "reseñas negativas (1 o 2 estrellas)"),
            (f"{d_neg['respondida'].mean():.0%}" if len(d_neg) else "—", "negativas con respuesta del banco"),
        ]))

        c1, c2 = st.columns(2, gap="large")
        with c1:
            titulo("Sentimiento por app", "Según las estrellas: 1-2 negativo, 3 neutro, 4-5 positivo.")
            fig, alto = g_sentimiento(d, "app")
            mostrar(fig, max(alto, 330), "sent_app")
        with c2:
            titulo("Reseñas negativas por mes", "Trazos con la paleta oficial de cada banco (meses con ≥30 reseñas).")
            mostrar(g_mensual(d, sel_apps), 330, "mensual")

        if len(d_neg):
            filas = {}
            for app in [a for a in apps if a in sel_apps]:
                sub = d_neg[d_neg["app"] == app]
                if len(sub):
                    filas[app] = tabla_tipos(sub["categorias"], len(sub))
            titulo("De qué se quejan en cada app",
                   "Porcentaje de las reseñas negativas de cada app que mencionan cada tipo de reclamo. "
                   "Una reseña puede tener varios tipos.",
                   color="#6366F1")
            tabla = pd.DataFrame(filas).T
            tabla.columns = [c.replace(" y ", " y<br>") for c in tabla.columns]
            fig, alto = g_calor(tabla)
            mostrar(fig, alto + 30, "tipos_app")

            c3, c4 = st.columns(2, gap="large")
            with c3:
                titulo("Temas encontrados con LDA", "Tema principal de cada reseña negativa.", color="#818CF8")
                con_tema = d_neg.dropna(subset=["tema"])
                if len(con_tema):
                    peso = con_tema["tema"].astype(int).value_counts(normalize=True) * 100
                    peso.index = [f"T{k}: {etiquetas_temas[k]}" for k in peso.index]
                    fig, alto = g_barras(peso, color="#818CF8")
                    mostrar(fig, alto, "temas")
            with c4:
                titulo("Palabras más repetidas en las quejas", "Lemas, sin nombres de bancos ni palabras vacías.", color="#F43F5E")
                excluir = {"no", "ni", "nada", "sin"} | STOP_DOMINIO
                conteo = Counter(t for txt in d_neg["texto_tema"] for t in txt.split()
                                 if t not in excluir and not t.startswith("emoji_"))
                fig, alto = g_barras(pd.Series(dict(conteo.most_common(12))), color="#F43F5E", sufijo="")
                mostrar(fig, alto, "palabras")

            with st.expander("Ver las palabras principales de cada tema"):
                st.html(tabla_temas_ui(etiquetas_temas, R["temas"]["palabras"]))

        titulo("Explorar reseñas", "Filtra por palabra clave, sentimiento o categoría de reclamo.", color="#10B981")
        b1, b2, b3 = st.columns([3, 1.3, 2])
        buscar = b1.text_input("Buscar una palabra", placeholder="ej. token, comisión, selfie")
        sent_sel = b2.selectbox("Sentimiento", ["todos"] + ORDEN_SENT)
        tipo_sel = b3.selectbox("Tipo de reclamo", ["todos"] + list(CATEGORIAS) + [SIN_CATEGORIA])
        vista = d
        if buscar:
            vista = vista[vista["content"].str.contains(buscar, case=False, regex=False, na=False)]
        if sent_sel != "todos":
            vista = vista[vista["sentimiento"] == sent_sel]
        if tipo_sel != "todos":
            vista = vista[vista["categoria_principal"] == tipo_sel]
        st.caption(f"{len(vista):,} reseñas encontradas (se muestran las 250 más recientes en la tabla)")
        st.html(tabla_resenas_ui(vista.sort_values("fecha", ascending=False), limite=250))

# ================================================================ TAB 4: buscador
with tab4:
    st.markdown("Escribe un problema con tus propias palabras. La app busca las reseñas que cuentan algo "
                "parecido, aunque no usen las mismas palabras.")
    consulta = st.text_input("Problema", placeholder="ej. me descontaron plata y nadie me explica",
                             label_visibility="collapsed", icon=":material/search:")
    q1, q2 = st.columns([3, 1], gap="large")
    apps_q = q1.pills("Buscar en", apps, selection_mode="multi", default=apps, key="apps_q")
    k = q2.slider("Resultados", 3, 20, 8)
    if consulta.strip():
        res = similares(consulta, int(k), apps_q or apps)
        if len(res) == 0:
            st.info("Ninguna palabra de la búsqueda aparece en las reseñas. Prueba con otras palabras.")
        else:
            izq, der = st.columns([1.7, 1], gap="large")
            with izq:
                titulo("Reclamos más similares", f"Top {len(res)} reseñas ordenadas por similitud semántica (LSA).")
                st.html("".join(tarjeta_resena(f, mostrar_sim=True) for _, f in res.iterrows()))
            with der:
                titulo("Sentimiento de estas reseñas", "Distribución por banco dentro de los resultados.")
                fig, alto = g_sentimiento(res, "app")
                mostrar(fig, max(alto, 250), "sent_similares")
                st.caption("La similitud va de 0 a 1. Se calcula con el coseno entre los vectores LSA "
                           "de la búsqueda y de cada reseña.")

# ================================================================ TAB 5: proyecto
with tab5:
    sm = metricas["sentimiento"]
    titulo("El problema", "Contexto del análisis sobre banca móvil en el Perú.")
    st.markdown("Miles de usuarios cuentan en Google Play los problemas que tienen con las apps de sus bancos. "
                "Esa información es pública, pero está en texto libre, con jerga y emojis, y es demasiada para "
                "leerla a mano. Esta app usa PLN para clasificar las reseñas, encontrar los reclamos más "
                "frecuentes y comparar los bancos.")

    titulo("Cómo se hizo", "Arquitectura de 5 etapas del proyecto.", color="#6366F1")
    etapas = [
        ("Recolección", f"{metricas['n_resenas']:,} reseñas públicas de Google Play con google-play-scraper."),
        ("Preprocesamiento", "Regex, emojis a texto, jerga, tokenización, stopwords, lematización y stemming."),
        ("Modelos", "TF-IDF, clasificador de sentimiento, reglas de reclamo, LDA y LSA."),
        ("Análisis", "Comparación entre bancos, temas por app y evolución mensual."),
        ("Aplicación", "Esta app en Streamlit, con los modelos guardados desde el notebook."),
    ]
    st.html('<div class="flujo">' + "".join(
        f'<div><span class="num">Paso {i}</span><b>{t}</b><small>{d}</small></div>'
        for i, (t, d) in enumerate(etapas, 1)) + "</div>")

    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        titulo("Datos recolectados", "Cobertura temporal y volumen por aplicativo bancario.", color="#10B981")
        st.html(tabla_datos_bancos_ui(apps, metricas))
        st.caption("Reseñas en español, de Perú, ordenadas de la más reciente a la más antigua. "
                   "No se guardan nombres ni fotos de los usuarios.")
    with c2:
        titulo("Técnicas de PLN", "Componentes lingüísticos y matemáticos implementados.", color="#38BDF8")
        st.html(tabla_tecnicas_ui([
            ["Preprocesamiento", "Regex, emojis, jerga, tokenización, stopwords, lematización (spaCy), stemming (Snowball)"],
            ["TF-IDF", "Cada reseña como vector de unigramas y bigramas"],
            ["Análisis de sentimiento", f"{sm['modelo']} con {sm['preprocesamiento'].lower()}"],
            ["Tipo de reclamo", f"{len(CATEGORIAS)} tipos definidos con expresiones regulares"],
            ["Modelado de temas", f"LDA con {metricas['temas']['n_temas']} temas sobre reseñas negativas"],
            ["Similitud semántica", f"LSA ({metricas['lsa']['componentes']} componentes) y similitud coseno"],
            ["Extracción de información", "Montos, fechas, versiones, celulares y entidades (regex) + NER de spaCy"],
        ]))

    titulo("Qué tan bien funciona el modelo de sentimiento", "Desempeño en validación cruzada y conjunto de prueba.")
    st.html(kpis([
        (f"{sm['f1_macro_test']:.3f}", "F1 macro en la prueba (20 % de los datos)"),
        (f"{sm['exactitud_test']:.3f}", "exactitud en la prueba"),
        (f"{sm['f1_macro_cv']:.3f}", "F1 macro en validación cruzada"),
    ]))
    titulo("F1 macro según el preprocesamiento y el modelo", "Validación cruzada de 5 particiones.", color="#6366F1")
    comp = pd.DataFrame(sm["comparacion"]).pivot(index="preprocesamiento", columns="modelo", values="f1_macro")
    comp = comp.reindex([v for v in ["Sin preprocesar", "Limpieza + stopwords", "Stemming", "Lematización"]
                         if v in comp.index])
    fig, alto = g_calor(comp, "{:.3f}")
    mostrar(fig, alto + 20, "comparacion")
    if sm.get("pysentimiento"):
        titulo("Comparación con un modelo preentrenado", "pysentimiento (RoBERTuito) en una muestra de prueba.", color="#10B981")
        st.html(tabla_generica_ui(pd.DataFrame(sm["pysentimiento"])))

    titulo("Limitaciones", "Consideraciones metodológicas del estudio.", color="#F43F5E")
    st.markdown("- La etiqueta de sentimiento sale de las estrellas, y hay usuarios que ponen 5 estrellas con una queja.\n"
                "- Se descargaron las reseñas más recientes, así que el periodo no es igual para todas las apps.\n"
                "- Las reglas de reclamo solo detectan las palabras que se definieron.\n"
                "- Las reseñas de Google Play no representan a todos los clientes, solo a los que escriben.")
    st.caption(f"Modelos entrenados el {metricas['fecha_ejecucion']}. Proyecto PC3 de Procesamiento de "
               "Lenguaje Natural, UNALM.")