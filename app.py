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
from src.estilo import COLORES_APP, COLORES_SENT, ORDEN_SENT, TINTA, TINTA_SEC, GRILLA

st.set_page_config(page_title="Reseñas de banca móvil", page_icon="💬", layout="wide")

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
# Etiquetas de la lectura anotada: clave -> (nombre visible, color)
ETIQUETAS = {
    "reclamo": ("palabra de reclamo", "#d9480f"),
    "monto": ("monto", "#2b8a3e"),
    "fecha": ("fecha o tiempo", "#1971c2"),
    "version": ("versión", "#5c940d"),
    "celular": ("celular o sistema", "#7048e8"),
    "entidad": ("entidad financiera", "#c2255c"),
    "operador": ("operador móvil", "#0b7285"),
    "ner": ("NER: persona, organización o lugar", "#495057"),
}
NER_NOMBRE = {"PER": "persona", "ORG": "organización", "LOC": "lugar"}
TEXTO_SENT = {"negativo": "Negativo", "neutro": "Neutro", "positivo": "Positivo"}

# ---------------------------------------------------------------- estilos
CSS = """
<style>
:root {
  --linea: color-mix(in srgb, currentColor 13%, transparent);
  --suave: color-mix(in srgb, currentColor 4%, transparent);
  --sec: color-mix(in srgb, currentColor 66%, transparent);
  --primario: #3d3a9e;
  --estrella: #e8a317;
  --titulo: "Bricolage Grotesque", "Hanken Grotesk", sans-serif;
}
.block-container { padding-top: 4.6rem; max-width: 1240px; }
h1, h2, h3, h4, h5 { letter-spacing: -0.015em; }

/* encabezado */
.hero h1 { font-family: var(--titulo); font-size: clamp(2rem, 4.2vw, 3.1rem); font-weight: 800;
           line-height: 1.04; margin: 0 0 .5rem; padding: 0; max-width: 18ch; }
.hero p { color: var(--sec); font-size: 1.08rem; margin: 0; max-width: 62ch; }
.apps { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 10px; margin: 1.4rem 0 .4rem; }
.app { position: relative; border: 1px solid var(--linea); border-radius: 14px;
       padding: 12px 14px 13px 18px; background: var(--suave); overflow: hidden; }
.app::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 5px; background: var(--c); }
.app .nom { font-weight: 700; font-size: .98rem; }
.app .prom { font-family: var(--titulo); font-size: 1.7rem; font-weight: 700; line-height: 1.15;
             margin-top: 2px; }
.app .det { font-size: .8rem; color: var(--sec); margin-top: 6px; }
.negbar { height: 5px; border-radius: 3px; background: var(--linea); margin-top: 4px; overflow: hidden; }
.negbar > span { display: block; height: 100%; background: #e34948; border-radius: 3px; }

/* estrellas */
.stars { position: relative; display: inline-block; font-size: .95rem; letter-spacing: 1px;
         color: color-mix(in srgb, currentColor 20%, transparent); line-height: 1; vertical-align: 2px; }
.stars::before { content: "★★★★★"; }
.stars > span { position: absolute; inset: 0 auto 0 0; width: var(--p); overflow: hidden;
                white-space: nowrap; color: var(--estrella); }

/* lectura anotada */
.anotado { font-size: 1.22rem; line-height: 2.25; padding: 14px 22px 16px;
           border-left: 4px solid var(--primario); background: var(--suave);
           border-radius: 0 14px 14px 0; margin: .3rem 0 .6rem; }
.anotado mark { background: color-mix(in srgb, var(--c) 16%, transparent); color: inherit;
                padding: 3px 4px; border-radius: 5px; box-decoration-break: clone;
                -webkit-box-decoration-break: clone; }
.anotado mark .lbl { font-size: .66rem; font-weight: 700; margin-left: 5px; padding: 2px 6px;
                     border-radius: 4px; background: var(--c); color: #fff; vertical-align: 2px;
                     white-space: nowrap; }
.leyenda { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .82rem; color: var(--sec); }
.leyenda span::before { content: ""; display: inline-block; width: 10px; height: 10px;
                        border-radius: 3px; background: var(--c); margin-right: 6px; vertical-align: -1px; }

/* resultados */
.res-titulo { font-size: .88rem; color: var(--sec); margin-bottom: 4px; }
.res-valor { font-family: var(--titulo); font-size: 1.65rem; font-weight: 700; line-height: 1.15; }
.res-nota { font-size: .85rem; color: var(--sec); margin-top: 6px; }
.pfila { display: grid; grid-template-columns: 74px 1fr 42px; align-items: center; gap: 8px;
         font-size: .84rem; margin-top: 7px; }
.pfila .track { height: 8px; border-radius: 4px; background: var(--linea); overflow: hidden; }
.pfila .track > span { display: block; height: 100%; border-radius: 4px; }
.pfila b { text-align: right; font-weight: 600; }

/* chips */
.chip { display: inline-block; font-size: .78rem; padding: 2px 10px; border-radius: 999px;
        border: 1px solid var(--linea); background: var(--suave); margin: 4px 6px 0 0; }
.chip.neg { background: color-mix(in srgb, #e34948 13%, transparent); border-color: color-mix(in srgb, #e34948 35%, transparent); }
.chip.neu { background: color-mix(in srgb, #9b9a94 15%, transparent); }
.chip.pos { background: color-mix(in srgb, #2a78d6 13%, transparent); border-color: color-mix(in srgb, #2a78d6 35%, transparent); }

/* pasos del preprocesamiento */
.paso { display: grid; grid-template-columns: 30px 1fr; gap: 12px; padding: 11px 0;
        border-bottom: 1px dashed var(--linea); }
.paso:last-child { border-bottom: none; }
.paso .n { width: 26px; height: 26px; border-radius: 50%; background: var(--primario); color: #fff;
           font-weight: 700; font-size: .8rem; display: grid; place-items: center; margin-top: 1px; }
.paso .t { font-weight: 600; font-size: .92rem; }
.paso .d { font-size: .8rem; color: var(--sec); margin-bottom: 3px; }
.paso .orig { font-size: .92rem; }
.tok { display: inline-block; padding: 1px 8px; margin: 4px 4px 0 0; border-radius: 6px;
       background: var(--suave); border: 1px solid var(--linea); font-size: .84rem; }
.tok.fuera { text-decoration: line-through; opacity: .42; }
.tok.cambio { border-color: var(--primario); background: color-mix(in srgb, var(--primario) 10%, transparent); }

/* tarjetas de reseña */
.rcard { border: 1px solid var(--linea); border-left: 4px solid var(--c); border-radius: 4px 12px 12px 4px;
         padding: 11px 16px 12px; margin-bottom: 10px; }
.rcard .cab { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; font-size: .82rem; color: var(--sec); }
.rcard .cab .app-n { font-weight: 700; color: inherit; font-size: .9rem; }
.rcard .txt { margin: 6px 0 2px; line-height: 1.5; font-size: .96rem; }
.sim { display: inline-flex; align-items: center; gap: 6px; margin-left: auto; }
.sim .track { width: 70px; height: 5px; border-radius: 3px; background: var(--linea); overflow: hidden; }
.sim .track > span { display: block; height: 100%; background: var(--primario); }

/* indicadores */
.kpis { display: flex; flex-wrap: wrap; border: 1px solid var(--linea); border-radius: 14px; margin: .4rem 0 1rem; }
.kpi { flex: 1 1 170px; padding: 13px 20px 14px; border-right: 1px solid var(--linea); }
.kpi:last-child { border-right: none; }
.kpi .v { font-family: var(--titulo); font-size: 1.85rem; font-weight: 700; line-height: 1.12; }
.kpi .l { font-size: .84rem; color: var(--sec); }

/* recorrido del proyecto */
.flujo { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0; margin: .4rem 0 1rem; }
.flujo div { padding: 10px 16px 12px 0; border-top: 3px solid var(--primario); margin-right: 14px; }
.flujo b { display: block; font-family: var(--titulo); font-size: 1.05rem; margin: 6px 0 2px; }
.flujo small { color: var(--sec); font-size: .84rem; line-height: 1.35; display: block; }
.flujo .num { font-size: .8rem; color: var(--primario); font-weight: 700; }

.nota { font-size: .85rem; color: var(--sec); margin: -.2rem 0 .6rem; }
div[data-testid="stTabs"] button p { font-size: .98rem; }
</style>
"""

# Plantilla común para todos los gráficos de Plotly
pio.templates["resenas"] = go.layout.Template(layout=dict(
    font=dict(family="Hanken Grotesk, sans-serif", size=13, color=TINTA),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=list(COLORES_APP.values()),
    xaxis=dict(automargin=True, gridcolor=GRILLA, zeroline=False, linecolor=GRILLA, ticks="", title_font=dict(color=TINTA_SEC),
               tickfont=dict(color=TINTA_SEC)),
    yaxis=dict(automargin=True, ticklabelstandoff=6, gridcolor=GRILLA, zeroline=False, linecolor=GRILLA, ticks="", title_font=dict(color=TINTA_SEC),
               tickfont=dict(color=TINTA)),
    hoverlabel=dict(bgcolor="white", bordercolor=GRILLA, font=dict(family="Hanken Grotesk, sans-serif",
                                                                    color=TINTA, size=13)),
    legend=dict(orientation="h", y=-0.16, x=0, title=None, font=dict(color=TINTA_SEC)),
    margin=dict(l=6, r=10, t=10, b=10),
    bargap=0.42, barcornerradius=4,
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
apps = [a for a in COLORES_APP if a in df["app"].unique()]
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
    marcas = []   # (inicio, fin, tipo, etiqueta)
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
    # si dos marcas se cruzan, gana la de mayor prioridad y luego la más larga
    # las marcas siempre abarcan palabras completas ("autoriz" -> "autorización")
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


# ---------------------------------------------------------------- piezas visuales
def estrellas(valor):
    return f'<span class="stars" style="--p:{max(0, min(valor, 5)) / 5 * 100:.0f}%"><span>★★★★★</span></span>'


def chip_sent(s):
    clase = {"negativo": "neg", "neutro": "neu", "positivo": "pos"}.get(s, "")
    return f'<span class="chip {clase}">{s}</span>'


def tarjeta_resena(fila, mostrar_sim=False):
    color = COLORES_APP.get(fila["app"], "#888")
    fecha = pd.to_datetime(fila["fecha"]).strftime("%d/%m/%Y") if pd.notna(fila.get("fecha")) else ""
    sim = ""
    if mostrar_sim:
        sim = (f'<span class="sim">similitud {fila["similitud"]:.2f}'
               f'<span class="track"><span style="width:{max(fila["similitud"], 0) * 100:.0f}%"></span></span></span>')
    tipo = fila.get("categoria_principal", "")
    chip_tipo = f'<span class="chip">{html.escape(tipo)}</span>' if tipo and tipo != SIN_CATEGORIA else ""
    return (f'<div class="rcard" style="--c:{color}"><div class="cab"><span class="app-n">{fila["app"]}</span>'
            f'{estrellas(fila["score"])}<span>{fecha}</span>{sim}</div>'
            f'<div class="txt">{html.escape(str(fila["content"]))}</div>'
            f'{chip_sent(fila["sentimiento"])}{chip_tipo}</div>')


def kpis(items):
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items) + "</div>"


def titulo(texto, nota=None):
    st.markdown(f"##### {texto}")
    if nota:
        st.html(f'<div class="nota">{nota}</div>')


def mostrar(fig, alto, key):
    fig.update_layout(height=alto)
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False})


# ---------------------------------------------------------------- gráficos
def g_sentimiento(datos, grupo):
    prop = (pd.crosstab(datos[grupo], datos["sentimiento"], normalize="index")
              .reindex(columns=ORDEN_SENT).fillna(0) * 100)
    if grupo == "app":
        prop = prop.reindex([a for a in apps if a in prop.index])
    fig = go.Figure()
    for s in ORDEN_SENT:
        fig.add_bar(y=prop.index, x=prop[s], name=s, orientation="h",
                    marker=dict(color=COLORES_SENT[s], line=dict(color="#fafafc", width=2)),
                    text=[f"{v:.0f}%" if v >= 8 else "" for v in prop[s]], textposition="inside",
                    insidetextfont=dict(color="white"),
                    hovertemplate="%{y}<br>" + s + ": %{x:.1f}%<extra></extra>")
    fig.update_layout(barmode="stack", barcornerradius=0, bargap=0.35, legend=dict(traceorder="normal"),
                      xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False),
                      yaxis=dict(autorange="reversed"))
    return fig, 90 + 46 * len(prop)


def g_barras(serie, color="#3d3a9e", sufijo="%"):
    serie = serie.sort_values()
    etiqueta = [f"{v:.1f}{sufijo}" if sufijo else f"{v:,.0f}" for v in serie.values]
    fig = go.Figure(go.Bar(x=serie.values, y=serie.index, orientation="h", marker_color=color,
                           text=etiqueta, textposition="outside", cliponaxis=False,
                           textfont=dict(color=TINTA_SEC),
                           hovertemplate="%{y}: %{x:.1f}" + sufijo + "<extra></extra>"))
    fig.update_layout(xaxis=dict(range=[0, (serie.max() if len(serie) else 1) * 1.2], showticklabels=False,
                                 showgrid=False), margin=dict(r=40))
    return fig, 40 + 34 * len(serie)


def g_calor(tabla, formato="{:.0f}%"):
    fig = go.Figure(go.Heatmap(
        z=tabla.values, x=list(tabla.columns), y=list(tabla.index),
        colorscale=[[0, "#eef0fb"], [1, "#2d2a85"]],
        text=[[formato.format(v) for v in fila] for fila in tabla.values], texttemplate="%{text}",
        hovertemplate="%{y} · %{x}: %{z:.1f}<extra></extra>", showscale=False, xgap=3, ygap=3))
    fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False, side="top"))
    return fig, 80 + 50 * len(tabla)


def g_mensual(datos, apps_sel):
    mensual = (datos.assign(mes=datos["fecha"].dt.to_period("M").dt.to_timestamp(),
                            es_neg=datos["sentimiento"] == "negativo")
                    .groupby(["app", "mes"]).agg(n=("es_neg", "size"), pct=("es_neg", "mean")).reset_index())
    mensual = mensual[mensual["n"] >= 30]
    fig = go.Figure()
    finales = []
    for app in [a for a in apps if a in apps_sel]:
        sub = mensual[mensual["app"] == app]
        if len(sub):
            fig.add_scatter(x=sub["mes"], y=sub["pct"] * 100, name=app, mode="lines+markers",
                            line=dict(color=COLORES_APP[app], width=2), marker=dict(size=8),
                            customdata=sub["n"],
                            hovertemplate=app + ": %{y:.1f}% (n=%{customdata})<extra></extra>")
            finales.append([app, sub["mes"].iloc[-1], sub["pct"].iloc[-1] * 100])
    # nombre de cada app al final de su línea, sin que se encimen
    finales.sort(key=lambda f: f[2])
    for i in range(1, len(finales)):
        finales[i].append(max(finales[i][2], finales[i - 1][-1] + 3.5))
    if finales:
        finales[0].append(finales[0][2])
    for app, x, y, y_lbl in finales:
        fig.add_annotation(x=x, y=y_lbl, text=app, showarrow=False, xanchor="left", xshift=10,
                           font=dict(color=TINTA, size=12))
    fig.update_layout(yaxis=dict(ticksuffix="%"), hovermode="x unified", showlegend=True,
                      margin=dict(r=90), xaxis=dict(showgrid=False))
    return fig


def tabla_tipos(categorias, n_total):
    conteo = categorias.str.split("; ").explode()
    conteo = conteo[conteo.ne("")].value_counts()
    return (conteo / max(n_total, 1) * 100).reindex(CATEGORIAS).fillna(0)


# ================================================================ ENCABEZADO
resumen = df.groupby("app").agg(n=("app", "size"), prom=("score", "mean"),
                                neg=("sentimiento", lambda s: (s == "negativo").mean() * 100))
fichas = "".join(
    f'<div class="app" style="--c:{COLORES_APP[a]}"><div class="nom">{a}</div>'
    f'<div class="prom">{resumen.loc[a, "prom"]:.1f} {estrellas(resumen.loc[a, "prom"])}</div>'
    f'<div class="det">{resumen.loc[a, "neg"]:.0f}% de reseñas negativas</div>'
    f'<div class="negbar"><span style="width:{resumen.loc[a, "neg"]:.0f}%"></span></div>'
    f'<div class="det">{resumen.loc[a, "n"]:,} reseñas</div></div>'
    for a in apps)
st.html(f'<div class="hero"><h1>¿Qué dicen los usuarios de su banca móvil?</h1>'
        f'<p>{metricas["n_resenas"]:,} reseñas de Google Play de cinco apps peruanas, analizadas con '
        f'procesamiento de lenguaje natural: sentimiento, tipo de reclamo, temas y reseñas parecidas.</p>'
        f'<div class="apps">{fichas}</div></div>')

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
        st.markdown("#### Lectura anotada")
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
            st.markdown("#### Cómo se procesó el texto")
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
            st.html("".join(f'<div class="paso"><span class="n">{i}</span><div><div class="t">{t}</div>'
                            f'<div class="d">{d}</div>{c}</div></div>' for i, (t, d, c) in enumerate(pasos, 1)))
            st.caption(f"El modelo de sentimiento usa la versión: {R['sent']['preprocesamiento'].lower()}.")

        with der:
            st.markdown("#### Reseñas parecidas en la base")
            st.html('<div class="nota">Búsqueda con LSA y similitud coseno: encuentra reseñas que cuentan algo '
                    'parecido aunque usen otras palabras.</div>')
            parecidas = similares(analizado, 5)
            if len(parecidas):
                st.html("".join(tarjeta_resena(f, mostrar_sim=True) for _, f in parecidas.iterrows()))
            else:
                st.info("No hay reseñas parecidas en la base.")

            st.markdown("#### Información extraída")
            info = extraer_informacion(analizado, nlp=nlp)
            filas = [{"dato": k, "valores": ", ".join(v) if v else "—"} for k, v in info.items()]
            st.dataframe(pd.DataFrame(filas), hide_index=True)
    else:
        st.info("Elige un ejemplo o escribe una reseña y presiona **Analizar reseña**.", icon=":material/arrow_upward:")

# ================================================================ TAB 2: archivo
with tab2:
    st.markdown("Sube un archivo con reseñas o comentarios para analizarlos todos juntos. "
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
                st.session_state["resultado_archivo"] = (res, col_grupo)

    if "resultado_archivo" in st.session_state:
        res, col_grupo = st.session_state["resultado_archivo"]
        st.html(kpis([
            (f"{len(res):,}", "textos analizados"),
            (f"{(res['sentimiento'] == 'negativo').mean():.0%}", "negativos"),
            (f"{(res['sentimiento'] == 'positivo').mean():.0%}", "positivos"),
            (f"{(res['tipo_reclamo'] != SIN_CATEGORIA).mean():.0%}", "con algún tipo de reclamo"),
        ]))
        g1, g2 = st.columns(2, gap="large")
        with g1:
            titulo("Sentimiento predicho")
            dist = res["sentimiento"].value_counts(normalize=True).reindex(ORDEN_SENT).fillna(0) * 100
            fig = go.Figure(go.Bar(x=ORDEN_SENT, y=dist.values, marker_color=[COLORES_SENT[s] for s in ORDEN_SENT],
                                   text=[f"{v:.0f}%" for v in dist.values], textposition="outside",
                                   cliponaxis=False, hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
            fig.update_layout(yaxis=dict(ticksuffix="%", range=[0, dist.max() * 1.2 + 1]), bargap=0.5)
            mostrar(fig, 300, "dist_archivo")
        with g2:
            titulo("Tipos de reclamo", "Porcentaje de los textos negativos que mencionan cada tipo.")
            neg_arch = res[res["sentimiento"] == "negativo"]
            fig, alto = g_barras(tabla_tipos(neg_arch["categorias"], len(neg_arch)))
            mostrar(fig, alto, "tipos_archivo")
        if col_grupo != "(ninguna)" and col_grupo in res.columns:
            titulo(f"Sentimiento por {col_grupo}")
            top = res[col_grupo].astype(str).value_counts().head(12).index
            sub = res[res[col_grupo].astype(str).isin(top)].assign(_g=lambda d: d[col_grupo].astype(str))
            fig, alto = g_sentimiento(sub, "_g")
            mostrar(fig, alto, "grupo_archivo")
        titulo("Resultados por texto")
        st.dataframe(res, hide_index=True)
        st.download_button("Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8"),
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
            (f"{len(d):,}", "reseñas"),
            (f"{d['score'].mean():.2f} {estrellas(d['score'].mean())}", "estrellas en promedio"),
            (f"{len(d_neg) / len(d):.0%}", "reseñas negativas (1 o 2 estrellas)"),
            (f"{d_neg['respondida'].mean():.0%}" if len(d_neg) else "—", "negativas con respuesta del banco"),
        ]))

        c1, c2 = st.columns(2, gap="large")
        with c1:
            titulo("Sentimiento por app", "Según las estrellas: 1-2 negativo, 3 neutro, 4-5 positivo.")
            fig, alto = g_sentimiento(d, "app")
            mostrar(fig, max(alto, 300), "sent_app")
        with c2:
            titulo("Reseñas negativas por mes", "Solo meses con 30 reseñas o más por app.")
            mostrar(g_mensual(d, sel_apps), 330, "mensual")

        if len(d_neg):
            filas = {}
            for app in [a for a in apps if a in sel_apps]:
                sub = d_neg[d_neg["app"] == app]
                if len(sub):
                    filas[app] = tabla_tipos(sub["categorias"], len(sub))
            titulo("De qué se quejan en cada app",
                   "Porcentaje de las reseñas negativas de cada app que mencionan cada tipo de reclamo. "
                   "Una reseña puede tener varios tipos.")
            tabla = pd.DataFrame(filas).T
            tabla.columns = [c.replace(" y ", " y<br>") for c in tabla.columns]
            fig, alto = g_calor(tabla)
            mostrar(fig, alto + 30, "tipos_app")

            c3, c4 = st.columns(2, gap="large")
            with c3:
                titulo("Temas encontrados con LDA", "Tema principal de cada reseña negativa.")
                con_tema = d_neg.dropna(subset=["tema"])
                if len(con_tema):
                    peso = con_tema["tema"].astype(int).value_counts(normalize=True) * 100
                    peso.index = [f"T{k}: {etiquetas_temas[k]}" for k in peso.index]
                    fig, alto = g_barras(peso, color="#6250d6")
                    mostrar(fig, alto, "temas")
            with c4:
                titulo("Palabras más repetidas en las quejas", "Lemas, sin nombres de bancos ni palabras vacías.")
                excluir = {"no", "ni", "nada", "sin"} | STOP_DOMINIO
                conteo = Counter(t for txt in d_neg["texto_tema"] for t in txt.split()
                                 if t not in excluir and not t.startswith("emoji_"))
                fig, alto = g_barras(pd.Series(dict(conteo.most_common(12))), color="#e34948", sufijo="")
                mostrar(fig, alto, "palabras")

            with st.expander("Ver las palabras principales de cada tema"):
                st.dataframe(pd.DataFrame({"tema": [f"T{k}" for k in range(len(etiquetas_temas))],
                                           "nombre": etiquetas_temas,
                                           "palabras principales": [", ".join(p) for p in R["temas"]["palabras"]]}),
                             hide_index=True)

        titulo("Explorar reseñas")
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
        st.caption(f"{len(vista):,} reseñas encontradas (se muestran hasta 500)")
        st.dataframe(vista.sort_values("fecha", ascending=False)
                          [["fecha", "app", "score", "sentimiento", "categoria_principal", "content"]].head(500),
                     hide_index=True,
                     column_config={"fecha": st.column_config.DateColumn("fecha", format="DD/MM/YYYY"),
                                    "score": st.column_config.NumberColumn("estrellas", format="%d ★"),
                                    "categoria_principal": "tipo de reclamo", "content": "reseña"})

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
                st.html("".join(tarjeta_resena(f, mostrar_sim=True) for _, f in res.iterrows()))
            with der:
                titulo("Sentimiento de estas reseñas")
                fig, alto = g_sentimiento(res, "app")
                mostrar(fig, max(alto, 240), "sent_similares")
                st.caption("La similitud va de 0 a 1. Se calcula con el coseno entre los vectores LSA "
                           "de la búsqueda y de cada reseña.")

# ================================================================ TAB 5: proyecto
with tab5:
    sm = metricas["sentimiento"]
    st.markdown("#### El problema")
    st.markdown("Miles de usuarios cuentan en Google Play los problemas que tienen con las apps de sus bancos. "
                "Esa información es pública, pero está en texto libre, con jerga y emojis, y es demasiada para "
                "leerla a mano. Esta app usa PLN para clasificar las reseñas, encontrar los reclamos más "
                "frecuentes y comparar los bancos.")

    st.markdown("#### Cómo se hizo")
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
        st.markdown("#### Datos")
        st.dataframe(pd.DataFrame([{"app": a, "reseñas": metricas["por_app"].get(a, 0),
                                    "desde": metricas["periodo"][a][0], "hasta": metricas["periodo"][a][1]}
                                   for a in apps]), hide_index=True)
        st.caption("Reseñas en español, de Perú, ordenadas de la más reciente a la más antigua. "
                   "No se guardan nombres ni fotos de los usuarios.")
    with c2:
        st.markdown("#### Técnicas de PLN")
        st.dataframe(pd.DataFrame([
            ["Preprocesamiento", "Regex, emojis, jerga, tokenización, stopwords, lematización (spaCy), stemming (Snowball)"],
            ["TF-IDF", "Cada reseña como vector de unigramas y bigramas"],
            ["Análisis de sentimiento", f"{sm['modelo']} con {sm['preprocesamiento'].lower()}"],
            ["Tipo de reclamo", f"{len(CATEGORIAS)} tipos definidos con expresiones regulares"],
            ["Modelado de temas", f"LDA con {metricas['temas']['n_temas']} temas sobre reseñas negativas"],
            ["Similitud semántica", f"LSA ({metricas['lsa']['componentes']} componentes) y similitud coseno"],
            ["Extracción de información", "Montos, fechas, versiones, celulares y entidades (regex) + NER de spaCy"],
        ], columns=["técnica", "cómo se aplica"]), hide_index=True)

    st.markdown("#### Qué tan bien funciona el modelo de sentimiento")
    st.html(kpis([
        (f"{sm['f1_macro_test']:.3f}", "F1 macro en la prueba (20 % de los datos)"),
        (f"{sm['exactitud_test']:.3f}", "exactitud en la prueba"),
        (f"{sm['f1_macro_cv']:.3f}", "F1 macro en validación cruzada"),
    ]))
    titulo("F1 macro según el preprocesamiento y el modelo", "Validación cruzada de 5 particiones.")
    comp = pd.DataFrame(sm["comparacion"]).pivot(index="preprocesamiento", columns="modelo", values="f1_macro")
    comp = comp.reindex([v for v in ["Sin preprocesar", "Limpieza + stopwords", "Stemming", "Lematización"]
                         if v in comp.index])
    fig, alto = g_calor(comp, "{:.3f}")
    mostrar(fig, alto + 20, "comparacion")
    if sm.get("pysentimiento"):
        titulo("Comparación con un modelo preentrenado", "pysentimiento (RoBERTuito) en una muestra de prueba.")
        st.dataframe(pd.DataFrame(sm["pysentimiento"]), hide_index=True)

    st.markdown("#### Limitaciones")
    st.markdown("- La etiqueta de sentimiento sale de las estrellas, y hay usuarios que ponen 5 estrellas con una queja.\n"
                "- Se descargaron las reseñas más recientes, así que el periodo no es igual para todas las apps.\n"
                "- Las reglas de reclamo solo detectan las palabras que se definieron.\n"
                "- Las reseñas de Google Play no representan a todos los clientes, solo a los que escriben.")
    st.caption(f"Modelos entrenados el {metricas['fecha_ejecucion']}. Proyecto PC3 de Procesamiento de "
               "Lenguaje Natural, UNALM.")
