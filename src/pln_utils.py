"""
pln_utils.py
Preprocesamiento de texto y técnicas de PLN del proyecto.

El notebook (entrenamiento) y la app de Streamlit (predicción) importan este
mismo archivo. Así los dos aplican exactamente los mismos pasos al texto.
"""
import re
import unicodedata
from functools import lru_cache

import emoji
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.es.stop_words import STOP_WORDS

# =====================================================================
# 1. LIMPIEZA Y NORMALIZACIÓN (expresiones regulares)
# =====================================================================
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
RE_MENCION = re.compile(r"@\w+")
RE_ALARGAMIENTO = re.compile(r"([a-záéíóúñ])\1{2,}")        # buenooo -> bueno
RE_RISA = re.compile(r"\b([jh][aei])\1+[jh]?\b|\bx+d+\b")  # jajaja, jejeje, xd
RE_NUMERO = re.compile(r"\d+")
RE_NO_LETRA = re.compile(r"[^a-záéíóúüñ_\s]")
RE_ESPACIOS = re.compile(r"\s+")
RE_TOKEN = re.compile(r"[a-záéíóúüñ_]+")

EMOTICONES = {
    r":-?\)+|:-?d\b|=\)|\(:": " emoji_feliz ",
    r":-?\(+|=\(|\):": " emoji_triste ",
    r"<3": " emoji_corazon ",
}

# Jerga y abreviaturas frecuentes en reseñas peruanas -> forma estándar
JERGA = {
    "q": "que", "k": "que", "ke": "que", "xq": "porque", "xk": "porque",
    "pq": "porque", "porq": "porque", "x": "por", "d": "de", "pa": "para",
    "tb": "también", "tmb": "también", "tbn": "también", "tambn": "también",
    "pls": "por favor", "plis": "por favor", "porfa": "por favor",
    "xfa": "por favor", "xfavor": "por favor", "dsd": "desde",
    "msj": "mensaje", "msje": "mensaje", "sms": "mensaje",
    "app": "aplicación", "apps": "aplicación", "aplicacion": "aplicación",
    "aplicativo": "aplicación", "aplicacio": "aplicación",
    "cel": "celular", "celu": "celular", "fono": "teléfono",
    "bn": "bien", "mxo": "mucho", "mñn": "mañana", "nunk": "nunca",
    "nd": "nada", "toy": "estoy", "cta": "cuenta", "ctas": "cuentas",
    "trx": "transacción", "transf": "transferencia", "nro": "número",
    "num": "número", "info": "información", "actualizacion": "actualización",
    "contrasena": "contraseña", "pesima": "pésima", "pesimo": "pésimo",
}

# spaCy marca como stopwords palabras que cambian el sentido de una reseña
# ("no", "bien", "mal", "nunca"...). Las sacamos de la lista.
CONSERVAR = {
    "no", "ni", "nunca", "nada", "sin", "tampoco", "jamás", "nadie",
    "ninguno", "ninguna", "ningún", "bien", "mal", "buen", "bueno", "buena",
    "buenos", "buenas", "mejor", "mejores", "peor", "peores", "mucho",
    "muchos", "poco", "pocos", "siempre", "gran", "tarde", "demasiado",
    # spaCy también trae "cuenta" como stopword, pero en reseñas de bancos es clave
    "cuenta", "cuentas",
}
EXTRA_STOP = {"pe", "pes", "oe", "ps", "pues", "osea", "etc", "hola", "eh", "ah", "ya"}
STOPWORDS = (set(STOP_WORDS) | EXTRA_STOP) - CONSERVAR

# Palabras del dominio que aparecen en casi todas las reseñas y no aportan
# para distinguir temas (se quitan para el modelo de temas y la similitud)
STOP_DOMINIO = {
    "aplicacion", "app", "banco", "bcp", "yape", "interbank", "bbva",
    "scotiabank", "scotia", "plin", "continental", "risa",
}

_stemmer = SnowballStemmer("spanish")


def quitar_tildes(texto: str) -> str:
    """Quita tildes y diéresis, pero conserva la ñ."""
    texto = texto.replace("ñ", "\0")
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", texto).replace("\0", "ñ")


def _emoji_a_texto(caracter, datos=None):
    """Convierte un emoji en un token, ej. 😡 -> emoji_cara_cabreada."""
    nombre = emoji.demojize(caracter, language="es").strip(":").lower()
    nombre = re.sub(r"[^a-záéíóúüñ]+", "_", nombre).strip("_")
    return f" emoji_{nombre} "


def _reducir_alargamiento(m):
    letra = m.group(1)
    return letra * 2 if letra in "rlc" else letra   # carrrro -> carro


def limpiar_texto(texto) -> str:
    """Limpieza con expresiones regulares: emojis, URLs, alargamientos, números y signos."""
    if not isinstance(texto, str):
        return ""
    for patron, reemplazo in EMOTICONES.items():
        texto = re.sub(patron, reemplazo, texto, flags=re.IGNORECASE)
    texto = emoji.replace_emoji(texto, replace=_emoji_a_texto)
    texto = texto.lower()
    texto = RE_URL.sub(" ", texto)
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_MENCION.sub(" ", texto)
    texto = RE_ALARGAMIENTO.sub(_reducir_alargamiento, texto)
    texto = RE_RISA.sub(" risa ", texto)
    texto = RE_NUMERO.sub(" ", texto)
    texto = RE_NO_LETRA.sub(" ", texto)
    return RE_ESPACIOS.sub(" ", texto).strip()


def tokenizar(texto: str) -> list:
    return RE_TOKEN.findall(texto)


def normalizar_jerga(tokens: list) -> list:
    salida = []
    for t in tokens:
        salida.extend(JERGA.get(t, t).split())
    return salida


def quitar_stopwords(tokens: list, quitar_dominio=False) -> list:
    salida = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if quitar_dominio:
        salida = [t for t in salida if quitar_tildes(t) not in STOP_DOMINIO]
    return salida


def stemizar(tokens: list) -> list:
    return [t if t.startswith("emoji_") else _stemmer.stem(t) for t in tokens]


def texto_normalizado(texto) -> str:
    """Texto limpio, con jerga normalizada y sin tildes (base para reglas regex)."""
    return quitar_tildes(" ".join(normalizar_jerga(tokenizar(limpiar_texto(texto)))))


# =====================================================================
# 2. LEMATIZACIÓN CON spaCy
# =====================================================================
@lru_cache(maxsize=1)
def cargar_spacy(modelo="es_core_news_sm"):
    import spacy
    return spacy.load(modelo)


def _lemas_doc(doc, quitar_dominio=False) -> list:
    lemas = []
    for tok in doc:
        texto = tok.text
        if not texto.strip():
            continue
        if texto.startswith("emoji_"):
            lemas.append(texto)
            continue
        lema = tok.lemma_.lower()
        if texto in CONSERVAR:
            lemas.append(texto)
        elif texto in STOPWORDS or lema in STOPWORDS or len(lema) < 2:
            continue
        else:
            lemas.append(lema)
    lemas = [quitar_tildes(l) for l in lemas]
    if quitar_dominio:
        lemas = [l for l in lemas if l not in STOP_DOMINIO]
    return lemas


def preprocesar_lote(textos, modo="lema", quitar_dominio=False, nlp=None, batch_size=256):
    """
    Preprocesa una lista de textos y devuelve una lista de strings listos para TF-IDF.
    modo = "original" (solo minúsculas), "limpio" (limpieza + stopwords),
           "lema" (lematización) o "stem" (stemming).
    """
    if modo == "original":   # línea base: solo minúsculas, sin ningún otro paso
        return [t.lower() if isinstance(t, str) else "" for t in textos]
    bases = [" ".join(normalizar_jerga(tokenizar(limpiar_texto(t)))) for t in textos]
    if modo == "lema":
        nlp = nlp or cargar_spacy()
        salida = []
        with nlp.select_pipes(disable=[p for p in ("parser", "ner") if p in nlp.pipe_names]):
            for doc in nlp.pipe(bases, batch_size=batch_size):
                salida.append(" ".join(_lemas_doc(doc, quitar_dominio)))
        return salida
    salida = []
    for base in bases:
        tokens = quitar_stopwords(base.split(), quitar_dominio)
        if modo == "stem":
            tokens = stemizar(tokens)
        salida.append(" ".join(quitar_tildes(t) for t in tokens))
    return salida


def pasos_preprocesamiento(texto, nlp=None) -> dict:
    """Muestra cada paso del preprocesamiento para un texto (se usa en la app)."""
    nlp = nlp or cargar_spacy()
    limpio = limpiar_texto(texto)
    tokens = tokenizar(limpio)
    normal = normalizar_jerga(tokens)
    sin_stop = quitar_stopwords(normal)
    with nlp.select_pipes(disable=[p for p in ("parser", "ner") if p in nlp.pipe_names]):
        lemas = _lemas_doc(nlp(" ".join(normal)))
    return {
        "1. Texto original": texto,
        "2. Limpieza (regex, emojis, minúsculas)": limpio,
        "3. Tokenización": tokens,
        "4. Normalización de jerga": normal,
        "5. Sin stopwords": sin_stop,
        "6. Lematización (spaCy)": lemas,
        "7. Stemming (Snowball)": [quitar_tildes(t) for t in stemizar(sin_stop)],
    }


# =====================================================================
# 3. ETIQUETA DE SENTIMIENTO A PARTIR DE LAS ESTRELLAS
# =====================================================================
def sentimiento_por_estrellas(score) -> str:
    if score <= 2:
        return "negativo"
    if score == 3:
        return "neutro"
    return "positivo"


# =====================================================================
# 4. CLASIFICACIÓN DEL TIPO DE RECLAMO (reglas con expresiones regulares)
# =====================================================================
# Se aplican sobre texto_normalizado(): minúsculas, sin tildes, jerga corregida.
# El orden del diccionario también es la prioridad cuando hay empate.
CATEGORIAS = {
    "Seguridad y fraude": [
        r"\brob(o|aron|ado|ada)\b", r"\bfraud", r"\bestaf", r"\bhacke",
        r"\bsin (mi )?(autoriz|consentim)", r"\bno reconoc", r"\bphishing\b",
        r"\bseguridad\b", r"\bvaciaron\b", r"\bsustra", r"\bsuplant", r"\bclon",
    ],
    "Transferencias y pagos": [
        r"\btransfer", r"\btransaccion", r"\byape(o|ar|ado|aron|aste)\b", r"\bplin\b",
        r"\bpag(o|os|ar|ue|ado)\b", r"\benvi(ar|o|e|e) (dinero|plata)\b",
        r"\bno (me )?(llega|llego|llegan|llegaron)\b", r"\bdevoluc", r"\breembols",
        r"\bextorn", r"\binterbancari", r"\brecarga", r"\bqr\b", r"\bdeposit",
    ],
    "Acceso y validación": [
        r"\blogin\b", r"\biniciar sesion\b", r"\bingres(ar|o)\b", r"\bentrar\b",
        r"\bcontraseña", r"\bclave\b", r"\btoken\b", r"\bhuella\b", r"\bbiometr",
        r"\bvalid(ar|acion|a)\b", r"\bidentidad\b", r"\bselfie\b", r"\bdni\b",
        r"\bcodigo de (verificacion|seguridad)\b", r"\bbloque", r"\bafilia",
        r"\bregistr", r"\bdispositivo\b",
    ],
    "Comisiones y límites": [
        r"\bcomision", r"\bcobr(o|an|aron|ar|aran)\b", r"\blimite", r"\btope\b",
        r"\bmonto maximo\b", r"\binteres(es)?\b", r"\btasa\b", r"\bcosto\b",
        r"\bdescuent",
    ],
    "Actualizaciones y compatibilidad": [
        r"\bactuali[zc]", r"\bversion\b", r"\bcompatib", r"\broot", r"\bandroid\b",
        r"\bhuawei\b", r"\bsistema operativo\b", r"\bnueva interfaz\b",
    ],
    "Fallas técnicas": [
        r"\bse (cuelga|traba|cierra|congela|cae|colgo|cerro)\b", r"\berror\b",
        r"\bfall(a|as|o|ando)\b", r"\blent[oa]\b", r"\bdemor", r"\bcaid[oa]\b",
        r"\bcay(o|endo)\b",
        r"\bmantenimiento\b", r"\bpantalla (en )?blanc", r"\bbug", r"\bno carga\b",
        r"\bno abre\b", r"\bno funciona\b", r"\bno sirve\b", r"\bno responde\b",
        r"\bcarga(ndo)?\b",
    ],
    "Atención al cliente": [
        r"\batencion\b", r"\basesor", r"\bcall ?center\b", r"\blinea\b",
        r"\bbanca por telefono\b", r"\breclam", r"\bagencia\b", r"\bchat\b",
        r"\bsoporte\b", r"\bsolucion", r"\bno (me )?(responden|contestan|ayudan)\b",
    ],
}
_CATEGORIAS_RE = {c: [re.compile(p) for p in ps] for c, ps in CATEGORIAS.items()}
SIN_CATEGORIA = "Sin categoría"


def clasificar_reclamo(texto) -> dict:
    """Devuelve todas las categorías detectadas y la principal (la de más coincidencias)."""
    norm = texto_normalizado(texto)
    conteo = {}
    for cat, patrones in _CATEGORIAS_RE.items():
        n = sum(1 for p in patrones if p.search(norm))
        if n:
            conteo[cat] = n
    if not conteo:
        return {"categorias": [], "principal": SIN_CATEGORIA}
    orden = list(CATEGORIAS)
    principal = sorted(conteo, key=lambda c: (-conteo[c], orden.index(c)))[0]
    return {"categorias": list(conteo), "principal": principal}


# =====================================================================
# 5. EXTRACCIÓN DE INFORMACIÓN (regex + NER de spaCy)
# =====================================================================
RE_MONTO = re.compile(
    r"(?i)(s/\.?\s?\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s?(?:soles|lucas|d[oó]lares))")
RE_FECHA = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
RE_HACE = re.compile(r"(?i)\bhace\s+(?:\d+|un|una|dos|tres|varios|varias)\s+"
                     r"(?:minutos?|horas?|d[ií]as?|semanas?|meses|mes|años?)\b")
RE_FECHA_PALABRA = re.compile(
    r"\b(hoy|ayer|anoche|anteayer|lunes|martes|miercoles|jueves|viernes|sabado|domingo|"
    r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|setiembre|septiembre|octubre|"
    r"noviembre|diciembre|fin de mes|quincena)\b")
RE_VERSION = re.compile(r"(?i)\bversi[oó]n\s?\d+(?:\.\d+)*")

DISPOSITIVOS = ["samsung", "huawei", "xiaomi", "redmi", "motorola", "iphone", "honor",
                "oppo", "realme", "nokia", "zte", "tecno", "infinix", "android", "ios",
                "tablet"]
ENTIDADES = ["yape", "plin", "bcp", "interbank", "bbva", "scotiabank", "banco de la nacion",
             "bim", "agora", "tunki", "izipay", "niubiz", "mibanco", "banbif", "pichincha",
             "falabella", "ripley", "caja", "sunat", "indecopi", "sbs", "reniec"]
OPERADORES = ["movistar", "entel", "bitel"]


# Palabras que el NER de spaCy (entrenado con noticias) confunde con nombres propios
# cuando una reseña empieza con mayúscula: "Hola", "Pésimo", "Uff"...
RUIDO_NER = {"hola", "uff", "pesimo", "pesima", "gracias", "mejoren", "excelente", "buenas",
             "buenos", "buenas tardes", "buenos dias", "ok", "por favor", "porfavor", "app",
             "aplicacion", "la verdad", "sinceramente", "malisimo", "malisima", "horrible",
             "genial", "super", "muy", "señores", "estimados", "tengo", "ahora", "ya"}
_CONOCIDAS = set(DISPOSITIVOS) | set(ENTIDADES) | set(OPERADORES)


def preparar_para_ner(texto) -> str:
    """Quita URLs, emojis y signos repetidos, pero conserva mayúsculas (el NER las usa)."""
    base = emoji.replace_emoji(RE_URL.sub(" ", str(texto)), replace=" ")
    return re.sub(r"([!?.,])\1+", r"\1", base)


def entidades_ner(doc) -> list:
    """Entidades PER/ORG/LOC de un doc de spaCy, sin el ruido típico de las reseñas."""
    salida = []
    for e in doc.ents:
        clave = quitar_tildes(e.text.lower().strip(" .,!?"))
        if e.label_ not in {"PER", "ORG", "LOC"} or len(clave) < 3:
            continue
        if clave in RUIDO_NER or clave in STOPWORDS:
            continue
        # una sola palabra al inicio de la oración: solo vale si es una marca o entidad conocida
        if len(e) == 1 and e.start == e.sent.start and clave not in _CONOCIDAS:
            continue
        salida.append(f"{e.text.strip(' .,!?')} [{e.label_}]")
    return salida


def _buscar_lista(norm, lista):
    return [x for x in lista if re.search(rf"\b{x}\b", norm)]


def extraer_informacion(texto, nlp=None, usar_ner=True) -> dict:
    """Extrae montos, fechas, versiones, dispositivos, entidades y NER de spaCy."""
    if not isinstance(texto, str):
        texto = ""
    norm = texto_normalizado(texto)
    info = {
        "Montos": RE_MONTO.findall(texto),
        "Fechas / tiempo": RE_FECHA.findall(texto) + RE_HACE.findall(texto)
                           + RE_FECHA_PALABRA.findall(norm),
        "Versión de la app": RE_VERSION.findall(texto),
        "Dispositivos / sistema": _buscar_lista(norm, DISPOSITIVOS),
        "Entidades financieras": _buscar_lista(norm, ENTIDADES),
        "Operadores móviles": _buscar_lista(norm, OPERADORES),
    }
    if usar_ner:
        nlp = nlp or cargar_spacy()
        info["NER spaCy (PER/ORG/LOC)"] = entidades_ner(nlp(preparar_para_ner(texto)))
    # quitar duplicados manteniendo el orden
    return {k: list(dict.fromkeys(v)) for k, v in info.items()}


# =====================================================================
# 6. SIMILITUD SEMÁNTICA (LSA + similitud coseno)
# =====================================================================
def buscar_similares(consulta, vectorizador, svd, matriz, k=5, nlp=None):
    """
    Proyecta la consulta al espacio LSA y devuelve (índices, similitudes) de las
    k reseñas más parecidas. 'matriz' son las reseñas ya proyectadas y normalizadas.
    """
    import numpy as np
    from sklearn.preprocessing import normalize

    texto = preprocesar_lote([consulta], modo="lema", quitar_dominio=True, nlp=nlp)
    x = vectorizador.transform(texto)
    if x.nnz == 0:   # ninguna palabra de la consulta está en el vocabulario
        return np.array([], dtype=int), np.array([])
    zq = normalize(svd.transform(x))
    sims = matriz @ zq.ravel()
    top = np.argsort(sims)[::-1][:k]
    top = top[sims[top] > 0]
    return top, sims[top]
