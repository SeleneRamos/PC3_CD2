# Analizador de reseñas de apps bancarias peruanas

Proyecto PC3 del curso de Procesamiento de Lenguaje Natural (UNALM).

Aplicación en Streamlit que analiza reseñas públicas de Google Play de cinco apps bancarias peruanas (Yape, BCP, Interbank, BBVA y Scotiabank). Para cada reseña detecta el sentimiento, el tipo de reclamo, el tema y la información que menciona (montos, fechas, celulares, entidades), y permite buscar reseñas que cuentan un problema parecido.

## Técnicas de PLN

| Técnica | Herramienta |
|---|---|
| Preprocesamiento: regex, emojis a texto, jerga, tokenización, stopwords, lematización y stemming | `re`, `emoji`, spaCy, NLTK (Snowball) |
| Representación TF-IDF (unigramas y bigramas) | scikit-learn |
| Análisis de sentimiento (comparación de Naive Bayes, regresión logística y SVM) | scikit-learn |
| Clasificación del tipo de reclamo con reglas | expresiones regulares |
| Modelado de temas (LDA, elección de k por coherencia UMass y perplejidad) | scikit-learn |
| Similitud semántica (LSA + coseno) | scikit-learn |
| Extracción de información y NER | regex + spaCy |

## Estructura

```
resenas-banca-peru/
├── app.py                         # aplicación de Streamlit
├── PC3_PLN_resenas_banca.ipynb    # notebook documentado (todas las fases)
├── requirements.txt
├── src/
│   ├── recoleccion.py             # descarga de reseñas de Google Play
│   ├── pln_utils.py               # preprocesamiento y técnicas de PLN
│   └── estilo.py                  # colores de los gráficos
├── data/
│   ├── raw/                       # resenas_raw.csv (lo crea el notebook)
│   └── processed/                 # resenas_procesadas.csv (lo crea el notebook)
└── models/                        # modelos entrenados (los crea el notebook)
```

## Cómo ejecutarlo

### 1. Subir el proyecto a GitHub

Crear un repositorio nuevo (por ejemplo `resenas-banca-peru`) y subir todos los archivos de esta carpeta.

### 2. Ejecutar el notebook en Google Colab

1. Abrir `PC3_PLN_resenas_banca.ipynb` en Colab.
2. En la primera celda de código, quitar los `#`, cambiar `USUARIO` por el usuario de GitHub y ejecutarla. Si Colab pide reiniciar la sesión después de instalar, reiniciar y seguir desde la celda de importaciones.
3. Ejecutar todo (`Entorno de ejecución > Ejecutar todas`). La descarga de reseñas tarda unos minutos.
4. Al final se crean `data/raw/`, `data/processed/` y `models/`. Descargarlos y subirlos al repositorio:

```python
!zip -r resultados.zip data models
from google.colab import files; files.download("resultados.zip")
```

### 3. Publicar la app en Streamlit Cloud

1. Entrar a https://share.streamlit.io con la cuenta de GitHub.
2. `Create app` → elegir el repositorio, la rama `main` y el archivo `app.py`.
3. En `Advanced settings`, elegir Python 3.12.
4. `Deploy`. La primera vez tarda unos minutos porque instala spaCy y su modelo en español.

### Ejecutar en local (opcional)

```bash
pip install -r requirements.txt
jupyter notebook PC3_PLN_resenas_banca.ipynb   # genera data/ y models/
streamlit run app.py
```

## Datos y ética

Se usan solo reseñas públicas de Google Play, con fines académicos. No se guardan el nombre ni la foto de los usuarios.

## Integrantes

- [completar]
