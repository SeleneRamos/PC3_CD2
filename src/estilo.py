"""Colores y estilo de gráficos (los mismos en el notebook y en la app)."""

# Cada app conserva siempre su color, en todos los gráficos.
# Se eligieron colores cercanos a cada marca y se revisó que se distingan
# entre sí, también para personas con daltonismo.
COLORES_APP = {
    "Yape": "#6250d6",        # violeta
    "BCP": "#eb6834",         # naranja
    "Interbank": "#1baf7a",   # verde
    "BBVA": "#2a78d6",        # azul
    "Scotiabank": "#e87ba4",  # rosado
}
# Sentimiento: rojo (negativo), gris (neutro), azul (positivo)
COLORES_SENT = {"negativo": "#e34948", "neutro": "#9b9a94", "positivo": "#2a78d6"}
ORDEN_SENT = ["negativo", "neutro", "positivo"]
ORDEN_APPS = list(COLORES_APP)
TINTA = "#1b1e2b"
TINTA_SEC = "#565b70"
GRILLA = "#e4e6ee"


def estilo_matplotlib():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRILLA,
        "axes.labelcolor": TINTA_SEC,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": TINTA,
        "axes.grid": True,
        "grid.color": GRILLA,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": TINTA_SEC,
        "ytick.color": TINTA_SEC,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.size": 10,
    })
