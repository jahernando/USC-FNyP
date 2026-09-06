"""
Constantes y utilidades compartidas por todos los temas.

Convenio de unidades
--------------------
En los notebooks se usan **unidades naturales** ($\\hbar = c = 1$), con las energías
en GeV o en MeV según el tema (se indica en cada caso). El paso a S.I. se hace con
el factor de conversión ``HBARC_GEV_FM`` = 0.197 GeV fm.

``scipy.constants`` se importa como ``units`` para poder usar tanto los prefijos
(``units.pico``, ``units.femto``, ...) como los valores CODATA
(``units.value("proton mass energy equivalent in MeV")``).
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as units

# ---------------------------------------------------------------------------
# Factores de conversión
# ---------------------------------------------------------------------------

HBARC_GEV_FM = 0.197                  # hbar*c en GeV fm   (fm = 1e-15 m)
HBARC_GEV_M = 0.197 * units.femto     # hbar*c en GeV m
HBAR_EV_S = units.hbar / units.eV     # hbar en eV s

BARN = 1e-28                          # 1 barn en m^2

#: Una sección eficaz en unidades naturales (GeV^-2) pasa a barn multiplicando
#: por este factor:  sigma[m^2] = sigma[GeV^-2] * (hbar c [GeV m])^2
GEV2_TO_BARN = HBARC_GEV_M ** 2 / BARN

# ---------------------------------------------------------------------------
# Masas en MeV — CODATA / PDG 2024
# ---------------------------------------------------------------------------

M_E = units.value("electron mass energy equivalent in MeV")    # 0.51100
M_P = units.value("proton mass energy equivalent in MeV")      # 938.272
M_N = units.value("neutron mass energy equivalent in MeV")     # 939.565

M_MU = 105.658                        # muón
M_TAU = 1776.86                       # tauón
M_PI = 139.570                        # pión cargado
M_K = 493.677                         # kaón cargado
M_B0 = 5279.72                        # mesón B0

# Vidas medias, en segundos — PDG 2024
TAU_MU = 2.1969811 * units.micro
TAU_TAU = 290.3 * units.femto
TAU_N = 878.4
TAU_PI = 26.033 * units.nano
TAU_K = 12.380 * units.nano
TAU_B0 = 1.519 * units.pico

# ---------------------------------------------------------------------------
# Parámetros electrodébiles
# ---------------------------------------------------------------------------

M_W = 80.34                           # GeV
M_Z = 91.19                           # GeV
GAMMA_Z = 2.45                        # GeV, anchura total del Z
G_FERMI = units.value("Fermi coupling constant")    # GeV^-2
SIN2_THETA_W = units.value("weak mixing angle")     # sen^2(theta_W)

N_COLORES = 3                         # cargas de color de los quarks


def version():
    """Imprime la fecha de ejecución, como control de versión de la clase."""
    print(' Last version ', time.asctime())


def plot_xy(x, y, xlabel='', ylabel='', title='', ylim=None, yscale=None,
            label=None, ax=None, **kwargs):
    """Dibuja una curva con la cosmética habitual de estos apuntes.

    Evita repetir en cada notebook las llamadas a ``grid``, ``xlabel``, ``ylabel``...
    La física —la función que se dibuja— se escribe en la celda del notebook.

    Parameters
    ----------
    x : array_like
        Valores del eje x.
    y : array_like or callable
        Valores del eje y, o una función que se evalúa sobre ``x``.
    xlabel, ylabel, title : str, optional
        Etiquetas de los ejes y título.
    ylim : tuple, optional
        Límites del eje y.
    yscale : {'log', 'linear'}, optional
        Escala del eje y.
    label : str, optional
        Etiqueta de la curva; si se da, se dibuja la leyenda.
    ax : matplotlib.axes.Axes, optional
        Ejes sobre los que dibujar. Por defecto, los actuales.
    **kwargs
        Se pasan a ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = plt.gca() if ax is None else ax
    yy = y(np.asarray(x)) if callable(y) else y
    ax.plot(x, yy, label=label, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yscale is not None:
        ax.set_yscale(yscale)
    if label is not None:
        ax.legend()
    ax.grid(alpha=0.3)
    return ax
