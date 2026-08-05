"""
Tema 1 — Introducción a la Física de Partículas.

Secciones eficaces diferenciales de la dispersión elástica sobre un núcleo:

* **Rutherford**: partícula cargada sin espín sobre un núcleo puntual e infinitamente
  masivo. Tratamiento clásico (órbita hiperbólica, análogo al problema de Kepler).
* **Mott**: electrón relativista, que sí tiene espín 1/2, sobre un núcleo con retroceso.

Aquí se trabaja en el S.I. (energías en julios, masas en kg) porque las expresiones
se escriben con :math:`\\varepsilon_0` explícito; el resultado se devuelve en barn/sr.
"""

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

from .common import BARN


def sigma_rutherford(Z1, Z2, E, theta):
    """Sección eficaz diferencial de Rutherford, :math:`d\\sigma/d\\Omega`, en m^2/sr.

    .. math::
        \\frac{d\\sigma}{d\\Omega} =
        \\left( \\frac{Z_1 Z_2 e^2}{16 \\pi \\varepsilon_0 E} \\right)^2
        \\frac{1}{\\sin^4(\\theta/2)}

    Parameters
    ----------
    Z1 : float
        Carga de la partícula incidente, en unidades de |e|.
    Z2 : float
        Carga del núcleo blanco, en unidades de |e|.
    E : float
        Energía cinética de la partícula incidente [J].
    theta : float or ndarray
        Ángulo de dispersión [rad].

    Notes
    -----
    Diverge en :math:`\\theta \\to 0`: es el reflejo del alcance infinito del
    potencial de Coulomb.
    """
    factor = (Z1 * Z2 * const.e**2 / (16 * np.pi * const.epsilon_0 * E))**2
    return factor / (np.sin(theta / 2)**4)


def sigma_mott(Z, E, M, theta):
    """Sección eficaz diferencial de Mott con retroceso del núcleo, en m^2/sr.

    .. math::
        \\frac{d\\sigma}{d\\Omega} =
        \\left( \\frac{Z e^2}{8 \\pi \\varepsilon_0 E} \\right)^2
        \\frac{1}{\\sin^4(\\theta/2)} \\; \\frac{E'}{E} \\;
        \\left[ 1 - \\beta^2 \\sin^2(\\theta/2) \\right]

    con la energía del electrón dispersado

    .. math:: E' = \\frac{E}{1 + \\frac{2E}{Mc^2}\\sin^2(\\theta/2)}.

    Parameters
    ----------
    Z : float
        Número atómico del blanco.
    E : float
        Energía **total** del electrón incidente [J].
    M : float
        Masa del núcleo [kg].
    theta : float or ndarray
        Ángulo de dispersión [rad].

    Notes
    -----
    Respecto a Rutherford aparecen dos factores:

    * :math:`E'/E`, el retroceso del núcleo;
    * :math:`1 - \\beta^2\\sin^2(\\theta/2)`, el efecto del espín del electrón, que
      suprime la dispersión hacia atrás (a :math:`\\theta = \\pi` y :math:`\\beta \\to 1`
      se anula: la conservación de la helicidad prohíbe la retrodispersión).

    Ojo con el prefactor: la forma general es :math:`(Z e^2 / 4\\pi\\varepsilon_0 \\, 2pv)^2`.
    Para una partícula no relativista :math:`pv = 2T` y sale el :math:`16\\pi\\varepsilon_0 T`
    de :func:`sigma_rutherford`; para un electrón ultrarrelativista :math:`pv \\to E` y sale
    :math:`8\\pi\\varepsilon_0 E`. No reutilizar aquí el prefactor de Rutherford, que está
    escrito para energía **cinética**, mientras que ``E`` aquí es la energía **total**.
    """
    # energía del electrón dispersado, teniendo en cuenta el retroceso del núcleo
    E_prime = E / (1 + (2 * E / (M * const.c**2)) * np.sin(theta / 2)**2)
    # velocidad del electrón en unidades de c (E es la energía total)
    beta = np.sqrt(1 - (const.m_e * const.c**2 / E)**2)
    factor = (Z * const.e**2 / (8 * np.pi * const.epsilon_0 * E))**2
    return (factor / np.sin(theta / 2)**4
            * (E_prime / E) * (1 - beta**2 * np.sin(theta / 2)**2))


def plot_secciones_eficaces(E_alpha=5e6, E_electron=5e6, M_nucleus=197):
    """Compara Rutherford (α) y Mott (β) sobre oro en función del ángulo.

    Parameters
    ----------
    E_alpha : float
        Energía cinética de la partícula α [eV].
    E_electron : float
        Energía total del electrón [eV].
    M_nucleus : float
        Masa del núcleo blanco [u]. Por defecto oro, A = 197.
    """
    theta = np.linspace(0.01, np.pi, 100)

    E_alpha = E_alpha * const.e          # de eV a julios
    E_electron = E_electron * const.e    # de eV a julios
    M_nucleus = M_nucleus * const.u      # de u a kg

    sigma_R = sigma_rutherford(2, 79, E_alpha, theta)   # alfa (Z=2) sobre oro (Z=79)
    sigma_M = sigma_mott(79, E_electron, M_nucleus, theta)

    plt.plot(theta / np.pi, sigma_R / BARN, label='Rutherford (α)')
    plt.plot(theta / np.pi, sigma_M / BARN, label=r'Mott ($\beta$)')
    plt.yscale('log')
    plt.xlabel(r'Ángulo de dispersión, $\theta/\pi$')
    plt.ylabel(r'$d\sigma/d\Omega$ [barn/sr]')
    plt.title('Secciones eficaces: Rutherford (α) y Mott (β)')
    plt.legend()
    plt.grid(alpha=0.3)
    return plt.gca()
