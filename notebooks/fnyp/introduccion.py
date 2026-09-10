"""
Tema 1 — Introducción a la Física de Partículas.

Secciones eficaces diferenciales de la dispersión elástica sobre un núcleo:

* **Rutherford**: partícula cargada sin espín sobre un núcleo puntual e infinitamente
  masivo. Tratamiento clásico (órbita hiperbólica, análogo al problema de Kepler).
* **Mott**: electrón relativista, que sí tiene espín 1/2, sobre un núcleo con retroceso.

Y la dependencia de la vida media con el sistema de referencia
(:func:`plot_dilatacion_temporal`).

Aquí se trabaja en el S.I. (energías en julios, masas en kg) porque las expresiones
se escriben con :math:`\\varepsilon_0` explícito; las secciones eficaces se
devuelven en m^2/sr.
"""

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

import scipy.constants as units

from .common import M_MU, TAU_MU


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


# ---------------------------------------------------------------------------
# Dilatación temporal: la vida media y la longitud de desintegración
# ---------------------------------------------------------------------------

#: Momento típico de un muón cósmico al nivel del mar, en MeV (PDG, Cosmic Rays).
P_MUON_COSMICO = 4000.

#: Altura típica de producción de los muones cósmicos, en m.
H_ATMOSFERA = 15000.


def plot_dilatacion_temporal(particulas=None, p_min=10., p_max=1e6,
                             marcar_muon_cosmico=True, verbose=True):
    """Dibuja la vida media y la longitud de desintegración frente al momento.

    Dos paneles, los dos en escala log-log:

    * **izquierda**, la vida media en el sistema del laboratorio,
      :math:`\tau = \gamma \tau_0 = \tau_0 \sqrt{1 + (p/mc)^2}`. Tiene una
      meseta en :math:`\tau_0` mientras :math:`p \ll mc` y crece linealmente
      a partir de ahí: el codo marca dónde empieza a notarse la dilatación.

    * **derecha**, la longitud de desintegración
      :math:`\lambda = \gamma\beta \, c\tau_0 = (p/mc) \, c\tau_0`, que es
      *exactamente* proporcional al momento — una recta de pendiente 1 para
      cualquier partícula. Lo que distingue a unas de otras es la ordenada,
      :math:`c\tau_0 / mc`.

    Parameters
    ----------
    particulas : list of tuple or None
        Lista de ``(nombre, masa_MeV, tau_s)``. Por defecto, solo el muón.
    p_min, p_max : float
        Rango de momento, en MeV.
    marcar_muon_cosmico : bool
        Si es ``True``, señala el muón cósmico típico (``p`` = 4 GeV) y el
        espesor de la atmósfera. Solo tiene sentido con el muón en la lista.
    verbose : bool
        Si es ``True``, imprime los números del muón cósmico.

    Returns
    -------
    dict
        ``p`` (MeV) y, por partícula, ``tau`` (s) y ``lambda`` (m).
    """
    if particulas is None:
        particulas = [('$\\mu^\\pm$', M_MU, TAU_MU)]

    p = np.logspace(np.log10(p_min), np.log10(p_max), 400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    salida = dict(p=p)

    for nombre, masa, tau0 in particulas:
        gamma = np.sqrt(1. + (p / masa) ** 2)     # gamma = E/mc^2
        tau = gamma * tau0                        # dilatación temporal
        lam = (p / masa) * units.c * tau0         # gamma*beta*c*tau0 = (p/mc)*c*tau0

        linea, = ax1.plot(p / 1000., tau, lw=1.8, label=nombre)
        ax2.plot(p / 1000., lam, lw=1.8, color=linea.get_color(), label=nombre)

        # la meseta no relativista y el codo en p = mc
        ax1.axhline(tau0, ls=':', lw=1, color=linea.get_color(), alpha=0.6)
        ax1.axvline(masa / 1000., ls=':', lw=1, color=linea.get_color(), alpha=0.6)
        if len(particulas) == 1:
            ax1.text(p_min / 1000. * 1.4, tau0 * 1.25, r'$\tau_0$ (en reposo)',
                     fontsize=8, color=linea.get_color())
            ax1.text(masa / 1000. * 1.3, tau0 * 4.0, r'$p = mc$',
                     fontsize=8, color=linea.get_color())

        salida[nombre] = dict(tau=tau, **{'lambda': lam})

    if marcar_muon_cosmico:
        gamma_c = np.sqrt(1. + (P_MUON_COSMICO / M_MU) ** 2)
        lam_c = (P_MUON_COSMICO / M_MU) * units.c * TAU_MU
        ax2.axhline(H_ATMOSFERA, ls='--', lw=1.2, color='0.4')
        ax2.text(p_min / 1000. * 1.4, H_ATMOSFERA * 0.30,
                 'espesor de la atmósfera, 15 km', fontsize=8, color='0.3')
        ax2.plot(P_MUON_COSMICO / 1000., lam_c, 'o', ms=7, color='crimson', zorder=5)
        ax2.annotate(f'muón cósmico\n$p$ = 4 GeV, $\\gamma$ = {gamma_c:.0f}\n'
                     f'$\\lambda$ = {lam_c/1000.:.0f} km',
                     xy=(P_MUON_COSMICO / 1000., lam_c),
                     xytext=(0.42, 0.12), textcoords='axes fraction',
                     fontsize=8, color='crimson',
                     arrowprops=dict(arrowstyle='->', color='crimson', lw=1))
        if verbose:
            beta_c = P_MUON_COSMICO / (M_MU * gamma_c)
            print(f' muón cósmico típico:  p = {P_MUON_COSMICO/1000:.0f} GeV')
            print(f'   gamma = {gamma_c:6.1f},  beta = {beta_c:.6f}')
            print(f'   vida media en el laboratorio  = {gamma_c*TAU_MU*1e6:8.1f} us'
                  f'   (en reposo, {TAU_MU*1e6:.2f} us)')
            print(f'   longitud de desintegración    = {lam_c/1000.:8.1f} km'
                  f'   (sin dilatación, {units.c*TAU_MU/1000.:.3f} km)')

    for ax, ylab, tit in ((ax1, r'vida media $\tau$ (s)',
                           r'la vida media que mide el laboratorio'),
                          (ax2, r'longitud de desintegración $\lambda$ (m)',
                           r'cuánto vuela antes de desintegrarse')):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'momento $p$ (GeV/c)')
        ax.set_ylabel(ylab)
        ax.set_title(tit, fontsize=10)
        ax.grid(alpha=0.3)          # solo rejilla mayor: la menor en log-log
                                    # dispara el tamaño de la figura guardada
        if len(particulas) > 1:
            ax.legend(fontsize=8)

    fig.tight_layout()
    return salida
