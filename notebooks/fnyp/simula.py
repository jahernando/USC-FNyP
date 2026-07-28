"""
Experimentos simulados para los Talleres.

Cada función monta un pequeño experimento Monte Carlo, dibuja el resultado y
**mide** una cantidad física a partir de los datos simulados, igual que haría un
experimento real. La idea es que el alumno cambie los parámetros (sobre todo el
número de sucesos) y vea qué ocurre.

Todas las funciones aceptan ``seed`` para que el resultado sea reproducible, y
devuelven las cantidades medidas en un diccionario.

Experimentos
------------
:func:`atenuacion`      un haz atraviesa un blanco extenso -> mide la longitud
                        de interacción :math:`\\lambda`
:func:`vida_media`      desintegraciones de una muestra -> mide :math:`\\tau` y su
                        error estadístico
:func:`espectro_beta`   compara la desintegración a dos y a tres cuerpos -> el
                        argumento que llevó a postular el neutrino
:func:`rutherford`      dispersión de partículas :math:`\\alpha` sobre una lámina
                        de oro -> compara el modelo de Thomson con el núcleo
                        puntual de Rutherford
"""

import numpy as np
import matplotlib.pyplot as plt

#: e^2 / (4 pi eps0), en MeV fm. Es la constante de la fuerza de Coulomb en las
#: unidades naturales de la física nuclear.
K_COULOMB = 1.43996  # MeV fm


# ---------------------------------------------------------------------------
# 1. Blanco extenso: atenuación del haz
# ---------------------------------------------------------------------------

def atenuacion(n=20000, lambda_real=12., espesor=50., seed=7, verbose=True):
    """Un haz de ``n`` partículas atraviesa un blanco: mide la longitud de interacción.

    Cada partícula interacciona a una profundidad distribuida exponencialmente con
    parámetro ``lambda_real``. Contando cuántas sobreviven a cada profundidad se
    reconstruye :math:`\\phi(x) = \\phi_0 e^{-x/\\lambda}` y, ajustando una recta a
    :math:`\\ln \\phi`, se recupera :math:`\\lambda`.

    Parameters
    ----------
    n : int
        Número de partículas del haz.
    lambda_real : float
        Longitud de interacción "verdadera" [cm]; es lo que el ajuste debe recuperar.
    espesor : float
        Espesor del blanco [cm].
    seed : int
        Semilla del generador, para reproducibilidad.
    verbose : bool
        Si es ``True``, imprime el resultado y dibuja la figura.

    Returns
    -------
    dict
        ``lambda_medida``, ``lambda_real`` y ``supervivientes``.
    """
    rng = np.random.default_rng(seed)
    x_int = rng.exponential(lambda_real, n)      # profundidad de la 1ª interacción

    xs = np.linspace(0, espesor, 40)
    n_sup = np.array([np.sum(x_int > x) for x in xs], dtype=float)

    ok = n_sup > 10                              # descartamos la cola sin estadística
    pendiente, ordenada = np.polyfit(xs[ok], np.log(n_sup[ok]), 1)
    lambda_medida = -1. / pendiente

    if verbose:
        plt.errorbar(xs, n_sup, yerr=np.sqrt(n_sup), fmt='o', ms=4,
                     label='simulación')
        plt.plot(xs, np.exp(ordenada + pendiente * xs), '-',
                 label=r'ajuste $\phi_0 e^{-x/\lambda}$')
        plt.yscale('log')
        plt.xlabel('espesor recorrido $x$ (cm)')
        plt.ylabel('partículas que no han interaccionado')
        plt.grid(alpha=0.3)
        plt.legend()
        print(f' lambda de entrada = {lambda_real:6.2f} cm')
        print(f' lambda medida     = {lambda_medida:6.2f} cm   (con n = {n} sucesos)')

    return dict(lambda_medida=lambda_medida, lambda_real=lambda_real,
                supervivientes=n_sup)


# ---------------------------------------------------------------------------
# 2. Vida media
# ---------------------------------------------------------------------------

def vida_media(n=2000, tau_real=2.197, seed=3, verbose=True):
    """Simula la desintegración de ``n`` partículas y mide su vida media.

    Los tiempos de desintegración se generan según :math:`e^{-t/\\tau}`. El
    estimador de máxima verosimilitud de :math:`\\tau` es simplemente la media de
    los tiempos, y su error es :math:`\\tau/\\sqrt{n}`: para mejorar la medida un
    factor 10 hay que multiplicar por 100 el número de sucesos.

    Parameters
    ----------
    n : int
        Número de partículas de la muestra.
    tau_real : float
        Vida media "verdadera". Por defecto, la del muón en microsegundos.
    seed : int
        Semilla del generador.
    verbose : bool
        Si es ``True``, imprime el resultado y dibuja el histograma.

    Returns
    -------
    dict
        ``tau_medida``, ``error``, ``tau_real`` y ``tiempos``.
    """
    rng = np.random.default_rng(seed)
    t = rng.exponential(tau_real, n)

    tau_medida = t.mean()                 # estimador de máxima verosimilitud
    error = tau_medida / np.sqrt(n)       # su error estadístico

    if verbose:
        plt.hist(t, bins=40, range=(0, 6 * tau_real), histtype='step', lw=1.5)
        plt.yscale('log')
        plt.xlabel(r'tiempo de desintegración $t$ ($\mu$s)')
        plt.ylabel('sucesos')
        plt.grid(alpha=0.3)
        desv = abs(tau_medida - tau_real) / error
        print(f' tau de entrada = {tau_real:6.4f}')
        print(f' tau medida     = {tau_medida:6.4f} +- {error:6.4f}'
              f'   ({desv:.1f} sigmas de la de entrada, con n = {n})')

    return dict(tau_medida=tau_medida, error=error, tau_real=tau_real, tiempos=t)


# ---------------------------------------------------------------------------
# 3. Espectro beta: dos cuerpos frente a tres cuerpos
# ---------------------------------------------------------------------------

def espectro_beta(Q=0.782, m_e=0.511, verbose=True):
    """Compara el espectro del electrón en una desintegración a 2 y a 3 cuerpos.

    En la desintegración a dos cuerpos :math:`n \\to p \\, e^-` la cinemática fija
    por completo la energía del electrón: el espectro sería **una raya**. En la
    desintegración a tres cuerpos :math:`n \\to p \\, e^- \\bar{\\nu}_e` la energía se
    reparte entre el electrón y el neutrino, y el espectro es **continuo** entre 0
    y :math:`Q`.

    Se usa la forma del espacio fásico de Fermi, sin la corrección coulombiana:

    .. math:: \\frac{dN}{dT} \\propto p_e \\, E_e \\, (Q - T)^2

    Lo que Chadwick midió en 1914 fue el espectro continuo. Como la energía debía
    conservarse, Pauli postuló en 1930 una partícula neutra, muy ligera y casi
    indetectable que se llevaba la energía que faltaba: el neutrino.

    Parameters
    ----------
    Q : float
        Energía liberada [MeV]. Por defecto la de la desintegración del neutrón.
    m_e : float
        Masa del electrón [MeV].
    verbose : bool
        Si es ``True``, dibuja los dos espectros y comenta el resultado.

    Returns
    -------
    dict
        ``T`` (energía cinética), ``espectro_3cuerpos`` y ``T_2cuerpos``.
    """
    T = np.linspace(1e-4, Q, 500)          # energía cinética del electrón
    E = T + m_e                            # energía total
    p = np.sqrt(E**2 - m_e**2)             # momento
    dNdT = p * E * (Q - T)**2
    dNdT = dNdT / dNdT.max()

    # dos cuerpos: la energía del electrón queda fijada por la cinemática.
    # Con Q << m_n el electrón se lleva prácticamente toda la energía liberada.
    T_2 = Q

    if verbose:
        plt.plot(T, dNdT, label=r'3 cuerpos: $n \to p \, e^- \, \bar{\nu}_e$')
        plt.vlines(T_2, 0, 1.05, color='crimson', lw=2,
                   label=r'2 cuerpos: $n \to p \, e^-$')
        plt.xlabel(r'energía cinética del electrón $T$ (MeV)')
        plt.ylabel(r'$dN/dT$ (normalizado)')
        plt.ylim(0, 1.15)
        plt.grid(alpha=0.3)
        plt.legend()
        print(f' 2 cuerpos: una raya en T = {T_2:5.3f} MeV')
        print(f' 3 cuerpos: continuo entre 0 y Q = {Q:5.3f} MeV')

    return dict(T=T, espectro_3cuerpos=dNdT, T_2cuerpos=T_2)


# ---------------------------------------------------------------------------
# 4. Dispersión de Rutherford
# ---------------------------------------------------------------------------

def rutherford(n=200000, E_MeV=5., Z_proyectil=2, Z_blanco=79,
               b_max_fm=5000., sigma_thomson_grados=1., seed=5, verbose=True):
    """Dispersión de partículas α sobre oro: modelo de Thomson frente al de Rutherford.

    **Rutherford**: toda la carga positiva se concentra en un núcleo puntual. Para
    un parámetro de impacto :math:`b`, la trayectoria es una hipérbola y el ángulo
    de dispersión vale

    .. math:: \\theta = 2 \\arctan\\left( \\frac{d}{2b} \\right),
              \\qquad d = \\frac{Z_1 Z_2 e^2}{4\\pi\\varepsilon_0 E},

    donde :math:`d` es la distancia de máximo acercamiento en un choque frontal.
    Los parámetros de impacto se sortean uniformemente sobre el área de un disco de
    radio ``b_max_fm`` (que representa el apantallamiento de los electrones
    atómicos).

    **Thomson** ("pudín de pasas"): la carga positiva está repartida por todo el
    átomo, así que cada colisión desvía muy poco y el resultado neto es una suma de
    muchas desviaciones pequeñas — una gaussiana de anchura de aproximadamente un
    grado. Retrodispersar es, sencillamente, imposible.

    Parameters
    ----------
    n : int
        Número de partículas α lanzadas.
    E_MeV : float
        Energía cinética de las α [MeV]. Geiger y Marsden usaron unos 5 MeV.
    Z_proyectil, Z_blanco : int
        Cargas del proyectil (α, Z=2) y del blanco (oro, Z=79).
    b_max_fm : float
        Parámetro de impacto máximo [fm]. Representa el apantallamiento de la carga
        del núcleo por los electrones atómicos. Es un parámetro libre de la
        simulación: **fija la normalización absoluta, no la física**. Lo que importa
        aquí es el contraste entre los dos modelos, no reproducir el 1 de cada 8000
        que midieron Geiger y Marsden (para eso hay que contar los núcleos de toda
        la lámina, ver el Taller).
    sigma_thomson_grados : float
        Anchura de la dispersión múltiple en el modelo de Thomson [grados].
    seed : int
        Semilla del generador.
    verbose : bool
        Si es ``True``, imprime los resultados y dibuja las distribuciones.

    Returns
    -------
    dict
        ``d_fm``, ``frac_rutherford`` y ``frac_thomson`` (fracciones de sucesos
        dispersados a más de 90 grados) y los ángulos simulados.
    """
    rng = np.random.default_rng(seed)

    # distancia de máximo acercamiento en un choque frontal
    d = Z_proyectil * Z_blanco * K_COULOMB / E_MeV                    # fm

    # parámetro de impacto uniforme en el área del disco:  P(b) db ~ b db
    b = b_max_fm * np.sqrt(rng.random(n))
    theta_ruth = np.degrees(2 * np.arctan(d / (2 * b)))

    # Thomson: muchas desviaciones pequeñas -> gaussiana centrada en 0
    theta_thom = np.abs(rng.normal(0., sigma_thomson_grados, n))

    frac_r = np.mean(theta_ruth > 90.)
    frac_t = np.mean(theta_thom > 90.)

    if verbose:
        bins = np.linspace(0, 180, 91)
        plt.hist(theta_ruth, bins=bins, histtype='step', lw=1.5,
                 label='Rutherford (núcleo puntual)')
        plt.hist(theta_thom, bins=bins, histtype='step', lw=1.5,
                 label='Thomson (carga difusa)')
        plt.yscale('log')
        plt.xlabel(r'ángulo de dispersión $\theta$ (grados)')
        plt.ylabel('sucesos')
        plt.grid(alpha=0.3)
        plt.legend()
        print(f' distancia de máximo acercamiento d = {d:5.1f} fm')
        print(f' fracción dispersada a theta > 90 grados:')
        uno_de_cada = f'1 de cada {int(round(1/frac_r)):d}' if frac_r > 0 else 'ningún suceso'
        print(f'   Rutherford : {frac_r:9.2e}   ->  {uno_de_cada}')
        print(f'   Thomson    : {frac_t:9.2e}   ->  '
              f'{"ningún suceso" if frac_t == 0 else ""}')

    return dict(d_fm=d, frac_rutherford=frac_r, frac_thomson=frac_t,
                theta_rutherford=theta_ruth, theta_thomson=theta_thom)
