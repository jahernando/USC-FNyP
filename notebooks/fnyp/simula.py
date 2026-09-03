"""
Experimentos simulados para los Talleres.

Cada función monta un pequeño experimento Monte Carlo, dibuja el resultado y
**mide** una cantidad física a partir de los datos simulados, igual que haría un
experimento real. La idea es que el alumno cambie los parámetros (sobre todo el
número de sucesos) y vea qué ocurre.

Por defecto la semilla del generador es aleatoria (``seed=None``): repetir el
experimento con las mismas condiciones da un conjunto de datos distinto y, por
tanto, una medida distinta dentro del error. Pasando ``seed=<entero>`` se
reproduce exactamente la misma toma de datos.

Todas las funciones devuelven las cantidades medidas en un diccionario.

Experimentos
------------
:func:`atenuacion`      un haz atraviesa un blanco extenso -> mide la longitud
                        de interacción :math:`\\lambda`
:func:`vida_media`      desintegraciones de una muestra -> mide :math:`\\tau` y su
                        error estadístico
:func:`vida_media_evolucion`
                        la misma toma de datos leída por partes -> se ve crecer el
                        histograma y converger la medida como :math:`1/\\sqrt{n}`
:func:`vida_media_animada`
                        lo mismo, pero animado, para proyectar en clase
:func:`vida_media_sin_memoria`
                        el tiempo que le queda a un superviviente -> las partículas
                        no envejecen, comparado con una población que sí lo hace
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

def atenuacion(n=20000, lambda_real=12., espesor=50., seed=None, verbose=True):
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
    seed : int or None
        Semilla del generador. Por defecto ``None``: cada ejecución es una toma de
        datos distinta, como en un experimento real. Fija un entero si quieres
        repetir exactamente la misma medida.
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

def vida_media(n=2000, tau_real=2.197, seed=None, verbose=True):
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
    seed : int or None
        Semilla del generador. Por defecto ``None``: cada ejecución es una toma de
        datos distinta. Fija un entero para repetir la misma medida.
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


def vida_media_evolucion(ns=(10, 100, 1000, 10000), tau_real=2.197, seed=None,
                         verbose=True):
    """Muestra cómo crece el histograma y cómo converge la medida al acumular sucesos.

    Es **una sola toma de datos** que se va leyendo por partes: el histograma de
    ``ns[1]`` sucesos contiene los ``ns[0]`` anteriores, y así sucesivamente. Por eso
    se ve *crecer* el histograma, igual que en un experimento que acumula estadística,
    y no cuatro experimentos independientes.

    Lo que el alumno debe observar:

    * la **forma** exponencial emerge del ruido a partir de unos cientos de sucesos;
    * la medida de :math:`\\tau` se acerca al valor verdadero, pero el error decrece
      solo como :math:`1/\\sqrt{n}`: cada dígito extra de precisión cuesta un factor
      100 en tiempo de toma de datos.

    Parameters
    ----------
    ns : sequence of int
        Números de sucesos acumulados que se muestran, en orden creciente.
    tau_real : float
        Vida media "verdadera". Por defecto, la del muón en microsegundos.
    seed : int or None
        Semilla del generador. Por defecto ``None``: cada ejecución es una toma de
        datos distinta. Fija un entero para repetir la misma medida.
    verbose : bool
        Si es ``True``, dibuja los histogramas y la curva de convergencia.

    Returns
    -------
    dict
        ``ns``, ``taus``, ``errores`` y ``tiempos`` (la muestra completa).
    """
    ns = sorted(int(n) for n in ns)
    rng = np.random.default_rng(seed)
    t = rng.exponential(tau_real, ns[-1])          # una única toma de datos

    taus = np.array([t[:n].mean() for n in ns])
    errores = taus / np.sqrt(ns)

    if verbose:
        bins = np.linspace(0, 6 * tau_real, 41)

        fig, axes = plt.subplots(1, len(ns), figsize=(3.4 * len(ns), 3.2),
                                 sharex=True)
        for ax, n, tau, err in zip(np.atleast_1d(axes), ns, taus, errores):
            # densidad, para que los histogramas sean comparables entre sí
            ax.hist(t[:n], bins=bins, density=True, histtype='step', lw=1.5)
            ax.plot(bins, np.exp(-bins / tau_real) / tau_real, 'k--', lw=1,
                    label='exponencial real')
            ax.set_title(f'n = {n}\n' + r'$\tau$ = ' + f'{tau:.2f} $\\pm$ {err:.2f}',
                         fontsize=10)
            ax.set_xlabel(r'$t$ ($\mu$s)')
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel('sucesos (normalizado)')
        np.atleast_1d(axes)[0].legend(fontsize=8)
        fig.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        ax2.errorbar(ns, taus, yerr=errores, fmt='o-', capsize=4)
        ax2.axhline(tau_real, color='crimson', ls='--',
                    label=r'$\tau$ verdadera')
        ax2.set_xscale('log')
        ax2.set_xlabel('sucesos acumulados $n$')
        ax2.set_ylabel(r'$\tau$ medida ($\mu$s)')
        ax2.grid(alpha=0.3)
        ax2.legend()
        fig2.tight_layout()

        print(f' tau de entrada = {tau_real:6.4f}')
        for n, tau, err in zip(ns, taus, errores):
            desv = abs(tau - tau_real) / err
            print(f'   n = {n:>7d}  ->  tau = {tau:6.4f} +- {err:6.4f}'
                  f'   ({desv:.1f} sigmas)')

    return dict(ns=np.array(ns), taus=taus, errores=errores, tiempos=t)


def vida_media_animada(n_max=10000, n_min=10, frames=40, tau_real=2.197,
                       fps=5, seed=None):
    """Versión animada de :func:`vida_media_evolucion`, para proyectar en clase.

    Misma idea —una sola toma de datos leída por prefijos crecientes— pero como
    animación: el histograma se ve llenarse suceso a suceso y la medida de
    :math:`\\tau` estabilizarse en el título.

    Devuelve un objeto ``HTML`` con los controles de reproducción, así que la
    celda **debe terminar en esta llamada** (sin punto y coma) para que se muestre.

    .. warning::
       La animación se incrusta como un PNG por fotograma: ~1-2 MB de salida. Úsala
       en vivo (RISE, VS Code) y **no guardes el notebook con su salida** si no
       quieres engordar el fichero y los diffs del repositorio. Para los apuntes y
       el Book, usa :func:`vida_media_evolucion`.

    Parameters
    ----------
    n_max, n_min : int
        Sucesos acumulados en el último y en el primer fotograma.
    frames : int
        Número de fotogramas, espaciados logarítmicamente entre ``n_min`` y ``n_max``.
    tau_real : float
        Vida media "verdadera". Por defecto, la del muón en microsegundos.
    fps : int
        Fotogramas por segundo de la reproducción.
    seed : int or None
        Semilla del generador. Por defecto ``None``: cada ejecución es una toma de
        datos distinta.

    Returns
    -------
    IPython.display.HTML
        La animación con sus controles.
    """
    from matplotlib.animation import FuncAnimation      # solo si se usa la animación
    from IPython.display import HTML

    rng = np.random.default_rng(seed)
    t = rng.exponential(tau_real, n_max)                # una única toma de datos

    ns = np.unique(np.logspace(np.log10(n_min), np.log10(n_max),
                               frames).astype(int))
    bins = np.linspace(0, 6 * tau_real, 41)
    curva = np.exp(-bins / tau_real) / tau_real

    fig, ax = plt.subplots(figsize=(6, 4))

    def dibuja(n):
        ax.clear()
        ax.hist(t[:n], bins=bins, density=True, histtype='stepfilled',
                alpha=0.6, lw=1.5)
        ax.plot(bins, curva, 'k--', lw=1, label='exponencial real')
        tau, err = t[:n].mean(), t[:n].mean() / np.sqrt(n)
        ax.set_title(f'n = {n}' + r'   $\tau$ = ' + f'{tau:.3f} $\\pm$ {err:.3f} '
                     + r'$\mu$s')
        ax.set_xlabel(r'tiempo de desintegración $t$ ($\mu$s)')
        ax.set_ylabel('sucesos (normalizado)')
        ax.set_ylim(0, 1.15 * curva.max())
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')

    anim = FuncAnimation(fig, dibuja, frames=ns, interval=1000 // fps, repeat=True)
    html = anim.to_jshtml(default_mode='once')
    plt.close(fig)                                      # evita el frame estático extra
    return HTML(html)


def vida_media_sin_memoria(n=200000, tau_real=2.197, edades=(1., 3.),
                           seed=None, verbose=True):
    """Muestra que una partícula **no envejece**: un superviviente es como uno nuevo.

    Es el experimento que responde a la pregunta "si el muón lleva ya un rato sin
    desintegrarse, ¿no le tocará ya?". Se simulan ``n`` partículas y se mira, para
    los supervivientes a distintas edades, **cuánto tiempo les queda por vivir**.

    Se comparan dos poblaciones con la *misma* vida media:

    * **sin memoria**: tiempos exponenciales, como los núcleos y las partículas.
      La distribución del tiempo restante de los supervivientes es idéntica a la
      de la muestra recién preparada. Los histogramas caen uno sobre otro.
    * **que envejece**: tiempos de Weibull de forma :math:`k=3`, que es lo que
      describe el desgaste (una bombilla, un rodamiento, un ser vivo). Ahí los
      supervivientes sí tienen menos vida por delante, y los histogramas se
      separan.

    El contraste es el argumento: si las partículas envejecieran, el panel de la
    izquierda se parecería al de la derecha. No lo hace.

    Parameters
    ----------
    n : int
        Número de partículas simuladas en cada población.
    tau_real : float
        Vida media "verdadera", común a las dos poblaciones. Por defecto, la del
        muón en microsegundos.
    edades : sequence of float
        Edades, **en unidades de** :math:`\\tau`, a las que se seleccionan los
        supervivientes.
    seed : int or None
        Semilla del generador. Por defecto ``None``: cada ejecución es una toma de
        datos distinta.
    verbose : bool
        Si es ``True``, dibuja los histogramas e imprime las vidas medias restantes.

    Returns
    -------
    dict
        ``medidas`` (vida media restante por población y edad), ``tiempos_exp``,
        ``tiempos_env`` y ``tau_real``.
    """
    rng = np.random.default_rng(seed)

    t_exp = rng.exponential(tau_real, n)

    k = 3.0                                  # forma de Weibull: desgaste
    t_env = rng.weibull(k, n)
    t_env *= tau_real / t_env.mean()         # misma vida media que la exponencial

    def restantes(t, edad):
        """Tiempo que les queda a los que aún no se han desintegrado en ``edad``."""
        t0 = edad * tau_real
        return t[t > t0] - t0

    poblaciones = (('sin memoria (exponencial)', t_exp),
                   (f'que envejece (Weibull k={k:.0f})', t_env))
    edades_todas = (0.0,) + tuple(float(e) for e in edades)

    medidas = {}
    for nombre, t in poblaciones:
        filas = []
        for edad in edades_todas:
            r = restantes(t, edad)
            if r.size == 0:                  # nadie llega: la poblacion se ha extinguido
                filas.append(dict(edad=edad, n=0, media=np.nan, error=np.nan))
                continue
            media = r.mean()
            filas.append(dict(edad=edad, n=int(r.size), media=media,
                              error=media / np.sqrt(r.size)))
        medidas[nombre] = filas

    if verbose:
        bins = np.linspace(0, 6 * tau_real, 61)

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
        for ax, (nombre, t) in zip(axes, poblaciones):
            for edad in edades_todas:
                r = restantes(t, edad)
                if r.size == 0:
                    continue                 # nadie sobrevive: no hay nada que dibujar
                etiqueta = ('muestra recién preparada' if edad == 0 else
                            f'sobrevivieron a $t > {edad:g}\\,\\tau$')
                ax.hist(r, bins=bins, density=True,
                        histtype='step', lw=1.6, label=etiqueta)
            ax.set_yscale('log')
            ax.set_title(nombre, fontsize=10)
            ax.set_xlabel(r'tiempo que le queda por vivir ($\mu$s)')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel('sucesos (normalizado)')
        fig.tight_layout()

        print(f' vida media de entrada = {tau_real:6.4f}\n')
        for nombre, filas in medidas.items():
            print(f' {nombre}')
            for f in filas:
                etq = ('muestra recién preparada' if f['edad'] == 0 else
                       f"sobrevivieron a t > {f['edad']:g} tau")
                if f['n'] == 0:
                    print(f"   {etq:<28s} n = {f['n']:>7d}   no sobrevive ninguna")
                    continue
                print(f"   {etq:<28s} n = {f['n']:>7d}"
                      f"   les quedan {f['media']:6.4f} +- {f['error']:6.4f}")
            print()

    return dict(medidas=medidas, tiempos_exp=t_exp, tiempos_env=t_env,
                tau_real=tau_real)


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
               b_max_fm=5000., sigma_thomson_grados=1., seed=None, verbose=True):
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
    seed : int or None
        Semilla del generador. Por defecto ``None``: cada ejecución es una toma de
        datos distinta. Fija un entero para repetir la misma medida.
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
