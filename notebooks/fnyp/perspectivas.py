"""
Código auxiliar de los Talleres del tema *Perspectivas* (experimental y teórica).

Mismo criterio que el resto del paquete: aquí va el código que el alumno **no**
necesita ver —la construcción de las matrices, la simulación, la cosmética de
las figuras—; las comprobaciones y las fórmulas que resuelven cada cuestión se
quedan visibles en la celda del notebook.

Parte experimental
------------------
:func:`masa_transversa`  simula :math:`W \\to e + \\bar{\\nu}` y dibuja la masa
                         transversa -> el borde (pico jacobiano) en :math:`m_W`,
                         y cómo lo difuminan el :math:`p_T` del W y la resolución
:func:`dedx`             :math:`\\mathrm{d}E/\\mathrm{d}x` de Bethe-Bloch frente al
                         momento -> las bandas de una TPC y dónde se cruzan
:func:`pico_sobre_fondo` un pico de :math:`H \\to \\gamma\\gamma` sobre el fondo al
                         acumular luminosidad -> la significancia crece como
                         :math:`\\sqrt{\\mathcal{L}}`, salvo que un sistemático la frene

Parte teórica
-------------
:func:`matrices_dirac`   las matrices :math:`\\gamma^\\mu`, :math:`\\gamma^5` y los
                         proyectores de quiralidad, en la representación de
                         Pauli-Dirac
:func:`espinores`        los cuatro espinores :math:`u_s(p), v_s(p)` con **p** en
                         el eje z, y :math:`\\gamma^\\mu p_\\mu`
:func:`kappa_espinor`    :math:`\\kappa = \\mathrm{p}/(E+m)` y el peso de las dos
                         componentes de abajo frente al momento, para e y p ->
                         el límite ultrarrelativista lo fija la masa
:func:`latex`, :func:`muestra`, :func:`ok`
                         cosmética: matrices y espinores escritos en LaTeX, con
                         la marca de si la comprobación se cumple
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Math, display

#: masa del electrón [MeV], para la Bethe-Bloch
M_E_MEV = 0.511

#: constante de la Bethe-Bloch, K = 4 pi N_A r_e^2 m_e c^2 [MeV cm^2 / g]
K_BETHE = 0.307075


# ---------------------------------------------------------------------------
# 1. Experimental: masa transversa y pico jacobiano
# ---------------------------------------------------------------------------

def masa_transversa(n=200000, m_w=80.4, pt_w=15., sigma=4., seed=None,
                    verbose=True):
    """Simula :math:`W \\to e + \\bar{\\nu}` y dibuja la masa transversa.

    Del neutrino solo se conoce su momento **transverso**, así que no hay masa
    invariante. En su lugar se usa la **masa transversa**

    .. math:: m^2_T = 2 \\, p^e_T \\, p^{miss}_T \\, (1 - \\cos \\Delta\\phi)

    que cumple :math:`m_T \\leq m_W` y se acumula justo debajo de :math:`m_W`.

    El W se desintegra isótropamente en su sistema propio, con
    :math:`p^* = m_W/2`. Con el W en reposo el electrón y el neutrino salen
    espalda contra espalda, :math:`\\Delta\\phi = \\pi`, y entonces

    .. math:: m_T = 2 \\, p^e_T = m_W \\sin\\theta^*

    Como :math:`\\cos\\theta^*` se reparte uniforme, la densidad de :math:`m_T`
    diverge en :math:`m_T \\to m_W`: es el **pico jacobiano**, el borde con el que
    se descubrió el W en 1983 y con el que hoy se mide su masa.

    Se dibujan tres casos, cada uno peor que el anterior:

    1. W en reposo y medida perfecta: el borde es una pared vertical en
       :math:`m_W`;
    2. W con momento transverso ``pt_w``, retroceso medido, y :math:`p_T` medido
       con una resolución gaussiana ``sigma``: el borde se difumina;
    3. lo mismo, pero **sin medir el retroceso hadrónico**, tomando
       :math:`{\\bf p}^{miss}_T = -{\\bf p}^e_T`: el borde se desborda.

    El :math:`p_T` del W **por sí solo no rompe el borde** mientras el retroceso
    esté medido, porque entonces :math:`{\\bf p}^{miss}_T = {\\bf p}^\\nu_T` y
    :math:`m_T \\leq m_W` sigue siendo exacta: es la clave ``solo el pT del W`` de
    ``fraccion``. Lo que rompe el borde es la **medida** —la resolución y, sobre
    todo, un detector que deje escapar el retroceso—. Por eso se construyen
    herméticos.

    Parameters
    ----------
    n : int
        Número de desintegraciones simuladas.
    m_w : float
        Masa del W [GeV].
    pt_w : float
        Momento transverso del W en los casos 2 y 3 [GeV].
    sigma : float
        Resolución gaussiana por componente de :math:`p_T` en los casos 2 y 3
        [GeV].
    seed : int, optional
        Semilla del generador; por defecto, aleatoria.
    verbose : bool
        Si es ``True``, dibuja los tres histogramas y comenta el resultado.

    Returns
    -------
    dict
        ``mt_ideal``, ``mt_medida``, ``mt_sin_retroceso`` (arrays) y
        ``fraccion``, la fracción de sucesos por encima de :math:`m_W` en cada
        caso.
    """
    rng = np.random.default_rng(seed)

    # desintegración isótropa en el sistema del W: e y nu, espalda contra espalda
    cos_t = rng.uniform(-1., 1., n)
    phi = rng.uniform(0., 2 * np.pi, n)
    sin_t = np.sqrt(1. - cos_t**2)
    p = 0.5 * m_w                                 # |p*| de cada uno
    pe = p * np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])
    pn = -pe

    def al_laboratorio(pt_boost):
        """Impulso del W en el eje x; devuelve los p_T de e y de nu."""
        if pt_boost == 0.:
            return pe[:2], pn[:2]
        e_w = np.sqrt(m_w**2 + pt_boost**2)
        beta, gamma = pt_boost / e_w, e_w / m_w
        out = []
        for q in (pe, pn):
            energia = p                            # e y nu sin masa: E* = |p*|
            qx = gamma * (q[0] + beta * energia)
            out.append(np.array([qx, q[1]]))
        return out

    def mt(pt_e, pt_n):
        mod_e, mod_n = np.hypot(*pt_e), np.hypot(*pt_n)
        cos_dphi = (pt_e[0] * pt_n[0] + pt_e[1] * pt_n[1]) / (mod_e * mod_n)
        return np.sqrt(np.clip(2. * mod_e * mod_n * (1. - cos_dphi), 0., None))

    pt_e0, pt_n0 = al_laboratorio(0.)
    pt_e1, pt_n1 = al_laboratorio(pt_w)
    pt_e2 = pt_e1 + rng.normal(0., sigma, pt_e1.shape)
    pt_n2 = pt_n1 + rng.normal(0., sigma, pt_n1.shape)

    casos = (('ideal: W en reposo y medida perfecta', mt(pt_e0, pt_n0)),
             (f'W con $p_T$ = {pt_w:.0f} GeV y $\\sigma(p_T)$ = {sigma:.0f} GeV',
              mt(pt_e2, pt_n2)),
             ('... y sin medir el retroceso hadrónico', mt(pt_e2, -pt_e2)))
    fraccion = {nombre: float(np.mean(x > m_w)) for nombre, x in casos}
    fraccion['solo el pT del W'] = float(np.mean(mt(pt_e1, pt_n1) > m_w))

    if verbose:
        bins = np.linspace(0., 1.6 * m_w, 120)
        for nombre, x in casos:
            plt.hist(x, bins=bins, histtype='step', lw=1.8, label=nombre)
        plt.axvline(m_w, color='0.3', ls='--', lw=1.2)
        plt.text(m_w, plt.ylim()[1] * 0.95, r'  $m_W$', color='0.3', va='top')
        plt.xlabel(r'masa transversa $m_T$ (GeV)')
        plt.ylabel('sucesos')
        plt.legend(loc='upper left')
        plt.grid(alpha=0.3)
        for nombre, x in casos:
            print(f' {nombre:38} : máximo en mT = {_moda(x, bins):5.1f} GeV,'
                  f' {100 * fraccion[nombre]:6.2f} % por encima de mW')
        print(f' {"(el pT del W por sí solo, sin resolución)":38} :'
              f' {100 * fraccion["solo el pT del W"]:30.2f} % por encima de mW')

    return dict(mt_ideal=casos[0][1], mt_medida=casos[1][1],
                mt_sin_retroceso=casos[2][1], fraccion=fraccion)


def _moda(x, bins):
    """Centro del bin más poblado."""
    cuentas, bordes = np.histogram(x, bins=bins)
    i = int(np.argmax(cuentas))
    return 0.5 * (bordes[i] + bordes[i + 1])


# ---------------------------------------------------------------------------
# 2. Experimental: dE/dx y las bandas de una TPC
# ---------------------------------------------------------------------------

#: masas [MeV] de las partículas que deja una colisión y se ven en una TPC
MASAS = {'e': 0.511, r'$\mu$': 105.66, r'$\pi$': 139.57,
         'K': 493.68, 'p': 938.27}


def dedx(p_min=0.1, p_max=10., Z_A=0.5, I_eV=188., resolucion=0.07,
         n_puntos=1500, seed=None, verbose=True):
    """Pérdida de energía por ionización frente al momento: las bandas de una TPC.

    Bethe-Bloch, en la forma habitual para partículas pesadas y con
    :math:`T_{max} \\simeq 2 m_e c^2 \\beta^2\\gamma^2`:

    .. math::
        -\\frac{\\mathrm{d}E}{\\mathrm{d}x} = K \\, \\frac{Z}{A} \\,
        \\frac{z^2}{\\beta^2} \\left[
        \\ln \\frac{2 m_e c^2 \\beta^2 \\gamma^2}{I} - \\beta^2 \\right]

    A momento fijo, :math:`\\beta\\gamma = p/m` depende de la **masa**: cada
    partícula recorre su propia banda y por eso una TPC identifica. Las bandas se
    juntan a alto momento, donde todas están en la meseta relativista: ahí el
    :math:`\\mathrm{d}E/\\mathrm{d}x` deja de separar.

    Para el electrón la fórmula es solo indicativa —su cinemática y sus pérdidas
    radiativas son otras—, pero sitúa bien su banda: a estos momentos el electrón
    ya está en la meseta. La Bethe-Bloch tampoco vale por debajo de
    :math:`\\beta\\gamma \\simeq 0.1`, que para el protón es :math:`p \\simeq 0.1`
    GeV: el extremo izquierdo de la figura es el límite de validez, no física.

    Parameters
    ----------
    p_min, p_max : float
        Rango de momento [GeV].
    Z_A : float
        Razón Z/A del gas (≈ 0.5 para casi todo).
    I_eV : float
        Potencial medio de ionización [eV]; 188 eV para el argón.
    resolucion : float
        Resolución relativa del :math:`\\mathrm{d}E/\\mathrm{d}x` medido, que
        convierte cada curva en una banda.
    n_puntos : int
        Puntos simulados por especie.
    seed : int, optional
        Semilla del generador.
    verbose : bool
        Si es ``True``, dibuja las curvas y la nube de puntos medidos.

    Returns
    -------
    dict
        ``p`` (momento), ``curvas`` (dict de arrays) y ``separacion``: por cada
        par de bandas contiguas, el momento hasta el que se distinguen a más de
        2 sigma y, si lo hay, el momento al que se cruzan.
    """
    rng = np.random.default_rng(seed)

    def bethe(p_gev, m_mev):
        bg = 1e3 * p_gev / m_mev                   # beta*gamma = p/m
        beta2 = bg**2 / (1. + bg**2)
        log = np.log(2e6 * M_E_MEV * bg**2 / I_eV)
        return K_BETHE * Z_A / beta2 * (log - beta2)

    p = np.logspace(np.log10(p_min), np.log10(p_max), 400)
    curvas = {nombre: bethe(p, m) for nombre, m in MASAS.items()}

    # separación entre bandas contiguas, en unidades de la resolución
    orden = sorted(MASAS, key=lambda k: MASAS[k])
    separacion = {}
    for a, b in zip(orden[:-1], orden[1:]):
        ya, yb = bethe(p, MASAS[a]), bethe(p, MASAS[b])
        n_sigma = np.abs(ya - yb) / (resolucion * 0.5 * (ya + yb))
        buenos = n_sigma >= 2.
        if not buenos.any():
            separacion[f'{a}/{b}'] = None
            continue
        p_lim = float(p[buenos][-1])
        # por debajo del límite las dos bandas pueden llegar a cruzarse
        malos = np.where(~buenos & (p < p_lim))[0]
        cruce = float(p[malos[np.argmin(n_sigma[malos])]]) if len(malos) else None
        separacion[f'{a}/{b}'] = (p_lim, cruce)

    if verbose:
        for nombre, m in MASAS.items():
            pp = 10**rng.uniform(np.log10(p_min), np.log10(p_max), n_puntos)
            yy = bethe(pp, m) * rng.normal(1., resolucion, n_puntos)
            puntos, = plt.plot(pp, yy, '.', ms=1.2, alpha=0.3)
            plt.plot(p, curvas[nombre], lw=1.6, label=nombre,
                     color=puntos.get_color())
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('momento $p$ (GeV)')
        plt.ylabel(r'$\mathrm{d}E/\mathrm{d}x$ (MeV cm$^2$/g)')
        plt.ylim(0.8 * min(c.min() for c in curvas.values()), None)
        plt.legend(ncol=5, loc='upper right', fontsize=10)
        plt.grid(alpha=0.3, which='both')
        for par, dato in separacion.items():
            if dato is None:
                print(f' {par:9} : no se distinguen a más de 2 sigma')
                continue
            p_lim, cruce = dato
            aviso = '' if cruce is None else f', pero se cruzan en {cruce:5.2f} GeV'
            print(f' {par:9} : se distinguen (> 2 sigma) hasta p ='
                  f' {p_lim:5.2f} GeV{aviso}')

    return dict(p=p, curvas=curvas, separacion=separacion)


# ---------------------------------------------------------------------------
# 2b. Experimental: un pico sobre el fondo y su significancia
# ---------------------------------------------------------------------------

def pico_sobre_fondo(lumis=(1., 10., 100.), m_h=125., sigma_m=1.7, s_por_fb=18.,
                     b_por_fb=5500., pendiente=0.03, m_min=100., m_max=160.,
                     ancho_bin=0.5, sistematico=0., seed=None, verbose=True):
    """Un pico de :math:`H \\to \\gamma\\gamma` sobre el fondo, acumulando luminosidad.

    El espectro de masa invariante :math:`m_{\\gamma\\gamma}` tiene un fondo
    exponencial que cae suavemente y, encima, un pico gaussiano de anchura
    ``sigma_m`` —la resolución del calorímetro— en :math:`m_H`. Los números por
    defecto son los del canal :math:`\\gamma\\gamma` de ATLAS en 2011-2012: unos 18
    sucesos de señal y unos 5500 de fondo por fb\\ :sup:`-1` entre 100 y 160 GeV.

    Como en :func:`fnyp.simula.vida_media_evolucion`, es **una sola toma de datos**
    que se va acumulando: cada luminosidad contiene los sucesos de las anteriores.
    En cada paso se estima el fondo bajo el pico ajustando una exponencial a las
    **bandas laterales** (fuera de :math:`m_H \\pm 2\\sigma_m`), como en el análisis
    real, y se calcula la significancia de conteo

    .. math:: Z = \\frac{N - B}{\\sqrt{B + (\\delta B)^2}}, \\qquad \\delta B = s \\, B

    con :math:`s` = ``sistematico`` la incertidumbre relativa sobre el fondo. Sin
    sistemático, :math:`S \\propto \\mathcal{L}` y :math:`\\sqrt{B} \\propto
    \\sqrt{\\mathcal{L}}`, así que :math:`Z \\propto \\sqrt{\\mathcal{L}}`. Con él,
    :math:`Z` satura en :math:`S/(sB)`: tomar más datos deja de ayudar.

    Es una simplificación deliberada: ni categorías, ni ajuste de verosimilitud, ni
    error estadístico del propio ajuste de las bandas laterales.

    Parameters
    ----------
    lumis : sequence of float
        Luminosidades integradas [fb^-1] a las que se dibuja el histograma.
    m_h, sigma_m : float
        Posición y anchura (resolución) del pico [GeV].
    s_por_fb, b_por_fb : float
        Sucesos de señal y de fondo esperados por fb^-1, en todo el rango.
    pendiente : float
        Pendiente de la exponencial del fondo [1/GeV].
    m_min, m_max, ancho_bin : float
        Rango y anchura de bin del histograma [GeV].
    sistematico : float
        Incertidumbre relativa sobre el fondo bajo el pico (0.01 = 1 %).
    seed : int, optional
        Semilla del generador; por defecto, aleatoria.
    verbose : bool
        Si es ``True``, dibuja los histogramas y la curva de significancia.

    Returns
    -------
    dict
        ``pasos`` (luminosidades), ``z_obs`` y ``z_esp`` (significancia observada
        y esperada en cada paso), ``s_ventana`` y ``b_ventana`` (señal y fondo
        esperados por fb^-1 en la ventana), ``l_5sigma`` (luminosidad para 5 sigma
        esperadas, ``inf`` si no se alcanza), ``bordes`` e ``histogramas``.
    """
    from scipy.special import erf

    rng = np.random.default_rng(seed)
    lumis = sorted(float(l) for l in lumis)

    bordes = np.arange(m_min, m_max + 0.5 * ancho_bin, ancho_bin)
    centros = 0.5 * (bordes[1:] + bordes[:-1])

    # sucesos esperados por bin y por fb^-1: exponencial normalizada + gaussiana
    forma_b = np.exp(-pendiente * (bordes - m_min))
    mu_b = b_por_fb * (forma_b[:-1] - forma_b[1:]) / (forma_b[0] - forma_b[-1])
    cdf = 0.5 * (1. + erf((bordes - m_h) / (np.sqrt(2.) * sigma_m)))
    mu_s = s_por_fb * np.diff(cdf)

    ventana = np.abs(centros - m_h) < 2. * sigma_m
    s_w, b_w = float(mu_s[ventana].sum()), float(mu_b[ventana].sum())

    def z_esperada(lum):
        s, b = s_w * lum, b_w * lum
        return s / np.sqrt(b + (sistematico * b)**2)

    # una sola toma de datos: se suman incrementos de Poisson paso a paso
    pasos = np.unique(np.concatenate([np.geomspace(min(0.3, lumis[0]), lumis[-1], 40),
                                      lumis]))
    cuentas = np.zeros_like(centros)
    z_obs, histogramas, estimas = [], {}, {}
    anterior = 0.
    for lum in pasos:
        cuentas = cuentas + rng.poisson((mu_b + mu_s) * (lum - anterior))
        anterior = lum
        # fondo bajo el pico: exponencial ajustada a las bandas laterales
        lat = ~ventana & (cuentas > 0)
        coef = np.polyfit(centros[lat], np.log(cuentas[lat]), 1, w=np.sqrt(cuentas[lat]))
        ajuste = np.exp(np.polyval(coef, centros))
        n_w, b_est = cuentas[ventana].sum(), ajuste[ventana].sum()
        z_obs.append((n_w - b_est) / np.sqrt(b_est + (sistematico * b_est)**2))
        if np.isclose(lumis, lum).any():
            histogramas[lum] = cuentas.copy()
            estimas[lum] = (ajuste, n_w, b_est, z_obs[-1])
    z_obs = np.array(z_obs)
    z_esp = z_esperada(pasos)

    denominador = s_w**2 - 25. * (sistematico * b_w)**2
    l_5sigma = 25. * b_w / denominador if denominador > 0 else np.inf

    if verbose:
        # arriba el espectro; abajo datos menos fondo, donde el pico sí se ve
        fig, axes = plt.subplots(2, len(lumis), figsize=(3.4 * len(lumis), 4.6),
                                 sharex=True, squeeze=False,
                                 gridspec_kw=dict(height_ratios=(2, 1)))
        for j, lum in enumerate(lumis):
            ax, ax_r = axes[0, j], axes[1, j]
            h, ajuste = histogramas[lum], estimas[lum][0]
            ax.errorbar(centros, h, yerr=np.sqrt(h), fmt='o', ms=2, lw=0.8, color='k')
            ax.plot(centros, ajuste, 'C3--', lw=1.2, label='fondo (bandas laterales)')
            ax.set_title(f'$\\mathcal{{L}}$ = {lum:g} fb$^{{-1}}$', fontsize=10)
            ax_r.errorbar(centros, h - ajuste, yerr=np.sqrt(h), fmt='o', ms=2, lw=0.8,
                          color='k')
            ax_r.plot(centros, mu_s * lum, 'C0-', lw=1.5, label='señal esperada')
            ax_r.axhline(0., color='C3', ls='--', lw=1)
            ax_r.set_xlabel(r'$m_{\gamma\gamma}$ (GeV)')
            for a in (ax, ax_r):
                a.axvspan(m_h - 2 * sigma_m, m_h + 2 * sigma_m, color='C0', alpha=0.12)
                a.grid(alpha=0.3)
        axes[0, 0].set_ylabel(f'sucesos / {ancho_bin:g} GeV')
        axes[1, 0].set_ylabel('datos $-$ fondo')
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].legend(fontsize=8, loc='lower left')
        fig.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(5.5, 3.4))
        ax2.plot(pasos, z_obs, 'o', ms=3.5, color='k', label='observada')
        ax2.plot(pasos, z_esp, 'C0-', lw=1.5, label='esperada')
        if sistematico > 0:
            ax2.plot(pasos, s_w * pasos / np.sqrt(b_w * pasos), 'C0:', lw=1.2,
                     label='esperada, sin sistemático')
        for z, texto in ((3., 'evidencia'), (5., 'descubrimiento')):
            ax2.axhline(z, color='0.5', ls='--', lw=1)
            ax2.text(pasos[0], z + 0.1, f' {z:.0f}$\\sigma$: {texto}', fontsize=8, color='0.3')
        ax2.set_xscale('log')
        ax2.set_xlabel(r'luminosidad integrada $\mathcal{L}$ (fb$^{-1}$)')
        ax2.set_ylabel(r'significancia $Z$ ($\sigma$)')
        ax2.grid(alpha=0.3, which='both')
        ax2.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0., 0.9))
        fig2.tight_layout()

        print(f' en la ventana {m_h - 2 * sigma_m:.1f}-{m_h + 2 * sigma_m:.1f} GeV, por fb^-1:'
              f'  S = {s_w:.1f},  B = {b_w:.0f}')
        for lum in lumis:
            _, n_w, b_est, z = estimas[lum]
            print(f'   L = {lum:6g} fb^-1 : N = {n_w:8.0f},  B estimado = {b_est:8.0f},'
                  f'  N - B = {n_w - b_est:6.0f}  ->  Z = {z:4.1f} sigma'
                  f'  (esperada {z_esperada(lum):4.1f})')
        if np.isfinite(l_5sigma):
            print(f' 5 sigma esperadas con L = {l_5sigma:.0f} fb^-1')
        else:
            print(f' 5 sigma esperadas: nunca. Z satura en S/(s B) = '
                  f'{s_w / (sistematico * b_w):.1f} sigma')

    return dict(pasos=pasos, z_obs=z_obs, z_esp=z_esp, s_ventana=s_w, b_ventana=b_w,
                l_5sigma=l_5sigma, bordes=bordes, histogramas=histogramas)


# ---------------------------------------------------------------------------
# 3. Teoría: las matrices de Dirac en la representación de Pauli-Dirac
# ---------------------------------------------------------------------------

def matrices_dirac():
    """Matrices :math:`\\gamma^\\mu`, :math:`\\gamma^5` y proyectores de quiralidad.

    En la representación de **Pauli-Dirac**, en bloques :math:`2\\times2`:

    .. math::
        \\gamma^0 = \\begin{pmatrix} I & 0 \\\\ 0 & -I \\end{pmatrix}, \\;\\;
        \\gamma^k = \\begin{pmatrix} 0 & \\sigma_k \\\\ -\\sigma_k & 0 \\end{pmatrix},
        \\;\\; \\gamma^5 = i \\gamma^0\\gamma^1\\gamma^2\\gamma^3

    :math:`\\gamma^5` se calcula aquí a partir de las otras cuatro, no se escribe
    a mano: así la comprobación del notebook es una comprobación de verdad.

    Returns
    -------
    dict
        ``sigma`` (las tres de Pauli), ``gamma`` (array ``(4,4,4)``, índice
        :math:`\\mu` primero), ``gamma5``, ``I2``, ``I4``, ``metrica``
        :math:`g^{\\mu\\nu}`, y los proyectores ``P_L``, ``P_R``.
    """
    I2 = np.eye(2, dtype=complex)
    cero = np.zeros((2, 2), dtype=complex)
    sigma = np.array([[[0, 1], [1, 0]],
                      [[0, -1j], [1j, 0]],
                      [[1, 0], [0, -1]]], dtype=complex)

    def bloques(a, b, c, d):
        return np.block([[a, b], [c, d]])

    gamma = np.array([bloques(I2, cero, cero, -I2)] +
                     [bloques(cero, s, -s, cero) for s in sigma])
    gamma5 = 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    I4 = np.eye(4, dtype=complex)

    return dict(sigma=sigma, gamma=gamma, gamma5=gamma5, I2=I2, I4=I4,
                metrica=np.diag([1., -1., -1., -1.]),
                P_L=0.5 * (I4 - gamma5), P_R=0.5 * (I4 + gamma5))


def _numero(z, dec=3):
    """Un número complejo en LaTeX: enteros y ``i`` limpios, el resto con ``dec`` cifras."""
    z = complex(z)
    re = 0. if abs(z.real) < 1e-12 else z.real
    im = 0. if abs(z.imag) < 1e-12 else z.imag

    def real(x):
        return f'{int(round(x))}' if abs(x - round(x)) < 1e-12 else f'{x:.{dec}g}'

    if im == 0.:
        return real(re)
    imag = {1.: 'i', -1.: '-i'}.get(im, real(im) + 'i')
    if re == 0.:
        return imag
    return real(re) + ('' if imag.startswith('-') else '+') + imag


def latex(x, dec=3):
    """Escalar, vector (en columna) o matriz, en LaTeX (``pmatrix``)."""
    x = np.asarray(x)
    if x.ndim == 0:
        return _numero(x, dec)
    filas = x.reshape(-1, 1) if x.ndim == 1 else x
    cuerpo = r' \\ '.join(' & '.join(_numero(z, dec) for z in fila)
                          for fila in filas)
    return r'\begin{pmatrix} ' + cuerpo + r' \end{pmatrix}'


def ok(cierto):
    """Marca LaTeX de una comprobación: tic verde o aspa roja."""
    return (r'\quad \color{green}{\checkmark}' if cierto
            else r'\quad \color{red}{\times}')


def muestra(*piezas):
    """Escribe en una línea, en LaTeX, varias expresiones separadas por espacios."""
    display(Math(r' \qquad '.join(piezas)))


def espinores(p, m):
    """Los cuatro espinores con **p** en el eje z, y :math:`\\gamma^\\mu p_\\mu`.

    .. math::
        u_1 = N \\begin{pmatrix} 1 \\\\ 0 \\\\ \\kappa \\\\ 0 \\end{pmatrix}, \\;\\;
        u_2 = N \\begin{pmatrix} 0 \\\\ 1 \\\\ 0 \\\\ -\\kappa \\end{pmatrix}, \\;\\;
        v_1 = N \\begin{pmatrix} 0 \\\\ -\\kappa \\\\ 0 \\\\ 1 \\end{pmatrix}, \\;\\;
        v_2 = N \\begin{pmatrix} \\kappa \\\\ 0 \\\\ 1 \\\\ 0 \\end{pmatrix}

    con :math:`\\kappa = \\mathrm{p}/(E+m)` y :math:`N = \\sqrt{E+m}`, que son los
    de la transparencia. Deben cumplir
    :math:`(\\gamma^\\mu p_\\mu - m) u_s = 0` y
    :math:`(\\gamma^\\mu p_\\mu + m) v_s = 0`.

    Parameters
    ----------
    p : float
        Módulo del momento, en el eje z (mismas unidades que ``m``).
    m : float
        Masa.

    Returns
    -------
    dict
        ``u1``, ``u2``, ``v1``, ``v2``, ``pslash`` (:math:`\\gamma^\\mu p_\\mu`),
        ``E``, ``kappa`` y ``N``.
    """
    d = matrices_dirac()
    E = np.sqrt(p**2 + m**2)
    k, N = p / (E + m), np.sqrt(E + m)

    u1 = N * np.array([1., 0., k, 0.], dtype=complex)
    u2 = N * np.array([0., 1., 0., -k], dtype=complex)
    v1 = N * np.array([0., -k, 0., 1.], dtype=complex)
    v2 = N * np.array([k, 0., 1., 0.], dtype=complex)

    pslash = E * d['gamma'][0] - p * d['gamma'][3]      # p^mu = (E, 0, 0, p)

    return dict(u1=u1, u2=u2, v1=v1, v2=v2, pslash=pslash, E=E, kappa=k, N=N)


# ---------------------------------------------------------------------------
# 4. Teoría: kappa y las componentes pequeñas del espinor
# ---------------------------------------------------------------------------

#: casos marcados en la figura: (etiqueta, masa [GeV], momento [GeV], offset)
#: partículas de la figura de kappa: nombre -> masa [GeV]
MASAS_KAPPA = {r'$e$': 0.511e-3, r'$p$': 0.938}

#: casos marcados en la figura de kappa: (etiqueta, partícula, p [GeV], offset)
CASOS_KAPPA = ((r'$e$, 1 MeV', r'$e$', 1e-3, (-44, 6)),
               (r'$e$, 1 GeV', r'$e$', 1., (-20, 8)),
               (r'$p$, 1 GeV', r'$p$', 1., (8, -12)),
               (r'$p$, 7 TeV', r'$p$', 7e3, (-24, 8)))

#: peso de las componentes de abajo marcado en la figura
PESO_REF = 0.4


def _kappa(p, m):
    """:math:`\\kappa = \\mathrm{p}/(E+m)`, con p y m en las mismas unidades."""
    return p / (np.sqrt(p**2 + m**2) + m)


def _peso(k):
    """Peso de las dos componentes de abajo, :math:`\\kappa^2/(1+\\kappa^2)`."""
    return k**2 / (1. + k**2)


def _kappa_de_peso(w):
    """Inversa de :func:`_peso`: :math:`\\kappa = \\sqrt{w/(1-w)}`."""
    w = np.clip(w, 0., 0.999)
    return np.sqrt(w / (1. - w))


def kappa_espinor(verbose=True):
    """:math:`\\kappa` frente al momento, para el electrón y el protón.

    Con **p** en el eje z, :math:`u_1 = N(1, 0, \\kappa, 0)` con
    :math:`\\kappa = \\mathrm{p}/(E+m)`. El peso de las dos componentes de abajo
    —las que enciende el *boost*— es

    .. math:: w = \\frac{\\kappa^2}{1 + \\kappa^2}

    Se dibuja solo :math:`\\kappa`; el eje de la derecha es el mismo leído como
    peso :math:`w` (no es otra curva: :math:`w` es función de :math:`\\kappa`).

    Las dos curvas tienen la **misma forma**, desplazada en :math:`\\log p` por
    :math:`m_p/m_e \\simeq 1836`: ambas dependen solo de
    :math:`\\beta\\gamma = \\mathrm{p}/m` (:math:`\\kappa = \\beta\\gamma/(\\gamma + 1)`).
    En reposo :math:`\\kappa = 0` y queda el espinor de Pauli; en el límite
    ultrarrelativista :math:`\\kappa \\to 1` y arriba y abajo pesan igual
    (:math:`w \\to 1/2`). Ese es el régimen en el que la quiralidad se confunde
    con la helicidad, y en el que trabaja la física de partículas.

    Parameters
    ----------
    verbose : bool
        Si es ``True``, dibuja las curvas, marca algunos casos y el momento en
        el que el peso de abajo llega al 40 %.

    Returns
    -------
    dict
        ``p`` (GeV), ``kappa`` y ``peso`` (dicts por partícula), ``p_ref``
        (momento en GeV con peso 40 %, por partícula) y ``casos``.
    """
    p = np.logspace(-5., 4., 600)                  # momento [GeV]
    kappa = {nombre: _kappa(p, m) for nombre, m in MASAS_KAPPA.items()}
    peso = {nombre: _peso(k) for nombre, k in kappa.items()}

    # peso 40 %  ->  kappa_ref  ->  beta*gamma = 2 kappa / (1 - kappa^2)
    k_ref = _kappa_de_peso(PESO_REF)
    bg_ref = 2. * k_ref / (1. - k_ref**2)
    p_ref = {nombre: bg_ref * m for nombre, m in MASAS_KAPPA.items()}

    casos = {}
    for etiqueta, nombre, pi, _ in CASOS_KAPPA:
        m = MASAS_KAPPA[nombre]
        ki = _kappa(pi, m)
        casos[etiqueta] = (pi, pi / m, ki, _peso(ki))

    if verbose:
        fig, ax = plt.subplots()
        for nombre in MASAS_KAPPA:
            linea, = ax.plot(p, kappa[nombre], lw=2, label=nombre)
            ax.axvline(p_ref[nombre], color=linea.get_color(), lw=1, ls=':')
        ax.axhline(k_ref, color='0.6', lw=1, ls='--',
                   label=f'peso abajo = {100 * PESO_REF:.0f} %')
        for etiqueta, _, pi, offset in CASOS_KAPPA:
            ki = casos[etiqueta][2]
            ax.plot([pi], [ki], 'o', ms=6, color='0.25')
            ax.annotate(etiqueta, (pi, ki), textcoords='offset points',
                        xytext=offset, fontsize=9, color='0.25')
        ax.set_xscale('log')
        ax.set_xlabel('momento p (GeV)')
        ax.set_ylabel(r'$\kappa = \mathrm{p}/(E+m)$')
        ax.set_ylim(0., 1.02)
        eje_w = ax.secondary_yaxis('right', functions=(_peso, _kappa_de_peso))
        eje_w.set_ylabel(r'peso de las componentes de abajo, $\kappa^2/(1+\kappa^2)$')
        eje_w.set_yticks([0., 0.1, 0.2, 0.3, 0.4, 0.45, 0.5])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3, which='both')
        for nombre, pr in p_ref.items():
            print(f' {nombre.strip("$"):5} : peso abajo = {100 * PESO_REF:.0f} % en p = {1e3 * pr:8.1f} MeV'
                  f'  (p/m = {bg_ref:.2f})')
        for etiqueta, (pi, xi, ki, wi) in casos.items():
            print(f' {etiqueta.replace("$", ""):12} : p/m = {xi:9.1f}, kappa = {ki:5.3f},'
                  f' peso abajo = {100 * wi:4.1f} %')

    return dict(p=p, kappa=kappa, peso=peso, p_ref=p_ref, casos=casos)
