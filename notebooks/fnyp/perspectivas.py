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

Parte teórica
-------------
:func:`matrices_dirac`   las matrices :math:`\\gamma^\\mu`, :math:`\\gamma^5` y los
                         proyectores de quiralidad, en la representación de
                         Pauli-Dirac
:func:`espinores`        los cuatro espinores :math:`u_s(p), v_s(p)` con **p** en
                         el eje z, y :math:`\\gamma^\\mu p_\\mu`
:func:`kappa_espinor`    :math:`\\kappa = \\mathrm{p}/(E+m)` y el peso de las dos
                         componentes de abajo -> el límite ultrarrelativista
"""

import numpy as np
import matplotlib.pyplot as plt

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

    # desintegracion isotropa en el sistema del W: e y nu, espalda contra espalda
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
            print(f' {nombre:38} : maximo en mT = {_moda(x, bins):5.1f} GeV,'
                  f' {100 * fraccion[nombre]:6.2f} % por encima de mW')
        print(f' {"(el pT del W por si solo, sin resolucion)":38} :'
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
        par de bandas contiguas, el momento hasta el que se distinguen a mas de
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

    # separacion entre bandas contiguas, en unidades de la resolucion
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
        # por debajo del limite las dos bandas pueden llegar a cruzarse
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
                print(f' {par:9} : no se distinguen a mas de 2 sigma')
                continue
            p_lim, cruce = dato
            aviso = '' if cruce is None else f', pero se cruzan en {cruce:5.2f} GeV'
            print(f' {par:9} : se distinguen (> 2 sigma) hasta p ='
                  f' {p_lim:5.2f} GeV{aviso}')

    return dict(p=p, curvas=curvas, separacion=separacion)


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
CASOS_KAPPA = ((r'$e$, 1 MeV', 0.511e-3, 1e-3, (8, -4)),
               (r'$p$, 1 GeV', 0.938, 1., (8, -12)),
               (r'$e$, 1 GeV', 0.511e-3, 1., (-16, -16)),
               (r'$p$, 7 TeV', 0.938, 7e3, (-24, -28)))


def kappa_espinor(verbose=True):
    """:math:`\\kappa` y el peso de las dos componentes de abajo del espinor.

    Con **p** en el eje z, :math:`u_1 = N(1, 0, \\kappa, 0)` con
    :math:`\\kappa = \\mathrm{p}/(E+m)`. El peso de las dos componentes de abajo
    —las que enciende el *boost*— es

    .. math:: w = \\frac{\\kappa^2}{1 + \\kappa^2}

    Ambas dependen **solo** de :math:`\\mathrm{p}/m`, no de la partícula: en
    reposo :math:`\\kappa = 0` y queda el espinor de Pauli; en el límite
    ultrarrelativista :math:`\\kappa \\to 1` y arriba y abajo pesan igual
    (:math:`w \\to 1/2`). Ese es el régimen en el que la quiralidad se confunde
    con la helicidad, y en el que trabaja la física de partículas.

    Parameters
    ----------
    verbose : bool
        Si es ``True``, dibuja las dos curvas y marca algunos casos.

    Returns
    -------
    dict
        ``x`` (:math:`\\mathrm{p}/m`), ``kappa``, ``peso`` y ``casos``.
    """
    x = np.logspace(-2., 4., 600)                  # p/m
    k = x / (np.sqrt(1. + x**2) + 1.)              # kappa = p/(E+m)
    w = k**2 / (1. + k**2)

    casos = {}
    for etiqueta, m, p, _ in CASOS_KAPPA:
        xi = p / m
        ki = xi / (np.sqrt(1. + xi**2) + 1.)
        casos[etiqueta] = (xi, ki, ki**2 / (1. + ki**2))

    if verbose:
        plt.plot(x, k, lw=2, label=r'$\kappa = \mathrm{p}/(E+m)$')
        plt.plot(x, w, lw=2, ls='--',
                 label=r'peso de las componentes de abajo, $\kappa^2/(1+\kappa^2)$')
        plt.axhline(0.5, color='0.6', lw=1, ls=':')
        for etiqueta, _, _, offset in CASOS_KAPPA:
            xi, ki, _w = casos[etiqueta]
            plt.plot([xi], [ki], 'o', ms=6, color='0.25')
            plt.annotate(etiqueta, (xi, ki), textcoords='offset points',
                         xytext=offset, fontsize=9, color='0.25')
        plt.xscale('log')
        plt.xlabel(r'$\mathrm{p}/m$')
        plt.ylabel(r'$\kappa$,  peso')
        plt.ylim(0., 1.05)
        plt.legend(loc='upper left', fontsize=9)
        plt.grid(alpha=0.3, which='both')
        for etiqueta, (xi, ki, wi) in casos.items():
            print(f' {etiqueta:12} : p/m = {xi:9.1f}, kappa = {ki:5.3f},'
                  f' peso abajo = {100 * wi:4.1f} %')

    return dict(x=x, kappa=k, peso=w, casos=casos)
