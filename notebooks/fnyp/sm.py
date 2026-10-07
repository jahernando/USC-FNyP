"""
Código auxiliar del tema del *Modelo Estándar*.

Mismo criterio que el resto del paquete: aquí va la cosmética de las figuras; la
física —el potencial, dónde está el mínimo, la curvatura— se escribe y se discute
en el notebook.

:func:`potencial_higgs`      el potencial :math:`V(\\phi) = \\mu^2\\phi^2/2 + \\lambda\\phi^4/4`
                             para varios :math:`\\mu^2`, de positivo a negativo, y cómo
                             cambian el mínimo :math:`v` y la masa de la oscilación
                             alrededor de él -> la rotura espontánea de simetría
:func:`sombrero`             el mismo potencial para un campo complejo, en 3D: del
                             cuenco (:math:`\\mu^2 > 0`) al sombrero mexicano (:math:`\\mu^2 < 0`)
:func:`potencial_interactivo` las dos figuras con un deslizador para :math:`\\mu^2`
                             (necesita ``ipywidgets``; no funciona en el Book)
"""

import numpy as np
import matplotlib.pyplot as plt

#: paleta Okabe-Ito (apta para daltónicos), de positivo a negativo en mu^2
OKABE_ITO = ['#0072B2', '#56B4E9', '#000000', '#E69F00', '#D55E00']
ESTILOS = ['-', '--', '-', '-.', ':']


def V(phi, mu2, lam=1.):
    """Potencial de un campo escalar real: mu^2 phi^2 / 2 + lambda phi^4 / 4."""
    return mu2 * phi**2 / 2 + lam * phi**4 / 4


def minimo(mu2, lam=1.):
    """Valor del campo en el vacío: 0 si mu^2 >= 0, sqrt(-mu^2/lambda) si mu^2 < 0."""
    return np.sqrt(np.clip(-np.asarray(mu2, float), 0, None) / lam)


def masa(mu2, lam=1.):
    """Masa de la oscilación alrededor del mínimo, m^2 = V''(v):
    mu^2 si mu^2 > 0;  -2 mu^2 = 2 lambda v^2 si mu^2 < 0."""
    mu2 = np.asarray(mu2, float)
    return np.sqrt(np.where(mu2 >= 0, mu2, -2 * mu2))


def potencial_higgs(mu2s=(1., 0.4, 0., -0.4, -1.), lam=1., marca=None, axs=None):
    """Izda.: V(phi) para varios mu^2, con su mínimo. Dcha.: v y la masa frente a mu^2.

    ``marca`` señala un mu^2 concreto en la figura de la derecha (lo usa el modo
    interactivo)."""
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1, ax2 = axs
    phis = np.linspace(-1.7, 1.7, 400)
    for k, mu2 in enumerate(mu2s):
        c, ls = OKABE_ITO[k % 5], ESTILOS[k % 5]
        ax1.plot(phis, V(phis, mu2, lam), c=c, ls=ls, lw=2, label=rf'$\mu^2 = {mu2:+.1f}$')
        v = minimo(mu2, lam)
        ax1.plot([-v, v] if v > 0 else [0.], [V(v, mu2, lam)] * (2 if v > 0 else 1), 'o', c=c, ms=7)
    ax1.set_ylim(-0.35, 1.0)
    ax1.axhline(0, c='grey', lw=0.5)
    ax1.set_xlabel(r'$\phi$')
    ax1.set_ylabel(r'$V(\phi)$')
    ax1.set_title(r'el potencial y su mínimo ($\lambda = 1$)')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    m2 = np.linspace(-1.2, 1.2, 400)
    ax2.plot(m2, minimo(m2, lam), c='#0072B2', lw=2, label=r'mínimo, $v$')
    ax2.plot(m2, masa(m2, lam), c='#D55E00', lw=2, ls='--', label=r'masa de la oscilación, $\sqrt{V^{\prime\prime}}$')
    ax2.axvline(0, c='grey', lw=0.5)
    ax2.text(0.6, 0.05, 'simetría\nintacta', ha='center')
    ax2.text(-0.6, 0.05, 'simetría\nrota', ha='center')
    if marca is not None:
        ax2.axvline(marca, c='k', ls=':')
    ax2.set_xlabel(r'$\mu^2$')
    ax2.set_title(r'$v$ y la masa frente a $\mu^2$')
    ax2.legend(fontsize=9, loc='upper center')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    return axs


def sombrero(mu2=-1., lam=1., ax=None):
    """V(|phi|) para un campo complejo, en 3D, con el vacío marcado."""
    if ax is None:
        ax = plt.figure(figsize=(5.5, 4.5)).add_subplot(projection='3d')
    tope = 0.3                                          # se corta el potencial en V = tope
    r_max = np.sqrt((-mu2 + np.sqrt(mu2**2 + 4 * lam * tope)) / lam)
    r, th = np.meshgrid(np.linspace(0, r_max, 16), np.linspace(0, 2 * np.pi, 41))
    x, y = r * np.cos(th), r * np.sin(th)
    ax.plot_wireframe(x, y, V(r, mu2, lam), color='#0072B2', lw=0.7)
    v = minimo(mu2, lam)
    ax.scatter([v], [0], [V(v, mu2, lam)], c='#D55E00', s=60, depthshade=False)
    ax.view_init(elev=28, azim=-60)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_zlim(-0.3, tope)
    ax.set_xlabel(r'Re $\phi$')
    ax.set_ylabel(r'Im $\phi$')
    ax.set_zticks([])
    ax.set_title(rf'$\mu^2 = {mu2:+.2f}$: ' + ('un mínimo, en el centro' if mu2 >= 0
                                                 else 'un círculo de mínimos'))
    return ax


def potencial_interactivo(lam=1.):
    """El potencial y el sombrero con un deslizador para mu^2 (necesita ipywidgets)."""
    try:
        from ipywidgets import FloatSlider, interact
    except ModuleNotFoundError:
        print(' Hace falta ipywidgets: pip install ipywidgets. Mientras, usa sombrero(mu2) con varios valores.')
        return

    def dibuja(mu2):
        fig = plt.figure(figsize=(15, 4.2))
        ax1, ax2 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2)
        potencial_higgs((mu2,), lam, marca=mu2, axs=(ax1, ax2))
        sombrero(mu2, lam, ax=fig.add_subplot(1, 3, 3, projection='3d'))
        plt.show()

    interact(dibuja, mu2=FloatSlider(value=1., min=-1., max=1., step=0.05, description='μ²'))
