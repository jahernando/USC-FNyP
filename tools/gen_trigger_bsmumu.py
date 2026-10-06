"""Genera los datos del ejercicio de trigger de muones del boletín III (26/27).

Dos muestras de sucesos con los dos muones dentro de la aceptancia del detector:

* ``trigger_bs2mumu.csv``     B_s -> mu+ mu-   (señal: muones desplazados)
* ``trigger_jpsi_prompt.csv`` J/psi -> mu+ mu- prompt (fondo: muones del vértice primario)

Columnas, una fila por suceso: ``p1, pt1, ip1, p2, pt2, ip2`` (p y pT en GeV, IP en micras).
No se dan ángulos, para que no se pueda reconstruir la masa invariante: el trigger
que se diseña es de un solo muón.

Modelo (simplificado, de juguete pero con los órdenes de magnitud de LHCb):

* espectro de la madre dN/dpT ~ pT / (1 + (pT/p0)^2)^n, eta uniforme en [2, 5];
* desintegración isótropa a dos cuerpos en el reposo de la madre;
* B_s con tiempo de vida exponencial (tau = 1.520 ps); el J/psi prompt nace en el vértice primario;
* aceptancia del muón: 2 < eta < 5 y p > 6 GeV (tiene que atravesar el hierro hasta las cámaras);
* IP = distancia de la traza al vértice primario, con la resolución de LHCb
  sigma_IP = (15 + 29/pT[GeV]) um en cada una de las dos direcciones transversas a la traza.

Uso:  python tools/gen_trigger_bsmumu.py   (escribe en notebooks/bols/datos/)
"""

from pathlib import Path

import numpy as np

SEED = 2627
N_EVENTS = 5000                  # sucesos aceptados por muestra
M_MU = 0.10566                   # GeV
SAMPLES = {
    # nombre: (masa GeV, c*tau um, p0 GeV, n)
    'bs2mumu':     (5.36692, 455.7, 5.5, 3.0),
    'jpsi_prompt': (3.09690,   0.0, 3.3, 3.0),
}
ETA_RANGE = (2.0, 5.0)
P_MIN = 6.0                      # GeV

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / 'bols' / 'datos'


def sample_pt(rng, n, p0, nexp):
    # inversa de la CDF de dN/dpT ~ pT (1 + (pT/p0)^2)^(-n)
    u = rng.random(n)
    return p0 * np.sqrt((1 - u) ** (1 / (1 - nexp)) - 1)


def generate(rng, n, mass, ctau, p0, nexp):
    pt = sample_pt(rng, n, p0, nexp)
    eta = rng.uniform(*ETA_RANGE, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    pvec = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], axis=1)
    E = np.sqrt((pvec ** 2).sum(1) + mass ** 2)

    # desintegración isótropa en el reposo de la madre
    ps = np.sqrt(mass ** 2 / 4 - M_MU ** 2)
    cth = rng.uniform(-1, 1, n)
    sth = np.sqrt(1 - cth ** 2)
    ph = rng.uniform(0, 2 * np.pi, n)
    d = ps * np.stack([sth * np.cos(ph), sth * np.sin(ph), cth], axis=1)
    es = np.full(n, mass / 2)

    beta = pvec / E[:, None]
    b2 = (beta ** 2).sum(1)
    gam = E / mass

    def boost(p3, e):
        bp = (beta * p3).sum(1)
        coef = (gam - 1) * bp / b2 + gam * e
        return p3 + coef[:, None] * beta

    mu1, mu2 = boost(d, es), boost(-d, es)

    # vértice de desintegración (um)
    if ctau > 0:
        L = rng.exponential(ctau, n) * (np.sqrt((pvec ** 2).sum(1)) / mass)
        vtx = pvec / np.linalg.norm(pvec, axis=1)[:, None] * L[:, None]
    else:
        vtx = np.zeros_like(pvec)

    out = []
    for mu in (mu1, mu2):
        p = np.linalg.norm(mu, axis=1)
        ptm = np.hypot(mu[:, 0], mu[:, 1])
        etam = np.arcsinh(mu[:, 2] / ptm)
        u = mu / p[:, None]
        ipv = vtx - (vtx * u).sum(1)[:, None] * u
        # dos direcciones ortonormales perpendiculares a la traza
        a = np.cross(u, np.array([0.0, 0.0, 1.0]))
        a /= np.linalg.norm(a, axis=1)[:, None]
        b = np.cross(u, a)
        sig = 15 + 29 / ptm
        ipv = ipv + (rng.normal(0, 1, n) * sig)[:, None] * a + (rng.normal(0, 1, n) * sig)[:, None] * b
        out.append((p, ptm, np.linalg.norm(ipv, axis=1), etam))

    acc = np.ones(n, bool)
    for p, _, _, etam in out:
        acc &= (etam > ETA_RANGE[0]) & (etam < ETA_RANGE[1]) & (p > P_MIN)
    cols = np.stack([out[0][0], out[0][1], out[0][2], out[1][0], out[1][1], out[1][2]], axis=1)
    return cols[acc], acc.mean()


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (mass, ctau, p0, nexp) in SAMPLES.items():
        cols, eff = generate(rng, 4 * N_EVENTS, mass, ctau, p0, nexp)
        cols = cols[:N_EVENTS]
        # el orden de los dos muones es aleatorio (no se distingue mu+ de mu-)
        header = 'p1,pt1,ip1,p2,pt2,ip2'
        np.savetxt(OUT / f'trigger_{name}.csv', cols, delimiter=',', header=header, comments='',
                   fmt=['%.2f', '%.3f', '%.1f', '%.2f', '%.3f', '%.1f'])
        print(f'{name:12s}  aceptancia = {eff:.3f}  -> {len(cols)} sucesos')


if __name__ == '__main__':
    main()
