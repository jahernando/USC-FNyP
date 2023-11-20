#!/usr/bin/env python
# coding: utf-8

# # Quark y Hadrones
# 
# 
# Jose A. Hernando
# 
# *Departamento de Física de Partículas. Universidade de Santiago de Compostela*
# 
# Septiembre-Octubre 2021
# 

# In[1]:


import time
print(' Last version ', time.asctime() )


# In[2]:


# general imports
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# numpy and matplotlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy.constants as units


# ## Índice
# 
# 
#   * Introducción - hadrones comunes
#   
#   * Quarks y color. 
#   
#   * Estructura de los nucleones
#   
#   * Desintegraciones débiles cargadas en hadrones
#      
#   * sobre QCD
#      

# ### Introducción
# 
# 
# Los hadrones son las partículas que sienten la interacción fuerte.
# 
# Son partículas compuestas se dividen en:
# 
#    * **bariones**: combinación de tres quarks con carga eléctrica entera.
#    
#    * **mesones**: una combinación de quark-anti-quark con carga eléctrica entera. 
#    
# El protón es el único barión estable: $\tau_p \gt 3.6 \, 10^{29}$ y.   
# 

# Algunos bariones relevantes:
# 
# ¡Revisar la composición de quarks!
# 
# | nombre    | &emsp; quark composition |  $J^{P}$ | &emsp; &emsp;  mass (MeV) &emsp;&emsp; | &emsp;  &emsp; &emsp; life-time &emsp;  | &emsp; &emsp; &emsp; &emsp; main decays &emsp; &emsp; |
# | :-- | :--: | :--: | :----:  | :--: | :-- |
# | $p$            | (uud) | $1/2^+$ | $938.27$              | $>3.6 \, 10^{29}$ y        | None |
# | $n$            | (udd) | $1/2^+$ | $939.56$              | $874.4\pm0.6$ s            | $p +e +\bar{\nu}_e$ (100%)|
# | $\Delta$ | (uuu) | $3/2^+$ | $1232 \pm 2$         | $118 \pm 2$ MeV             | $p + \pi^-, n+ \pi^0$ |
# | $\Lambda$      | (uus) | $1/2^+$ | $1115.683 \pm 0.006$ | $263 \pm2$ ps               | $p + \pi^0, n+ \pi^+$|
# | $\Sigma^0$     | (uds) | $1/2^+$ | $1182.64 \pm 0.02$   | $7.4 \pm 0.7 \, 10^{-20}$ s | $\Lambda+\gamma$|
# | $\Omega^-$     | (sss) | $3/2^+$ | $1672.45 \pm 0.29$   | $82.1\pm 1.1$ ps            | $\Lambda+K^+, ...$|
# 
# donde $J^P$, indica el spin $J$, y $P$ el signo que se introduce en la función de ondas al aplicar el operador paridad.

# Algunos mesones relevantes:
# 
# | nombre    | &emsp; quark composition |  $J^{P}$ | &emsp; &emsp;  mass (MeV) &emsp;&emsp; | &emsp;  &emsp; &emsp; life-time &emsp;  | &emsp; &emsp; &emsp; &emsp; main decays &emsp; &emsp; |
# | :-- | :--: | :--: | :----:  | :--: | :-- |
# | $\pi^0$   | $(\bar{u}u, \bar{d}d)$ | $0^-$   | $134.97$       | $8.43\pm0.6 \, 10^{-17}$  s| $2\gamma$  (98%)|
# | $\pi^\pm$ | ($u\bar{d}, \bar{u}d$) | $0^{-}$ | $139.57$       | $26.033\pm0.05$ ns | $\mu + \bar{\nu}_\mu$ (98%) |
# | $\rho$    | ($u\bar{u}, \bar{d}d$) | $1^-$   | $7775.5\pm0.4$ | $194.4\pm1$ MeV | $2\pi$| 
# | $K^\pm$   | $(\bar{s}u, s\bar{u})$ | $0^{-}$ | $493.677\pm 0.016$| $12.39\pm0.02$ ns | $\mu+ \bar{\nu}_\mu$ (63%), $\pi^- \pi^0$ (20 %) |
# | $K^0_S$   | $(\bar{s}d, \bar{d}s)$ | $0^{-}$  | $497.648\pm0.022$ | $89.5\pm0.05$ ps | $\pi^++\pi^-$ (69%), $2\pi^0 $ (30%)|
# | $K^0_L$   | $(\bar{s}d, \bar{d}s)$ | $0^{-}$  | $497.648\pm0.022$ | $51.14\pm0.21$ ns | $\pi^\pm + e^\pm + \nu_e$ (40%), $\pi^\pm + \mu^\pm + \bar{\nu}_\mu$ (27%)|
# | $\phi$    | $(s\bar{s})$          | $1^-$    | $1019.46\pm0.019$ | $4.26\pm0.05$ MeV | $K+\bar{K}, \pi^++\pi^-+\pi^0$|
# | $D^\pm$   | $(c\bar{d},d\bar{c})$ |$0^{-}$ | $1869.3\pm0.4$ | $0.500\pm 0.007$ ps | $K+...$ |
# | $D^0, \bar{D}^0$    | $(c\bar{u},u\bar{c})$ |$0^{-}$ | $1864.5\pm0.4$    | $0.410\pm 0.002$ ps | $K+...$ |
# | $B^0, \bar{B}^0$    | $(b\bar{d},d\bar{b})$ |$0^{-}$ | $5279.4\pm0.5$    | $1.530\pm 0.009$ ps | $D+...$ |
# | $B^0_s, \bar{B}^0_s$| $(b\bar{s},s\bar{b})$ |$0^{-}$ | $5367.5\pm1.8$    | $1.466\pm 0.059$ ps | $D^\pm_s+...$ |
# | $J/\Psi$            | $(c\bar{c})$          |$1^{-}$ | $3096.92\pm0.011$ | $93.4\pm2.1$ keV    | hadrons (87%), $\mu+\mu^+, e+e^+$ (6%) |
# | $\Upsilon$      | $(b\bar{b})$          |$1^{-}$ | $9460.3\pm0.3$ | $31.98\pm2.63$ keV | $l+l^+$ |

# ### Descubrimientos de los hadrones, quarks y el color
# 
#    * 1932 Chadwick descubre el neutrón.
#    
#    * 1935 Yukawa postula que la fuerza entre nucleones se media con un $\pi$.
#    
#    * 1947 Powell descubre en las emunsiones nucleares el $\pi$ y el $\mu$.
#    
#    * 1947 Observación del $\pi$, $K$.
#    
#    * 1961 Gell-Mann predice la existencia del $\Omega^-$ (que se descubre en 1968) con el modelo de SU(3) de sabor.
#    
#    * 1964 Gell-Mann y Zweig postulan la existencia de los quarks.
#    
#    * 1968 Descubrimiento de los quarks y la estructura de los nucleones en SLAC mediante aniquilación $e+e^+$.
#    
#    * 1974 Richter, Ting et al, Descubrimiento del $J/\Psi$ en SLAC y Brookhaven. 
#    
#    * 1977 Lederman et al, Descubrimiento de $\Upsilon$ en Fermilab.
#    
#    * 1979 Descubrimiento del gluón en sucesos de tres jets (DESY).
#    
#    * 1995 Descubrimiento del top en Fermilab en CDF.

# **Hitos importantes en la compresión de los hadrones:**
# 
#   * 1963 Cabibbo propone la mezcla de estados de masas y débiles en los quarks.
#   
#   * 1964 Fitch, Cronin et al, violación de CP en sistema de kaones neutrons. 
#   
#   * 1973 Kobayashi, Maskawa para explicar la violación CP, proponen la matriz de mezclas que lleva su nombre.
#   
#   * 1998 confirmación de violación CP en el sistema de $B_0$ neutros en BaBar (SLAC) y Belle (KEKB).

# ## Quarks y color
# 
# ### aniquilación $e+e^+\to f+\bar{f}$
# 
# La aniquilación $e+e^- \to q+\bar{q}$ es una de las pruebas principales de la existencia de los quarks de sabor y del número de color.
# 
# Vamos a calcular su sección eficaz y compararla con $e+e^+ \to \mu+\mu^+$ [MT6.2].
# 
# Los siguientes diagramas muestras las dos aniquilaciones, $e + e^+ \to \mu+ \mu^+$ y $ e + e^+ \to q+\bar{q}$ que están mediados por los portadores neutros $\gamma, Z$.
# 
# | | 
# | :--: |
# |  <img src="./imgs/feynman_new_eemumu_eeqq.png" width = 450 align="center"> |
# | Diagramas de Feynman $e+e^+ \to \mu+\mu^+$ (izda), $e+e^+\to q+\bar{q}$ (derecha)|
# 

# Los propagadores asociados con fotón y el $Z^0$ son respectivamente:
# 
# $$\frac{g_{\mu\nu}}{q^2}, \;\; \frac{g_{\mu\nu}}{q^2 - m_Z^2},$$
# 
# donde $m_Z = 91.2$ GeV es la masa del $Z$.
# 
# La importancia de los mediadores depende del cuadri-momento transferido, $q$.
# 
#   * Cuando $q^2 \le m^2_Z$. El fotón domina y el propagador del $Z$ es simplemente $\propto 1/m^2_Z$.
# 
#   * Cuando $q^2 \sim m^2_z$ la contribución de $Z$ domina (es la región de la resonancia del $Z$).
#   
#   * En la zona intermedia, los dos diagramas contribuyen.
#   

# #### Sección eficaz $e+e^+\to f+\bar{f}$.
# 
# Consideremos en rango $q$ de aniquilación que domina el fotón (QED). 
# 
# La sección eficaz puede calcularse "sencillamente" usando las distintas combinaciones de helicidad de los fermiones.
# 
# En el CM, en condiciones relativistas, $\sqrt{s} \gg m_\mu$, podemos utilizar la aproximación de los espinores de helicidad se identifican con los de quiralidad. 
# 

# De las 4 posibles corrientes $(e,e^+)$ con helicidades distintas: $LL, LR, RL, RR$, solo $LR, RL$ son no nulas. Lo mismo pasa pra las corrientes $(\mu,\mu^+)$. 
# 
# La figura muestra las posibles combinaciones de helicidad:
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_eemumu_helicities_2.png" width = 700 align="center"> |
# | Las posibles combinaciones de helicidad de $e^++e \to \mu^+ + \mu$ en el CM en el régimen relativista |
# 
# donde $\theta$ es el ángulo $\mu$ respecto al eje $z$ dado por la dirección de $e$ en el CM (en la literatura aparece frecuentemente como $\theta^*$).

# | | 
# | :--: |
# |  <img src="./imgs/hadrons_eemumu_cuadrimomenta.png" width = 250 align="center"> |
# | $e^++e \to \mu^+ + \mu$ con cuadrimomentos asignados|
# 
# Las corrientes electromagnéticas del electrón y muón son:
# 
# $$j^\mu_e   = e \, \bar{v}(p_b) \gamma^\mu u(p_a), \;\;\; j^\nu_\mu = e \, \bar{u}(p_c) \gamma^\nu v(p_d)$$
# 
# El elemento de matriz para cada combinación de helicidad es:
# 
# $$\mathcal{M}_{fi} = - e^2  \bar{v}(p_b) \gamma^\mu u(p_a) \frac{g_{\mu\nu}}{q^2} \bar{u}(p_c) \gamma^\nu v(p_d),$$
# 
# donde $e$ es la carga eléctrica, asociada a cada vértice por QED y $q^2 = s$ es el cuadrimomento transferido.

# Calcularemos el elemento $M_{fi}$ asociado a la combinación $RL \to RL$, considerando que la energía $\sqrt{s}$ es suficientemente elevada como para despreciar las masas de los fermiones.
# 
# los cuadrimomentos en el CM son:
# 
# $$p_a = (E, 0, 0, E), \;\; p_b = (E, 0, 0, -E), \\
# p_c = (E, E\sin\theta, 0, E \cos \theta), \;\; p_d (E, -E \sin \theta, 0, -E \cos \theta)$$
# 
# donde $E$ es la energía en el CM y $\theta$ el ángulo de $\mu$ respecto la dirección $z$ dada por el $e$.

# 
# Y utilizaremos les espinores de helicidad:
# 
# $$u_+(p_a) = \sqrt{E} \begin{pmatrix} 1 \\ 0 \\ 1 \\ 0\end{pmatrix}, \;\;
# v_-(p_b) = \sqrt{E} \begin{pmatrix} 0 \\ -1 \\ 0 \\ -1\end{pmatrix}, \;\;
# u_+(p_c) = \sqrt{E} \begin{pmatrix} c \\ s \\ c \\ s\end{pmatrix}, \;\;
# v_-(p_d) = \sqrt{E} \begin{pmatrix} s \\ -c \\ s \\ -c\end{pmatrix}, \;\;$$
# 
# donde recordemos $c = \cos \theta/2, s = \sin \theta/2$.

# Las corrientes $RL \to RL$ quedan:
# 
# $$j^\mu_e|_{RL}   = e \bar{v}_{-}(p_b) \gamma^\mu u_{+}(p_a) = e 2E\, (0, -1, -i, 0), \\
# j^\nu_\mu|_{RL} = e \bar{u}_{+}(p_c) \gamma^\nu v_{-}(p_d) = e 2E\, (0, -\cos\theta, i, \sin \theta)$$
# 
# Recordemos que $q^2 = s = (2E)^2$ en el CM.
# 
# Esto es:
# 
# $$\mathcal{M}_{fi} = e^2 \, \frac{(2E)^2}{s} (1 + \cos \theta) = 4 \pi \alpha \, (1 + \cos \theta)$$
# 
# Notar que en la figura anterior (RL$\to$RL derecha) los spins se solapan para $\theta = 0$ y se oponen cuando $\theta = \pi$.

# Procedríamos igual para el resto de combinaciones. 
# 
# Los factores de $M_{fi}$ asociados al acoplo de las corrientes cargadas son:
# 
# ||||| 
# | :--: | :--: | :--: | :--: |
# | $LR \to LR$ | $LR \to RL$ | $RL \to LR$ | $RL \to RL$ |
# | $e^2 s (1+\cos \theta)$|$e^2 s (1 - \cos \theta)$ | $e^2 s (1-\cos \theta)$| $e^2 s (1 + \cos \theta)$ |
# 
# 
# Calculamos el promedio del módulo al cuadrado de cuatro elementos de matriz:
# 
# $$\langle |M_{fi}|^2 \rangle = \frac{1}{4}\frac{e^4 s^2}{s^2}  \left[2 (1+\cos \theta)^2 + 2 (1-\cos \theta)^2 \right]  = e^4 (1 + \cos^2\theta)$$
# 
# Notar que la sección eficaz diferencial depende de $(1+\cos^2 \theta)$ y  es *simétrica* bajo paridad, $\theta \to \pi - \theta$ (como debe ser porque QED conserva paridad).

# La sección eficaz será:
# 
# $$\sigma = \frac{1}{64 \pi^2 s} \frac{p^*_f}{p^*_i} \int_\Omega \langle |M_{fi}|^2 \rangle \, \mathrm{d}\Omega^*$$
# En este caso $p^*_f = p^*_i = E$ 
# 
# La integración en el ángulo sólido:
# 
# $$\int_{\Omega} (1+\cos^2 \theta) \, \mathrm{d}\Omega^* = 2\pi \int_{-1}^{+1} (1 + \cos^2 \theta) \mathrm{d}(\cos \theta) = \frac{16\pi}{3}$$
# 
# Obtenemos:
# 
# $$\sigma = \frac{1}{64\pi^2s} e^4 \frac{16\pi}{3} = \frac{e^4}{12 \pi s} = \frac{(4 \pi)^2 \alpha^2}{12 \pi s} = \frac{4\pi \alpha^2}{3s}$$

# *Cuestión*
# 
# Calcula el coeficiente en bars si $s$ viene dado en GeV.

# In[3]:


alpha  = units.alpha
sigma0 = (4 * np.pi * alpha**2)/3 # GeV^2 si s in GeV^2
hbarc  = 0.197 * units.femto # GeV m
sigma0_si = sigma0 * (hbarc**2)
barn   = 1e-28 # m^2
print('sigma {:e} barns GeV^2 '.format(sigma0_si/barn))


# #### Inciso : Consideraciones sobre el espín
# 
# Los elementos de matriz de las combinaciones de helicidad presentan una depedencia angular que podemos enteder a partir de las combinaciones de spín. 
# 
# A partir de las figuras con las combinaciones de helicidad, podemos ver que:
# 
#   * El espín de la corriente muónica es $| 1, \pm 1 \rangle_\theta$, definido por la dirección del $\mu$ con un ángulo $\theta$ respecto el $e$. La tercera componente es +1 en el caso de la combinación RL y -1 en el caso de LR. 
# 
#   * El espín de la corriente electrónica es $|1, \pm 1\rangle$ respecto la dirección de $e$, la tercerca componente es +1 para RL y -1 para LR.

# 
# Ambos están relacionados por una rotación de los spines con ángulo $\theta$.
# 
# Esperamos por lo tanto que los elementos de transición serán proporcionales a:
# 
# |      | | | | 
# | :--: | :--: | :--: | :--: |
# | $LR \to LR$ | $LR \to RL$ | $RL \to LR$ | $RL \to RL$ |
# | $\langle -1, 1 | 1, -1 \rangle_\theta = d^1_{-1, -1}(\theta)$ | $\langle -1, 1 | 1, 1 \rangle_\theta = d^1_{-1, 1}(\theta)$ | $\langle 1, 1 | 1, -1 \rangle_\theta = d^1_{1, -1}(\theta)$ | $\langle 1, 1 | 1, 1 \rangle_\theta = d^1_{1, 1}(\theta)$ | 
# | $(1+\cos \theta)/2$ |$(1 - \cos \theta)/2$ | $(1-\cos \theta)/2$| $(1 + \cos \theta)/2$ |
# 

# ### Sección eficaz hadrónica
# 
# La situación es idéntica para la producción $e+e^+ \to q+\bar{q}$. A excepción de que en el vértice de la corriento $(q, \bar{q})$ el factor de la carga es $+Q_q e$, donde $Q_q$ es la carga del quark en unidades de $e$ (carga).
# 
# La figura muestra la distribución en $|\cos\theta|$ de los eventos $e+e^+\to q + \bar{q}$ en el CM del experimento CELLO (en DESY) en los 80's en la región $38.6 \le \sqrt{s} \le 46.5$ GeV. Los dos quarks se manifiestan como dos jets (ver más adelante).
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_eeqq_CELLO.png" width = 300 align="center"> |
# |  Distribución angular $e+e^+\to$ 2 jets en CELLO (DESY, 1987) [MT, AB, CELLO]|
# 
# que siguen la dependencia  $1+\cos^2\theta$.

# ### Número de colores
# 
# Como vimos, el vértice de la corriente de los quarks entra en la sección eficaz como $Q^2_q e^2$.
# 
# Por lo tanto sección eficaz a todos los quarks, $e+e^+ \to q+\bar{q}$ (hadrones), dependerá para una energía disponible, $\sqrt{s}$, de los quarks de sabor que se puedan crear con esa energía, de su carga.
# 
# Como veremos los quarks se presentan en tres colores, por eso introducimos por cada quark disponible un factor $N_C = 3$.
# 
# La sección total es:
# 
# $$\sigma(e+e^+\to \mathrm{hadrons}) = \frac{4\pi \alpha^2}{s} \, N_c \, \sum_q Q^2_q,$$
# 
# donde $q = u, d, s, c, b$ dependiendo de la energía $\sqrt{s}$.

# 
# La razón $R_\mu$ entre la producción de hadrones y pares $\mu, \mu^+$ dependerá de:
# 
# $$R_\mu = \frac{\sigma(e+e^+ \to q+\bar{q})}{\sigma(e+e^+ \to \mu+\mu^+)} = N_c \, \sum_q Q^2_q$$
# 
# | | | | 
# | :--: | :--: | :--: |
# | $u,d,s$ | $u, d, s, c$ | $u, d, s, c, b$ |
# | $3 \left(\frac{4}{9}+\frac{1}{9}+\frac{1}{9} \right) = 2$ | $2 + \frac{4}{3} = \frac{10}{3}$ | $2 + \frac{5}{3} = \frac{11}{3}$ |

# *Cuestión*: ¿Cómo serían estos factores si $N_C = 1$ ?

# La figura muestra los datos experimentales y la predicción del modelo de quarks sencillo (línea punteada) y con correciones de QCD (línea continua).
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_eeqq_rmu.png" width = 600 align="center"> |
# | sección eficaz $\sigma(e+e^+ \to \mathrm{hadrons})$ (arriba) y $R_\mu$ (abajo) en función de $\sqrt{s}$, datos y teoría (líneas) [PDG]|
# 

# La figura muestra los datos experimentales y la predicción del modelo de quarks sencillo (línea punteada) y con correciones de QCD (línea continua) en las tres regiones relevantes: para los quarks ligeros ($u, d, s$), el quark $c$ y el quark $b$.
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_R_regions.png" width = 600 align="center"> |
# | $R_\mu$ vs $\sqrt{s}$, datos y teoría (líneas), para quars ligeros (arriba), $+c$ (medio), +$b$ (abajo) [PDG]|

# 
# Ovservamos:
# 
#   * la sección eficaz decrece con $1/s$
#   
#   * aparecen resonancias de mesones ligeros (con quarks $u, d, s$), y pesados, ($J/\Psi, \Psi$), y $\Upsilon$ conforme producimos quarks $c, b$ respectivamente. 
#   
#   * la Razón $R_\mu$ se incrementa con la aparición de un nuevo quark, con su carga y con sus tres colores. Lo que demuestra **la existencia de los quarks, de sus cargas fracionadas y de que se presentan en 3 colores**.
#    
#   * la correción adicional de QCD ajusta a los datos. 
#   
# En el capítulo del Modelo Estárdar discutiremos la resonancia del $Z$ (en la derecha en las figuras).
# 
# Nota adicional: El efecto de la corrección en QCD se debe a que los gluones transmiten color. El buen ajuste demuestra que efectivamente es así.

# ## Estructura de los nucleones
# 
# La estructura interna del protón (o neutrón) se estudia mediante experimentos de dispersión:
# 
# |      |      |      |
# | :--: | :--: | :--: |
# | &emsp; electrones (QED)   &emsp;|  &emsp; &emsp; débil  &emsp; &emsp; |  &emsp; &emsp; fuerte  &emsp; &emsp;|
# |  $e + p$ |  &emsp; $\nu + p$ | &emsp; $p + p$|
# 
# cada uno es sensible a diversas fuerzas (QED y débil), débil y fuerte respectivamente.
# 
# Los procesos de dividen en:
# 
# |     | |
# | :--: | :--: |
# | &emsp;&emsp;   elásticos &emsp;&emsp;  | &emsp;&emsp;   inelásticos &emsp;&emsp;   |
# | $e + p \to e + p$ | $e + p \to e + X$ |
#   
# Donde $X$ indica la producción de diversos hadrones.

# La dispersión inelásticas se torna dominante comforme la energía aumenta.
# 
# A una determinada energía, los procesos inelásticos se corresponden a procesos elásticos de la sonda con los partones (quarks de valencia).
# 
# Conforme la energía de la sonda $e, \nu, p$ aumenta, podemos exploramos escalas distintas del protón.
# 
#   * El protón como partícula 'puntual' con spín $1/2$ y carga $+e$.
#   
#   * La distribución de carga, $\rho(r)$, y momento magnético del protón.
#   
#   * La estructura de quarks de valencia (uud), se llamaron partones.
#   
#   * Los quarks de valencia y el mar de gluones y quarks.

# La estructura del protón se descubrió gracias a las pruebas $e+ p \to e + X$ principalmente en SLAC en los 60's [ver MT8, AB6.3].
# 
# Aquí vamos a estudiar la estructura a partir de las dispersiónes con $\nu_\mu$ dado que solo interaccionan débilmente. (que se realizaron en los 80's) [ver MT12.2-3]
# 
# Los siguientes diagramas muestran las interacciones $\nu_\mu, \bar{\nu}_\mu$ con $q, \, \bar{q}$, con $q=u, d$, del nucleón.
# 
# | | 
# | :--: |
# |  <img src="./imgs/feynman_nuq_scattering.png" width = 600 align="center"> |
# | Diagramas de Feynman de la dispersión de $\nu_\mu, \bar{\nu}_\mu$ con $q, \bar{q}$ en corrientes cargadas|
# 
# 

# Consideramos que estamos en el rango relativista (despreciamos las masas) y con $q^2 \ll m^2_W$. Recordemos que $m_W = 80.4$ GeV.
# 
# $$\mathcal{M}_{fi} = -\frac{g^2_W}{2 m^2_W} j_\nu \cdot j_d$$
# 
# Los vertices introducen un factor $g_W/\sqrt{2}$ y el propagador del $W$ podemos aproximarlo:
# 
# $$-i\frac{g_{\mu\nu}}{q^2 - m^2_W} = i \frac{g_{\mu\nu}}{m^2_W},$$
# 

# En corriente de neutrinos ($\nu_\mu$), que es $j_\nu$, y de los quarks $(d, y)$, j_d, solo intervienen los fermiones a izquierdad y anti-fermiones a derechas por que las interacciones débiles cargadas tienen estructura V-A.
# 
# Las corrientes son en el CM:
# 
# $$j^\mu_\nu   = \bar{u}_{-}(p_b) \gamma^\mu u_{-}(p_a) = 2E \, (c, s, -is, c), \\
# j^\nu_d = \bar{u}_{-}(p_c) \gamma^\nu u_{-}(p_d) = 2E \, (c, -s, -is, -c),$$
# 
# donde $p_a, p_c$ son los cuadrimomentos de $\nu_\mu, \mu$ respectivamente, y $p_c, p_d$ del $d, u$. Recordemos que $c = \cos \theta/2, s = \sin \theta/2$, donde $\theta$ es el ángulo que forma el $\mu$ respecto al $\nu_\mu$ en el CM. La dirección de éste último define el eje $z$.
# 
# El elemento de matriz es:
# $$\mathcal{M}_{fi} = \frac{g^2_W}{m^2_W} s$$
# 
# Notar que el elemento de matríz is isotrópo.
# 
# Para calcular el promedio del elemento de matriz, contamos que hay dos estados de espín para el quark incidente y solo uno para el neutrino.
# 
# $$\langle |\mathcal{M}_{fi}|^2 \rangle = \frac{1}{2} \left(\frac{g^2_W}{m^2_W}\right)^2 s^2$$

# 
# La sección eficaz en el CM será (ver [A2]):
# 
# $$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega^*} = \frac{1}{(8\pi)^2 s} \frac{p^*_c}{p^*_a} \langle |\mathcal{M}_{fi}|^2 \rangle = \frac{1}{(4 \pi)^2 s} \left(\frac{\sqrt{2}g^2_W}{8m^2_W} \right)^2 s^2 = \frac{G^2_F}{4\pi^2} s,$$
# 
# donde $\sqrt{s}$ es la energía en CM. 
# 
# Notar que la sección eficaz diferencial es isótropa.
# 
# La sección eficaz total:
# 
# $$\sigma = \frac{G^2_F}{\pi} s \simeq {s (\mathrm{GeV}^2)}.$$
# 
# Depende de $s = (2E)^2 = m_N E_\nu$. Estos experimentos son de blanco fijo, un haz de neutrinos incide sobre un blando de material. $E_\nu$ es la energía de los neutrinos en el laboratorio y $m_N$ la masa del nucleón.
# 
# Nota adicional: conforme la energía aumenta y $q \sim m^2_W$, la presencia del propagador completo del $W$ en el elemento de matriz controla que la sección eficaz no crezca indefinidamente con $E_\nu$.
# 

# La siguiente figura muestra las medidas de las secciones eficaces $\sigma(\nu_\mu + N \to \mu + X)/E_\nu$, y $\sigma(\bar{\nu}_\mu N \to \mu^+ + X)/E_\nu$ en función de $E_\nu$ realizadas por varios experimentos.
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_xsection_nuN.png" width = 550 align="center"> |
# | sección eficaces experimentales  de $\sigma(\nu_\mu +N \to \mu + X)/E_\nu, \; \sigma(\bar{\nu}_\mu) + N \to \mu^+ + X)/E_\nu$ vs $E_\nu$ [PDG]|

# Observamos:
# 
#  * la sección eficaz aumenta linealmente con $E_\nu$, con $s$
#  
#  * la sección eficaz con $\nu_\mu$ es un factor ~2 de la de los $\bar{\nu}_\mu$
#  
# El valor promedio:
# 
# $$\sigma(\nu + N)/E_\nu = 0.677 \pm 0.014 \, 10^{-38} \; \mathrm{cm^2 GeV^{-1}} \\ \sigma(\bar{\nu} + N)/E_\nu = 0.334 \pm 0.008 \, 10^{-38} \; \mathrm{cm^2 GeV^{-1}}$$
# 
# Y la razón:
# 
# $$\frac{\sigma(\nu+N)}{\sigma(\bar{\nu}+N)} = 1.984 \pm 0.012.$$
# 

# Para entender esta discrepancia necesitamos estudiar también las interacciones cargadas $\nu_\mu + \bar{u} \to \mu + \bar{d}$ y $\bar{\nu}_\mu + u \to \mu^+ + d$
# 
# La siguiente figura muestra en el CM las configuraciones de spín de las corrientes cargadas de $\nu_\mu, \, \bar{\nu}_\mu$ con $q, \bar{q}$, donde $q = u, b$: 
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_nuq_spins.png" width = 700 align="center"> |
# | Spin configuration en el CM de las corrientes cargadas $\nu_\mu, \bar{\nu}_\mu$ con $q, \bar{q}$|
# 
# Las distribuciones angulares de los elementos de matriz $\mathcal{M}_{fi}$, con $\theta$ en ángulo que dispersión del $\mu (\mu^+)$ respecto al $\nu_\mu (\bar{\nu}_\mu)$ en el CM dependerán de la configuración de spín inicial y final de las dos corrientes.
# 
# |      | | | | 
# | :--: | :--: | :--: | :--: |
# |  $\nu_\mu + d  \to \mu + u$  |  $\nu_\mu + \bar{u} \to \mu + \bar{d}$  |  $\bar{\nu}_\mu + u \to \mu^+ + d$  |   $\bar{\nu}_\mu + \bar{d} \to \mu^+ + \bar{u}$ |
# | $S_z = 0$ | $\langle 1, -1| 1, -1 \rangle_\theta$| $\langle 1, +1| 1, +1 \rangle_\theta$ |$S_z = 0$ | 
# |  &emsp;&emsp; &emsp;&emsp;&emsp; 1  &emsp;&emsp; &emsp;&emsp;&emsp; |  &emsp;&emsp; &emsp;&emsp;&emsp; $(1+\cos \theta)/2$  &emsp;&emsp; &emsp;&emsp;&emsp;|  &emsp;&emsp; &emsp;&emsp;&emsp; $(1+\cos \theta)/2$   &emsp;&emsp; &emsp; &emsp;&emsp; |  &emsp;&emsp; &emsp;&emsp;&emsp; 1  &emsp;&emsp;&emsp; &emsp;&emsp;| 
# 

# La relación entre la secciones eficaz diferencial de $\bar{\nu}_\mu + u$ y $\nu_\mu + d$ es:
#     
# $$\frac{\mathrm{d}\sigma(\bar{\nu}+N)}{\mathrm{d}\Omega^*} = \frac{1}{4}(1+\cos\theta^*)^2 \frac{\mathrm{d}\sigma(\nu+N)}{\mathrm{d}\Omega^*}$$
# 
# Si integramos:
# $$\int_{\Omega}(1+\cos \theta^*) \mathrm{d}\Omega^* = \int_0^\pi\int_{-1}^{+1} (1+x)^2 \; \mathrm{d}\phi\mathrm{d}x = \frac{16}{3} \pi$$
# 
# Por lo tanto:
# 
# $$\sigma(\bar{\nu}_\mu + N) = \frac{G^2_F}{3 \pi} s$$
# 
# Y la razón:
# 
# $$\frac{\sigma(\nu_\mu + N)}{\sigma(\bar{\nu}_\mu + N)} = 3$$

# Los experimentos de dispersión $e+p$ ya habían descubierto que la composición del protón es muy compleja. Esta formado por los quarks de valencia ($uud$), pero también por gluones y por pares quark, anti-quark del mar.
# 
# Sea $f_q$ la fracción de quarks en el nucleón y $f_{\bar q}$ la de anti-quarks. Podemos dar las secciones eficaces en un nucleón compuesto por $q$ y $\bar{q}$ por:
# 
# $$\sigma(\nu_\mu + N \to \mu + X) = \frac{G^2_F}{\pi} s \left(f_q + \frac{1}{3} f_{\bar{q}} \right), \\
# \sigma(\bar{\nu}_\mu + N \to \mu^+ + X) = \frac{G^2_F}{\pi} s \left(\frac{1}{3} f_q + f_{\bar{q}} \right),$$
# 
# ¡Notar que los neutrinos no interacción con los gluones sólo con los quarks y anti-quarks!
# 
# Con los valores experimentales anteriores obtenemos:
# $$f_q \simeq 0.41, \;\; f_{\bar{q}} \simeq 0.1$$
# 
# Concluimos que **¡El protón está compuesto en 41% de quarks y 10% de anti-quarks, el resto son gluones!**
# 
# Lo que está en acuerdo con los experimentos $e + p \to e + X$.

# ## Corrientes débiles cargadas en hadrones
# 
# En las desintegraciones débiles de los leptones vimos que la constante de acoplo de cada vértice era universal $G^{(e)}_F = G^{(\mu)}_F = G^{(\tau)}_F$.
# 
# A partir de las medidades de las desintegraciones $\beta$ obtenemos la constante de acoplo del vértice asociado a la corriente de los quarks, $G^{(\beta)}_F$, que podemos comparar con $G^{(\mu)}_F$ del vértice de las desintegraciones $\mu \to \nu_\mu + e + \bar{\nu}_e$:
# 
# | | 
# | :--: |
# |  <img src="./imgs/feynman_mu_beta_decay.png" width = 350 align="center"> |
# | Diagramas de Feynman de desintegración $\beta$ y del $\mu$|
# 
# $$G^{(\beta)}_F = 1.1066 \pm 0.001 \, 10^{-5} \, \mathrm{GeV^{-2}}, \\
# G^{(\mu)}_F = 1.1663787 \pm 0.0000006 \, 10^{-5} \, \mathrm{GeV^{-2}}.$$
# 
# que es un 5% más pequeña.

# Cabibbo (1963) propuso que las interacciones débiles son universales para los auto-estados débiles $d', s'$ mientras que observamos los estados de masas $d, s$ dentro de los hadrones. Esta matriz sólo afectaría a los quarks de tipo 'abajo'. 
# 
# Entre ambos mediaría una matriz de rotación unitaria:
# 
# $$\begin{pmatrix} d' \\ s' \end{pmatrix} = 
# \begin{pmatrix} \cos \theta_c & \sin \theta_c \\ - \sin \theta_c & \cos \theta_ c \end{pmatrix}
# \begin{pmatrix} d \\ s \end{pmatrix},$$
# donde $\theta_C$ es el ángulo de Cabibbo.
# 
# De tal forma que los vértices de las corrientes cargadas entre quarks serán:
# 
# | | 
# | :--: |
# |  <img src="./imgs/feynman_cabibbo.png" width = 350 align="center"> |
# | Vértices débiles con el ángulo de Cabbibo|
# 

# La constange $G_F^{(\beta)}$ estaría suprimida por un factor $\cos^2 \theta_C$.
# 
# También explica la diferencia entre las anchuras de desintegración de $\pi^- \to \mu + \bar{\nu}_\mu$ y $K^- \to \mu +  \bar{\nu}_\mu$.
# 
# | | 
# | :--: |
# |  <img src="./imgs/feynman_piK_cabibbo.png" width = 450 align="center"> |
# | Vértices débiles en la desitegración $\pi^, K^-$|
# 
# La razón de desintegración entre ambos estará suprimida por un factor $\tan^2 \theta_c$.
# 
# El valor del $\theta_C \sim 13$º.

# La extensión a 3 generaciones da lugar a la matriz, $V$, de Cabibbo-Kobayashi-Maskawa (CKM).
# 
# $$\begin{pmatrix} d' \\ s' \\ b' \end{pmatrix} = 
# \begin{pmatrix} V_{ud} & V_{us} & V_{ub} \\ 
#                 V_{cd} & V_{cs} & V_{cb} \\
#                 V_{td} & V_{ts} & V_{tb} \end{pmatrix}
# \begin{pmatrix} d \\ s \\ b \end{pmatrix}$$
# 
# Donde $V$ es una matriz unitaria: $V^\dagger V = I$, que queda definida por tres ángulos de rotación y una fase compleja.
# 
# Las corrientes débiles entre quarks son:
# 
# $$j^\mu_{d\to u} = \frac{g_w}{\sqrt{2}} \bar{u}(p_u) \gamma^\mu V_{ub} \frac{1}{2}(I - \gamma^5) u(p_d), \\ 
# j^\mu_{u\to d} = \frac{g_w}{\sqrt{2}} V^*_{ub} \bar{u}(p_d) \gamma^\mu \frac{1}{2}(I - \gamma^5) u(p_u), $$
# 
# donde $p_u, p_b$ son los cuatrimomentos de $u, b$ respectivamente. 
# 
# Notar que cuando  el quark de 'abajo' aparece en la corriente con el espinor adjunto, ${\bar u}$, ésta lleva asociada el factor $V^*_{ij}$

# La matriz de CKM da lugar a una **fenomenología muy rica entre hadrones** donde existen **ligaduras** que deben cumplirse, por ejemplo, que la matriz sea unitaria.
# 
# Por ejemplo, las siguientes desintegraciones permiten determinar por ejemplo $|V_{us}|, |V_{ub}|$
# 
# | | 
# | :--: |
# |  <img src="./imgs/feynman_K0_B0_decays.png" width = 450 align="center"> |
# | Diagramas de Feynman $K^0 \to \pi^- + e^+ + \nu_e, \; B^0 \to \pi^- + e^+ + \nu_e$|
# 

# Los valores experimentales de los elementos absolutios de la matriz CKM:
#   
# $$\begin{pmatrix} |V_{ud}| \simeq 0.974& |V_{us}| \simeq 0.225 & |V_{ub}| \simeq 0.004\\ 
#                 |V_{cd}| \simeq 0.225& |V_{cs}| \simeq 0.973 & |V_{cb}| \simeq 0.041 \\
#                 |V_{td}| \simeq 0.009& |V_{ts}| \simeq 0.040 & |V_{tb}| \simeq 0.999 \end{pmatrix}$$
# 
# Observamos:
#   
#   * los valores diagonales son próximos a 1.
#   
#   * la interacción entre las generaciones 1-2 es más intensa que entre la 1-3 y 2-3.
# 

# Si escribimos la matriz en términos de los tres ángulos de mezcla $\theta_{12}, \theta_{13}, \theta_{23}$ y una fase compleja $\delta$:
# 
# $$
# V_{CKM} =  
# \begin{pmatrix} 1 & 0 & 0 \\ 0 & c_{23} & s_{23} \\ 0 & -s_{23} & c_{23} \end{pmatrix}
# \begin{pmatrix} c_{13} & 0 & s_{13}e^{-i\delta} \\ 0 & 1 & 0 \\  -s_{13} e^{i\delta}& 0 & c_{13} \end{pmatrix}
# \begin{pmatrix} c_{12} & s_{12} & 0 \\ -s_{12} & c_{12} & 0 \\ 0 & 0 & 1 \end{pmatrix}
# $$
# donde $c_{ij} = \cos \theta_{ij}, \; s_{ij} = \sin \theta_{ij}$.
# 
# Los valores experimentales:
# $$
# s_{12} = 0.2265 \pm 0.0005, \; s_{13} = 0.0036\pm 0.0001, \; s_{23} = 0.04505 \pm 0.0008, \delta = 1.196\pm0.045
# $$
# 
# Los ángulos de mezcla son "pequeños". Existe una **fase compleja** $\delta$, una constante de la Naturaleza, que dará lugar a una fenomenología muy rica de procesos de **violación CP en hadrones** [ver MT14, AB8].

# La siguiente figura muestra las medidas experimentales (áreas coloredas) al 99 % CL de diversos experimentos. Todos ellos coinciden en un punto (el del vértice marcado con el ángulo $\alpha$) en el plano complejo.
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_CKM_plot.png" width = 450 align="center"> |
# | plano $(\bar \rho, \bar \eta)$ y medidas experimentales al 99% CL [PDG]|

# ## Características de las interacciones fuertes
# 
# ### Confinamiento
# 
# Los quarks no se presentan aislados, siempre en parejas en mesones o tríos en bariones. 
# 
# Sería facil indenticar una partícula cargada con carga fraccionaria $2/3, -1/3$ y no ha sucedido. Los quarks no son libres.
# 
# Este fenómeno se conoce como **confinamiento**.
# 
# No existe una explicación analítica, pero podemos entenderlo como el hecho de que al separar dos quarks estos están ligados fuertemente por los gluones, que actuarían como una "cuerda" estirada, que al remperse crearía un quark-antiquark en sus extremos de rotura (ver [MT10.4])
# 
# Matemáticamente veremos que la condición de confinamiento se traduce en que los quarks tienen que ser singletes de color.

# ### Hadronización
# 
# Conforme rompemos un hadrón en un choque de alta energía, producimos una cascada de hadronización, que se denomina **jet**
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_hadronization.png" width = 400 align="center"> |
# | Esquema de hadronización [MT]|
# 
# Experimentalmente un jet es un corro de partículas que salen en un "cono" en la dirección del quark (o gluón) inicial.

# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_jets_OPAL.png" width = 550 align="center"> |
# | Sección trasnversal de dos eventos: 2 jets $e+e^+ \to q+\bar{q}$ (izda) y 3 jets $e+e^+\to q+\bar{q}+g$ (derecha) en OPAL (90's) en LEP [MT, OPAL Collab]|
# 
# Los jets se caracterizan por su quark primigenio como $c, b$ dependiendo de las partículas y características del jet. 
# 
# En la figura se muestran dos sucesos $e+e+\to 2$ jets y $3$ jets en el plano transverso tomados por OPAL del LEP . Se aprecia como en el caso de 2 jets eston sos opuestos. Los sucesos 3 jets se producen por la radiación de un gluón (que hadroniza produciendo un jet), son procesos $q,q,\bar{g}$.
# 
# Para identificar jets en los datos de ATLAS y CMS, se emplean habitualmente algoritmos basados en el *aprendizaje profundo* (*deep learning*), como *redes neuronales* (*neural networks)*.
# 
# El proceso de modelización de la hadronización es complejo y basado en la fenomenología.

# ### Libertad asintótica
# 
# Paradójicamente las constantes de acoplo $\alpha$ ¡no son constantes!, dependen de la $q^2$ de forma logarítmica. Su explicación se escapa al contenido del curso (ver [MT10.5, AB5.8].
# 
# Esencialmente la constante de acoplo que asociamos a los vértices en los diagramas de Feynman es una constante efectiva (observada).
# 
# En el caso de que considerásemos en el propagador (por ejemplo del fotón) todos los posibles diagramas de Feynman de orden superior que podrían intervenir, nos encontraríamos que la suma de amplitudes no converge. 
# 
# G.'t Hoof demostró en los 70's que una **teoría gauge** puede reabsorver esos diagramas, proveer una amplitud finita y como consecuencia la constante de acomplo, $\alpha$, depende de $q^2$ de forma logarítmica. Este mecanismo se denomina **renormalización**.
# 
# La constante de acomplo fuerte $\alpha_W \sim 1$ a bajas energía, pero a las energías del LHC $\alpha_W \sim 0.17$, lo que permite realizar determinados cálculos perturbativos en QCD para el LHC.
# 
# Conforme la energía es mayor, menor es la constante de acoplo fuerte, y por lo tanto los quarks están más libres dentro de los hadrones, aunque nunca libres completamente, este fenómeno se denomina **libertad asintótica**.

# Valores experimentales de la constante de acoplo fuerte $\alpha_s(q^2)$ en función de $q$
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_alphas_running.png" width = 350 align="center"> |
# | valores experimentales de $\alpha_W(q^2)$ vs q [PDG]|

# ### hadrones blancos (o sin color)
# 
# Vimos antes al medir $R_\mu$ que los quarks se presentaban en **tres colores**, que llamaros rojo, verde, y azúl.
# En realidad son los nombres de cargas, no tiene nada que ver con los colores habituales.
# 
# La teoría detrás de las interacciones fuertes, Quantum Chromo Dynamics, QCD, se desarrolló en los 70's por Gross, Politzer, Wilczek, Fritzsch, Gell-Mann, 't Hooft entre otros.
# 
# La QCD multiplica los **quark** de sabor por **tres colores**, rojo, verde y azul, que podemos asignarlos como las componentes de un triplete.
# 
# $$c_r = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \;\;
# c_g = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \;\;
# c_b = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}. $$
# 
# Similar a los dupletes de spín:
# 
# $$\chi_\uparrow   = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \;\;
# \chi_\downarrow = \begin{pmatrix} 0 \\ 1  \end{pmatrix}.$$
# 
# De forma matemática podemos entender que QCD extiendo el espinor de Dirac, $u(p)$ al añadirle un triplete de color, $c_i$: 
# 
# $$u(p) \to u(p) \, c_i, $$
# 
# con $i = r, g, b$.

# El duplete corresponde a la representación irreducible ${\bf 2}$ del grupo SU(2), que tenía tres generadores, $T_i = \frac{1}{3}\sigma_i$, construidos partir de las matrices de Pauli, que obedecían a las reglas de conmutación del momento angular. 
# 
# En SU(2) solo dos operadores conmutan, $T^2, T_3$, el spin total y su tercera componente. Las combinaciones de estados daban lugar a nuevos estados que se relacionaban con los originales via los coeficientes de Clesch-Gordan, que a su vez se obtenían de la aplicación de los operadores escalera.
# 
# El triplete de color es la representación irreducible de ${\bf 3}$ del grupo SU(3), que tiene 8 generadores, $T_i = \frac{1}{2} \lambda_i$, construidos a partir de las matrices de Gell-Mann.
# 
# En SU(3) solo tres operadores conmutan $T^2, T_3, Y$. Donde  $T_3$ es la tercera componente de color, $Y$ es la hypercarga $Y$ de color. Cada estado lo podemos etiquetar ahora con dos valores, $T_3, Y$.

# Cuando combinamos dupletes de SU(2) obteníamos un triplete y un singlete: ${\bf 2} \otimes {\bf 2} = {\bf 1}\oplus  {\bf 0}$
# 
# En la figura se muestra gráficamente la combinación de un triplete con un anti-triplete, dos tripletes y tres tripletes de color:
# 
# | | 
# | :--: |
# | <img src="./imgs/groups_su3_33b.png" width = 500 align="center"> |
# | <img src="./imgs/groups_su3_no33.png"  width = 480 align="center"> |
# | <img src="./imgs/groups_su3_333.png" width = 500 align="center"> |
# | combinaciones de quark-antiquark (arriba), de dos quarks (medio) y tres quarks (abajo) [MT]|
# 

# QCD exige que los hadrones sean **singletes de color**, partículs **sin color** o *blancas*. 
# 
# Las combinaciones que dan singletes son:
# 
# $${\bf 3} \otimes {\bf 3} \otimes {\bf 3}   = {\bf 10} \oplus {\bf 8} \oplus {\bf 8} \oplus {\bf 1} \\
# {\bf 3} \otimes {\bf \bar{3}}  = {\bf 8} \oplus {\bf 1}$$
# 
# Podemos formar una partícula *sin color*:
# 
#    * con un **trío de quarks**, que son los **bariones**
#    
#    * cun un par **quark, anti-quark**, que son los **mesones**.
# 
# Esto explica por qué solo hemos observado bariones y mesones.

# ### Los diagramas de Feynman en QCD
# 
# La siguiente figura muestra el diagrama de Feynman asociado a la dispersion por color $q_b + q_r \to q_r + q_b$, donde gráficamente asociamos colores al gluón que intercambia los colores rojo y azúl, $(r,b)$ (ver [MT10.7, AB6.4]).
# 
# | |
# | :--: |
# | <img src="./imgs/feynman_new_qqqq_colorlines.png" width = 400 align="center"> |
# | Diagrama de Feynman $q_b +q_r \to q_r + q_b$ (izda) y sus líneas de color (derecha)|
# 
# el índice $\alpha=1, \dots, 8$ corre sobre todos los posibles gluones.
# 
# La figura de la derecha muestra como los gluones intercambian cargas de color y son *coloreados*, al contrario que el fotón que no tiene carga eléctrica.
# 
# *Nota*: En los diagramas de Feynman los gluones se dibujan con un muelle.

# El hecho de que los gluones intercambien color, posibilita que interaccionen entre ellos, por ejemplo, con un vértice $ggg$ como el de la figura:
# 
# | |
# | :--: |
# | <img src="./imgs/feynman_ggg_colorlines.png" width = 400 align="center"> |
# 
# 
# Al interaccionar entre ellos el rango de acción es muy corto, fm, y es el responable del confinamiento de los quarks y gluones en singletes de color.
# 

# La siguiente figura muestra los diagramas de Feynman asociados a las dispersiones fuertes $q_i + q_k \to q_j + q_l$, $q_i + \bar{q}_{\bar j} \to q_k + \bar{q}_{\bar{l}}$, donde los índices $i, j, k, l$ corresponden a las cargas $r, g, b$ y ${\bar j}, \bar{l}$ a las anti-cargas ${\bar r}, \bar{g}, \bar{b}$.
# 
# | |
# | :--: |
# | <img src="./imgs/feynman_qqqq_qqbqbq.png" width = 400 align="center"> |
# |Diagramas de Feynman de las interaciones de color entre quarks y anti-quarks|

# 
# El factor asociado a los vértices de color en los diagramas de Feynman es:
# 
# $$-i \frac{g_S}{2} \lambda^a,$$
# donde $g_S$ es la constante de acoplo fuerte, y $\lambda^a$ es la matriz de Gell-Mann asociada a cada uno de los 8 gluones, gluón $a = 1, \dots, 8$.
# 
# El propagador de los gluones es:
# 
# $$-i \frac{g_{\mu\nu}}{q^2} \delta^{ab}$$
# 
# donde $\delta^{ab}$ garantiza que un mismo gluón media entre los dos vértices
# 

# La corriente de los quarks que cambian de color $i \to k$ y quedan mediadas por el gluón $a$ queda:
# 
# $$j^\mu_q  = \bar{u}(p_c) c^\dagger_k \left( -i \frac{g_s}{2} \lambda^a \right)   \gamma^\mu  c_i u(p_a),$$
# 
# donde $p_a$ es el cuadrimomento del quark entrante y $p_c$ el del saliente.
# 
# Notar que en la corriente sólo hemos introducido un factor de color que corresponde al elemento $k, i$ de la matriz $\lambda^a$:
# 
# $$c^\dagger_k \lambda^a c_i = \lambda^a_{ki}$$
# 
# Mientras que la del antiquark que cambia $\bar{j} \to \bar{l}$:
# 
# $$j^\mu_{\bar{q}}  = \bar{v}(p_b) c^\dagger_j \left( -i \frac{g_s}{2} \lambda^a \right)  c_l \gamma^\mu v(p_d),$$
# 
# donde $p_b$ es el cuadrimomento del antiquark entrante y $p_d$ el saliente. 
# 
# Notar que los índices de color están ahora intercambiados:
# 
# $$c^\dagger_j \lambda^a c_l = \lambda^a_{jl}$$

# ### QCD y QED
# 
# QEC y QCD son teorías similares: 
# 
#    * Son teorías con simetrías gauge locales que conservan paridad, carga y sabor.
#    
#    * Los mediadores son campos vectoriales sin masa.
#    
# Sin embargo son radicalemente diferentes:
# 
#    * El fotón no tiene carga eléctrica y su rango es infinito. No hay interacciones entre fotones. 
#    
#    * Los gluones trasfieren color y por ello interaccionan entre ellos, dando lugar a que su rango de acción sea de fm, y solo sean libres partículas singletes de color.
#    
#    * El fotón proviene del generador del grupo U(1) que es la simetría local gauge del electromagnetismo.
#    
#    * Los fotones provienen de los generadores del grupo SU(3), las matrices de Gell-Mann, que es la simetría gauge local de QCD. Existen tres cargas en QCD, o colores, $r, g, b$ y sus correspondientes anticargas $\bar{r}, \bar{g}, \bar{b}$.
#    
# Nota: podemos entender $r$ como una carga y pensar en $r = +, \bar{r} = -1$, de forma similar a la carga eléctrica.

# ### Referencias
# 
#   * [MT] Mark Tomsom, "Modern Particle Physics", Cambridge U. press.
#   
#   * [AB] Alessandro Bettini, "Introduction to Elementary Particle Physcs", Cambridge U. press.
#   
#   * [PDG](https://pdg.lbl.gov/) Particle Data Group.
#   
#   * [CELLO] Behrend, H. J. et al. 1987. Phys. Lett., B183, 400–411.
#   
# 
