#!/usr/bin/env python
# coding: utf-8

# # Sobre los quarks y los hadrones
# 
# Jose A. Hernando
# 
# *Departamento de Física de Partículas. Universidade de Santiago de Compostela*
# 
# Noviembre 2023
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


# ## Objetivo
# 
#   * Presentar los principales hadrones y bariones, y sus características: confinamiento, hadronización y libertad asintótica.
# 
#   * Mostrar las existencia de los quaks y el color via $e+e^+ \to q + \bar{q}$.
#   
#   * Mostart las desintegraciones débiles cargadas en hadrones y cómo las familias de los quarks se mezclan.
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

# ### los quarks
# 
# Los quarks son los constituyentes de los hadrones.
# 
# Existen 6 quarks, agrupados en tres dupletes de isoespín debil.
# 
# $$
# \begin{pmatrix} u \\ d \end{pmatrix}, \begin{pmatrix} c \\ s \end{pmatrix}, 
# \begin{pmatrix} t \\ b \end{pmatrix} \\
# $$
# 
# La tabla de quarks
# 
# |     | generación   |              | Q (e)  | 
# |:--  | :--          | :--          | :--    |
# |  u  (2 MeV) | c (1.2 GeV) | t (170 GeV)  | $\frac{2}{3}$   | 
# |  d  (5 MeV)  | s (93 MeV)  | b (4.2 GeV)  |  $\frac{-1}{3}$ | 
# 

# Cada quark está un estado de un triplete de color ($r, g, b$, rojo, verde y azúl):
# 
# $$
# c_r = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \;\;
# c_g = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \;\;
# c_b = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}. 
# $$
# 
# Por lo tanto hay 18 quarks y 18 anti-quarks.
# 
# Los estados de quark son, por helicidad $\pm$, y color, $a = r, g, b$:
# 
# $$
# u_{\pm}(p) \, c_a
# $$
# 
# Pero cada hadrón está en un singlete de color. Lo que limita las combinaciones pertimidad de quarks en bariones y mesones.

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
# No existe una explicación analítica, pero podemos entenderlo como el hecho de que al separar dos quarks estos están ligados fuertemente por los gluones, que actuarían como una "cuerda" estirada, que al romperse crearía un quark-antiquark en sus extremos de rotura (ver [MT10.4])
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
# La estructura y elementos del *jet* depende del quark (quark) que lo inició. Los *jets* de quarks pesados $b, t$ son diferentes en topología a los de los ligeros $u, d, s$.

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
# Paradójicamente **las constantes de acoplo ¡no son constantes!**, dependen de la escala de la interacción, su dependencia es suave, logarítmica con el $q^2$, el cuadrado de momento transferido. Su explicación se escapa al contenido del curso (ver [MT10.5, AB5.8] y su explicación está relacionada con la escala a la que observamos, que convierte la constante en "efectiva". Si medimos la constante a una determinada escala podemos estimar cuánto vale a una escala superior.
# 
# La teoría cuántica de campos convivió durante décadas con un problema matemático grave: en el caso de que considerásemos en el propagador (por ejemplo del fotón) todos los posibles diagramas de Feynman de orden superior que podrían intervenir, nos encontraríamos que la suma de amplitudes no converge, ¡es infinita! 
# 
# G.'t Hoof demostró en los 70's que una **teoría gauge** puede reabsorver esos diagramas, proveer una amplitud finita y como consecuencia la constante de acomplo, $\alpha$, depende de $q^2$ de forma logarítmica. Este mecanismo se denomina **renormalización**.

# Valores experimentales de la constante de acoplo fuerte $\alpha_s(q^2)$ en función de $q$
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_alphas_running.png" width = 350 align="center"> |
# | valores experimentales de $\alpha_S(q^2)$ vs q [PDG]|
# 
# La constante de acomplo fuerte $\alpha_S \sim 1$ a bajas energía, pero a las energías del LHC $\alpha_W \sim 0.17$, lo que permite realizar determinados cálculos perturbativos en QCD para el LHC.
# 
# Conforme la energía es mayor, menor es la constante de acoplo fuerte, y por lo tanto los quarks están más libres dentro de los hadrones, aunque nunca libres completamente, este fenómeno se denomina **libertad asintótica**.
# 
# Contrariamente la constante de acoplo electromagnética, $\alpha$  ,y débil,  $\alpha_W$ decrecen con $q^2$. 
# 
# *Nota*: Debe existir una escala de energía $\Lambda$ donde las constantes, al menos dos, sean iguales. En las teorías supersimétricas, las tres constantes se igualan a la escala de Planc.

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

# La estructura del protón se descubrió gracias a las pruebas $e+ p \to e + X$ principalmente en SLAC en los 60's [ver MT8, AB6.3]. La interacción es compleja porque es electro-debil.
# 
# También podemos estudiar la estructura de los nucleones a partir de las dispersiónes con $\nu_\mu$ dado que *solo interaccionan débilmente*, y es más sencillo. Estos experimentos se realizaron en los 80's [ver MT12.2-3].
# 
# Los siguientes diagramas muestran las interacciones $\nu_\mu, \bar{\nu}_\mu$ con $q, \, \bar{q}$, con $q=u, d$, del nucleón.
# 
# | | 
# | :--: |
# |  <img src="./imgs/feynman_nuq_scattering.png" width = 600 align="center"> |
# | Diagramas de Feynman de la dispersión de $\nu_\mu, \bar{\nu}_\mu$ con $q, \bar{q}$ en corrientes cargadas|
# 
# 

# Los cálculos de la sección eficaz, usando las reglas de los diagramas de Feynman, son relativamente sencillos en TCQ:
# 
# En el caso: $\nu + q \to \mu + q'$:
# 
# $$
# \langle |\mathcal{M}_{fi}|^2 \rangle = \frac{1}{2} \left(\frac{g^2_W}{m^2_W}\right)^2 s^2, \;\;\;
# \frac{\mathrm{d}\sigma}{\mathrm{d}\Omega^*} = \frac{G^2_F}{4\pi^2} s, \;\;\;
# \sigma = \frac{G^2_F}{\pi} s.
# $$
# 
# Y con los anti-neutrinos,  $\bar{\nu} + q \to \mu + q'$::
# 
# $$
# \frac{\mathrm{d}\sigma}{\mathrm{d}\Omega^*} = \frac{1}{4}(1+\cos\theta^*)^2 \frac{G^2_F}{4\pi^2} s, \;\;\;
# \sigma = \frac{G^2_F}{3 \pi} s
# $$
# 
# Este es, la razón entre ambas es simplemente:
# 
# $$
# \frac{\sigma(\nu_\mu + N)}{\sigma(\bar{\nu}_\mu + N)} = 3
# $$
# 

# Los datos experimentales, de diversos experimentos a diferentes energías, son los siguientes
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_xsection_nuN.png" width = 550 align="center"> |
# | sección eficaces experimentales  de $\sigma(\nu_\mu +N \to \mu + X)/E_\nu, \; \sigma(\bar{\nu}_\mu) + N \to \mu^+ + X)/E_\nu$ vs $E_\nu$ [PDG]|
# 
# Observamos:
# 
#  * la sección eficaz aumenta linealmente con $E_\nu$, con $s$
#  
#  * la sección eficaz con $\nu_\mu$ es un factor ~2 de la de los $\bar{\nu}_\mu$
# 

#  
# El valor promedio:
# 
# $$\sigma(\nu + N)/E_\nu = 0.677 \pm 0.014 \, 10^{-38} \; \mathrm{cm^2 GeV^{-1}} \\ \sigma(\bar{\nu} + N)/E_\nu = 0.334 \pm 0.008 \, 10^{-38} \; \mathrm{cm^2 GeV^{-1}}$$
# 
# Y la razón:
# 
# $$\frac{\sigma(\nu+N)}{\sigma(\bar{\nu}+N)} = 1.984 \pm 0.012.$$
# 

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

# ## 6 Quarks de 3 colores 
# 
# ### aniquilación $e+e^+\to f+\bar{f}$
# 
# La aniquilación $e+e^- \to q+\bar{q}$ es una de las pruebas principales de la existencia de los quarks de sabor y del número de color.
# 
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
# La importancia de los mediadores depende del cuadrado cuadri-momento transferido, $q^2$.
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
# El promedio del elemento de matriz, $e+e^+\to \mu+\mu^+$ puede calcularse en TQC usando los diagramas de Feynman (ver [MT6.2]), damos simplemente su valor. en El CM
# 
# $$
# \langle |M_{fi}|^2 \rangle = e^4 (1 + \cos^2\theta^*) = (4 \pi)^2 \alpha^2 (1 + \cos^2\theta^*)
# $$
# 

# 
# La situación es idéntica para la producción $e+e^+ \to q+\bar{q}$. A excepción de que en el vértice de la corriente $(q, \bar{q})$ el factor asociado ala carga es $+Q_q e$, donde $Q_q$ es la carga del quark en unidades de $e$ (carga).
# 
# La figura muestra la distribución en $|\cos\theta|$ de los eventos $e+e^+\to q + \bar{q}$ en el CM del experimento CELLO (en DESY) en los 80's en la región $38.6 \le \sqrt{s} \le 46.5$ GeV. 
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_eeqq_CELLO.png" width = 300 align="center"> |
# |  Distribución angular $e+e^+\to$ 2 jets en CELLO (DESY, 1987) [MT, AB, CELLO]|
# 
# que siguen la dependencia  $1+\cos^2\theta$.
# 
# Por supuesto, los dos quarks se manifiestan como dos jets.

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


# ### Número de colores
# 
# Como vimos, el vértice de la corriente de los quarks entra en la sección eficaz como $Q^2_q e^2$.
# 
# Por lo tanto sección eficaz a todos los quarks, $e+e^+ \to q+\bar{q}$ (hadrones), dependerá para una energía disponible, $\sqrt{s}$, de los quarks de sabor que se puedan crear con esa energía, de su carga.
# 
# Pero cada quark se presenta además en tres colores, por lo que por cada quark disponible un factor $N_c = 3$.
# 
# El factor que tenemos que añadir para un  quark $q$ es:
# 
# $$
# N_c Q^2_q
# $$

# La sección total es:
# 
# $$\sigma(e+e^+\to \mathrm{hadrons}) = \frac{4\pi \alpha^2}{s} \, N_c \, \sum_q Q^2_q,$$
# 
# donde $q = u, d, s, c, b$ dependiendo de la energía $\sqrt{s}$.
# 
# Cuando $\sqrt{s}$ supera el umbrar de producción de un quark, $q$, la sección eficaz $\sigma(e+e^+\to q+\bar{q})$ se incrementará en $N_c Q^2_q$

# Podemos comparar mejor con la sección eficaz $\sigma(e+e^+ \to \mu+\mu^)$.
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

# ## Desintegraciones débiles de hadrones.
# 
# Sabemos que el bosón $W$ se acopla con un leptón de arriba y abajo pero siempre de la misma generación.
# 
# $$
# \begin{pmatrix}\nu_e \\ e \end{pmatrix}, \; \begin{pmatrix}\nu_\mu \\ \mu \end{pmatrix}, \;
# \begin{pmatrix}\nu_\tau \\ \tau \end{pmatrix}
# $$
# 
# Vimos (en el tema leptones) que la constante de acoplo es la misma para las tres generaciones $g_W$, lo que llamamos **universalidad** leptónica.
# 
# | |
# | :--: |
# | <img src="./imgs/feynman_wleptons.png" width = 600 align="center">|
# 

# Los quarks se acoplan con el $W^\pm$ en pares de arriba y abajo entre las distintas generaciones:
# 
# $$
# \begin{pmatrix}u \\ d \end{pmatrix}, \; \begin{pmatrix} u \\ s \end{pmatrix}, \; \begin{pmatrix}u \\ b \end{pmatrix} \\
# \begin{pmatrix}c \\ d \end{pmatrix}, \; \begin{pmatrix} c \\ s \end{pmatrix}, \; \begin{pmatrix}c \\ b \end{pmatrix} \\
# \begin{pmatrix}t \\ d \end{pmatrix}, \; \begin{pmatrix} t \\ s \end{pmatrix}, \; \begin{pmatrix}t \\ b \end{pmatrix}
# $$
# 

# 
# La constante de acoplo ya no es universal si no que depende del par arriba y abajo que se acopla, $g_W V_{ij}$, (donde $i = \{u, c, t\}, j = \{d, s, b\}$).
# 
# | |
# | :--: |
# |<img src="./imgs/feynman_vertexW.png" width = 600 align="center">|
# |Vértices del $W$ con parejas de quarks|
# 
# Los elementos $V_{ij}$ forman una matriz unitaria llamada de **matrix CKM** (Cabibbo, Kobayashi, Maskawa). 
# 
# El acoplo es más intenso entre pares de la misma generación.

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

# Cabibbo (1963) propuso que las interacciones débiles son universales para los auto-estados débiles $d', s'$, pero que experimentalmente observamos los estados de masas $d, s$ dentro de los hadrones. Entre ambos media una matriz de mezcla unitaria.
# 
# Si solo consideramos dos quarks abajo, la matriz es simplemente:
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
# Esto explica la diferencia entre las anchuras de desintegración de $\pi^- \to \mu + \bar{\nu}_\mu$ y $K^- \to \mu +  \bar{\nu}_\mu$.
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
# Las corrientes débiles entre quarks $u \to b$ son:
# 
# $$
# j^\mu_{u\to d} =  \bar{u}(p_d) \, \left[ \frac{g_w}{\sqrt{2}} V^*_{ub} \gamma^\mu \frac{1}{2}(I - \gamma^5) \right] u(p_u), 
# $$
# 
# donde $p_u, p_b$ son los cuatrimomentos de $u, b$ respectivamente. 

# La matriz de CKM da lugar a una **fenomenología muy rica entre hadrones** donde existen **ligaduras** que deben cumplirse.
# 
# En particular la matriz debe ser unitaria.
# 
# Por ejemplo, las siguientes desintegraciones permiten determinar por ejemplo $|V_{us}|, |V_{ub}|$
# 
# | | 
# | :--: |
# |  <img src="./imgs/feynman_K0_B0_decays.png" width = 450 align="center"> |
# | Diagramas de Feynman $K^0 \to \pi^- + e^+ + \nu_e, \; B^0 \to \pi^- + e^+ + \nu_e$|
# 

# Los valores experimentales de los elementos absolutos de la matriz CKM:
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
# Los ángulos de mezcla son "pequeños". Existe una **fase compleja** $\delta$, una constante de la Naturaleza, que dará lugar a una fenomenología muy rica de procesos de **violación de carga y paridad en hadrones** [ver MT14, AB8] que escapan an nivel de este curso.

# La siguiente figura muestra las medidas experimentales (áreas coloredas) al 99 % CL de diversos experimentos. Todos ellos coinciden en un punto (el del vértice marcado con el ángulo $\alpha$) en el plano complejo, que está relacionado con $\delta$.
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_CKM_plot.png" width = 450 align="center"> |
# | plano $(\bar \rho, \bar \eta)$ y medidas experimentales al 99% CL [PDG]|

# ## Sobre la Cromo Dinámica Cuántica, QCD.
# 
# Vimos antes al medir $R_\mu$ que los quarks se presentaban en **tres colores**, que llamaros rojo, verde, y azúl.
# En realidad son los nombres de cargas, no tiene nada que ver con los colores habituales.
# 
# La teoría detrás de las interacciones fuertes, Quantum Chromo Dynamics, QCD, se desarrolló en los 70's por Gross, Politzer, Wilczek, Fritzsch, Gell-Mann, 't Hooft entre otros.
# 
# En QCD cada **quark** tiene un posible **color**: rojo, verde y azul, que podemos asignarlos como las componentes de un triplete.
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
# De forma matemática podemos entender que QCD extiendo el espinor de Dirac, $u(p)$ al añadirle un triplete de color, $c_a$: 
# 
# $$u(p) \to u(p) \, c_i, $$
# 
# con $a = r, g, b$.

# El duplete corresponde a la representación irreducible ${\bf 2}$ del grupo SU(2), que tenía tres generadores, $T_i = \frac{1}{3}\sigma_i$, dados por matrices de Pauli, que obedecían a las reglas de conmutación del momento angular. 
# 
# En SU(2) solo dos operadores conmutan, $T^2, T_3$, el spin total y su tercera componente. Los estados irreducibles los etiquetábamos con $T_3$. Las combinaciones de estados daban lugar a nuevos estados que se relacionaban con los originales via los coeficientes de Clesch-Gordan, que a su vez se obtenían de la aplicación de los operadores escalera.
# 
# El triplete de color es la representación irreducible de ${\bf 3}$ del grupo SU(3), que tiene 8 generadores, $T_i = \frac{1}{2} \lambda_i$, dados por las matrices de Gell-Mann.
# 
# En SU(3) solo tres operadores conmutan $T^2, T_3, Y$. Donde  $T_3$ es la tercera componente de color, $Y$ es la hypercarga de color. Cada estado lo podemos etiquetar ahora con dos valores, $T_3, Y$.

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

# **Los gluones al tener color, interaccionan entre ellos**, por ejemplo, con un vértice $ggg$ como el de la figura:
# 
# | |
# | :--: |
# | <img src="./imgs/feynman_ggg_colorlines.png" width = 400 align="center"> |
# 
# 
# Esto ahce que su rango de acción es muy corto, fm.
# 

# La siguiente figura muestra los diagramas de Feynman asociados a las dispersiones fuertes $q_i + q_k \to q_j + q_l$ y $q_i + \bar{q}_{\bar j} \to q_k + \bar{q}_{\bar{l}}$, donde los índices $i, j, k, l$ corresponden a las cargas $r, g, b$ y ${\bar j}, \bar{l}$ a las anti-cargas ${\bar r}, \bar{g}, \bar{b}$.
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
# $$j^\mu_q  = \bar{u}(p_c) \left[ c^\dagger_k \left( -i \frac{g_s}{2} \lambda^a \right)   \gamma^\mu  c_i \right] u(p_a),$$
# 
# donde $p_a$ es el cuadrimomento del quark entrante y $p_c$ el del saliente.
# 
# Notar que en la corriente sólo hemos introducido un factor de color que corresponde al elemento $k, i$ de la matriz $\lambda^a$:
# 
# $$c^\dagger_k \lambda^a c_i = \lambda^a_{ki}$$

# ### QCD y QED
# 
# QEC (Electro Dinámica Cuántica) y QCD son teorías similares: 
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
