#!/usr/bin/env python
# coding: utf-8

# # Sobre los leptones
# 
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


# ## Objetivos
# 
#    * Los leptones sienten la fuerza electro-débil y se clasifican en tres familias que se comportan de forma universal.
#    
#    * las corrientes cargadas presentan violación de carga y paridad. 
#    
#    *  la violación de paridad está asociada a una propiedad que se llama quiralidad
# 
#    * los neutrinos solo sienten la fuerza débil y oscilan entre ellos.
#    

# ## Características de los leptones
# 
# Los leptones se presentan en tres generaciones agrupados en dupletes:
# 
# $$
# \begin{pmatrix}\nu_e \\ e \end{pmatrix}, \; \begin{pmatrix}\nu_\mu \\ \mu \end{pmatrix}, \;
# \begin{pmatrix}\nu_\tau \\ \tau \end{pmatrix}
# $$
# 
# Sus propiedades:
# 
# |       | $e$ | $\mu$ | $\tau$ | $\nu$ |
# | :--   | :-- | :--   | :--    | :--   |
# | masa  | 0.511 MeV    | 105.7 MeV  |  1777 MeV|   $\le$ 1 eV     |  
# | carga |   -1  | -1 | -1 | 0 |
# | vida media ($\tau$)| estable | $2.197$ $\mu$s | $290.6$ fs | estable|

# *Cuestión*: ¿Cuál es el recorrido medio del $\mu, \, \tau$ con momento $p = 10$ GeV en el vacío?
# 
# Aplicando la transformación de Lorentz de un tiempo propio del sistema en reposo al sistema del observador:
# 
# $$
# d = \, \gamma \, \beta \, \tau \, c, \;\; \gamma = \frac{E}{m}, \; \beta = \frac{p}{E}
# $$

# In[3]:


mass = 0.106 # GeV
tau  = 2.2 * units.micro # s
p    = 10 # GeV
ene  = np.sqrt(mass**2 + p**2)
gamma = ene/mass
beta  = p/ene
distance = gamma * beta * tau * units.c
print(' distance {:e} m'.format(distance))


# ### Desintegraciones
# 
# 
# Los siguientes diagramas de Feynman muestran las desintegraciones de los leptones cargados:
# 
# | |
# | :-- |
# <img src="./imgs/feynman_lepton_decays.png" width = 500 align = "center">
# |Desintegraciones del $\mu$ (arriba) y del $\tau$ (abajo)|

# Desintegraciones principales:
# 
# |    | Fracción de desintegración |
# |:-- | --:                                          |
# | $\mu  \to e + \bar{\nu_e} + \nu_\mu$  | $\sim 100$ %|
# | $\tau \to \mu + \bar{\nu_\mu} + \nu_\tau$| $17.4$ % |
# | $\tau \to e + \bar{\nu_e} + \nu_\tau$| $17.8$ % |
# | $\tau \to \mathrm{hadrons}+ \nu_\tau$| $\sim 64$ % |

# #### Conservación del número leptónico.
# 
# Experimentalmente en toda interacción se conservan en número total de leptones menos anti-leptones de la misma generación y también el total. 
# 
# Asignamos un número leptónico a cada generación (es negativo para las antipartículas). 
# 
# |                                                 | $L_e$   | $L_\mu$ | $L_\tau$  |  $L$ |
# | :--                                             | --      | --      | --        | --   |
# | $(e, \nu_e), \; (e^+, \bar{\nu}_e)$            | +1, -1  |  0      | 0         | +1, -1 |
# | $(\mu, \nu_\mu), \; (\mu^+, \bar{\nu}_\mu)$     | 0       |  +1,-1  | 0         | +1, -1 |
# | $(\tau, \nu_\tau), \; (\tau^+, \bar{\nu}_\tau)$ | 0       |  0      | +1,-1     | +1, -1 |    
# 
# **El número leptónico de sabor** (por generación) **y el total se conservan.**

# 
# *Cuestión*: Verifica que el número leptónico de sabor y el total se conservan en las desintegraciones de $\mu, \tau$.
# 

# 
# Los experimentos que han buscado la violación del número léptonico solo han puesto límites a los valores de vida media de los procesos que lo violan. Por ejemplo $\mu \to e + \gamma$
# 
# La conservación del la carga eléctrica fuerza que el $e$, la partícula cargada más ligera sea estable. 
# 
# *Nota adicional* Como veremos posteriormente las oscilaciones de neutrinos violan el número leptónico de sabor pero en la propagación de los neutrinos.
# 
# *Cuestión*: ¿La desintegración $\mu \to \nu_\mu + e + e^+$ viola alguna regla de conservación? ¿Y la desintegración $\tau \to \mu + \mu + \mu^+$?

# ### Descubrimientos de los leptones
#     
#   * El 1930 C. Anderson descubrió **el positrón** en una cámara de burbujas con un campo magnético 1 T observó que de los productos con rayos cósmicos aparecían trazas que por la curvatura eran como el electrón pero con carga positiva.
#     
#   * En 1937 Aderson y Nedermayer descubrieron partículas penetrantes en los rayos cósmicos. En 1947 se observaron en emulsiones fotográficas con rayos cósmicos la desintegraciones del pión y del **muón**, que se correspondía a la partícula penetrante.
#     
#   * en 1975 M. Perl en el colisionador $e^+e^-$ de SLAC descubrió **el tau** en la producción de pares $e+e^+ \to \tau+\tau^+$, donde éstos se desintegraban leptónicamente $\tau \to\mu +\bar{\nu}_\mu + \nu_\tau$
#     

#   * En 1030 **Pauli** propuso **el neutrino** como solución desperada para resolver el problema de la no conservación de energía y momento en las desintegraciones $\beta$, la existencia de una partícula neutra de pequeña masa que escaparía indetectada.
#    
#   * En 1956 **Reines y Cowan** detectaron el $\nu_e$. A partir del flujo intenso de neutrinos del reactor nuclear de Savannah, Carolina del Sur, detectaron la interacción inversa $\bar{\nu}_e + p \to e^+ +n $ en un detector con líquido centelleador y agua como blanco. 
#    
#   * En 1962 Schwartz, Lederman y Steinberger descubrieron el $\nu_\mu$ con el haz de $\nu_\mu$ producido en el AGS de Boorkhaven, Nueva York, con un detector de placas anternadas de material pasivo y cámaras de chispas. Demostraton que los $\nu_\mu$ son distintos a los $\nu_e$ y que aparecen en parejas con su fermiones cargados $\mu$.
#    
#   * En 2001 la colaboración DONUT de Fermilab en emulsiones observó la primera interacción de un $\nu_{\tau}$.

# **Otros hitos relevantes relacionados con los leptones**
#      
#    * 1956 Experimento de Ms **Wu**. Descubrimiento de **la violación de paridad**.
# 
#    * 1958 Goldhaber descubre que el neutrino tiene helicidad negativa.
#    
#    * 1970 **David et al, y Bahcall**, descubren **el problema de los neutrinos solares**.
#    
#    * 1998 Tajita et al, Descubrimiento de **oscilaciones de neutrinos** en el experimento **Super-Kamiokande**.
#    
#    * 2012 McDonald et al, Solución del problema de los neutrinos solares, en el experimen to SNO.

# #### Descubrimiento del neutrino
# 
# El espectro continuo de energía de la desintegraciones $\beta$ era un misterio en la década de los 20 en del siglo XX.
# 
# $$
# ^A_Z X \to ^A_{Z+1}X' + e \;\; (?)
# $$
# 
# | |
# |:--:|
# |<img src="./imgs/intro_bspectrum_1935.png" width=300 align="center">|
# | Espectro $\beta$|
# 
# Si el nucleo padre está en reposo y sólo se emiten dos partículas, el núcleo hijo y el electrón, $\beta$, éste debería ser mono-energético. La energía a repartir entre el núcleo hijo y el electrón es $Q$ y salen en el CM con el mismo momento $p^*$.
# 
# N. Bohr: *”At the present stage of atomic theory, however, we may say that we have no argument... for upholding the energy principle in the case of 𝛽-ray disintegrations.”*
# 

# [Pauli](https://en.wikipedia.org/wiki/Wolfgang_Pauli) postuó la existencia del neutrino en una histórica carta.
# 
# | |
# | :--: |
# |<img src="./imgs/intro_Pauli_letter.jpg" width=600 align="center">|
# ||
# 

# B. Pontecorvo [[>]](https://www.youtube.com/watch?v=yXrHnsBgQSw&t=9s) sugirió utilizar el flujo de neutrinos generado en los reactores nucleares.
# 
# | | |
# | :--: | :--: |
# | <img src="./imgs/intro_cowan_reines_detector.png" width=300 align="center"> | <img src="./imgs/intro_cowan_reines_telegram.png" width=300 align="center">|
# 
# 
# F. Reines and C. L. Cowan detectaron los neutrino con la interacción inversa usando el flujo de neutrinos, $\Phi \simeq 10^{17}$ $\nu\mathrm{/(m^2 \, s)}$, generado en el reactor nuclear de Savanna River, SC, EEUU.
# 
# Se trataba de un contador de positrones (detectaban simultaneamente en los centelleadores gammas de 511 keV provinientes de la desintegración del positrón) que operaba cuando el reactor estaba en funcionamiento o parada.

# ## Características de las interacciones débiles
# 
# ### Universalidad leptónica
# 
# La **Universalidad leptónica** nos dice que la intensidad del acoplo de los vértices de las corrientes cagadas es igual para las tres generaciones de leptones.
# 
# Si suponemos que en las corrientes de Fermi la constante $G^\alpha_F$, con $\alpha = e, \mu, \tau$ es *distinta para cada sabor*, veremos que experimentalmente esas constantes son en realidad iguales para todos los sabores, y que valen $G_F$, la constante de Fermi.
# 
# La anchura de desintegración del $\mu$, que se calcula en un curso más avanzado, es:
#  
# $$
# \Gamma (\mu \to e + \bar{\nu}_e + \nu_\mu) \equiv  \frac{1}{\tau_\mu} = \frac{G^e_F G^\mu_F m^5_\mu}{192 \pi^3},
# $$
# 
# *Cuestión*: verifica que las dimensiones en la fórmula de la anchura de desintegración son correctas.
# 

# Para la desintegraciones $\tau \to e + \bar{\nu}_e + \nu_\tau$ intervendra las constante $G^\tau_F$ en la corriente tauónica y $G^e_F$ en la eléctrica.
# 
# La anchura de desintegración parcial:
# 
# $$
# \Gamma (\tau \to e + \bar{\nu}_e + \nu_\tau) = \frac{G^\tau_F G^e_F m^5_\tau}{192 \pi^3},
# $$
# 
# La anchura de desintegración total es la suma de las parciales: $\Gamma = \sum_i \Gamma \equiv 1/\tau_\tau$, y la fracción de desintegración, $Br$, se relaciona con la anchura total y parcial por:
# 
# $$
# Br(\tau \to e + \bar{\nu}_e + \nu_\tau) = \frac{\Gamma(\tau \to e + \bar{\nu}_e + \nu_\tau)}{\Gamma_\tau} = \tau_\tau \, \Gamma(\tau \to e + \bar{\nu}_e + \nu_\tau)
# $$
# 
# donde $\tau_\tau$ es la vida media del $\tau$.
# 
# Luego:
# 
# $$
# \Gamma(\tau \to e + \bar{\nu}_e + \nu_\tau) = \frac{1}{\tau_\tau} \, Br(\tau \to e + \bar{\nu}_e + \nu_\tau) 
# $$
# 

# A partir de la razón entre las dos anchuras:
# 
# $$
# \frac{\Gamma(\tau \to \nu_\tau + e + \bar{\nu}_e)}{\Gamma(\mu \to \nu_\mu + e + \bar{\nu}_e)} = \frac{\tau_\mu}{\tau_\tau} \mathcal{Br}(\tau \to \nu_\tau + e + \bar{\nu}_e) = \frac{G^\tau_F m^5_\tau}{G^\mu_F m^5_\mu}
# $$
# 
# Podemos despejar la razón entre la constantes de Fermi para cada sabor, y sustituyendo por sus valores experimentales obtenemos:
# 
# $$
# \frac{G^\tau_F}{G^\mu_F} = \frac{m^5_\mu \tau_\mu}{m^5_\tau \tau_\tau} \, Br(\tau \to e + \bar{\nu}_e + \nu_\tau) = 1.0023 \pm 0.033
# $$
# 

# In[4]:


tau_tau  = 290.6 * units.femto # s
mass_tau = 1777  # MeV
tau_mu   = 2.197 * units.micro # s
mass_mu  = 105.7   # MeV
br_tau_e = 0.178
ratio    = (tau_mu * mass_mu**5)/(tau_tau * mass_tau**5) * br_tau_e
print('Gamma ratio {:4.3f}'.format(ratio))


# La constande de Fermi es común a todos los vértices de las corrientes cargadas entre leptones:
# 
# $$
# G_F = 1.166 \times 10^{-5} \;\; \mathrm{GeV}^{-2}.
# $$
# 
# La constante de Fermi esta relacionada con el intensidad del acoplo $g_W$ del modelo estándar por:
# 
# $$
# \frac{G_F}{\sqrt{2}} = \frac{g^2_W}{8 m^2_W},
# $$
# 
# donde $m_W = 80.385 \pm 0.015$ GeV es la masa del $W$.

# *Cuestión*: A partir de las anchura de desintegración $\Gamma(\tau \to \mu + \nu_\mu + \bar{\nu}_\tau)$ y $\Gamma(\tau \to e + \nu_e + \bar{\nu}_\tau)$ calcular $G^e_F/G^\mu_F$. La desviación respecto de la unidad se compensa con un factor de espacio de fase.

# ### Violación de Paridad
# 
# Recordemos que es la operación de inversión por paridad, ${\hat P}$: 
# 
# $$
# {\hat P}  \, : \; {\bf x} \to - {\bf x}, \;\;\;  \Psi({\bf x}, t) \to  \Psi(-{\bf x}, t).
# $$
# 
# Recordemos también cómo se clasifican distintas magnitudes bajo paridad:
# 
# |        | Rango | Paridad | Ejemplo |
# |:--     | --    | -- | :-- |
# | scalar | 0     | +1 | temperatura |
# |pseudo-scalar| 0 | -1 | helicidad |
# | vector | 1 | -1   | momento |
# |axial-vector | 1 | +1 | momento angular, campo magnético | 
# 
# 

# #### Violación de paridad
# 
# En 1957 Ms Wu y colaboradores descubrieron que las corrientes cargadas no conservan paridad. 
# 
# En el experimento polarizaban temporalmente $^{60}\mathrm{Co}$ mediante un campo magnético ${\bf B}$. El $^{60}\mathrm{Co}$ se desintegra $\beta$:
# 
# $$
# ^{60}\mathrm{Co} \to ^{60}\mathrm{Ni}^* + e + \bar{\nu}_e.
# $$
# 
# | |
# | :-- |
# | <img src="./imgs/leptons_Co_scheme.jpeg" width = 200 align = "center"> |
# |Esquema de la desintegración $\beta$ del $^{60}\mathrm{Co}$ polarizado|
# 
# y estudiaron la emisión de $e$ hacia arriba en los casos de campo magnético hacia arriba o hacia abajo.
# Si la paridad se conservase, el número de electrones emitidos en las dos configuraciones debería ser el mismo.

# Considera los siguientes gráficos, el de la derecha es la inversión por paridad del de la izquierda. 
# 
# El nucleo de $^{60}\mathrm{Co}$ tiene un momento magnético ${\bf \mu}$, dado por su espín, ${\bf S}$, que está alineado en la dirección de un campo magnético ${\bf B}$. 
# 
# Tanto ${\bf B}$, como ${\bf \mu}$ y ${\bf S}$, son vectores axiales, no cambian bajo paridad, mientras que el momento, ${\bf p}$; del electrón sí cambia por ser un vector.
# 
# Si se conserva la paridad, el electrón debe salir con igual probabilidad en los dos lados. Esto es con $\theta$ respecto a la vertical y con $\pi-\theta$.
# 
# En el caso extremo, cuando el electrón sale en la vertical, el electrón sale con la misma probabilidad hacia arriba que hacia abajo.
# 
# | | 
# | :--: | 
# | <img src="./imgs/leptons_Co_spins.png" width = 400 align = "center"> | <
# |Esquema de la desintegración $\beta$ del $^{60}\mathrm{Co}$ polarizado|
# 
# 
# *Nota* En la figura no se dibuja el neutrino que se emite en la desintegración $\beta$.

# | | 
# | :--: | 
# | <img src="./imgs/leptons_Co_results.png" width = 360 align = "center"> |
# | Razón de detección de $e$ en función del tiempo y la polarización de ${\bf B}$ [WU]|
# 
# 
# ¡Se observa que los electrones se emitían con preferencia en la dirección opuesta al campo magnético!
# 
# Luego las desintegraciones $\beta$ o **las corrientes cargadas violan paridad**.
# 
# Nota: el $^{60}\mathrm{Co}$ se desmagnetizaba al cabo de un tiempo, al perder temperatura la muestra. 
# 

# #### La corrientes bajo paridad
# 
# En la teoría de Fermi, el elemento de Matriz se construía con el producto de dos corrientes, la hadrónica y la fermíonica.
# 
# $$
# M_{fi} \propto j^\mu_{\mathrm{had}}  \, g_{\mu\nu} \, j^\nu_{\mathrm{lep}} = j_{\mathrm{had}}  \cdot  j_{\mathrm{lep}}
# $$
# 
# como cada corriente cambia bajo paridad:
# 
# $$
# \hat{P} \; : \; j^\mu = (\rho, {\bf j}) \to j^\mu (\rho, - {\bf j}),
# $$
# 
# pero $M_{fi}$ es invariante bajo paridad
# 
# $$
# \hat{P} \; : \; (\rho, {\bf j})_{\mathrm{had}} \cdot (\rho, {\bf j})_{\mathrm{lep}} = (\rho, - {\bf j})_{\mathrm{had}} \cdot (\rho, - {\bf j})_{\mathrm{lep}}
# $$
# 
# ¡Luego no representa la física de la corrientas cargadas!

# Nota:
# 
# En la representación Pauli-Dirac, la operación paridad es simplemente: $\hat{P} = \gamma^0$
# 
# La corriente de probabilidad $j^\mu = \bar{\Psi} \gamma^\mu \Psi \to {\bar \Psi} \gamma^0 \gamma^\mu \gamma^0 \Psi$ bajo paridad cambia:
# 
# $$
# \hat{P} \, j^\mu = (j^0, -{\bf j}^k)
# $$
# 
# dado que:
# $$
# \gamma^0 \gamma^k = - \gamma^k \gamma^0
# $$
# 
# donde $k = 1, 2, 3$

# Podemos construir el elemento de matrix $M_{fi}$ de forma génerica con las formas $ {\bar \Psi} \, \Gamma \, \Psi $, donde $\Gamma$ es una matrix $4 \times 4$ construida con la matrices de Dirac.
# 
# Solo existen cinco posibilidades de $\gamma$ que garantizan que el elmento de matriz es invariante Lorentz: 
# 
# |        | -------- Forma -------- | Componentes | Spin |
# |:--     | :--:    | :--: | :--: |
# | scalar | ${\bar \Psi} \Psi$   | 1 | 0 |
# |pseudo-scalar|  ${\bar \Psi} \gamma^5 \Psi$  | 1 | 0|
# | vector | ${\bar \Psi} \gamma^\mu \Psi$  | 4   | 1 |
# |pseudo-vector | ${\bar \Psi} \gamma^\mu \gamma^5 \Psi$ | 4 | 1 | 
# |tensor | ${\bar \Psi} (\gamma^\mu\gamma^\nu - \gamma^\nu \gamma^\mu) \Psi$ | 6 | 2 | 
# 
# 

# 
# A partir del estudio de las desintegraciones de los muones se determinó las $\Gamma$ que intervienen en el acoplo de las corrientes
# 
# | |
# | :-- |
# <img src="./imgs/leptons_espectrum_mudecay.png" width = 300 align = "center">
# |Espectro del electrón en $\mu$ decays, la línea es la teoría V-A [MU]|
# 
# 

# La forma que corresponde a las desintegraciones de muones (y las corrientes cargadas) es:
# 
# $$
# {\bar \Psi} \left[ \gamma^\mu \frac{1}{2}(I - \gamma^5) \right] \Psi
# $$
# 
# La corriente $j^\mu_V = \frac{1}{2}{\bar \Psi} \gamma^\mu \Psi$ es **vectorial** porque cambia como un vector frente a paridad. 
# 
# 
# La corriente $j^\mu_A = \frac{1}{2}{\bar \Psi} \gamma^\mu \gamma^5 \Psi$ es **axial** porque cambia como un axial-vector frente a paridad. 
# 
# Decimos que las corrientes cargadas tienen una **estructura $V-A$**

# ¿Cómo cambia $M_{fi}$ ahora bajo paridad?
# 
# $$
# \hat{P} \;  \; : j^\mu_V - j^\mu_A = (\rho_V - \rho_A, \; {\bf j}_V - {\bf j}_A) \to (\rho_V - \rho_A, - {\bf j}_V - {\bf j}_A)
# $$
# 
# Y ahora $M_{fi}$, al ser el producto de dos corrientes $V-A$, ¡cambia bajo paridad! 
# 
# $$
# (\rho_V - \rho_A, \; {\bf j}_V - {\bf j}_A)_{\mathrm{had}} \cdot (\rho_V - \rho_A, \; {\bf j}_V - {\bf j}_A)_{\mathrm{lep}} \neq 
# (\rho_V - \rho_A, \; -{\bf j}_V - {\bf j}_A)_{\mathrm{had}} \cdot (\rho_V - \rho_A, \; -{\bf j}_V - {\bf j}_A)_{\mathrm{lep}}
# $$
# 
# aunque sigue siendo un invariante Lorentz.

# #### Proyección de quiralidad
# 
# ¿Qué es la matriz:
# 
# $$
# P_L \equiv \frac{1}{2} (I-\gamma^5) \, \mathrm{?}
# $$
# 
# Es un proyector que actua sobre el sobre el espinor de Dirac:
# 
# $$
# \Psi_L = \frac{1}{2}(I - \gamma^5) \, \Psi
# $$
# 
# El proyector complementario es:
# 
# $$
# P_R = \frac{1}{2} (I+\gamma^5)
# $$
# 
# Los dos proyectores cumplen:
# 
# $$
# I = P_R + P_L, \;\; P^2_R = P_R, \;\; P^2_L = P_L, \; P_R P_L = P_L P_R = 0 
# $$
# 
# Estos proyectores se llaman de **quiralidad** y proyectan a **izquierda y derecha**.
# 

# De tal forma que un **espinor de Dirac** se descompone en dos proyecciones, **quiralidad a izquierdas y derechas**:
# 
# $$
# \Psi_R = P_R \, \Psi, \;\;\; \Psi_L = P_L \, \Psi, \;\;\; \Psi = \Psi_R + \Psi_L =  P_R \Psi + P_L \Psi = (P_R + P_L) \, \Psi
# $$
# 
# **La quiralidad se invierte al aplicar paridad**:
# 
# $$
# \hat{P} : P_L \leftrightarrow P_R, \;\; \Psi_L \leftrightarrow \Psi_R
# $$
# 
# 
# Nota: La transformación paridad $\hat{P} = \gamma^0$ cambia los proyectores de quiralidad:
# 
# $$
# \hat{P} \; P_L = \gamma^0 \frac{1}{2}(1 - \gamma^5) = \frac{1}{2} (1+\gamma^5) \gamma^0 = P_R \, \hat{P}
# $$

# 
# Podemos decir entonces que, **las corrientas cargadas**
# 
# * **Son de tipo V-A**, $\; {\bar \Psi} \, \frac{1}{2}\gamma^\mu (I - \gamma^5) \, \Psi$ 
# 
# * O **solo intervienen los espinores a izquierdas de los fermiones**, $u_L = \frac{1}{2} (I - \gamma^5) \, u$, y **de derechas a los anti-fermiones**, $v_R$ 
# 
# En la corrientes cargadas *solo* interviene la parte a izquierdas de los espinores de las partículas, la parte a derechas no juega ningún papel. Para las anti-partículas es al contrario, solo interviene su espinor a derechas.
# 
# En una teoría que conserva paridad la parte a izquierdas y a derechas de las partícuals debe entrar igual en el elemento de Matriz, como pasa con las interacciones electromagnéticas y fuertes.

# #### Proyección de los espinores de helicidad
# 
# En la representación de Pauli-Dirac, los proyectores de quiralidad son simplemente:
# 
# $$
# P_R = \frac{1}{2}\begin{pmatrix} I & I \\ I & I \end{pmatrix}, \;\;\;
# P_L = \frac{1}{2}\begin{pmatrix} I & -I \\ -I & I \end{pmatrix},
# $$
# 
# 
# Sean ahora los espinores de helicidad con ${\bf p} = p \, \hat{k}$ en la dirección $z$:
# 
# $$
# u_{+} = N \begin{pmatrix} 1 \\ 0 \\ \kappa  \\ 0 \end{pmatrix}, \;
# u_{-} = N \begin{pmatrix} 0 \\ 1 \\ 0 \\ -\kappa  \end{pmatrix}, \;
# v_{+} = N \begin{pmatrix} 0 \\ -\kappa  \\ 0 \\ 1 \end{pmatrix}, \;
# v_{-} = N \begin{pmatrix} \kappa  \\ 0 \\ 1 \\ 0 \end{pmatrix}. 
# $$
# donde $N$ es un factor de normalización y:
#  
# $$
# \kappa = \frac{p}{E+m}
# $$
# 

# Si aplicamos los proyectores de quiralidad sobre los espinores $u_{\pm}$ de helicidad, obtenemos:
# 
# $$
# P_R \, u_+ = \frac{1}{2}(1+\kappa) N \begin{pmatrix} 1 \\ 0 \\ 1  \\ 0 \end{pmatrix}, \;\;\;
# P_L \, u_+ = \frac{1}{2}(1-\kappa) N \begin{pmatrix} 1 \\ 0 \\ -1  \\ 0 \end{pmatrix}, \\
# P_R \, u_- = \frac{1}{2}(1-\kappa) N \begin{pmatrix} 0 \\ 1 \\ 0  \\ 1 \end{pmatrix}, \;\;\;
# P_L \, u_- = \frac{1}{2}(1+\kappa) N \begin{pmatrix} 0 \\ 1 \\ 0  \\ -1 \end{pmatrix}, \;\;\;
# $$
# 

# Así tenemos la tabla que relaciona las componentes de quiralidad y de helicidad de los espinores $u$:
# 
# | helicidad $\downarrow$ \ quiralidad  $\rightarrow$| --- R --- | --- L --- |
# | :--: | :--: | :--: | 
# | + | $\frac{1}{2}(1+\kappa)$ | $\frac{1}{2}(1-\kappa)$ |
# | - | $\frac{1}{2}(1-\kappa)$ | $\frac{1}{2}(1+\kappa)$ |
# | en los acoplos con el $W$  | anti-fermiones, $v_R$     | fermiones, $u_L$ |
# 
# Hemos dicho que en la interacciones débiles cargadas solo participan los espinores $u_L$, esto es la última columna, L.
# 
# Si repitiésemos los cálculos para los espinores $v_R$ obtendríamos los mismos coeficientes. En las corrientes débiles entraría la parte $v_R$ que correpondería a la columna del medio, R.
# 

# Consideremos ahora el caso de partículas sin masa, $m=0$, o ultra-relativistas, $m \ll E$, en ambos casos tenemos que $\kappa = 1$ y por lo tanto:
# 
# | helicidad $\downarrow$ \ quiralidad  $\rightarrow$ | --- R --- | --- L --- |
# | :--: | :--: | :--: | 
# | + | 1 | 0 |
# | - | 0 | 1 |
# | en los acoplos con el $W$    | anti-fermiones, $v_R$     | fermiones, $u_L$ |
# 
# Esto es **para partículas sin masa o ultra-relativistas la quiralidad y la helicidad coinciden**.
# 
# *En esos dos casos*, que son los más comunes a alta energías, en las corrientes cargadas solo participan los espinores de los fermiones $u_-$ con helicidad negativa y de los antifermiones $v_+$ con helicidad positiva.
# 

# ### Helicidad del neutrino
# 
# En 1957 Goldhaber y colaboradores midieron  en Brookhaven, NY, la helicidad del neutrino, que resultó ser negativa.
# 
# | |
# | :-- |
# <img src="./imgs/leptons_goldhaber_experiment.png" width = 200 align = "center">
# |Esquema del experimento de Goldhaber [GOL]|
# 
# $$
# ^{152}\mathrm{Eu}(J = 0) + e \to ^{152}\mathrm{Sm}^*(J = 1) + \nu_e \to ^{150}\mathrm{Sm}(J =0) + \gamma
# $$
# 

# La dirección $z$ es la "vertical" definida por el eje del campo ${\bf B}$.
# 
# El campo ${\bf B}$ polariza el electrón de la captura de $^{152}\mathrm{Eu}$ arriba o abajo, $S_z(e) = \pm 1/2$ dependiendo de la orientación del campo. 
# 
# Se seleccionan los fotones que salen hacia abajo por lo que el neutrino sale hacia arriba.
# 
# La tercera componente del spín $\pm 1/2$ en $z$ se conserva.
# 
# Lo que implica que la helicidad del neutrino es la del fotón (ver más adelante).

# Se cambia la polaridad del campo ${\bf B}$ y se determina el número de fotones en cada orientación del campo.
# 
# Si se producen neutrinos con las dos helicidades independientemente de la polaridad de ${\bf B}$ se medirán el mismo número de fotones.
# 
# | |
# | :-- |
# <img src="./imgs/leptons_goldhaber_spins.png" width = 200 align = "center">
# |helicidades en el experimento de Goldhaber|
# 

# 
# La siguiente tabla muestra las dos posibilidades de espín y helicidades posibles en el experimento de Goldhaber.
# 
# | ----- $S_z(e)$ -----  | ----- $S_z(\nu)$ ----- | ----- $S_z(\gamma)$ ----- | ----- $h(\nu)$ ----- | ----- $h(\gamma)$ -----|  observación |
# | :--:   | :--:        | :--:         |:--:           | :--:         | :--: |
# |  1/2   |  -1/2  |  1  |  -   |  -      |  si|
# |-1/2   |  1/2  |  -1  | +    |+     |  no|
# 
# En la Naturaleza, solo se realiza la primera posibilidad (fila en la tabla). Solo existen neutrinos con helicidad negativa y no existen con helicidad positiva. 
# 
# Ya vimos que para una partícula con masa nula, helicidad es quiralidad, luego neutrinos con helicidad negativa implica neutrinos a izquierdas.

# ### La masa del neutrino
# 
# 
# La masa del neutrino no se ha medido, se establece su límite, $\lt 1$ eV, con la forma final del espectro de energía de los electrones en desintegración $\beta$ del tritio,
# 
# $$
# ^3\mathrm{H} \to ^3\mathrm{He} + e + \bar{\nu}_e,
# $$
# que debería verse reducido por la masa del neutrino.
# 
# | |
# | :-- |
# <img src="./imgs/leptons_numass_tritium.jpeg" width = 500 align = "center">
# |Espectro del electrón en un decay de tritio si el neutrino tiene masa $\sim$ 1 eV|
# 
# El experimento actual es [KATRIN] que busca $m_{\nu_e} < 0.2$ eV.
# 
# 

# **El Modelo Estandar postulaba que el neutrino no tiene masa**. En todo caso su masa es ridículamente pequeña comparada con sus energías.
# 
# Los neutrinos solo interaccionan débilmente, y **en las interacciones débiles solo intervienen las proyecciones a izquierdas**, esto es $u_L$ y $v_R$. **Solo son reales esos dos espinores** para el neutrino:
# 
# $$
# u_L \equiv u_-, \;\;\; v_R \equiv v_+
# $$
# 
# En términos de la conjugación de Carga y Paridad encontramos:
# 
# | |      |  $\hat{P}$| |
# | :--: | :--: | :--: | :-- |
# |  | $\nu_L$| $\Rightarrow$ | $\nu_R$  (no existe)|
# | $\hat{C}$ | $\Downarrow$|  |  $\Downarrow$|
# |  | $\bar{\nu}_L$ (no existe) | $\Rightarrow$| $\bar{\nu}_R$ 
# 
# **La fuerza débil viola de forma máxima paridad, $P$, y conjugación de carga, $C$, pero no $CP$**.
# 

# Dado que **el neutrino**:
# 
#   * **tiene masa compatible con nula**.
#   
#   * **la helicidad es negativa** (para el anti-neutrino positiva).
#   
# **El Modelo Estándar postula** que **los neutrinos no tienen masa**, solo existen en **quiralidad a izquierdas**, y el antineutrino con quiralidad a derechas, 
# 
# $$
# \nu_L, \;\;\; {\bar \nu}_R.
# $$ 
# 

# **Las corrientes cargadas en leptones violan paridad, P, y carga, C; pero preservan CP**
# 
# | |
# | -- |
# <img src="./imgs/leptons_escher_CP.png" width = 450 align = "center">
# |Efecto de aplicar paridad (reflejo, izquierda-derecha) y carga (invertir grises, arriba-abajo)|
# 

# ### Desintegración del pión
# 
# | |
# | -- |
# <img src="./imgs/feynman_pion_decay.png" width = 300 align = "center">
# |Diagrama de Feynman de $\pi^+ \to l^+ + \nu_l$|
# 
# La desintegración del pión $\pi^+ \to l^+  + \nu_l$ ejemplifica los efectos de la estructura V-A de las corrientes cargadas.
# 
# La razón de las anchuras de desintegración a $l = \mu, e$ medida es:
# 
# $$
# \frac{\Gamma(\pi^+ \to e^+ + \nu_e)}{\Gamma(\pi^+ \to \mu^+ + \nu_\mu)} = 1.230 \pm 0.004 \; 10^{-4}
# $$
# 
# Sin embargo el espacio fásico disponible para el $e$ es mayor que el del $\mu$ dado que la razón entre ellos viene dada por $p^*_e/p^*_\mu$. 
# 
# ¿Por qué es así entonces?

# **C**uestión: Calcula $p^*_l, E^*_l$ en el CM.
# 
# **C**uestión: Calcula la razón entre el $p^*_e/p^*_\mu$.

# | |
# | :-- |
# <img src="./imgs/leptons_piondecay_spins.png" width = 300 align = "center">
# |Momento y spin de $\pi^+ \to l^+ + \nu_l$ en el CM|
# 
# Si consideramos el CM: El $\pi^+$ tiene $J= 0$, el neutrino, $\nu_l$, tiene helicidad (-), y para conservar $J$, $l^+$ debe tener helicidad (-), pero sabemos que solo la proyección con quiralidad a derechas del $l^+$ interviene en la desintegración. ¿Cuánto vale esa parte para el $e^+$ y el $\mu^+$?
# 
# Sabemos que el espinor $v_-$ helicidad (-) tiene como coeficiente de quiralidad a derechas:
# 
# $$
# \frac{1}{2}(1-\kappa) = \frac{1}{2}\left(1 - \frac{p_l}{E_l + m_l} \right)
# $$ 
# 

# 
# En el CM
# $$
# E_l = \frac{m^2_\pi + m^2_l}{2 m_\pi}, \; \; p_l = \frac{m^2_\pi - m^2_l}{2 m_\pi}, \;\;
# \frac{1}{2}(1-\kappa) = \frac{1}{2}\left(1 - \frac{p_l}{E_l + m_l} \right)= \frac{m_l}{m_\pi + m_l}
# $$
# 
# El $\mu$ no es relativista, $\beta = 0.27$, pero sí el electrón, $\beta = 0.99997$. 
# 
# El factor $(1-\kappa)/2$ es mucho más pequeño para $e$ que para el $\mu$.
# 
# El factor asociado a las proyecciones a derechas entre el $e$ y $\mu$ que aparece en $|M_{if}|^2$ será:
# 
# $$
# \frac{m^2_e}{m^2_\mu} \frac{(m_\pi + m_e)^2}{(m_\pi + m_\mu)^2}\sim 7 \, 10^{-5}
# $$
# 
# Así pues, a pesar de que el espacio fásico es mucho mayor para la desintegración al $e^+$, el $\pi^+$ se desintegra mayoritariamente a $\mu^+$, porque los leptones cargados de la desintegración deben tener helicidad negativa, u como en las corrientes V-A, solo interviene su componente de quiralidad a derechas, esta está muy suprimida en la helicidad negativa si la partícula es relativista, lo que es el caso deo positrón pero no para el muón. Por eso la desintegración a $e^+$ está muy suprimida respecto a $\mu^+$.

# In[5]:


mpi = 139.51  # MeV
mmu = 105.66  # MeV
me  =   0.511 # MeV
ene_cm = lambda ml : (mpi**2 + ml**2)/(2*mpi)
p_cm   = lambda ml : (mpi**2 - ml**2)/(2*mpi)
a_R    = lambda ml : (1 - p_cm(ml)/(ene_cm(ml) + ml))/2

beta_mu, beta_e = p_cm(mmu)/ene_cm(mmu), p_cm(me)/ene_cm(me)
print(' beta mu   = {:7.6f}   , beta   e = {:7.6f}    '.format(beta_mu, beta_e))
print(' energy mu = {:5.3f} MeV, energy e = {:5.3f} MeV'.format(ene_cm(mmu), ene_cm(me)))


# In[6]:


helicity_supres = a_R(me)/a_R(mmu)
espace_phase    = p_cm(me)/p_cm(mmu)
print(' supresion por helicidad = {:e}'.format(helicity_supres**2))
print(' espacio fásico          = {:e}'.format(espace_phase))
print(' total                   = {:e}'.format(espace_phase * helicity_supres**2))


# Distribución de la energía del espectro de electrones en las desintegraciones del $\pi$.
# 
# | |
# | :-- |
# <img src="./imgs/leptons_espectrum_pidecay.png" width = 400 align = "center">
# |Espectro de energías del $e$ en las desintegraciones del $\pi$ [AB]|
# 

# ## Oscilaciones de neutrinos
# 
# Sabemos experimentalmente que **los neutrinos oscilan entre sus sabores en su propagación**. 
# 
# Las oscilaciones **dependen de las masas** de los neutrinos. 
# 
# Pero recordemos que el Modelo Estándar postula que los neutrinos no tienen masas.
# 
# Esto hace que las oscilaciones de neutrinos sean por ahora la **única prueba experimental** de que **el SM necesita revisarse**.
# 

# 
# ### El puzzle solar
# 
# En la década de los 60, el experimento de R. David en la mina de Homestake, Dakota del Sur, medió el flujo de $\nu_e$  solares y resultaba ser $1/3$ del esperado según el modelo teórico del Sol desarrollado por J. Bahcall. 
# 
# Durante aproximadamente 40 años los resultados de los experimentos que medían el flujo de $\nu_e$ solares mostraban un deficit respecto al flujo esperado según el modelo solar.
# 
# En el 2004 El experimento SNO (en Canada) midió el flujo total (la suma de los flujos de los diferentes sabores) de neutrinos y verificó que estaba en acuerdo con el esperado por el modelo solar para $\nu_e$ producidos en el Sol. Por lo tanto los neutrinos que se producían en el Sol, todos $\nu_e$, estaban de acuerdo con el modelo teórico, pero al llegar a la Tierra, solo una fracción, 1/3, eran $\nu_e$.
# 
# Pontecorbo en los 50's adelantó que los neutrinos podrían oscilar entre sus diversos sabores si tenían masas distintas.
# 
# Ahora sabemos que los $\nu_e$ que se producen en la reacciones nucleares del Sol cambian de sabor en su camino hacia la Tierra.
# 
# 

# ### Neutrinos atmosféricos
# 
# En 1998 el experimento Super-Kamiokande en la mina de Kamioka, Japón, medió que el flujo de $\nu_\mu$ producido en las interacciones de los rayos cósmicos con la atmósfera dependía de la dirección de llegada, en concreto midió que existía una asimetría arriba y abajo, donde abajo correspondía a los neutrinos que habían atravesado la Tierra, y arriba sólo la atmósfera.
# 
# Ahora sabemos que los $\nu_\mu$ dependiendo de la distancia recorrida se transforman de sabor a los dos otros neutrinos, pero principalmente en $\nu_\tau$.

# La siguiente figura muestra el detector Super-Kamiokande. 
# 
# | |
# | :-- |
# <img src="./imgs/leptons_SK.png" width = 450 align = "center">
# |Super-Kamiokande a) dibujo del detector b) principio de detección [SK, MT13.2]|
# 
# SK es un tanque de agua de 50 kton, 36 m de altura y 34 de diámetro, con 11 k PMTs en las paredes.
# 

# SK detecta las corrientes cargadas de neutrinos $\nu_e \to e + X$ o $\nu_\mu \to \mu + X$, a partir del leptón cargado. 
# 
# En el agua, el leptón a partir de un umbral de energía emite luz Cherenkov que se detecta en gigantescos PMTs colocados en las paredes del detector. 
# 
# A partir de la luz y el tiempo de de su llegada a los PMTs se determina la energía y dirección del neutrino. 
# 
# Se pueden distingir $e$ de $\mu$ por los patrones de luz en los PMTs.

# Resultados de la colaboración SK. 
# 
# | |
# | :-- |
# <img src="./imgs/leptons_SK_results.png" width = 400 align = "center">
# |eventos de $\nu_e$ (iza), $\nu_\mu$ detectados por SK en función del ángulo [SK]|
# 

# En la figura, el eje $x$, correponde al $\cos \theta$, donde $\theta$ es el ángulo de llegada, $\cos \theta = 1$ indica arriba, y $\cos \theta = -1$, abajo (o sea, atravesando la Tierra).
# 
# Observamos:
# 
#  * Para los $\nu_e$ la predicción y la observación coinciden.
# 
#  * Para los $\nu_\mu$, de multi-GeV, hay un déficit para aquellos que provienen de 'abajo'.
# 
#  * La línea corresponde a las predicción en caso de oscilaciones.
# 
# 
#  Luego SK concluyó que los neutrinos producidos $\nu_\mu$ en las atmósfera oscilan en su recorrido hasta el detector.

# ### Oscilaciones en dos familias
# 
# Como los neutrinos son neutros, un neutrino de un sabor, por ejemplo $\nu_e$, puede estar compuesto por una combinación de neutrinos con diferentes masas. Cada neutrino se propagará libremente de acuerdo con su masa como una onda libre; y con la distancia -o el tiempo transcurrido-, la composición inicial de los diferentes tipos de neutrinos cambia.
# 
# Los neutrinos pueden oscilar si los estados de sabor no son los mismos que los estados de masas y las masas de los neutrinos son distintas.
# 
# Los estados de sabor son $\nu_\alpha$, con $\alpha = e, \mu, \tau$, y los estados de masas $\nu_i$ con masas $m_i$  donde $i = 1, 2, 3$. 
# 
# Consideremos por simplicidad solo dos sabores y dos masas. Los sabores serán: $\alpha, \beta$
# 
# La relación entre estados de masas y de sabor de neutrinos viene dada por una matriz de mezcla:
# 
# $$
# \begin{pmatrix} \nu_\alpha \\ \nu_\beta \end{pmatrix} = \begin{pmatrix} \cos \theta & \sin \theta \\ - \sin \theta & \cos \theta \end{pmatrix} \begin{pmatrix} \nu_1 \\ \nu_2 \end{pmatrix} 
# $$
# 

# Si producimos un neutrino de un determinado sabor, $\nu_\alpha$, en un tiempo $t_0  = 0$, con un momento $p$, este es una combinación de los neutrinos de masas $| \nu_1 \rangle, \, | \nu_2 \rangle$ que depende de un ángulo $\theta$.
#    
# $$
# | \nu_\alpha \rangle = \cos \theta | \nu_1 \rangle + \sin \theta | \nu_2 \rangle
# $$
#     
# La evolución de cada neutrino de masas corresponde a la de una partícula libre, esto es:
# 
# $$
# | \nu (t) \rangle  = \cos \theta e^{+i({\bf p x} - E_1t)}| \nu_1 \rangle + \sin \theta e^{+i({\bf p x} - E_2t)}| \nu_2 \rangle
# $$
# 
# donde $E_i = \sqrt{m^2_i + p^2}$, con $i = 1, 2$
# 

# En un determinato tiempo $t$, o equivalentemente a una distancia $L$, podemos detectar el neutrino con otro sabor $\beta$.
# 
# $$
# | \nu_\beta \rangle = -\sin \theta | \nu_1 \rangle + \cos \theta | \nu_2 \rangle
# $$
# 
# ¡Recordemos que las interacciones de los neutrinos tienen lugar en sus estados de sabor!
# 
# Calculamos la amplitud de transición:
# 
# $$
# A_{\alpha \beta}(t) = \langle \nu_\beta | \nu(t) \rangle = -\sin\theta\cos\theta e^{-iE_1t} + \sin\theta\cos\theta e^{-iE_2t} 
# $$
# 
# recordemos que una fase global en la amplitud, en este caso $e^{i{\bf px}}$ no afecta a la física del proceso.

# Si operamos:    
# $$
# A_{\alpha\beta}(t) = \frac{1}{2}\sin 2 \theta e^{-iE_1} \left( 1 + e^{-i(E_2 - E_1)t}\right)
# $$
# 
# de nuevo la fase global $e^{-iE_1t}$ no altera la física, y si desarrollamos:
# 
# $$
# E_2 - E_1 \simeq \frac{p}{2}  \left(1 + \frac{m_2^2}{p^2}\right) - \frac{p}{2}  \left(1 + \frac{m_1^2}{p^2}\right) = \frac{m_2^2 - m^2_1}{2p} \simeq \frac{\Delta m^2_{21}}{2E}
# $$
# 
# donde $\Delta m_{21}^2 = m_2^2 - m^2_1$, y dado que $m_i \sim 0$ aproximamos $p \sim E$
# 

# La probabilidad de aparición, o de oscilación, de un $\nu_\alpha$ como $\nu_\beta$ a un tiempo $t$ es:
# 
# $$
# P_{\alpha\beta}(t) = |A_{\alpha\beta}|^2 = \frac{1}{4} \sin^2 2 \theta (1 + e^{-i \frac{\Delta m^2_{21}t}{2E}}) (1 + e^{+i \frac{\Delta m^2_{21}t}{2E}})  \\
# = \frac{1}{4} \sin^2 2 \theta \left(2 - 2 \cos \frac{\Delta m^2_{21}t}{2E} \right)  \\
# = \sin^2 2 \theta \sin^2 \frac{\Delta m^2_{21}t}{4E}
# $$

# Podemos dar la probabilidad de oscilación a una distancia $L$, que corresponde a tiempo, $t$. 
# 
# Notar que los experimentos se situan a una distancia $L$ de la fuente de neutrinos.
# 
# $$
# P_{\alpha\beta}(L) = \sin^2 2 \theta \sin^2 \frac{\Delta m^2_{21}L}{4E}
# $$
# 
# Si damos unidades:
# 
# $$
# P_{\alpha\beta}(L) = \sin^2 2 \theta \sin^2 \frac{1.27 \Delta m^2_{21} [\mathrm{eV}^2]L [\mathrm{km}]}{4E[\mathrm{GeV}]}
# $$
# 
# Se trata de una probabilidad de forma oscilatoria, cuya amplitud están controlada por $\theta$, si $\theta = 0$ no hay oscilación, y su frecuencia por $\Delta m^2_{21}$, si los dos neutrinos tienen igual masa, tampoco hay oscilación.
# 
# Para observar la oscilación, dado un $\Delta m^2_{12}$ y una energía $E$ debemos colocar el detector a una distancia $L$ adecuada.
# 

# Consideremos el caso de $\Delta m^2 = 2 \times 10^{-3} \; \mathrm{eV}^2$ y $\theta = 47^o$
# 
# Vamos a calcular la probabilidad de oscilación $P_{\alpha\beta}(L)$
# 
# *Cuestion*: ¿A qué distancia deberíamos colocar el detector para observar la oscilación máxima?

# In[7]:


theta    = np.pi * 47/180. # rad
ls       = np.linspace(0, 2000, 100) # km
ene      = 1. # GeV
delta_m2 = 0.0025 # eV^2
posc     = lambda d: np.sin(2*theta)**2 * np.sin(1.27 * delta_m2 * d / ene)**2
plt.plot(ls, posc(ls))
plt.grid(); plt.xlabel(r'$L$ (km)'); plt.ylabel(r'$P_{\alpha\beta}(L)$'); plt.ylim((0, 1.));


# ### Oscilaciones en tres familias
# 
# Como existen tres neutrinos hay dos diferencias al cuadrado de sus masas, que controlan la fase de oscilación de los neutrinos solares, $\Delta m^2_{sol}$ y de los atmosféricos, $\Delta m^2_{atm}$ [PDG]:
# 
# $$
# \Delta m^2_{sol} \simeq 7.5 \ 10^{-5} \; \mathrm{eV}^2, \;\;\; |\Delta m^2_{atm}| \simeq 2.5 \; 10^{-3} \mathrm{eV}^2  
# $$
# 
# Notar que los neutrinos son quasi-degenerados en masa.

# La relación entre los estados de sabor $\nu_\alpha$ con $\alpha = e, \mu, \tau$ con los estados de masas $\nu_i$ con masas $m_i$ donde $i = 1, 2, 3$, viene dada por una matriz unitaria $U_{PMNS}$, conocida como PMNS (Pontecorvo-Maki-Nakagawa-Sakata)
# 
# La matiz $U$ tiene 3 ángulos de mezcla y una fase compleja (de forma similar a la matriz CKM).
# 
# Se han medido experimentalmente los tres ángulos [PDG], pero no la fase compleja $\delta$.
# 
# $$
# \theta_{sol} \simeq 33^0, \; \theta_{atm} \simeq 47^o \; \theta_{13} \simeq 8^o
# $$
# 
# 
# 

# ### El futuro de las oscilaciones de neutrinos
# 
# En la próxima década el detector Hyper-Kamiokande en Japón y DUNE en los EEUU determinaran seguramente la fase $\delta$ y fijarán nuestra comprensión de las oscilaciones de neutrinos.
# 
# HK estudiará las oscilaciones $\nu_\mu \to \nu_e$ y $\bar{\nu}_\mu \to \bar{\nu}_e$ con energías de 0.6 GeV y una distancia 295 km, desde su producción en J-Parc hasta su detección en HK en la mina de Kamioka.
# 
# | |
# | :-- |
# <img src="./imgs/leptons_HK.png" width = 300 align = "center">
# |Dibujo del detector HK comparado con Notre-Dame de Paris|
# 

# ### Bibliografía
# 
#  * [AB] Alessandro Bettini, "Introduction to Elementary Particle Physcs", Cambridge U. press. Temas 7 y 10.
# 
#  * [MT] Mark Tomsom, "Modern Particle Physics", Cambridge U. press. Temas 11, 12 y 13.
#     
#  * [PDG](https://pdg.lbl.gov/) Particle Data Group.
# 
#  * [WU] C.S. Wu et al, *Phys. Rev.* **D17** 2369 (1957)
# 
#  * [MU] M. Bardon, P. Norton, J. Peoples, A. M. Sachs and J. Lee-Franzini, *Phys. Rev.
# Lett.* **14**, 449 (1965).
# 
#  * [GOL] M. Goldhaber et al, *Phys Rev* **109** 1015 (1958)
# 
#  * [KATRIN] M. Aker et al. (KATRIN Collaboration)
# *Phys. Rev. Lett.* **123**, 221802 (2019)
# 
#  * [SK] Y. Ashie et al, SK Collaboration, *Phys. Rev.* **D71** 112005 (2005)
# 
