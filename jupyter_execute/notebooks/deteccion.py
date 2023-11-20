#!/usr/bin/env python
# coding: utf-8

# # Introducción a Física de Partículas
# 
# 
# ## Producción y detección
# 
# 
# Jose A. Hernando
# 
# *Departamento de Física de Partículas. Universidade de Santiago de Compostela*
# 
# Septiembre 2021
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


# ## Objetivos
# 
#   * Conocer los principios básicos de la interacción de las particulas con la materia
#     
#   * Principales tipos de detectores

# ## Aceleradores.
# 
# La historia de la Física de Partículas está ligada a la de aceleradores.
# 
# Para producir una partícula necesitamos al menos una energía igual a su masa: $E = m c^2$, y para explorarla una longitud de onda menor que su tamaño por la relación  $p = h/\lambda$.
# 
# Se pueden acelerar las partículas estables cargadas: $e, e^+, p, \bar{p}$. 
# 
# Podemos crear haces secundarios al golpear un haz de $p$ contra un blanco, lo genera partículas secundarias, mesones ($\pi, K$), que se pueden seleccionar en un rango de energía y momento. 
# 
# Si además se dejan desintegrarse y se filtran se obtiene un haz de neutrinos $\nu_e, \nu_\mu$.

# ### Experimentos de blanco fijo y de colisión
# 
# Existen dos tipos de experimentos principales:
# 
#   * De **blanco fijo**, *fixed-target*, donde un haz se hace incidir sobre un blanco.
#   
#   * de **colisión**, *collider*, donde dos haces se cruzan en determinados puntos.
#   
# La energía disponible en el cdm en el ambos experimentos en muy diferente. 
# 
# Para un experimento de blanco fijo, $ s= (E, {\bf p}) + (m_t, 0)$, donde $(E, {\bf p})$ es el cuadri-momento de la partícula del haz con masa $m_b$, y $(m_a, 0)$ la masa de la partícula del blanco (usualmente nucleones). 
# 
# Por lo tanto la Energía en el cdm, $\sqrt{s} = \sqrt{(E+m_b)^2 - {\bf p}^2} = \sqrt{m_b^2 + 2 m_b m_a E + m^2_a}$, que depende de $\sqrt{E}$ si pueden despreciarse las masas. 
# 
# Si aumentamos por un factor 100 la energía del beam, en el CM, aumenta aproximadamente por 10.

# Para un experimento de colisión, $s = (E_a, {\bf p}_a) + (E_b, {\bf p}_b)$, donde $(E_\alpha, p_\alpha)$ son los cuadri-momentos de las partículas de los dos haces. 
# 
# El el caso de partículas relativistas de igual masa que colisioan de frente ${\bf p}_b = - {\bf p}_a$ y $E_a = E_b  = E$, la energía en el cdm, $\sqrt{s} = 2E$. Se incrementa linealmente con la energía del haz. 

# ### Introducción a aceleradores
# 
# Los haces están compuestos de **paquetes** (*bunches*), con una densidad elevada de partículas, por ejemplo LHC cada paquete tiene del orden de $10^{11}$ protones.
# 
# Existen dos tipos de aceleradores principales:
# 
# * Lineales, donde el acelerador es un dispositivo recto, un tubo de haz en vacío, que los paquetes recorren una vez.
# 
# * Circulares, los paquetes guiran en el tubo de vacío del haz numerosas vueltas $10^5$.
# 

# #### Acelerador lineal
# 
# Animación de un detector lineal
# 
# | |
# | :--: |
# | <img src="./imgs/Linear_accelerator_animation_16frames_1.6sec.gif" width = 600 align="center"> |
# | Esquema de funcionamiento de un acelerador lineal [Wikipedia]|

# El detector lineal más grande es el Linac de SLAC, California, de 3 km de largo.
# 
# 
# | |
# |:--|
# |<img src="./imgs/det_slac_aerial.png" width = 300 align="center">|
# | vista area de Slac (y el acelerador lineal), California |
# 
# 

# #### Aceleradores circulares
# 
# Los aceleradores cíclicos funcionan de manera similar pero el haz circula por un anillo.
# 
# Las partículas relativistas al girar pierden energía por radiación (**synchrotron radiation**) proporcional a $1/m^4$. 
# 
# Este efecto es más dramático para $e$ que para $p$.
# 
# Para 'girar' a las partículas se crea un campo magnético dipolar $B$ perperdicular a su dirección. 
# 
# Por el electromagnetismo sabemos que $p = 0.3 B/ \rho$, donde $p$ es el momento, en TeV, $B$ el campo magnético, en T, y $\rho$ el radio, en km.
# 
# Para el Tevatron, en Fermilab, Chicago, $\rho = 1$ km. $B = 1.5$ T, y permite acelerar protones hasta 1 TeV.
# 
# Para obtener campos más intensos hay que recurrer a imanes superconductores.
# 
# Por otro lado, las partículas oscilan dentro de los paquetes y para mantenerlas juntas se utilizan campos magnéticos de cuadropolos.

# En los **anillos de almacenamiento** se hacen girar dos haces de partículas en sentido opuestos en el mismo anillo.
# 
# En determinados **puntos de colisión** mediante un conjunto de imanes se comprimen los paquetes y se cruzan para que tengan lugar las colisiones.
# 
# Este es el caso de LEP, el colisionador $e^+e^-$ del CERN, de 27 km de circunferencia, que funcionó a finales del siglo XX.
# 
# También es el caso del LHC, que ocupa el tunel donde antes estuvo LEP. LHC es un colisionador $pp$ con una energía $\sqrt{s} = 7-13$ TeV. 
# 
# Los dipolos superconductores del LHC operan a 8 T. Los paquetes contienen del orden de $10^{11}$ $p$ que recorren el anillo a una frecuencia de 40 MHz. El volumen de los paquetes es de 40 cm de largo y 1 mm de sección que se reduce a 10 $\mu$m en la zona de colisión.
# 

# Vista aérea del LHC donde se sobreimponen las líneas de los túneles de los aceleradores del CERN.
# Esquema de los aceleradores del CERN. [video](https://www.youtube.com/watch?v=pQhbhpU9Wrg)
# 
# | | |
# |:-- | :-- |
# |<img src="./imgs/det_LHC_aerial.png" width = 350 align="center"> | <img src="./imgs/det_CERN_accelerators_complex.png" width = 320 align="center">
# | vista aérea de la zona del LHC (dibujada), Ginebra | Esquema de los aceleradores del CERN |
# 

# #### Aceleradores hadrónicos y leptónicos
# 
# Aceleradores hadrónicos:
# 
# Pueden alcanzar más energía para un menor radio. 
# 
# Sus interacciones de las partículas son complejas, porque los protones no son elementales.
# 
# | Name      | Laboratory | Type      | Year      | Energy       | Type        | Discoveries | Experiments |
# | :--       | :--        | :--       | :--       | :--          | :--         | :--         | :-- |
# | Cosmotron | Brookhaven | $p$       | 1953-1968 | 3 GeV        | synchrotron | first mesons | |
# | Bevatron  | Berkely    | $p$       | 1954-1970 | 6.2  GeV     | synchrotron | strage particles, ${\bar p}, {\bar n}$ ||
# | AGS       | Boorkhaven | $p$       | 1960-     | 33 GeV       | synchrotron | $J/\Psi$, $\nu_\mu$, CP in kaons | |
# | SPS       | CERN       | $p$       | 1968-     | 450          | synchrotron | fixed target experiments|  |
# | Main Ring | Fermilab   | $p$       | 1970-     | 500 GeV      | syncrhotron | $\nu$ oscillation | MINOS, NOvA| 
# | SPS       | CERN       | $p\bar{p}$| 1981-1984 | 270-315 GeV  | collider (6.3 km) | $W^\pm$| UA1, UA2|
# | Tevatron  | Fermilab   | $p\bar{p}$| 1987-2012 | 1.96 TeV     | collider (6.3 km) | $t$ quark| CDF, D0|
# | LHC       | CERN       | $pp$      | 2008-     | 7-13 TeV     | collider (27 km) | Higgs| ATLAS, CMS, LHCb, ALICe |

# Aceleradores leptónicos:
# 
# Las colisiones $e^+e^-$ son entre partículas elementales via interaciones electro-débil, nunca fuerte.
# 
# | Name      | Laboratory | -- Type  -- | Year      | Energy  | Type | Discoveries | Experiments |
# | :--       | :--        | :--       | :--       | :--     | :--  | :--         | :-- |
# | Linac     | SLAC       | $e$       | 1955-1968 | 3 MeV   | linear (3 km)| | |
# | AdA       | Fracati    | $e^+e^-$  | 1961-1964 | 250 MeV | collider (6 m) | first collisions | |
# | SPEAR     | SLAC       | $e^+e^-$  | 1972-1990 | 3 GeV   | collider | charmoniun states | Mark-I|
# | SLC       | SLAC       | $e^+e^-$  | 1988-1998 | 45 GeV  | linear collider | $Z$ | SLD, Mark-II| 
# | LEP-I,II  | CERN       | $e^+e^-$  | 1989-2000 | 45, 90 GeV | collider (27 km) | $Z$, SM physics | ALEPH, DELPHI, OPAL, L3 |
# | PEP-II    | SLAC       | $e^+e^-$  | 1998-2008 |  9/3.1 GeV | collider | $B$ CP violation | BaBar |
# | KEKB       | KEK        | $e^+e^-$  | 1999-2009 |  8/3.5 GeV | collider (3 km)| $B$ CP violation | Belle |
# | SuperKEKB  | KEK        | $e^+e^-$  | 2016-     |  7/4 GeV   | collider (3 km) | | Belle-II |
# 

# Mixtos:
# 
# Los electrones de alta energía permiten explorar los protones.
# 
# | Name      | Laboratory | Type      | Year      | Energy  | Type | Discoveries | Experiments |
# | :--       | :--        | :--       | :--       | :--     | :--  | :--         | :-- |
# | HERA     | DESY       | $ep$      | 1992-2007 | 27.5/907 GeV   | collider (6 km)| nucleon structure| H1, ZEUS, HERA-B|

# ### Parámetros de un colisionador
# 
# Los dos parámetros fundamentales de un colisionador son:
#     
#  * **Energía** en el centro de masas, $\sqrt{s}$, que determina la masa de las partículas que podemos crear.
# 
#  * **Luminosidad**, o el número de partículas que se curzan entre ellas por unidad de área y tiempo. Determina el número de interacciones en cada cruze. 

# #### Luminosidad
# 
# Para dos vagones con $n_1, n_2$ partículas que se cruzan entre sí en la dirección $z$ con una frecuencia $\nu$ y que tienen una sección transversal $\sigma_T$, la luminosidad es:
# $$
# \mathcal{L} = \nu \frac{n_1 n_2}{ \sigma_T}
# $$
#  
# Si un tipo de interacción tiene una sección eficaz $\sigma$, esperamos una frecuencia de iteraciones $n = \sigma \mathcal{L}$.
# 
# Recordar que la sección eficaz $\sigma$ se mide en m$^2$ o en barns (1 barn = $10^{-24}$ cm$^2$).

# 
# En general $\mathcal{L}(t)$ depende del tiempo, de la operación del acelerador. A $\mathcal{L}(t)$ le llamamos **luminosidad instantanea**.
# 
# Y a lo largo de un tiempo (típicamente años), $\int_t  \mathcal{L}(t) \, \mathrm{d}t$, **luminosidad integrada**, que se mide en (m$^{-2}$ o b$^{-1}$ -barns-). 
#  
# A las interacciones que se producen en el cruce de los vagones se las conoce como **evento**. 
# 
# En número de interacciones por evento es una variable aleatoria con distribución Poisson de media $\sigma \mathcal{L}$.

# Parámetros característicos de algunos colisonadores recientes:
# 
# | Name     | Laboratory | -- Type --  | Year      | Energy, $\sqrt{s}$  | Luminosity, $\mathcal{L}$  (cm$^{-2}$,s$^{-1}$) |
# | :--      | :--        | :--        | :--       | :--                 | :-- |
# | PEP-II   | SLAC       | $e^+e^-$   | 1999-2009 | 10.5 GeV            | $1.2 \, 10^{34}$|
# | LEP      | CERN       | $e^+e^-$   | 1989-2000 | 90-209  GeV         | $10^{32}$ |
# | Tevatron | Fermilab   | $p\bar{p}$ | 1987-2012 | 1960  GeV           | $4 \, 10^{32}$ |
# | LHC      | CERN       | $p p$      | 2009-     | 8-13 TeV            | $10^{34}$| 
# 

# Luminosidad integrada del LHC a lo largo de los años de funcionamiento.
# 
# | |
# | :-- |
# |<img src="./imgs/det_LHC_intlumi.png" width = 500 align="center">|
# | Lunimosidad integrada del LHC [CERN]|

# **cuestión**: La colaboración ATLAS ha medido la sección eficaz de producción de Higgs a $\sqrt{s} = 13$ TeV en las colisiones $pp$ con un valor $\sigma_H = 57 \pm 6$ pb. ¿Cuántas interacciones dónde se produce un Higgs se esperan en ATLAS por segundo? ¿Cuántas con una luminosidad integrada $\int_t \mathcal{L} \mathrm{d}t = 36$ fb$^{-1}$? 

# ## Interacción de las partículas con la materia
# 
# Las interacciones de las partículas pueden dividirse:
# 
#    * interacciones de las partículas cargadas
#    
#    * interacciones electromagnéticas de electrones y fotones
#    
#    * interacciones fuertes de hadrones.
# 

# ### Interacciones de las partículas cargadas
# 
# las partículas cargadas relativistas interaccionan electromagnéticamente con los electrones de los átomos de la materia y pierden energía por ionización.
# 
# Esta pérdida por distancia recorrida viene dada por la fórmula de Bethe-Block (1930s):
# 
# $$
# \frac{\mathrm{d}E}{\mathrm{d}x} \simeq - \frac{4\pi \hbar^2 \alpha^2}{m_e} \frac{n_e}{\beta^2} \left[ \ln \left( \frac{2 m_e c^2 \beta^2 \gamma^2}{I_e}\right) - \beta^2\right]
# $$
# 
# donde $v = \beta c$ es la velocidad, $\gamma$ el factor de Lorentz, $n_e$ la de densidad de electrones, $I_e$ el potencial de ionización ($I_e \simeq 10 Z$ eV).
# 
# Se trata de un valor promedio. La distribución $\mathrm{d}E/\mathrm{d}x$ está relacionada con la fluctuación del número de colisiones de la partícula con los electrones de los átomos.

# La densidad de electrones $n_e = \rho N_A Z/A$, donde $\rho$ es la densidad, $A$ la masa atómica (g/mol), $Z$ el número atómico y $N_A$ el número de Avogadro. 
# 
# Si re-escribimos:
# $$
# \frac{1}{\rho}\frac{\mathrm{d}E}{\mathrm{d}x} \simeq - K \frac{Z}{A \beta^2}\left[ \ln \left( \frac{2  m_e c^2 \beta^2 \gamma^2}{I_e}\right) - \beta^2\right]
# $$
# Donde $K = \frac{4\pi \hbar^2 \alpha^2 N_A}{m_e} = 0.307$ MeV cm$^2$/mol
# 
# Como $Z/A$ es prácticamente constante para la mayoría de los átomos. La dependencia con el material es principalmente  proporcional a su densidad.

# | |
# |:--|
# |<img src="./imgs/det_eneloss_mip.jpg" width = 400 align = "center">|
# |pérdida de energía por ionización [PDG]|

# El recorrido medio de una partícula con $\beta$ hasta que se detiene en un medio se denomina **rango** de penetración.
# 
# La pérdida de la energía depende de la velocidad de la partícula $\beta$. Distinguimos tres regiones:
#  * La pérdida es más intensa para baja $\beta$ (de la dependencia $1/\beta^2$). La inionización es mayor al final de la trayectoria, cuando la velocidad es muy pequeña. Esta región se conococe como **pico de Bragg**.
#  * En el rango de $\beta\gamma$ de $1-10$, la energía perdida es mínima, esa región se denomina **mip** (*minimum ionizing particle*).
#      Un muón de 10 GeV en hierro pierde en promedio 13 MeV/cm y su rango es de varios metros.
#  * Para $\beta \gamma >100$, la perdida aumenta de forma logaritmica. A partir de aquí los efectos de radiación son relevantes.

# La fórmula calcula el valor promedio. La distribución $\mathrm{d}E/\mathrm{d}x$ está relacionada con la fluctuación del número de colisiones de la partícula con los electrones de los átomos.
# 
# | |
# |:--|
# |<img src="./imgs/det_eloss_alice_tpc.png" width = 400 align = "center">|
# | pérdida de energía en la TPC de ALICE [ALIC]|
# 
# Pérdida de energía en la TPC (*Time Projection Chamber*) de ALICE.
# 
# Los electrones pierden tambien energía por radiación **bremsstrahlung** (en la siguiente sección).

# ### Interacciones electromagnéticas de fotones y electrones
# 
# #### Interacciones de los electrones
# 
# Las partículas cargadas pueden radiar fotones por la interacción electromagnética con los protones de los núcleos. Esta radiación se llama **bremsstrahlung**.
# 
# $$e^- + (A, Z) \to e^- + \gamma + (A, Z)$$ 
# 
# Esta radiación empieza a ser dominante a partir de una **energía crítica** $E_c \sim 800/Z$ MeV, antes domina la inoización.

# Los diagramas de Feynman son:
# 
# | |
# | :-- |
# |<img src="./imgs/feynman_bremsstrahlung.png" width = 400 align = "center">|
# | Diagramas de Feynman de bremsstrahlung |
# 
# Donde el disco representa la corriente del núcleo con $Z$ protones.
# 
# Notar que la interacción es del orden $Z^2 \alpha^3$
# 

# Este proceso se puede calcular en QED (*Quantum Electro-Dynamics*) y su sección eficaz: 
# 
# $$\sigma_{b} \propto E/m^2.$$
# 
# Afecta más a los electrones que a los muones por un factor $(m_e/m_\mu)^2$. 
# 
# Los muones por debajo de $\mathcal{O}(100)$ GeV pierden energía principalmente por ionización. 
# 
# La pérdida de energía por bremsstralung por encima de $E_c$ puede expresarse:
# 
# $$
# \frac{\mathrm{d}E}{\mathrm{d}x} = - \frac{E}{X_0}, \;\; E(x) = E_0 \, e^{-x/X_0},
# $$
# donde $X_0$ se denomina **longitud de radiación** y $E_0$ es la energía inicial del electrón.

# 
# 
# | |
# | :-- |
# |<img src="./imgs/det_eloss_electrons.png" width = 400 align = "center">|
# |energía perdida para electrones|
# 
# $X_0$ depende del material. Notar que $X_0 = n \sigma_b$, donde $n$ es la densidad de núcleos.
# 
# | H$_2$ | H$_2$O | $_{12}$C | $_{54}$Xe | $_{26}$Fe  | $_{82}$Pb | 
# | :--   | :--    | :--     | :--  | :--         | :--       |
# | 7.6 km  | 36.1 cm | 18.8 cm | 15.47 cm | 1.76 cm | 0.56 cm |
# 
# $X_0$ es la distancia a la que electrón ha perdido en promedio $1/e$ de su energía.
# 

# #### Interacciones de los fotones
# 
# Las interacciones de los fotones con la materia dependen de su rango de energía, por debajo del MeV domina el efecto foto-eléctrico, en el rango de MeVs, la dispersión Compton, y por encima 10 MeV, la producción de pares.
# 
# | |
# | -- |
# | <img src="./imgs/det_xsec_photons_lead.png" width = 340 align = "center">|
# | sección eficaz de fotones en plomo [PDG]|
# 
# A altas energías el efecto dominante será la producción de pares.
# 
# 
# 

# Los diagramas de Feyman de producción de pares:
# 
# | |
# | -- |
# | <img src="./imgs/feynman_pair_production.png" width = 350 align = "center"> |
# |Diagramas de Feynman de la producción de pares|
# 
# La sección eficaz de producción de pares crece rápidamente desde el umbral de producción y puede aproximarse:
# 
# $$
# \sigma_\gamma \simeq \frac{7}{9} \frac{1}{n X_0},
# $$
# donde $n$ es la densidad de núcleos.
# 

# La cantidad $\lambda = 1/(n \sigma_\gamma)$ es el **camino libre medio**, que vale $\lambda \simeq 7/9 X_0$ y que nos indica la cantidad de fotones que se pierden en un haz monoenergético de intensdad $I$:
# 
# $$
# \frac{\mathrm{d}I}{\mathrm{dx}} = - \frac{I}{\lambda}, \;\; I(x) = I_0 e^{-x/\lambda}
# $$
# 
# Por lo tanto la longitud de radiación caracteriza la pérdida de energía de electrones y conversión de fotones en pares para partículas por encima $\sim 10$ MeV.
# 

# Los electrones o fotones de alta energía al atravesar un medio de $X_0$ producen una cascada electromagnética.
# 
# | |
# | -- |
# | <img src="./imgs/det_egamma_cascade.png" width = 250 align = "center"> |
# | Desarrollo esquemático de una cascada electromagnética |
# 
# 

# ### Interacciones fuertes de los hadrones
# 
# Los hadrones cargados (protones, piones, kaones) pierden energía por ionización. 
# 
# También por interacciones fuertes con los núcleos de la materia.
# 
# Las interacciones fuertes se caracterizan con la **longitud de interacción**, $\lambda_I$, que es la distancia media entre interacciones fuertes. $\lambda \gt X_0$
# 
# |           | Fe    |
# |:--        | :--   |
# $\lambda_I$ | 17 cm |
# $X_0$       | 1.76 cm |

# Las interacciones de hadrones producen una casada hadrónica.
# 
# Las cascadas son más variables que las electromagnéticas, dado que en ellas se pueden producir más tipos de partículas, y también $\pi^0$s que se desintegran electromagnéticamente $\pi^0 \to \gamma \gamma$, 
# 
# Lo que da lugar a su vez a una cascada electromagnética dentro de la hadrónica. Lo que da lugar a una dispersidad en la energía repartida entre los dos tipos de cascadas.
# 
# Una parte de la energía también se pierde en forma de excitaciones y roturas nucleares.

# ## Detectores de partículas
# 
# Los detectores de partículas usan como base la interacción de las partículas con la materia, principalmente la ionización y radiación.
# 
# En Física de Partículas intentamos determinar el momento, $p$, la energía, $E$ y la identidad de las partículas, (PID), esto es si la partícula es un $\mu$, un $e$, un $K$, $\pi$, ...
# 
# Los detectores se dividen en:
#     
#    * **Detectores de trazas**: determinan las trayectorias de las partículas cargadas.
#        Sirven para medir generalmente el momento y determinar los vértices de desintegración.
#        Los detectores de trazas están habitualmente inmersos en campos magnéticos con lo que el momento se determina a partir de su curvatura.
#     
#    * **Calorímetros**: sirven para la energía de las partículas, principalmente electrones/fotones y hadrones.
#  
#     

# ### detectores tipo
# 
# 
# Los detectores tipo en colisionadores tienen una estructura en capas cilíndricas.
# 
# En el interior están los detector de trazas más precisos (silicon detectors), submergidos en un campo magnético (selenoidal).
# 
# Le siguen el calorímetro electromagnéticos y el hadrónico.
# 
# Y finalmente los detectores de muones (que son las partículas más penetrantes)

# 
# |  |
# |:-- |
# | <img src="./imgs/det_CMS_subdetectors.png" width = 500 align = "center"> |
# | Esquema de un sector del *Compact Muon Selenoid* ([CMS](https://cms.cern/detector)) del LHC|
# 

# ### Detectores de trazas
# 
# En los detectores de trazas se detecta la ionización (electrones liberados) del paso de la partícula cargada a través del medio para determinar puntos de paso o *hits*. 
# 
# Los detectores están submergidos en un campo magnético, ${\bf B}$, (Teslas) que produce una curvatura de la partícula proporcional a su momento, ${\bf p}$(GeV)  en la dirección perperdicular a ${\bf B}$
# 
# Si entre ambos tiene un ángulo $\theta$:
# $$
# p \cos \theta = 0.3 B \rho,
# $$
# donde $\rho$ (m) es el radio de curvatura en el plano perperdicular.
# 
# Llamamos al momento en el plano perperdicular, **momento transverso**, $p_T = p \cos \theta$.
# 
# Para CMS con $B = 4$ T, y un $\pi^\pm$ de $p$ 100 GeV, $\rho \sim 100$ m.  

# ##### Detectores de silicio
# 
# Están basando en vaciar de portadores libres de carga una oblea de silico (aprox $~300$ $\mu$m de esperor) donde se ha dopado tiras (*strips*) de tipo $p$, separadas aprox $~50$ $\mu$m, para crear uniones $pn$.
# 
# | |
# | :-- |
# <img src="./imgs/det_silicon_detector.png" width = 400 align = "center">
# |Esquema de un sensor de micro-strips de silicio|
# 
# La ionización del paso de una partícula cargada crea pares electrón/hueco ($\mathcal{0}(1)$ eV).
# 
# Los electrones de ionización derivan hacia las tiras $p$, donde su carga es amplificada por la electrónica (*front end electronics*).

# Con estos detectores se pueden reconstruir las trayectorias con precisión $\sim 10$ $\mu$m e identificar vértices de desintegración de partículas que pueden recorrer $\sim  1$ cm.
# 
# | | |
# |-- | -- |
# | <img src="./imgs/det_DELPHI_vertex.jpeg" width = 350 align = "center">| <img src="./imgs/det_DELPHI_bdecay.gif" width = 350 align = "center"> |
# |DELPHI micro-vertex detector| DELPHI Event con b-tag [DELPHI]|
# 
# ATLAS y CMS utilizan detectores de silicio, que se desarrollaron a partir de 80's. Estos detectores fueron esenciales en los experimentos de LEP, BaBar, Belle entre otros y FERMI, en astropartículas.

# #### Detectores gaseosos
# 
# En los detectores gaseosos se utiliza la ionización ($\sim 30$ eV por ionización) del paso de las partículas cargadas en gases nobles Ar, Xe. 
# 
# Los detectores están en un rango de voltage proporcional (no hay efecto avalancha). 
# 
# Los electrones derivan bajo la presencia de un campo eléctrico ${\bf E}$ hasta el ánodo, que puede estar formado por hilos (*wires*).
# 
# Existen varios tipos de detectores: *wire chambers*, *multiproportial wire chambers (MPWC)*, *time proyection chambers (TPC)*.

# ##### Cámaras de proyección temporal (TPC)
# 
# Las TPC suelen tener forma de barril, pueden ser simétricas, con dos tapas como ánodos y un cátodo central de HV.
# 
# Bajo la presencia del campo eléctrico, los $e$ de ionización derivan hasta el ánodo.
# 
# Son recogidos por un detector segmentado (hilos) en dos direcciones ($R, \phi$) o ($x, y$), donde se amplifica su carga (electrónica frontal).
# 
# El tiempo de llegada de los electrones $\Delta t$ sirve de estimación de la posición en $z$ (el eje de la cámara).  

# Las TPCs son excelentes detectores de trazas
# 
# | | |
# |-- | -- |
# | <img src="./imgs/det_ALICE_TPC_scheme.png" width = 400 align = "center">| <img src="./imgs/det_ALICE_ppevent.jpeg" width = 400 align = "center"> |
# |ALICE TPC| ALICE pp event [ALICE]|
# 
# Las TPC se utilizan también en experimentos de búsqueda de materia oscura, XENON, detectores de neutrinos EXO, NEXT.

# ### Calorímetros
# 
# #### Calorímetros electromagnéticos
# 
# Los calorímetros electromagnéticos estás construidos con materiales de alto $Z$, alto $X_0$
# 
# Suelen tener una estructura alternada de material pasivo (por ejemplo Pb), donde se desarrollan las cascadas, y  material activo, donde se detecteca la ionización.
# 
# La resolución en energía, $\sigma_E$, está limitada por las fluctuaciones en la producción de las partículas (proporcional a $\sqrt{E}$) en la cascada, en general:
# 
# $$
# \frac{\sigma_E}{E} \sim \frac{3 - 10 \mathrm{\%}}{\sqrt{E \;\mathrm{(GeV)}}}
# $$
# 

# El calorímetro de CMS está formado por cristales de plomo y tungusteno PbWO$_4$, con $X_0 = 0.83$ cm, que es un centelleador inorgánico.
# 
# Los **centelleadores** contienen moléculas centelleadoas que en vez de ionizar, se excitan, y al de-excitarse emiten luz en el visible que puede detectarse con detectores de fotones (por ejemplo foto-multiplicadores). El número de fotones es proporcional a la energía absorvida, aproximadamente 100 eV por fotón de centelleo.
# 
# Los centelleadores deben tener una $X_0$ alta para evitar la conversión de los fotones.

# #### Calorímetros hadrónicos
# 
# Los calorímetros hadrónicos estan construidos con un material de alta densidad, aún así, las cascadas suelen ocupar mucho volumen, por ejemplo, un hadrón de 100 GeV genera una cascada de paroximadamente 2 m de profundidad y 50 cm de anchura.
# 
# Estan construidos habitualmente en capas de material pasivo y activo. El calorimetro de ATLAS está formado por placas y plástico centelleador.
# 
# La resolución en energía, $\sigma_E$, está limitada por la producción de las partículas, $\sqrt{E}$, por la fracción entre la cascada electromagnética y hadrónica, y por la energía perdida.
# 
# $$
# \frac{\sigma_E}{E} \sim \frac{50 \mathrm{\%}}{\sqrt{E \mathrm{(GeV)}}}
# $$

# ### Detectores de luz Cherenkov
# 
# El paso de partículas cargadas que se mueven a una velocidad, $v$, mayor que la luz en el medio, $v \gt c/n$, donde $n$ es el índice refracción, genera un frente de onda coherente con un determinado ángulo, $\theta$, respecto a la dirección ${\bf v}$, en el visible, conocida como luz Cherenkov.
# 
# | |
# | :-- |
# <img src="./imgs/det_cherenkov_schematic.png" width = 300 align = "center">
# |Esquema de emisión de luz Cherenkov|
# 
# $$
# \cos \theta = 1/\beta n
# $$
# 
# Si detectamos el cono de luz, podemos determinar la dirección, $\hat {\bf v}$ y ${\beta}$.

# Los neutrinos, como veremos, tienen una sección de interacción muy pequeña, $\sigma \sim 10^{-38}$ cm$^2$, por lo que para detectarlos se necesitan detectores de gran masa y fuentes con un flujo muy intenso de neutrinos.
# 
# | | |
# | :-- | :-- |
# | <img src="./imgs/det_SK_image.jpeg" width = 350 align = "center"> | <img src="./imgs/det_SK_ne_ring.jpeg" width = 300 align = "center">
# |Imagen del interir de SK| anillo del cono de una interacción $\nu_e$|
# 
# SuperKamiokande, en Japón, es un detector de neutrinos, es un gigantesco tanque de agua, $n = 1.33$, de 50 k toneladas, $40$ m de profundidad, $41$ m de diámetro y 13 k PMTs.
# 
# Detecta los neutrinos mediante la interacción $\nu_e(\nu_\mu) \to e(\mu) + X$, donde $e, \mu$ emiten radiación Cherenkov que se detecta en las paredes recubiertas con gigantescos PMTs. 

# ## Sistema de disparo, procesado y análisis de datos
# 
# ### sistema de disparo
# 
# Los detectores de partículas tienen un **sistema de disparo**, *trigger*, para seleccionar al momento los datos relevantes.
# 
# Por ejemplo el LHC tiene una frecuencia de colisiones de 40 MHz, 25 ns. El tamaño de la información de un evento en ATLAS es de Mbytes. Es imposible actualmente almacenar todas las colisiones que se producen en el LHC.
# 
# El trigger es un sistema de diversos niveles donde se buscan trazas (principalmente muones) de alto momento transverso y partículas desplazadas del vértice de interacción (que provienen posible de partículas de vida corta). Estas son las características principales de sucesos de interés.
# 
# El sistema de disparo de alto nivel, *High Level Trigger*, HTL, selecciona eventos pre-procesados *on-line* en una granja de ordenadores in-situ.

# 
# ### Procesado de datos
# 
# Los datos que pasen el trigger se **almacenan y procesan** posteriormente en centros de computación interconectados, *GRID*. A  partir de las señales de cada sensor (cuentas de adc, tiempo, etc) se reconstruyen trazas y cascadas, y de ahí partículas con su cuadri-momento, posición e identificador. Este procesado se realiza en programas específicos en lenguajes como C++ o Python.
# 
# El producto final son ficheros de datos procesados (*data summary tape*, DST) que se analizan con **técnicas estadísticas de análisis**, muchas veces multi-variables, como redes neuronales.
# 
# El proceso se completa con la producción de **eventos simulados** por técnicas de Monte-Carlo donde se simula con gran precisión el detector y la física. La producción de datos simulados se realiza también en centros de computación y el grid. Los datos simulados sirven de referencia y permiten estimar y preveer las capacidades de un detector.

# ### Errores, observación y evidencia
# 
# Las medidades de los experimentos se presentan como:
# 
#    * la **estimación de un observable**, por ejemplo: una sección eficaz, la vida media, la masa o una fracción de desintegración. 
#    
#    * **un límite en el observable** como la vida media, masa, etc, de una **búsqueda**.
# 
# Los primeros están sujetos a una estimación de errores. Los errores son de dos tipos (y se suelen dar por separado):
# 
#   * **estadísticos**: dependiendo de la cantidad de sucesos relevantes disponibles.
# 
#   * **sistemáticos**: que reflejan nuestras incertidumbres en parámetros que afectan al observable, que puede ser de diversas índole: de calibración, de estimación de eficiencias de selección, teóricos, etc.
#   
# La determinación de un límite se presenta con un nivel de confianza y los errores ya están incorporados en el propio límite. 

# 
# La interpretación de los errores y el nivel de confianza depende de si en análisis es **frequentista** o **bayesiano**.
# 
# Podemos decir que los frequentistas parten de una teoría, $\mu$, y verifican si concuerda con los datos, $x$, esto es calculan la 'probabilidad' $p(x|\mu)$; mientras que los bayesianos, intentar inferir si la teoría es correcta a partir de los datos $p(\mu | x)$.
# 
# Un bayesiano diría: "¡Hemos encontrado el Higgs!" y un frecuentista: "Los datos casan con la existencia de un Higgs".

# ### Biblografía
# 
# * [AB] "Elementary Particle Physics", A. Bettini, Cambridge. Tema 1.
#  
# * [MT] "Modern Particle Physics", Mark Thomson, Cambridge. Tema 1
# 
# * [MS] "Particle Physics", B.R. Martin, G. Shaw, John Wiley & Sons
# 
# * [BM] "Nuclear and Particle Physics, (an introduction)", B. R. Martin, Wiley
# 
# * [AL] "Elementary Particle Physics (an intuitive introduction)", A. Loarkoski, Cambridge
# 
