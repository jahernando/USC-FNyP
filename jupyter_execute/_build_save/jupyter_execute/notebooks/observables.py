#!/usr/bin/env python
# coding: utf-8

# # Observables en Física de Partículas
# 
# *Jose A. Hernando Morata*
# 
# *Departamento de Física de Partículas. Universidade de Santiago de Compostela*
# 
# *Noviembre 2023*
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
# 
# * Recordar la cinemátics relativista
# 
# * Discutir los principables observables y los parámetros de los experimentos
# 
#   - sección eficaz, amplitud de desintegración o tiempo de vida media, fracción de desintegración
# 
# 
# * Introducción a la Física de Partículas experimental
# 
#   - El paso de las partículas por la materia, los detectores
#   
#   - La dinámica: experimentos, las medidas y sus errores
# 
# * Introducción a la física teórica de partículas y los cálculos de los observables
# 
#   - Unos apuntes mínimos de la Teoría (Cuántica de Campos) y la ecuación de Dirac
# 
#   - La regla de oro de Fermy y los diagramas de Feynman: las corrientes, los vértices (acoplos) y los mediadores
#     

# ## Pasos previos
# 
# ### Unidades Naturales
# 
# Los físicos/as de Partículas trabajamos en **unidades naturales**, UN, donde
# 
# $$
# \hbar =  1, \;\;\; c = 1.
# $$
# 
# Así, la ecuación de Einstein queda:
# 
# $$
# E^2 = m^2 c^4 + p^2 c^2 \;\; \Rightarrow \;\; E^2 = m^2 + p^2 
# $$
# 
# Para convertir la unidades naturales al sistema internacional, SI, se utilizan los siguientes factores de conversión:
# 
# | ---- $\hbar$ ------| ---- $c$ ------ |  $\hbar c$  ---- | ---- GeV ----- |
# | :--:     | :--:  | :--: | :--: |
# | $1.055$ $10^{-34}$ Js | $2.998$ $10^8$ m/s| 0.197 GeV fm | $1.602$  $10^{-20}$ J | 
# 
# Energía, momento y masa las daremos en MeV o GeV, y habitualmente tiempo y distancia los damos en s y fm.

# La tabla muestra como las unidades de las magnitudes más comunes en NU:
# 
# | Cantidad | kg, m, s|  $\hbar, c$, GeV|  NU|
# |:--       | :--     |:---             | :--                  | 
# | Energía  | kg m$^2$ s$^{-1}$ | GeV          | GeV |
# | momento  | kg m s$^{-1}$    | GeV/$c$      | GeV |
# | masa     | kg               | GeV/$c^{2}$ | GeV |
# | tiempo   | s                | $\hbar$/GeV | GeV$^{-1}$|
# | distance | m                | ($\hbar c$/GeV) | GeV$^{-1}$ |
# | área     | m$^2$            | ($\hbar c$/GeV)$^{2}$ | GeV$^{-2}$ |
# 
# La magnitud de referencia es GeV.

# 
# Es útil recordar el factor de conversión: $\hbar c$ = 0.197 GeV fm.
# 
# Para convertir de NU a SI se añaden los factores $\hbar, c$ correspondientes y para cuadrar las magnitudes.
# 
# *Cuestión*: expresa el radio del protón $r = 4.1$ $\mathrm{GeV}^{-1}$ en el S.I.

# In[3]:


hbarc = 0.197  # GeV fm (femto = 10^-15) 
r_un  = 4.1    # 1/GeV
r_si  = r_un * hbarc
print(' radio del protón ', r_si, ' fm')


# 
# ### Recordatorio de la cinemática relativista
# 
# #### Transformaciones de Lorentz
# 
# La transformación de Lorentz nos relaciona el espacio-tiempo $(t, {\bf r})$ en un sistema inercial $\Sigma$ con el espacio-tiempo $(t', {\bf r}')$ en otros sistema inercial $\Sigma'$ que se desplaza respecto del primero con velocidad $v$ en la dirección $z$.
# 
# | |
# |:--:|
# |<img src="./imgs/fun_frames_relativity.png" width = 400 align="center">|
# | Sistemas inerciales, $\Sigma'$ se desplaza con velocidad $v$ respecto $\Sigma$ [MT2.2]|
# 
# Einstein postuló que la velocidad de la luz, $c$, es la misma en los dos sistemas y nada puede viajar más rápido que la luz $v \lt c$. 
# 
# La luz producida en $t = t' = 0$, cuando el origen de los dos sistemas coincide, cumple: 
# 
# $$
# c^2t^2 - {\bf r}^2 = c^2 t'^2 - {\bf r}'^2,
# $$
# 
# donde ${\bf r} = (x, y, z)$, y de igual manera para ${\bf r}' = (x', y', z')$.
# 
# 

# Se cumple esa condición si las coordenadas en los dos sistemas están relacionadas por la transformación de Lorentz.
# 
# $$
# t' = \gamma \left( t - \frac{v}{c^2} z \right), \; x' = x, \; y' = y, \, z' = \gamma (z - vt) 
# $$
# 
# donde introducimos, el **factor de Lorentz**, $\gamma$, y $\beta$:
# 
# $$
# \gamma = (1 - \beta^2)^{-1/2}, \; \; \beta = v/c 
# $$

# En NU:
# 
# $$
# t' = \gamma (1 - \beta z),  \; x' = x, \; y' = y, z' = \gamma (z - \beta t)
# $$
# 
# $$
# \begin{pmatrix} t' \\ x' \\ y' \\ z' \end{pmatrix} = 
# \begin{pmatrix} \gamma         & 0 & 0 & -\gamma \beta \\
#                  0             & 1 & 0 & 0 \\
#                  0             & 0 & 1 & 0 \\
#                  -\gamma \beta & 0 & 0 & \gamma 
# \end{pmatrix}
# \begin{pmatrix}t \\ x \\ y \\ z \end{pmatrix}
# $$
# 

# Consideremos una partícula de masa $m$ que para un observador va a velocidad $\beta$ (en $z$). 
# 
# El sistema del observador es $\Sigma$, mientras que el sistema donde la partícula está en reposo (va a su misma velocidad) es $\Sigma'$
# 
# La tranformación de Lorentz de $\Sigma' \to \Sigma$ viene dada por:
# 
# $$
# \begin{pmatrix} t \\ x \\ y \\ z \end{pmatrix} = 
# \begin{pmatrix} \gamma         & 0 & 0 & \gamma \beta \\
#                  0             & 1 & 0 & 0 \\
#                  0             & 0 & 1 & 0 \\
#                  \gamma \beta & 0 & 0 & \gamma 
# \end{pmatrix}
# \begin{pmatrix}t' \\ x' \\ y' \\ z' \end{pmatrix}
# $$
# 
# ¡Un cambio se signo en la velocidad!

# ##### cuatro-vectores
# 
# La magnitud ${\bf r}^2$, el módulo al cuadrado de un vector 3D, donde ${\bf r} = (x, y, z)$, es invariante bajo rotaciones.
# 
# Defininos el **cuadri-vector** como $x = (t, {\bf x}) = (t, x, y, z)$, y su *norma*, o producto escalar, al cuadrado, viene dada por:
# 
# $$
# x^2 = t^2 - {\bf x}^2
# $$
# 
# es un **invariante** bajo transformaciones de **Lorentz**.
# 
# Si definimos $g_{\mu\nu}$ como la **matriz diagonal de la métrica**:
# 
# $$
# g_{\mu\nu } = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}
# $$
# 
# El producto escalar al cuadrado definido entre dos **cuadri-vectores**, $a, b$, se expresa como:
# 
# $$
# a \, b =  g_{\nu\mu} a^\nu b^\mu  \equiv  \sum_{\mu=0}^3 \sum_{\nu = 0}^3 g_{\mu\nu} a^\nu b^\mu
# $$

# ##### cuadri-momento
# 
# Si consideramos una partícula de masa $m$, que para un observador se mueve a velocidad $\beta$ (en $z$). La partícula tiene energía $E$ y momento ${\bf p}$, y cumple la ecuación de Einstein (en UN):
# 
# $$
# E = m^2 + {\bf p}^2 \;\; \Rightarrow \;\; m^2 = E^2 - {\bf p}^2
# $$
# 
# que es válida en todos los sistemas inerciales (relacionados por transformaciones de Lorentz). 
# 
# Si definimos **cuadrimomento** de una partícula como 
# 
# $$
# p \equiv (E, \, {\bf p}),
# $$
# 
# El cuadrimomento **es un cuadrivector**.
# 
# Su norma al cuadrado es un invariante Lorentz, $p^2 = E - {\bf p}^2 = m^2$, y coincide con la masa de la partícula.

# En el sistema donde la partícula está en reposo, $\Sigma'$, el cuadrimomento es $p' = (m, {\bf 0})$.
# 
# Podemos obtener el cuadrimomento $p = (E, {\bf p})$ de la partícula en el sistema del observador, $\Sigma$, para el que la partícula tiene velocidad ${\beta}$ (en z) aplicando la transformación de Lorentz:
# 
# $$
# p = (m \, \gamma,0, 0,  m \gamma \beta), 
# $$
# 
# Esto es:
# 
# $$
# E = m \, \gamma \;\;\; {\bf p} = m \, \gamma  \, {\vec \beta}.
# $$
# 
# O podemos dar los parámetros de Lorentz a partir del cuadrimomento en el sistema del observador:
# 
# $$
# \gamma = \frac{E}{m}, \;\;\; {\vec \beta} = \frac{{\bf p}}{E}
# $$

# ### Interacciones y desintegraciones de partículas
# 
# #### Interacciones y colisiones
# 
# Podemos **crear nuevas partículas** al hacer **interaccionar o colisionar  partículas a alta energía**
# 
# Por ejemplo mediante la interacción $\beta$ inversa:
# 
# $$
# \bar{\nu}_e + p \to n + e^+
# $$
# 
# podemos crear un $n$ y un $e^+$, siempre que el $\bar{\nu}_e$ tenga la suficiente energía al colisionar con un protón.
# 
# Observa que en la interacción $\beta$ inversa se conserva la carga, Q, y los números leptónico, $L$ y bariónico, $B$.
# 
# ¡Y por su puesto, se conserva la energía, el momento y el momento angular total!

# 
# También podemos crear nuevas partículas (un par de fermión-antifermión) mediante **aniquilaciones** de un partículas con su antipartícula
# 
# por ejemplo: 
# 
# $$
# e + e^+ \to \mu + \mu^+.
# $$
# 
# Un par de electrón y positrón se aniquilan y producen un par de muón y antimuón. Veremos más adelante que este proceso está mediado o por un fotón o por un bosón $Z^0$.
# 
# La energía de los dos electrones necesaria para producir en el centro de masas, CM, los dos muones debe ser al menos $\sqrt{s} \ge 2 m_\mu$ (ver más adelante), ¡esto es los dos muones aparecerían en reposo!.
# 
# Notar que el anti-fermión de la aniquilación (o producción de pares) es siempre la antipartícula de fermión, en este caso $e^+$ es el anti-fermión de $e$, y el $\mu^+$ es el anti-fermión de $\mu$. 
# 
# Por lo tanto, siempre que creamos por aniquilación antimateria creamos a su vez la materia complementaria.

# #### interacciones en colisión o en blanco fijo
# 
# 
# Recordemos que existen dos tipos principales de experimentos en Física de Partículas:
# 
#   * **colisionador**: haces de $e, e^+, p, \bar{p}$ colisionan entre sí, por ejemplo [LEP](https://es.wikipedia.org/wiki/Large_Electron-Positron_collider) ($e +e^+$)
# 
#   * **blanco fijo** (*fixed target*): haces de $e, p, \mu, \nu$ chocan contra un blanco (átomos, núcleos). Los experimentos de neutrinos son de blancos fijos, por ejemplo [Gargamelle](https://en.wikipedia.org/wiki/Gargamelle).
# 
# 
# 
#  

# #### creación de partículas en colisiones o en blanco fijo
# 
# Las figuras muestran un dibujo del detector CMS (colisiones $(p, p)$ del LHC) y de SuperKamiokande (blanco fijo donde interaccioan neutrinos).
# 
# | | |
# |:--: | :--: |
# | <img src="./imgs/intro_CMS.jpg" width = 400 align="center">|  <img src="./imgs/intro_SK.jpg" width = 250 align="center">|
# | esquema del CMS [CERN] | dibujo de SK [SK] |
# 
# SuperKamiokande (SK) un gigantesco tanque de 50 kton de agua que actua como blanco donde interactuan los neutrinos solares, atmosféricos y de un haz desde JPARC. 
# Se está construyendo un detector mayor, HyperKamiokande, [HK](https://www-sk.icrr.u-tokyo.ac.jp/en/hk/), que operará en 2027.

# ### Interacción en colisión o en blanco fijo
# 
# La figura muestra el caso de colisión de dos partículas $a, b$ (izda) y de interacción de una partícula $a$ contra un blanco fijo $b$ (derecha).
# 
# | |
# |:--|
# | <img src="./imgs/intro_collider_target.png" width = 600 align="center">|
# | colisión central (izda) y en blanco (derecha) de dos partículas $a, \, b$.|
# 
# A partir de los cuadrimomentos de las partículas $p_a, \; p_b$, definimos la variable $s$, el cuadrado de la suma de los momentos iniciales,
# 
# $$
# s = (p_a + p_b)^2,
# $$
# 
# que es un invariante Lorentz.
# 

# En un mismo sistema inercial, la suma de cuadrimomentos iniciales es igual a los finales, por conservación de la energía y el momento antes y después de la interacción.
# 
# $$
# \sum_{i \in \mathrm{iniciales}} p_i = \sum_{j \in \mathrm{finales}} p_j \Rightarrow s = \left(\sum_{i \in \mathrm{iniciales}} p_i \right)^2 = \left(\sum_{j \in \mathrm{finales}} p_j \right)^2
# $$
# 
# Por lo tanto $s$ se conserva en la interacción.
# 
# Además, por ser invariante Lorentz, podemos calcular $s$ en el sistema inercial que más nos convenga.

# Consideremos el caso, $a+b \to c$, donde el choque en el centro de masas de $a$ y $b$, produce una tercera partícula $c$ de masa $m_c$.
# 
# El momento inicial total en el centro de masas es nulo, esto es ${\bf p}_b = - {\bf p}_a = {\bf p}^*$, por lo que $c$ queda en reposo, ${\bf p_c} = 0$.
# 
# El valor de $s$ antes de la colisión, $s = (p_a+p_b)^2$ y después, $s = p^2_c$, es el mismo:
# 
# $$
# s = (p_a + p_b)^2 = ((E_a, {\bf p}^*) + (E_b, -{\bf p}^*))^2 = (E_a+E_b, {\bf 0})^2 = (E_a+E_b)^2, \;\;\;\; s = (m_c, {\bf 0})^2 = m^2_c
# $$
# 
# Las cantidades con $p^*$ se refieren en general al sistema CM.
# 
# Esto es:
# 
# $$
# m_c = E_a + E_b 
# $$
# 
# Luego $\sqrt{s}$ correspondería a la masa de la particula producida, y en el caso general a la energía disponible para crear partículas, o a la **energía en el centro de masas**.

# 
# Veamos cuánto vale $\sqrt{s}$ en un experimento de colisión en el centro de masas, CM, y otro de blanco fijo.
# 
# 1. En una **colisión en el CM**: 
# 
# $$
# {\bf p}_b = - {\bf p_a}  = p^* \Rightarrow s = ((E_a, {\bf p_a}) + (E_b, - {\bf p_a}))^2 = (E_a+E_b)^2
# $$ 
#   
#   Si consideramos el caso en el que $m_a (m_b) \ll E_a (E_b)$, entonces $E_a = E_b = E$, obtenemos:
#  
# $$
# s = (2 E)^2 \Rightarrow \sqrt{s} \propto E
# $$ 
#   
# 2. En una interacción **en blanco fijo**:
#   
# $$
# s = ((E_a, {\bf p}_a) + (m_b, 0))^2 = m^2_a + m^2_b + 2 \, m_b \, E_a.
# $$
#   
#   En el caso de antes, donde las masas son despreciables respecto a su energía, $m_a \ll E_a , \;\; E_a = E$, obtenemos:
#   
# $$
# s = 2 m_b E_a \Rightarrow \sqrt{s} \propto \sqrt{E}
# $$
# 
# En el caso de los colisionadores, $\sqrt{s}$, crece con $E$ y en el caso del blanco fijo con $\sqrt{E}$.

# *Cuestión:* considera el caso de $E = 100$ GeV, ¿En qué tipo de experimento podríamos crear partículas de mayor masa?
# 
# *Ejemplo:* ¿Cuál es la energía mínima de un neutrino para que tenga lugar la interacción beta inversa en un blanco fijo?
# 
# $$
# \bar{\nu}_e + p \to n + e^+
# $$

# *Solución:*
# 
# El valor de $s$, es igual, calculado como el cuadrimomento total de las partículas iniciales *o* finales.
# 
# Para la iniciales, en el *sistema de laboratorio*, considerando la energía del neutrino $E$ y su masa nula.
# 
# $$
# p_{\bar{\nu}_e} = (E, E \, {\bf k}), \;\; p_p = (m_p, {\bf 0}) \Rightarrow s = (E+m_p, E \, \bf{k})^2
# $$
# 
# Para las finales, considerando el *sistema centro de masas*, y su energía mínima (que las dos se produzcan en reposo). El valor *mínimo* de $s$ es:
# 
# $$
# p_n = (m_n, {\bf 0}), \;\; p_{e^+} = (m_e, {\bf 0}) \Rightarrow s = (m_n + m_e, {\bf 0})^2
# $$
# 
# Del hecho que el valor de $s$ es igual antes y después de la interacción,
# 
# $$
# (E+m_p, E \,{\bf k})^2 = (m_n+m_e, {\bf 0})^2, \\
# (E+m_p)^2 - E^2 = (m_n+m_e)^2, \;\;\; 2m_p E +  m^2_p = (m_n+m_e)^2,
# $$
# 
# obtenemos que la energía umbral del neutrino para producir una interacción $\beta$ inversa es
# 
# $$
# E = \frac{(m_n+m_e)^2 - m^2_p}{2 m_p}.
# $$

# In[4]:


m_p = units.value("proton mass energy equivalent in MeV")
m_n = units.value("neutron mass energy equivalent in MeV")
m_e = units.value("electron mass energy equivalent in MeV") 
E = ((m_n + m_e)**2 - m_p**2)/(2 * m_p)
print("Energía umbral del nuetrino = {:4.3f} MeV".format(E))


# #### Desintegraciones de partículas
# 
# Las partículas inestables (todas a excepción de $p, e, \nu$), se desintegran a otras partículas
# 
# El siguiente esquema muestra la **desintegración** de una partícula $a$, en reposo, en dos $b, d$, con masas $m_a, m_b, m_c$.
# 
# | |
# |:--:|
# |<img src="./imgs/intro_drawing_decay2.png" width = 400 align="center">|
# | Esquema de desintegración de la partícula $a$ a dos $b, c$ en el CM|
# 
# En el CM los momentos de $b, c$ son iguales, $p^*$, y ese caso se cumple:
# 
# $$
# s = (m_a, {\bf 0})^2 = m^2_a = (p_b + p_c)^2 = (E_b+E_c, {\bf 0})^2
# $$
# 
# *Ejercicio:* Calcula cuánto vale $p^*$ en función de $m_a, m_b, m_c$.

# ##### Desintegraciones a varias partículas
# 
# Si consideramos ahora la una partícula con masa $m_a$  se desintegra a varias, $n$, partículas, cada una con cuadrimomento $p_i$ y masa $m_i$. Podemos calcular $s$ inicial en el sistema donde $a$ están en reposo, y $s$ final en el sistema del laboratorio. 
# 
# $$
# s = (m_a, {\bf 0})^2 =  m^2_a, \\
# s = \left(\sum_{i = 1}^n p_i \right)^2 = \left( \sum_i E_i, \, \sum_i {\bf p}_i \right)^2 = \left( \sum_i E_i \right)^2 - \left( \sum_i {\bf p}_i \right)^2,
# $$
# 
# donde $p_i$ es el cuadrimomento de la partícula $i$, con energía $E_i$ y momento ${\bf p}_i$, en el sistema de laboratorio. 
# 
# La variable $s$ se corresponde a la masa al cuadrado del sistema (que se llama redundantemente **masa invariate** al cuadrado). 
# 
# $$
# s = \left( \sum_{i  = 1}^n  p_i \right)^2
# $$
# 
# Si las $n$ partículas provienen de la desintegración de $a$ entonces su masa invariante nos permite **detectar la partícula** "a" que se desintegró.

# En la siguiente figura se muestra el espectro de la masa invariante de dos muones ($\mu, \mu^+$) obtenida con los datos del LHCb en el Run-I y Run-II. 
# 
# El pico a ~5.4 MeV, corresponde a la masa del $B_s$. El resto de eventos se consideran de contaminación.
# 
# | |
# |:--:|
# |<img src="./imgs/intro_massinv_Bsmumu.png" width = 700 align="center">|
# | Espectro de la mas invariante y observación de $B_s \to \mu + \mu^+$ en los datos del Run-I (izda) y Run-II (dcha) del experimento [LHCb] (2021)|
# 
# *Nota:* Que la estadística en el Run-I para determinar si existe un pico en el $B_s$ es insuficiente.

# #### Los parámetros de los colisionadores
# 
# Los dos parámetros fundamentales de un colisionador son:
#     
#  * **Energía** en el centro de masas, $\sqrt{s}$, que determina las partículas que se pueden crear en la colisión.
# 
#  * **Luminosidad**, $\mathcal{L}$, o el número de partículas que lanzamos entre ellas por unidad de área y tiempo. Establece el número de interacciones en cada cruze. 
# 
# La frecuencia de eventos, en Hz = $\mathrm{s}^{-1}$, que se producen por cruce, $\nu$ es el producto de la luminosidad $\mathcal{L}$ y una sección eficaz, $\sigma$:
#     
# $$
#  \nu = \mathcal{L} \, \sigma.
# $$
# 
# Como la luminosidad pueden depender del tiempo, llamamos luminosidad instantánea a $\mathcal{L}(t)$  y la integral en un tiempo $T$, $\int_0^T \mathcal{L}(t) \mathrm{d}t$, **luminosidad integrada**.
# 
# La luminosidad se mide en $\mathrm{cm}^{-2}\mathrm{s}^{-1}$ y la luminosidad integrada en $\mathrm{cm}^{-2}$, o en $\mathrm{b}^{-1}$ (generalmente en fb).

# #### Los parámetros de los colisionadores
# 
# Esta es una lista de algunos colisionadores:
# 
# | collider | Laboratory | Type  ----     | Period    | Energy, $\sqrt{s}$, (GeV) | Luminosity, $\mathcal{L}$, ($cm^{-2}s^{-1})$ ---|
# | :--:      | :--:      | :--:       | :--:      | :--:             | :--: |
# | PEP-II   | SLAC       | $e+e^+$   | 1999-2009 | 10.5             | $1.2 \, 10^{34}$|
# | LEP      | CERN       | $e+e^+$   | 1989-2000 | 90-209           | $10^{32}$ |
# | Tevatron | Fermilab   | $p+\bar{p}$ | 1987-2012 | 1960             | $4 \, 10^{32}$ |
# | LHC      | CERN       | $p+p$      | 2009-     | 7000-13000            | $10^{34}$| 

# Luminosidad integrada del LHC a lo largo de los años de funcionamiento.
# 
# | |
# | :-- |
# |<img src="./imgs/det_LHC_intlumi.png" width = 600 align="center">|
# | Lunimosidad integrada del LHC [CERN]|

# *Cuestion*: ¿Cuántos Higgs se han producido en el LHC Run-II si la sección eficaz de producción es $\sigma = 56 \pm 4$ pb?

# In[5]:


lumi  = 160. / units.femto # 1/barn
sigma =  56. * units.pico  #   barn
N     = lumi * sigma
print('Numero de Higgs producidos por el LHC en el Run-II a sqrt(s) = 13 TeV = {:e}'.format(N))


# #### Parámetros de los experimentos en blanco fijo
# 
# Los parámetros fundamentales de un experimento de blanco fijo son:
# 
#   * la **energía** del haz incidente, $E$, recordemos $\sqrt{s} = \sqrt{2 m_b E_a}$
#   
#   * el **flujo de haz**, $\Phi$, las partículas que atraviesan una unidad de área por unidad de tiempo.
#   
#   * el **número de partículas blanco**, $N$, que es proporcional a la masa del blanco.
#  
# La frecuencia de eventos, en Hz, en un experimento de blanco fijo viene dada por:
# 
# $$
# \nu = \phi \, \sigma N
# $$
# 
# donde, por ejemplo, sin el blanco son los protones de los núcleos del material con $(A, Z)$
# 
# $$
# N = \frac{Z N_A}{A} M
# $$
# 
# donde $M$ es la masa del blanco, en g, $A$ el número de masa g/mol, $N_A$ el número de Avogadro, y $Z$ el número de protones en cada núcleo.

# Conforme el haz penetrea en el material, su flujo disminuje.
# 
# Si la densidad de los elementos blanco es $n$, y el haz penetra normal (en la dirección $z$) con flujo inicial $\Phi$ al blanco, y éste tiene una sección trasversal $S$:
# 
# $$
# \mathrm{d}I(z) =- \sigma n \Phi S \mathrm{d}z \;\;\; \Rightarrow \;\;\; I(z) = I_0 e^{-n \sigma z} = I_0 e^{-z/\lambda}
# $$
# 
# donde $I_0 = \Phi S$. 
# 
# Llamamos **longitud de absorción** a la cantidad $\lambda = 1/(n \sigma)$, e indica a qué distancia la intensidad del haz se ha reducido en un factor $1/e.$

# ## Observables
# 
# 
# En física de partículas habitualmente estimamos (o ponemos límites en su valor) a:
# 
#   * la **sección eficaz**, $\sigma$, de una interacción (en barns, 1 barn $= 10^{-24}$ $\mathrm{cm}^2$). En algunos experimentos se mide la distribución angular de las partículas o **la sección eficaz diferencial**. 
#   
#   * el **tiempo de vida media**, $\tau$, (s), cuando la partícula es de vida "corta", o su inverso, la **anchura de desintegración**, $\Gamma$, (MeV), cuando la partícula es una resonancia.
#     
#   * la **fracción de desintegración**, $\mathcal{Br}$, en un canal (%). El porcentaje de veces que un partícula se desintegra en un canal, por ejemplo el $\tau$ se desintegra a $\mu + \bar{\nu}_\mu + \nu_\tau$ un 17.8% de las veces. 
#   

# ### Sección eficaz
# 
# | | 
# | :--: | 
# |<img src="./imgs/fun_drawing_xsection.png" width = 400 align="center">|
# 
# 
# La sección eficaz cuantifica como un área efectiva, $\mathrm{cm}^{-24}$ o b (barn), la razón de intacciones por segundo en el caso de un experimento de blanco fijo y de un flujo, $\Phi$, sobre un blanco con $N$ elementos o en el caso de un colisionador, dada una luminosidad $\mathcal{L}$ como.
# 
# $$
# \nu = \sigma \Phi  N, \;\;\; \nu = \sigma \, \mathcal{L}
# $$
# 
# donde $\nu$ es el promedio de las interacciones, que siguen una distribución de Poisson.
# 
# En general también estudiaremos la sección eficaz diferencial (en función de la energía y/o del ángulo sólido)

# La siguiente figura muestra la sección eficaz $\sigma(e+e^+ \to \mathrm{hadrons})$ vs $\sqrt{s}$ medida en el experimento OPAL (2001), uno de los 4 experimentos de LEP.
# 
# | | 
# | :--: | 
# | <img src="./imgs/intro_Zreso_OPAL.png" width = 400 align="center">|  
# | $\sigma(e+e^+\to \mathrm{hadrons})$ vs $\sqrt{s}$ (puntos) y predicción del SM (linea)  [OPAL]| de la interacción $\sigma(e+e^+ \to \mathrm{hadrons})$ vs $\sqrt{s}$ en la resonancia del $Z^0$.
# 
# 

# Notas que:
#     
#    * La sección eficaz está en nbarns.
#     
#    * La resonancia del $Z$ sigue una curva de Breit-Wigner, con media $m_z = 91.1852 \pm 0.0030$ GeV y anchura $\Gamma_Z = 2.4948 \pm 0.004$.
#    
# La sección eficaz es proporcional a la curva de Breit-Wigner:
# 
#  $$
#  \sigma(s) \propto \frac{1}{(s-m^2_Z)^2 + m^2_z \Gamma^2_Z}
#  $$
#  
# *cuestión* Estudiar los casos $\sqrt{s} = m_Z, \; \sqrt{s} = m_z \pm \frac{\Gamma_Z}{2}$
#  
# *Nota adicional*: La curva teórica de la figura anterior está corregida por la radiación de fotones en los estados inicial y  final.

# *Cuestión*: Calcula la razón de producción de $Z \to \mathrm{hadrons}$ en el pico de $Z$ en OPAL usando la luminosidad de LEP.

# In[6]:


sigma   = 31 * units.nano # bars
barn_si = 1e-24 # cm2
lumi    = 1e32 * barn_si  # /bar s 
R       = lumi * sigma
print('Razón de producción Z -> hadrons en OPAL = {:4.1f} Hz'.format(R))


# Si queremos determinar una sección eficaz, $\sigma$, en un experimento de colisionador, mediremos el número de interacciones, $N$ que se producen en un tiempo, $T$. Siendo $\mathcal{L}_{\mathrm{int}}$ la luminosidad integrada en el ese tiempo (lo que se conoce como un *run*), 
# 
# $$
# N = \sigma \mathcal{L}_{\mathrm{int}} \Rightarrow \sigma = \frac{N}{\mathcal{L}_{\mathrm{int}}}
# $$
# 
# Si no conocemos $\mathcal{L}_{\mathrm{int}}$, podemos usar el número medido $N'$ de otras interacciones cuya $\sigma'$ sea conocida en el mismo periodo de tiempo: 
# 
# $$
# \sigma = \frac{N}{N'} \sigma'
# $$
# 
# Habitualmente nuestros eventos son seleccionados con una efficiencia (por ejemplo no detectamos todos los eventos que se producen), en este caso $\epsilon, \epsilon'$ son las eficiencias de selección de eventos $N, N'$ respectivamente.
# 
# $$
# \sigma = \frac{N}{N'} \frac{\epsilon'}{\epsilon} \sigma'
# $$

# En un experimento de blanco fiijo, si hemos medido $n$ interacciones en un blanco de $N$ elementos sobre los que incidía un flujo $\Phi$ duante un tiempo $T$, la sección efizaz se estima simplemente como:
# 
# $$
# \sigma = \frac{n}{\Phi N T}
# $$
# 

# ### Anchura de desintegración
# 
# Las desintegraciones se caracterizan por su anchura o por su tiempo de vida media.
# 
# Un sistema de N partículas en un volumen V que se desintegran con una frecuencia fija:
# 
# $$
# \frac{\mathrm{d}N}{\mathrm{d}t} = - \Gamma N; \;\;\;\;
# N(t) = N \, e^{-\Gamma t}
# $$
# 
# A $\Gamma$ la llamamos **anchura de desintegración** y a su inverso, $\tau$ **tiempo de vida media**:
# 
# $$
# \tau = \frac{1}{\Gamma}, \;\; N(t) = N \, e^{-t/\tau}
# $$
# 
# Por ejemplo:
# 
#   * la vida media del $\tau$ es 0.296 ps
#     
#   * La anchura de desintegración del $Z$, $\Gamma_Z \simeq 2.5$ GeV.
# 

# ### Fracción de desintegración
# 
# En el caso de que la partícula se desintegra a varios, $n$, canales, $i = 0, \dots, n$.
# 
# Para cada uno habrá una **anchura parcial de desintegración**, $\Gamma_i$
# 
# La anchura total será la suma de las parciales:
# 
# $$
# \Gamma = \sum_{i=1}^{n} \Gamma_i
# $$
# 
# Y llamaremos fracción o razón de desintegración, $\mathcal{Br}_i$, de un canal al porcentaje de veces que una partícula se desintegra en ese canal, que corresponde a:
# 
# $$
# \mathcal{Br}_i = \frac{\Gamma_i}{\Gamma}
# $$

# 
# Las fracciones de desintegración, $\mathcal{Br}$, del leptón $\tau$ son:
# 
# |      |   |  |
# |:--:  | :--: | :--: |
# | $\tau \to e + \bar{\nu}_e + \nu_\tau$ | $\tau \to \mu + \bar{\nu}_\mu + \nu_\tau$ | $\tau \to \mathrm{hadrons} + \nu_\tau$ | 
# | ------- 17.4% -------- |-------- 17.8% ---------- | ---------- $\sim 64$% ------------|
# 
# La anchura total es la suma de las particales:
# 
# $$
# \Gamma = \Gamma(\tau \to e + \bar{\nu}_e + \nu_\tau) + \Gamma(\tau \to \mu + \bar{\nu}_\mu + \nu_\tau) + \Gamma(\tau \to \mathrm{hadrons} + \nu_\tau) 
# $$
# 
# Y la fracción de desintegración es:
# $$
# \mathcal{Br}(\tau \to \mu + \bar{\nu}_\mu + \nu_\tau) = \frac{\Gamma(\tau \to \mu + \bar{\nu}_\mu + \nu_\tau)}{\Gamma}
# $$
# 

# *Cuestión*: Calcula la anchura de desintegración en MeV del $\tau$
# 
# *Cuestión*: Calcula las anchuras de desintegración parciales del $\tau$.

# In[7]:


hbar = units.hbar/units.eV # eV s
tau   = 0.296 * units.pico  # s
gamma_tau = hbar / tau     # eV
print('Gamma \t = {:e} eV '.format(gamma_tau))
brs = (('e', 14.7), ('mu', 17.8), ('hadrons', 65))
for name, br in brs:
    print('Gamma_{:s} \t = {:e} eV'.format(name, gamma_tau * br * 1e-2))


# Las anchuras de desintegración las medimos directamente de la anchura de la resonancia en el espectro de energías.
# 
# O si la partícula "vuela" lo suficiente para ser detectada podemos determinar el vértice primario y secundario y conocido su momento estimar su tiempo de vida media.

# ## Una perspectiva experimental
# 
# Desde el punto de vista de su detección, podemos distinguir dos rangos principales de partículas :
# 
# * Las partículas *estables*, $p, e, \nu$; o con *vida media "larga"* cargadas, $\mu, \gamma, K, \pi^\pm$, (recorren $\mathcal{O}$(m))
#  
#    * Se detectan como **trayectorias o depositos de energía** en el detector.
#    
#    * La identidad de las partículas puede determinarse por su rango de **penetración, ionización, producción de luz** en los materiales y por sus **desintegraciones**.
#   
# 
# * La identificación de las partículas de *vida media "corta"*, $<\mathcal{O}$(ps), puede realizarse de dos formas: 
#   
#    * Si la partícula "vuela" del orden $\mathcal{O}$(cm), podemos determinar su **vértices de desintegración** con detectores precisos de trazas (ie. $\tau$).
#   
#    * Si la partícula tiene un tiempo de vida media extremadamente pequeño, se desintengra en su lugar de producción, podemos observarla como una **resonancia** en el espectro de energías de la producción o mediante la **masa invariante** de las partículas producidas en su desintegración.

# 
# Las interacciones de las partículas con la materia pueden clasificarse en:
# 
#    * interacciones de las partículas cargadas ($e$, $\mu$, $p$, $K^\pm$, $\pi^\pm$)
#    
#    * interacciones electromagnéticas de electrones y fotones ($e$, $\gamma$)
#    
#    * interacciones fuertes de hadrones ($\pi, K, ...$).

# #### Las partículas cargadas
# 
# Pueden excitar los átomos o moléculas o ionizar (separar los electrones)
# 
# La mayor parte de los detectores de trayectorias usan la ionización del medio tras el paso de las partículas cargadas.

# La pérdida de energía $\frac{\mathrm{d}E}{\mathrm{dx}}$ por distancia debido a la ionización sigue la ecuación de Bethe-Bloch (1930s):
# 
# $$
# \frac{1}{\rho}\frac{\mathrm{d}E}{\mathrm{d}x} \simeq - K \frac{Z}{A \beta^2}\left[ \ln \left( \frac{2  m_e c^2 \beta^2 \gamma^2}{I_e}\right) - \beta^2\right]
# $$
# 
# Donde $K = \frac{4\pi \hbar^2 \alpha^2 N_A}{m_e} = 0.307$ MeV $\mathrm{cm}^2$/mol, $\alpha$ es la constante de estructura fina, $m_e$ la masa del electrón, $N_A$ el número de Avogadro, $I_e$ un potencial de ionización que depende del material $(A, Z)$, $\rho$ la densidad, y $\gamma, \beta$ los parámetros de Lorentz asociados a la velocidad $\beta$ de la partícula.

# La pérdida de energía tiene una curva general en función principalmente de $\gamma \beta$ y del material
# 
# | |
# |:--|
# |<img src="./imgs/det_eneloss_mip.jpg" width = 400 align = "center">|
# |pérdida de energía por ionización [PDG]|
# 
#   

# El recorrido medio de una partícula con $\beta$ hasta que se detiene en un medio se denomina **rango** de penetración.
# 
# La pérdida de la energía depende de la velocidad de la partícula $\beta$. Distinguimos tres regiones:
#  * La pérdida es más intensa para baja $\beta$ (de la dependencia $1/\beta^2$). La inonización es mayor al final de la trayectoria, o **región de Bragg**.
#  * En el rango de $\beta\gamma$ de $1-10$, esa región se denomina **mip** (*minimum ionizing particle*).
#      Un muón de 10 GeV en hierro pierde en promedio 13 MeV/cm y su rango es de varios metros.
#  * Para $\beta \gamma >100$, la perdida aumenta de forma logaritmica. A partir de aquí los efectos de radiación son relevantes.

# #### Detectores gaseosos
# 
# En los detectores gaseosos se utiliza la ionización ($\sim 30$ eV por ionización) del paso de las partículas cargadas en gases nobles, por ejemplo Ar, Xe. 
# 
# Los detectores están, en general, en un rango de voltage proporcional (no hay efecto avalancha). 
# 
# El paso de la partícula deja un regero de electrones de ionización.
# 
# Esos electrones derivan bajo la presencia de un campo eléctrico ${\bf E}$ hasta el ánodo donde se recoge la carga.
# 
# Existen varios tipos de detectores: *wire chambers*, *multiproportial wire chambers (MPWC)*, *time proyection chambers (TPC)*.

# ##### Cámaras de proyección temporal (TPC)
# 
# Las TPC suelen tener forma de barril, pueden ser simétricas, con dos tapas como ánodos y un cátodo central de HV.
# 
# Bajo la presencia del campo eléctrico, los $e$ de ionización derivan hasta el ánodo.
# 
# Son recogidos por un detector segmentado (hilos) en dos direcciones ($r, \phi$) o ($x, y$), donde se amplifica su carga (electrónica frontal).
# 
# El tiempo de llegada de los electrones al ánodo, $\Delta t$, sirve para estimar la posición en $z$ (el eje de la cámara).  

# Las TPCs son excelentes detectores de trazas
# 
# | | |
# |-- | -- |
# | <img src="./imgs/det_ALICE_TPC_scheme.png" width = 400 align = "center">| <img src="./imgs/det_ALICE_ppevent.jpeg" width = 400 align = "center"> |
# | [ALICE TPC](https://alice-collaboration.web.cern.ch/menu_proj_items/tpc) | ALICE pp event [ALICE]|
# 
# Las TPC se utilizan también en experimentos de búsqueda de materia oscura, XENON, detectores de neutrinos EXO, NEXT.

# La fórmula calcula el valor promedio. La distribución $\mathrm{d}E/\mathrm{d}x$ está relacionada con la fluctuación del número de colisiones de la partícula con los electrones de los átomos.
# 
# | |
# |:--|
# |<img src="./imgs/det_eloss_alice_tpc.png" width = 400 align = "center">|
# | pérdida de energía en la TPC de ALICE [ALICE]|
# 
# Pérdida de energía en la TPC (*Time Projection Chamber*) de ALICE.
# 
# Los electrones pierden tambien energía por radiación **bremsstrahlung** (en la siguiente sección).

# En los detectores de trazas se detecta la ionización (electrones liberados) del paso de la partícula cargada a través del medio para determinar puntos de paso o *hits*. 
# 
# Los detectores están inmerso en un campo magnético, ${\bf B}$, (Teslas) que produce una curvatura de la partícula proporcional a su momento, ${\bf p}$ (GeV)  en la dirección perperdicular a ${\bf B}$. Si entre ambos tiene un ángulo $\theta$:
# 
# $$
# p_T \equiv p \, \sin \theta = 0.3 \, B \, \rho,
# $$
# 
# donde $\rho$ (m) es el radio de curvatura en el plano perperdicular, $B$ (T) el campo magnético.
# 
# Llamamos al momento en el plano perperdicular, **momento transverso**, $p_T = p \sin \theta$.
# 
# Para CMS con $B = 4$ T, y un $\pi^\pm$ de $p$ 100 GeV, $\rho \sim 100$ m.  

# ##### Detectores de silicio
# 
# Están basando en vaciar de portadores libres de carga una oblea de silico (aprox $~300$ $\mu$m de esperor) donde se ha dopado tiras (*strips*) de tipo $p$, separadas aprox $~50$ $\mu\mathrm{m}$, para crear uniones $pn$.
# 
# | |
# | :-- |
# <img src="./imgs/det_silicon_detector.png" width = 400 align = "center">
# |Esquema de un sensor de micro-strips de silicio|
# 
# La ionización del paso de una partícula cargada crea pares electrón/hueco ($\mathcal{0}(1)$ eV).
# 
# Los electrones de ionización derivan hacia las tiras $p$, donde su carga es amplificada por la electrónica (*front end electronics*).

# Con estos detectores se pueden reconstruir las trayectorias con precisión $\sim 10$ $\mu\mathrm{m}$ e identificar vértices de desintegración de partículas que pueden recorrer $\sim  1$ cm.
# 
# | | |
# |-- | -- |
# | <img src="./imgs/det_DELPHI_vertex.jpeg" width = 350 align = "center">| <img src="./imgs/det_DELPHI_bdecay.gif" width = 350 align = "center"> |
# |DELPHI micro-vertex detector| DELPHI Event con b-tag [DELPHI]|
# 
# ATLAS y [CMS](https://cms.cern/detector/identifying-tracks/silicon-strips) utilizan detectores de silicio, que se desarrollaron a partir de 80's. Estos detectores fueron esenciales en los experimentos de LEP, BaBar, Belle entre otros y FERMI, en astropartículas.

# #### Interacciones electromagnéticas
# 
# #### Interacciones de los electrones
# 
# Las partículas cargadas pueden radiar fotones por la interacción electromagnética con los protones de los núcleos. Esta radiación se llama **bremsstrahlung**.
# 
# $$e^- + (A, Z) \to e^- + \gamma + (A, Z)$$ 
# 
# Esta radiación empieza a ser dominante a partir de una **energía crítica** $E_c \sim 800/Z$ MeV, antes domina la ionización.
# 
# Este proceso se puede calcular en QED (*Quantum Electro-Dynamics*) y su sección eficaz: 
# 
# $$\sigma_{b} \propto E/m^2.$$
# 
# Afecta más a los electrones que a los muones por un factor $(m_e/m_\mu)^2$. 
# 
# Los muones por debajo de $\mathcal{O}(100)$ GeV pierden energía principalmente por ionización. 

# La pérdida de energía por bremsstralung por encima de una energía umbral, $E_c$, puede expresarse:
# 
# $$
# \frac{\mathrm{d}E}{\mathrm{d}x} = - \frac{E}{X_0}, \;\; E(x) = E_0 \, e^{-x/X_0},
# $$
# donde $X_0$ se denomina **longitud de radiación** y $E_0$ es la energía inicial del electrón.
# 
# | |
# | :-- |
# |<img src="./imgs/det_AB1.1_electron_xsec_Pb.png" width = 400 align = "center">|
# |energía perdida para electrones|
# 
# 
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
# | <img src="./imgs/det_AB1.1_gamma_xsec_Pb.png" width = 340 align = "center">|
# | sección eficaz de fotones en plomo [AB1.1]|
# 
# A altas energías el efecto dominante será la producción de pares.
# 
# $$
# \sigma_\gamma \simeq \frac{7}{9} \frac{1}{n X_0},
# $$
# donde $n$ es la densidad de núcleos.
# 
# La cantidad $\lambda = 1/(n \sigma_\gamma)$ es el **camino libre medio**, que vale $\lambda \simeq 7/9 X_0$ y que nos indica la cantidad de fotones que se pierden en un haz monoenergético de intensidad $I$:
# 
# $$
# \frac{\mathrm{d}I}{\mathrm{dx}} = - \frac{I}{\lambda}, \;\; I(x) = I_0 e^{-x/\lambda}
# $$
# 
# Por lo tanto la longitud de radiación, $X_0$ caracteriza la pérdida en un médio de energía de electrones y fotones por encima $\sim 10$ MeV.
# 

# #### Calorímetros electromagnéticos
# 
# $$
# $$
# 
# | | |
# | :--: | :--: |
# | <img src="./imgs/det_gamma_cascade.png" width = 300 align = "center">| <img src="./imgs/det_gamma_cascade_real.png" width = 300 align = "center"> |
# | Esquema de cascada [AB]| Cascada [AB]| 
# 
# 

# Los **calorímetros electromagnéticos** tienen una estructura alternada de material pasivo (por ejemplo Pb, alto $Z$, alto $X_0$), donde se desarrollan las cascadas, y  material activo, donde se detecteca la ionización (i.e centelleadores).
# 
# La energía depositada es proporcional a la energía incidente del $e$ o $\gamma$ y la resolución en energía, $\sigma_E$, está limitada por las fluctuaciones en la producción de las partículas (que es proporcional a $\sqrt{E}$) en la cascada):
# 
# $$
# \frac{\sigma_E}{E} \sim \frac{3 - 10 \mathrm{\%}}{\sqrt{E \;\mathrm{(GeV)}}}
# $$
# 
# Los **centelleadores** contienen moléculas que se excitan al paso de la partícular cargadas, y al de-excitarse emiten luz en el visible que puede detectarse con sensores de fotones (por ejemplo **foto-multiplicadores**).  
# 
# El número de fotones es proporcional a la energía absorbida, aproximadamente 100 eV por fotón de centelleo. 
# 
# Los centelleadores deben tener una $X_0$ alta para evitar la conversión de los fotones.

# #### PMT
# 
# Los photomultiplicadores son sensores que detectan fotones via efecto foto-electrico, arrancan un electrón de la primera capa del sensor y esté genera un avalancha en las siguientes capas.
# 
# 
# | |
# | -- |
# | <img src="./imgs/det_PMT_drawing.png" width = 550 align = "center">|
# | Esquema de funcionamiento de un PMT [WK]|
# 
# 
# Se utilizan como sensores que recogen la luz de cristales, plásticos centelleadores, etc. Se utilizan en calorímetros y en cámaras de muones, entre otros. 

# ### Experimentos
# 
# #### Experimentos $4\pi$
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
# Imágenes de [CMS](https://home.cern/resources/image/experiments/cms-images-gallery)
# 

# #### Experimentos de blanco fijo o hacia delante
# 
# Solo tiene instrumentalizados un brazo de la interacción.
# 
# Están constituidos en general por la zona de interacción, de detectores de vértices, seguidos de trazas (en un campo magnético), por los calorímetros electromagnético, hadrónico y finalmente por detectores de muones.
# 
# Es el caso de detector LHCb en el LHC.
# 
# |  |
# |:-- |
# | <img src="./imgs/det_LHCb_subdetectors.jpg" width = 500 align = "center"> |
# | Esquema de un sector del *LHCb* ([LHCb](https://lhcb.cern/detector)) del LHC|
# 
# Imágenes de [LHCb](https://home.cern/resources/image/experiments/lhcb-images-gallery)
# 

# #### Procesado y análisis de datos
# 
# Sistema de disparo (*trigger*) y de adquisición de datos
# 
# Es preciso reducir el rate de producción (i.e 40 MHz) y seleccionar los sucesos relevantes ($\mathcal{0}(10)$ k)
# 
# Puedes ver imágenes de la sala de control de [LHCb](https://cds.cern.ch/record/1299577) durante la toma de datos (*online*).
# 
# Procesado y reconstrucción
# 
# Los eventos se procesan para a partir de las señales de los sensores, primero se reconstruyen trazas, vértices y depósitos de energía y luego partículas y cuadrimomentos (*offline*). 
# 
# Para ello se utilizan programas en C++ o Python.
# 
# También se producen millones de eventos simulados en centros de computación, mediante técnicas de Monte-Carlo.

# #### Análisis
# 
# Los eventos de una toma de datos se seleccionan primero, y luego se analizan con técnicas de análisis de datos (i.e *Neural Networks*) donde la estadística juega un papel esencial. En general  se utilizan como control datos de calibración y simulados (i.e con técnicas de Monte-Carlo).
# 
# Las medidades de los experimentos se presentan como:
# 
# * La **estimación** de un observable, por ejemplo: una sección eficaz, la vida media, la masa o una fracción de desintegración. 
# 
#    Existen dos tipos de errores:
# 
#    * *estadísticos*: dependiendo de la cantidad de sucesos relevantes disponibles.
# 
#    * *sistemáticos*: que reflejan nuestras incertidumbres en parámetros que afectan al observable, que puede ser de diversas índole: de calibración, eficiencias de selección, teóricos, etc.
# 
#    Por ejemplo [la observación del Higgs en ATLAS](https://arxiv.org/abs/1207.7214), 
#    
# 
# * Un **límite** en el observable como la vida media, masa, sección eficaz, etc, de una **búsqueda**.
#   
#    La determinación de un límite se presenta con un nivel de confianza (i.e 90%).
# 
#    Por ejemplo, el límite en [la masa del neutrino en KATRIN](https://arxiv.org/abs/1909.06048)
# 
# 

# ### Breve introducción a los aceleradores
# 
# 
# En un acelerados las partículas estables ($e, e^+ p, \bar{p}$) se agrupan en paquetes (*bunches*), con un número elevados de partículas, $\mathcal{O}(10^{11})$, que circulan en un tubo de vacío.
# 
# Se les acelera mediante gradientes de potencial en cavidades resonantes, se les mantiene en órbita mediante campos magnéticos dipolares y cuadripolares.
# 
# Existen dos tipos de aceleradores principales:
# 
# * **Lineales**, donde el acelerador es un dispositivo recto que los paquetes recorren una vez.
# 
# * **Circulares**, colisonadores, en los los paquetes giran en el tubo de vacío del haz numerosas vueltas $\mathcal{O}(10^5)$.
# 

# ##### Acelerador lineal
# 
# Animación de un detector lineal y vista del acelerador lineal ($e, e^+$) en SLAC (California) de 3 km de largo.
# 
# | | |
# | :--: | :--: | 
# | <img src="./imgs/Linear_accelerator_animation_16frames_1.6sec.gif" width = 400 align="center"> | <img src="./imgs/det_slac_aerial.png" width = 200 align="center">|
# | Acelerador lineal [Wikipedia]| Linar acelarator (SLAC) |
# 

# ##### Aceleradores circulares
# 
# Vista aérea del LHC donde se sobreimponen las líneas de los túneles de los aceleradores del CERN.
# 
# | | |
# |:-- | :-- |
# | <img src="./imgs/det_CERN_accelerators_complex.png" width = 300 align="center"> | <img src="./imgs/det_LHC_aerial.png" width = 320 align="center"> | 
# | Esquema de los aceleradores del CERN [[>]](https://www.youtube.com/watch?v=pQhbhpU9Wrg)|  vista del LHC (CERN) | 
# 

# 
# Las partículas relativistas al girar pierden energía por radiación (**synchrotron radiation**) proporcional a $1/m^4$ ¡Este efecto es más dramático para $e$ que para $p$!
# 
# Para 'girar' a las partículas se utiliza un campo magnético dipolar $B$ perperdicular a su dirección. 
# 
# Por el electromagnetismo sabemos que $p = 0.3 B \, \rho$, donde $p$ es el momento, en TeV, $B$ el campo magnético, en T, y $\rho$ el radio, en km.
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
# Los dipolos superconductores del LHC operan a 8 T. Los paquetes contienen del orden de $10^{11}$ $p$ que recorren el anillo a una frecuencia de 40 MHz. 
# 
# El volumen de los paquetes es de 40 cm de largo y 1 mm de sección que se reduce a 10 $\mu\mathrm{m}$ en la zona de colisión.

# #### Haces secundarios de hadrones, $\mu, \, \nu_\mu$.
# 
# | | |
# |:--: | :--:  
# | <img src="./imgs/det_nubeam_schematic.jpg" width = 340 align="center"> | <img src="./imgs/det_nubeam_DUNE.jpg" width = 340 align="center"> | 
# | Esquema del haz de neutrinos de FermiLab | Haz de $\nu$ de FermiLab a DUNE (en construcción)|
# 
# Se obtienen de golpear un haz primario (principalmente $p$) contra un blanco.
# 
# Se seleccionan los partículas de interés mediante un conjunto de imanes tuneados para permitir el paso de partículas con una carga y un momento determinados.
# 
# Por ejemplo, los $\pi^\pm$ se desintegraran a un haz e muones $\mu^\pm$ y netrinos $\bar{\nu}_\mu, \nu_\mu$.
# 

# ## Una perspectiva teórica
# 
# La teoría de Física de Partículas se basa en la **Teoría Cuántica de Campos**.
# 
# que podemos caracterizar por varios elementos fundamentales:
# 
# 1. Toda tipo de partícula tiene asociada un **campo**.  Existen tres tipos de campos principales:
# 
# | | | |
# |:--: | :--: | :--: |
# | Bosón Escalar (S = 0 ), Klein-Gordon |  Espinor (S = 1/2), Dirac | Bosón vectorial (S = 1), Maxwell |
# | $(\partial^\mu\partial_\mu + m^2) \, \phi = 0$ | $(i\gamma^\mu \partial_\mu - m) \, \Psi = 0$ | $\partial_\mu (\partial^\mu A^\nu - \partial^\nu A^\mu) = j^\nu$ |
# 
# 
# 2. **Los campos están cuanficados**, se construyen con operadores creacción y destrucción de partículas a partir del vacío. ¡Las partículas son "excitaciones" de los campos! 
# 
# para el campo fermiónico:
# 
# $$
# a^{\dagger}_{p, s} \, |0 \rangle = |p, s \rangle, \;\;\; \{a^{\dagger}_{p,s}, a^{\dagger}_{p', s'} \} = (2\pi)^3 \, \delta^{(3)}({\bf p}- {\bf p}') \, \delta_{ss'}
# $$
# 
# 3. La interacción entre fuerzas y materia (por ejemplo electrón y fotón) se establece a partir de la conservación de la simetría llamada **simetría gauge local** que consiste en que la física no cambia si cambiamos en cada punto ($x$, local) el campo fermiónico por una fase $\theta(x)$,  $\Psi'(x) = e^{ig\theta(x)} \Psi(x)$.

# 
# Veremos más adelante que el *Modelo Estándar* necesita de dos elementos más:
# 
# 1.  **La teoría de grupos**. El electromagnetismo está asociado al grupo que se llama $U(1)$ o singletes, la interacción débil al $SU(2)$ o dupletes, y la fuerte al SU(3) o tripletes.
# 
# 3.  *Un campo escalar*, esto es de spín nulo, el campo del **bosón de Higgs**, que interacciona con el resto de partículas como si fuera un campo de inercia, y que además tiene un potencial singular, con la forma de un sombrero mexicano, y que el valor del campo en el vacío toma de forma espontánea un valor de entre todos los posibles equivalentes, podríamos decir que al azar un punto dentro del ala del sombrero, lo que se conoce como **rotura espontánea de simetría**.

# ### Sobre la ecuación de Dirac
# 
# Dirac propuso en 1932 una ecuación, que lleva su nombre, que describe la dinámica relativista de los fermiones.
# 
# $$
#  (i \gamma^\mu \partial_\mu - m) \, \Psi(x) = 0
# $$
# 
# Las $\gamma$'s (veremos en breve) son matrices complejas $4\times 4$ y $\partial_\mu$:
# 
# $$\partial_\mu = \left( \frac{\partial}{\partial t}, \frac{\partial}{\partial x},  
#                 \frac{\partial}{\partial y}, \frac{\partial}{\partial z}  \right).
# $$
# 
# La contracción de índices significa un sumatorio, esto es $\gamma^\mu\partial_\mu \equiv \sum_{\mu = 0}^4 \gamma^\mu \partial_\mu$ (en gereral a partir de ahora será así).
# 
# Dirac usó dos directrices para escribir su ecuación:
#  1. La coordenada espacial y temporal,  $\partial_0, \partial_i, \;\; i = 1, 2, 3$, aparecen linealmente y en igualdad de condiciones.
#  2. Considerando la asociación con los operadores energía y momento, $i\partial_0 \to E, \;\;\; -i\partial_i = p_i$, al elevar al cuadrado la ecuación obtenemos la ecuación de Einstein, $E^2-{\bf p}^2 = m^2$ 
# 
# 

# Pero para ello, los factores $\gamma^\mu$ no pueden conmutar, no pueden por lo tanto ser escalares, deben ser matrices complejas $4\times4$ que cumplan el álgebra siguiente:
# 
# $$
# \{ \gamma^\mu, \gamma^\nu \} \equiv \gamma^\mu \gamma^\nu + \gamma^\nu \gamma^\mu = 2 g^{\mu\nu}
# $$
# 
# conocida como de ágebra de Clifford. 
# 
# De forma equivalente, las matrices-$\gamma$ cumplen:
# 
# $$
# (\gamma^0)^2 = I, \; (\gamma^k)^2 = - I, \;\; \gamma^\mu \gamma^\nu = - \gamma^\nu \gamma^\mu \, (\mu \neq \nu),
# $$
# con $k=1, 2, 3$, y $\mu = 0, 1, 2, 3$.
# 

# Existen varias representaciones del álgebra y podemos usar la más conveniente. la Física no cambia, no depende, de la representación elegida.
# 
# La representación más común es la de Pauli-Dirac:
# 
# $$
# \gamma^0 = \begin{pmatrix} I & 0 \\ 0 & -I \end{pmatrix}, \;\;
# \gamma^k = \begin{pmatrix} 0 & \sigma_k \\ -\sigma_k & 0 \end{pmatrix}, \;\;
# \gamma^5 = \begin{pmatrix} 0 & I \\ I & 0 \end{pmatrix},
# $$
# 
# donde $I$ es la matriz identidad $2\times2$ y $\sigma_k$ la matrices de Pauli $2\times2$.
# 
# La quinta matrix se define $\gamma^5 \equiv i \gamma^0\gamma^1\gamma^2\gamma^3$ y como veremos juega un papel fundamental en la interacción débil.
# 

# 
# La solución de la ec. de Dirac es una función de ondas con cuatro componentes complejas, el espinor de Dirac:
# 
# $$
# \Psi = \begin{pmatrix} \psi_1 \\ \psi_2 \\ \psi_3 \\ \psi_4\end{pmatrix}.
# $$
# 
# 
# Las soluciones generales son cuatro ondas planas, que asociamos a los fermiones, $\Psi_i(x)$ y anti-fermiones $\Phi_i(x)$:
# 
# $$
# \Psi_i(x) = u_i(E, {\bf p}) \, e^{+i({\bf p x} - Et)}, \;\;\;\; \Phi_i(x) = v_i(E, {\bf p}) \, e^{-i({\bf p x} - Et)}
# $$
# 
# donde $u_i(E, {\bf p}), v_i(E, {\bf p})$ corresponden a los **espinores** (cuatro componentes en columna) de los fermiones y antifermiones respectivamente y el índice $i=1, 2$ está asociado con las dos posibles componentes de espín de $s=1/2$.

# A falta de un eje definido o un campo magnético, podemos tomar de forma natural el eje $z$ aquel en el que se mueve la partícula con momento $p$. En ese caso los espinores son:
# 
# $$
# u_{-} = N \begin{pmatrix} 0     \\ 1       \\ 0       \\ -\kappa \end{pmatrix}, \;
# u_{+} = N \begin{pmatrix} 1    \\  0      \\ \kappa  \\  0      \end{pmatrix}, \;
# v_{-} = N \begin{pmatrix} \kappa \\ 0       \\ 1      \\ 0        \end{pmatrix} \;,
# v_{+} = N \begin{pmatrix} 0     \\ -\kappa \\ 0       \\ 1       \end{pmatrix},
# $$
# 
# donde $\kappa = \frac{p}{E+m}$ y $N = \sqrt{E+m}$, un factor de normalización.
# 
# Son los espinores del fermión, $u_{\pm}$, con su espín alineado (o opuesto  a su momento, esto es tiene helicidad positiva $+$ o negativa $-$, y los del antifermión, $v_{\pm}$, con sus dos helicidades. 
# 
# 

# Gráficamente:
# 
# | |
# |:--:|
# |<img src="./imgs/dirac_spinors_helicities.png" width = 450 align="center">|
# | spín y momento de los spinores de helicidad|
# 
# Notar que $\kappa$ es un factor que cuantifica cuán *relativista* es la partícula. Si la partícula está en reposo $p = 0 \to \kappa = 0$, y si es ultra-relativista $E \simeq p \to \kappa = 1$. 
# 
# Recordemos que la **helicidad** es la proyección del espín en la dirección del momento, ${\bf p}$.
# 

# 
# Definimos el espinor **adjunto** como:
# 
# $$
# \bar{\Psi} = \Psi^{\dagger} \gamma^0; \;\;\; \bar{\Psi} = (\psi^*_1, \psi^*_2, -\psi^*_3, -\psi^*_4),
# $$
# 
# esté último en la representación Pauli-Dirac.
# 
# A partir de ahí podemos definir una **corriente**:
# 
# $$
# j^\mu =  \bar{\Psi} \gamma^\mu \Psi,\;\;\;  j^\mu = (j^0, {\bf j})
# $$
# 
# Que se comporta como un **vector** bajo transformaciones de Lorentz, esto es, es un **cuadri-vector**.
# 
# Esta corriente se conserva, cumple la **ecuación de conservación**:
# 
# $$
# \frac{\partial \rho}{\partial t} + \nabla \cdot {\bf j} = 0,
# $$
# 
# esto es, la cantidad de $\rho$ que entra en un volumen diferencial en una unidad de tiempo corresponde al flujo de la corriente ${\bf j}$ que atraviesa las paredes del volumen en ese tiempo.
# 
# Es este caso es la **corriente de probabilidad** del fermión.

# ### Diagramas de Feyman
# 
# R. Feynman desarrolló en los 50's la teoría QED (*Quantum Electro Dynamics)* basada en la TCQ (Teoría Cuántica de Campos).
# 
# A partir de ideas previas en física teórica:
# 
#   * las interacciones tienen lugar entre **dos corrientes de fermiones** (Fermi )
# 
#   * la fuerza se transmite mediante un portador o **mediador** (Yukawa).
#     
#   * **la regla de Oro de Fermi**. La frecuencia de la transición, $R$, es proporcional a $|M_{fi}|^2$ y a la densidad de estados disponibles $\rho(E)$.
#    
# $$
# R = (2\pi) |M_{fi}|^2 \rho(E)
# $$
#   
# Recordemos que $M_{fi} = \langle f | H_{int}| i \rangle$ es el elemento de la matriz de transición entre los estados iniciales, $| i \rangle$ y finales, $\langle f|$, mediante un hamiltoniano de interacción $H_{int}$.
#   
# 

# La parte cinématica puede resolverse "facilmente".
# 
# Los dos casos comunes (ver [Apéndice-fundamentos]) en el CM se muestran en el tabla:
#   
#   * anchura de desintegración de la partícula $a \to b + c$
#  
#   * sección eficaz de la interación de dos cuerpos $a +b \to c + d$
# 
# | | |
# | :--: | :--: |
# | $\Gamma(a \to b + c)$ | $\sigma(a+b \to c+d)$ | 
# | <img src="./imgs/intro_drawing_decay2.png" width = 250 align="center">| <img src="./imgs/intro_drawing_int2.png" width = 250 align="center">   |

# La parte cinemática en el cálculo de la anchura de desintegración, $\Gamma(a \to b + c)$
# 
# $$
# \Gamma = \frac{p^*}{32 \pi^2 m^2_a} \int_{\Omega} |M_{fi}|^2 \mathrm{d}\Omega^*
# $$
# 
# Y en el cánculo de la sección eficaz $\sigma(a + b \to c + d)$:
# 
# $$
# \sigma = \frac{1}{64 \pi^2 (E_a+E_b)^2} \frac{p^*_f}{p^*_i} \int_\Omega |M_{fi}|^2 \, \mathrm{d}\Omega^*
# $$ 
# 
# donde el índice "*" corresponde a las cantidades en el **centro de masas**.
# 
# La física de la interacción se encuentra en el término $M_{fi}$
# 

# Feynman propuso una serie de diagramas que tienen asociados unas reglas que permiten calcular $M_{fi}$.
# 
# Los **diagramas de Feynman** nos permiten además representar **gráficamente** una interacción.
# 
# y son válidos para **todas las fuerzas**.

# #### Teoría de Fermi
# 
# Recordemos que Fermi propuso una teoría para le desintegración $\beta$, 
# 
# $$
# n \to p + e + \bar{\nu}_e
# $$ 
# 
# que se basaba en acoplo **puntual** entre **dos corrientes fermiónicas** con una intensidad $G_F/\sqrt{2}$.
# 
# | | 
# | :--: | 
# | <img src="./imgs/intro_fermi_currents.png" width = 300 align="center">|  
# | Representación gráfica de la teoría de Fermi del acoplo entre dos corrientes|
# 
# 
# 

# Y el elemento $M_{fi}$ podía calcularse a partir de:
# 
# $$
# M_{fi} = \frac{G_F}{\sqrt{2}} \; \left[\bar{\Psi}_p \gamma^\mu \Psi_n \right] g_{\mu\nu} \left[ \bar{\Psi}_e \gamma^\nu \Psi_{\bar{\nu}_e} \right],
# $$
# 
# donde:
# 
#   * $G_F = 1.16 \, 10^{-5}$ $\mathrm{GeV}^{-2}$, es la constante de acoplo, o de Fermi, que cuantifica la intensidad de la interacción.
#   
#   * $j^\mu_{\mathrm{had}} = \bar{\Psi}_p \gamma^\mu \Psi_n$ es la corriente hadrónica.
#   
#   * $j^\nu_{\mathrm{lep}} = \bar{\Psi}_e \gamma^\nu \Psi_{\bar{\nu}_e}$ es la corriente leptónica.
# 
# 
# Como $j^\mu_{\mathrm{had}}, \, j^\nu_{\mathrm{lep}}$ son corrientes, cuadri-vectores, $M_{fi} = g_{\mu\nu} j^\mu_{\mathrm{had}} j^\nu_{\mathrm{lep}}$ es un invariante Lorentz.
# 
# Recordemos que $\gamma^\mu$ son las matrices de Dirac, $g_{\mu\nu}$ el tensor diagonal de la métrica, y cuando aparecen los índices arriba y abajo, se entiende que hay implícito un sumatorio.

# #### diagrama de Feynman
# 
# El siguiente **diagrama de Feyman** representa la interacción de dos corrientes fermiónicas
# 
# | |
# |:--:|
# |<img src="./imgs/feynman_default.png" width = 300 align="center">|
# | Diagrama de Feyman de una dispersión|
# 
# Identificamos:
# 
#    * las **corrientes fermiónicas** de las partículas, de $a$ a $c$, y de $b$ a $d$.
#    
#    * el **mediador** o portador, $X$
#    
#    * los **vértices de acoplo** con intensidad $g$.

# En más detalle:   
#    
#    * las **corrientes fermiónicas** estan representadas por líneas. Cada fermión es un segmento de la línea con una flecha asociada. Los fermiones tienen las flechas hacia la derecha y los antifermiones hacia la izquierda.
#    
#    * El **mediador** se representa con una línea en senos para el fotón, y los bosones débiles $W^\pm, Z$ y con un muelle para los gluones que median la fuerza fuerte. El mediador introduce un factor en el elemento de matriz denominado **propagador**.
#    
#    * La intensidad y características de la interacción está cuantificada por un factor asociado al **vértice**. La **intensidad de la interacción** viene dada por una **"constante" de acoplo** $g$. 
# 
# *Nota*: Es común en la literatura que la **línea temporal** corresponda **hacia la derecha**. Así los fermiones avanzan en el tiempo y los antifermiones en sentido inverso. Los mediadores se representan verticalmente, para indicar que no sabemos en qué dirección temporal se intercambió el mediador. 

# #### Vértices
# 
# Los vértices de los diagramas de Feynman para las distintas fuerzas son:
# 
# | |
# | :--: | 
# | <img src="./imgs/feynman_forces.png" width = 500 align="center">|
# 

# Para cada fuerza tenemos asociada una **constante de acoplo**:
#   
# |  |  |  | |
# | :--: | :--: | :--: | :--: |
# | electromagnética| débil cargada | débil neutra| fuerte |
# | $e $ | $g_W$ | $g_Z$ | $g_s$ | 
#   
# La constante asociada al electromagnetismo es la carga eléctrica, $e$.
#  
# La constante nos indica la **intensidad** de la fuerza 
#  

# Resulta conveniente utilizar **constantes adimensionales**, que denotamos $\alpha$, para cuantificar la intensidad de las fuerzas entre ellas.
# 
# Para el electromagnetimo utilizamos $\alpha$, la constante de estructura fina:
# $$
# \alpha = \frac{e^2}{4 \pi \epsilon_0}
# $$
# 
# Como en cada diagrama hay dos vértices, $M_{fi}$ es proporcional a $g^2 \propto \alpha$, y la propabilidad $|M_{fi}|^2 \propto g^4 \propto \alpha^2 $
# 
# Las constantes adimensionales de las fuerzas serían aproximadamente:
# 
# | electromagnetismo $\alpha$| débil $\alpha_W$| fuerte $\alpha_S$|
# | :--:  | :--: | :--: | 
# | 1/137 |  1/30 | 1 |
# 
# 

# En los vértices se conserva: 
#   
#    * la **carga eléctrica**. En el caso de $W^\pm$ el portador transfiere la carga.
#       
#    * el **sabor**, la identidad de las partículas, a excepción de $W^\pm$ donde cambia $f \to f'$
#   
#    * el **cuadrimomento**. Sea la partícula $a$ entrante y $c$ saliente de la corriente, el cuadrimomento transferido es $q = p_c-p_a$.
#      
#    * el número leptónico y leptónico de sabor, el número de quarks/antiquarks.
#    
#    * el número total de fermiones menos antifermiones.
# 

# #### Portadores
# 
# Los portadores transmiten la fuerza, y se les asociada un **propagador** que actua como un factor entre las corrientes.
# 
# En la tabla se muestra la expresión aproximada de los propagadores: 
# 
# |  |  |  | |
# | :--: | :--: | :--: | :--: |
# | electromagnética | débil cargada | débil neutra| fuerte --- |
# | $\frac{g_{\mu\nu}}{q^2}$ | $\frac{g_{\mu\nu}}{q^2 - m^2_W}$ | $\frac{g_{\mu\nu}}{q^2 - m^2_Z}$ | $\frac{g_{\mu\nu}}{q^2}$ | 
# 
# donde $q^2$ es el cuadrimomento transferido, en el caso del diagrama anterior, $q = p_c - p_a = p_d - p_c$, y $m_W = 80.4, m_z = 91.2$ GeV son las masas del $W$ y del $Z$ respectivamente. Notar que el tensor de la métrica $g_{\mu\nu}$ une las dos corrientes.

# 
# Podemos interpretar $q^2$ como la "masa efectiva" al cuadrado del mediador y éste actua con un factor $1/q^2$ en $M_{fi}$.
# 
# En la desintegración $\beta$ en un núcleo, $q^2 (\, \sim \mathcal{0}(MeV^2)) \ll \,  M^2_W \, (80 GeV)^2$, por lo tanto
# 
# $$
# \frac{g_{\mu\nu}}{q^2 - M^2_W} \to -\frac{g_{\mu\nu}}{M^2_W}.
# $$
# 
# ¡De ahí que Fermi asociara al vértice una constante!
# 

# ### Tipos de interacciones
# 
# Existen tres tipos de diagramas principales: **dispersión**, **aniquilación** y **desintegración**
# 
# | |
# | :--: |
# |<img src="./imgs/feynman_new_type_interactions.png" width = 450 align="center">|
# |De izda a derecha diagramas de dispersión,  aniquilación y desintegración|
# 
# El cuadrimomento transferido $q$ es:
# 
# |  |  |  |
# | :--: | :--: | :--: | 
# | --- dispersión ---| -- aniquilación --  | desintegración --|
# | $q = p_c - p_a$ | $q = p_a + p_b$ | $q = p_c - p_a$ | 
# 
# 

# El diagrama de Feynman para la desintegración $\beta$ de un neutrón es:
# 
# | |
# | :--: |
# |<img src="./imgs/feynman_new_beta_decay.png" width = 250 align="center">|
# |Diagrama de Feynman de la desintegración $\beta$|
# 
# notar que los quarks ($ud$) son meros "espectadores".
# 
# La relación exacta entre $G_F$ y $g_W, m_W$ es:
# 
# $$
# \frac{G_F}{\sqrt{2}} = \frac{g^2_W}{8 m^2_W}
# $$
# 
# *Nota adicional*: En la relación anterior aparecen algunos factores $2$ que tienen un origen histórico.

# *Cuestión*: Calcula el valor de $g_W$ a partir de los valores de $G_F$ y $m_W$.

# El siguiente diagrama muestra la aniquilación $e+e^+ \to \mu + \mu^+$ mediada por un fotón. 
# 
# | |
# | :--: |
# |<img src="./imgs/feynman_eemumu_momenta.png" width = 400 align="center">|
# |Diagrama de Feynman $e+e^+ \to \mu + \mu^+$ mediado por $\gamma$|
# 
# Notar que los partículas están etiquetadas para asignar los cuadrimomentos.
# 

# #### Diagramas de árbol y de lazo
# 
# Los diagramas de Feynman que hemos visto son de nivel árbol. 
# 
# Otros diagramas pueden contribuir al mismo proceso, por ejemplo, en la dispersión de dos electrones:
# 
# | |
# | :--: |
# | <img src="./imgs/feynman_tree_loop.png" width = 400 align="center"> |
# | Diagrama árbol (izda) y lazo (derecha) |
# 
# El elemento de matrix del primer diagrama es $\propto \alpha$ mientras que el segundo $\propto \alpha^2$
# 
# Si la constante de acoplo $\alpha$ es pequeña, como es el caso en electromagnetismo, podemos considerar que los diagramas de lazo son correciones y el diagrama árbol es de primer nivel  (*leading order*).
# 
# A lo largo de los temas consideraremos solo diagramas árbol

# 
# 
# ## Bibliografía
# 
#  * [AB] Alessandro Bettini, "Introduction to Elementary Particle Physcs", Cambridge U. press. Tema 1
# 
#  * [MT] Mark Tomsom, "Modern Particle Physics", Cambridge U. press. Tema 1
#     
#  * [PDG](https://pdg.lbl.gov/) Particle Data Group.
# 
#  * [LHCb] "Observation of the rare $B_s \to \mu + \mu^+$ decay from the combined analysis of CMS and LHCb data", LHCB, CMS collaborations, Nature522, pages 68–72 (2015)
# 
#  * [Peskin] M. Peskin, "Lectures on the Theory of the Weak Interaction", 2016 CERN-JINR European School of Particle Physics, [arXiv:1708.09043](https://arxiv.org/abs/1708.09043)
# 
#  * [OPAL] G. Abbiendi et al. OPAL Collaboration, "Precise Determination of the Z Resonance Parameters at LEP", Eur. Phys. J. C 19, 587 (2001) [arXiv:hep-ex/0012018v1](https://arxiv.org/abs/hep-ex/0012018).
