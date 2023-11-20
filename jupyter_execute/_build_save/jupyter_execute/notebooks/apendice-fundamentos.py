#!/usr/bin/env python
# coding: utf-8

# # Introducción a Física de Partículas
# 
# 
# ## Apéndice: Unidades, relatividad y ecuación de Dirac
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


# # Indice
# 
#  * Unidades
#  * Relatividad especial
#     * Transformación de Lorentz
#     * cuadrimomentos
#  * Amplitudes de desintegración y secciones eficaces
#     * versión relativista de la regla de oro de Fermi 
#     * amplitud de desintegración a dos cuerpos
#     * sección eficaz de la interacción de dos cuerpos
#  * Invariantes de Mandelstam

# ## Unidades
# 
# Las unidades del SI son kg, m, s.
# 
# Pero no son  convenientes en Física de Partículas, por ejemplo, $m_e = 9.1 \, 10^{-31}$ kg.
# 
# Se utilizan unidades más convenientes, por ejemplo, para la sección eficaz se usa el barn = $10^{-20}$ m$^2$. Los valores de las secciones eficaces están en pbarns o fbarns.
# 
# Las magnitudes convenientes son $\hbar, c$, GeV.
# 
# | ---- $\hbar$ ------| ---- $c$ ------ | ---- GeV ----- |
# | :--     | :--  | :-- |
# | $1.055$ $10^{-34}$ Js | $2.998$ $10^8$ m/s| $1.602$  $10^{-10}$ J |
# 

# En Física de Partículas se usan  las Unidades Naturales (NU) donde $\hbar = c = 1$
# 
# Así tenemos $E^2 = m^2 c^2 + p^2 c^2$ $\to$ $E^2 = m^2 + p^2$, que es la ecuación de Einstein.
# 

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
# Es útil recordar el factor de conversión: $\hbar c$ = 0.197 GeV fm.
# 
# Para convertir de NU a SI se añaden los factores $\hbar, c$ correspondientes y para cuadrar las magnitudes.

# *Cuestión*: expresa el radio del protón $r = 4.1$ GeV$^{-1}$ en el S.I.

# In[3]:


r = 4.1
hbarc = 0.197e-15
r_si = r * hbarc
print(' radio del protón ', r_si, ' m')


# ## Relatividad special
# 
# ### Transformación de Lorentz
# 
# La transformación de Lorentz nos relaciona el espacio-tiempo $(t, {\bf r})$ en un sistema inercial $\Sigma$ con el espacio-tiempo $(t' {\bf r}')$ en otros sistema inercial $\Sigma'$ que se desplaza respecto del primero con velocidad $v$ en la dirección $z$.
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
# 

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

# 
# La transformación inversa (de $\Sigma' \to \Sigma$) viene dada por:
# 
# $$
# \begin{pmatrix} t \\ x \\ y \\ z \end{pmatrix} = 
# \begin{pmatrix} \gamma         & 0 & 0 & +\gamma \beta \\
#                  0             & 1 & 0 & 0 \\
#                  0             & 0 & 1 & 0 \\
#                  +\gamma \beta & 0 & 0 & \gamma 
# \end{pmatrix}
# \begin{pmatrix}t' \\ x' \\ y' \\ z' \end{pmatrix}
# $$
# 
# La matriz, ${\bf \Lambda}^{-1}$, de la segunda transformación, es inversa de la primera, ${\bf \Lambda}$. Esto es: ${\bf \Lambda} {\bf \Lambda}^{-1} = {\bf I}$

# ### cuatro-vectores
# 
# La magnitud ${\bf r}^2$, de un vector 3D, donde ${\bf r} = (x, y, z)$, es invariante bajo rotaciones.
# 
# El producto escalar, $x^\mu x_\mu = t^2 - x^2 - y^2 - z^2$, definido entre cuadri-vectores dados por: $x^\mu = (t, x, y, z), \;\, x_\mu = (t, -x, -y, -z)$, es invariante respecto tranformaciones de Lorentz.
# 
# El cuadri-vector, $x^\mu$, se llama **contra-variante** y $x_\mu$, **co-variante**.
# 

# 
# Están relacionados via  $x_\mu = g_{\mu\nu} x^\nu$. 
# 
# Donde $g_{\mu\nu}$ es el **tensor diagonal de la métrica**:
# 
# $$
# g_{\mu\nu } = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}
# $$
# 
# El producto entre dos cuadrivectores $a^\nu, b^\mu$ se puede denotar:
# $$
#  g_{\nu\mu} a^\nu b^\mu  = a^\nu b_\nu  = a_\mu b^\mu,
# $$
# donde los índices repetidos implica el sumatorio en los mismos. 
# $$
# g_{\nu\mu} a^\nu b^\mu \equiv \sum_{\mu=0}^3 \sum_{\nu = 0}^3 g_{\mu\nu} a^\nu b^\mu, \;\;\; a_\mu b^\mu \equiv \sum_{\mu=0}^3 a_\mu b^\mu
# $$

# La tranformación de Lorentz de $\Sigma \to \Sigma'$ viene dada por:
# 
# para el cuadri-vector contra-variante:
# 
# $$x'^\mu = \Lambda^\mu _{\;\; \nu} x^\nu,$$ 
# 
# donde $\Lambda^\mu_{\;\; \nu}$ es el elemento de la matriz de la transformación $\Sigma \to \Sigma'$ y los índices repetidos se suman.
# 
# para el co-variante:
# $$x'_\mu = \Lambda_\mu ^{\;\; \nu} x_\nu,$$
# 
# donde $\Lambda_\mu^{\;\; \nu}$ corresponde a los índices de la matriz ${\bf \Lambda}^{-1}$

# ### Cuadri-momento
# 
# Definimos el **cuadrimomento** (como vector contra-variante); $p^\mu = (E, p_x, p_y, p_z)$
# 
# El producto escalar es un invariante Lorentz:
# 
# $$
# p^2 = p^\mu p_\mu = E^2 - {\bf p}^2 = m^2,
# $$ 
# dado que para ${\bf p} = {\bf 0}$, tenemos $p_\mu = (m, 0, 0, 0)$.
# 
# Esta expresion es *la relación de Einstein entre energía y momento*.
# 
# En general usaremos $p^2 = p^\mu p_\mu$, para distinguirlo del producto escalar en 3D, ${\bf p}^2  = |{\bf p}|^2$

# 
# La energía y el momento de una partícula relativista de masa $m$ y velocidad ${\bf v}$ son
# 
# $$
# E = \gamma m c^2, \;\;\; {\bf p } = \gamma  m {\bf v}
# $$
# 
# En NU:
# 
# $$
# E = \gamma m, \;\;\; {\bf p} = \gamma m {\bf \beta} 
# $$
# 
# De donde obtenemos:
# 
# $$
# \gamma = \frac{E}{m}, \;\;\; \beta = \frac{p}{E}
# $$

# En el caso de *un sistema de n partículas*, $i = 1, \dots, n$, el cuadrimomento total:
# 
# $$
# p = \sum_i p_i,
# $$
# cumple:
# $$
# p^2 = \left(\sum_i E_i \right)^2 - \left( \sum_i {\bf p}_i \right)^2
# $$
# 
# En el caso en el que una partícula se desintegre $a \to b + c$, se cumple:
# 
# $$
# (p_b + p_c)^2 = p_a^2 = m^2_a,
# $$
# que se demonima, redundantemente, **masa invariante**.

# Es útil trabajar en el sitema **centro de masas** (CM), donde se cumple:
# 
# $$
# \sum_i {\bf p}_i = {\bf 0}
# $$
# 
# Es habitual utilizar la notación, $E^*, {\bf p}^*$ para la energía y el momento dados en el CM.

# ## Regla de oro de Fermi
# 
# 
# La **regla de oro de Fermi** establece que la **frecuencia de transición**, $R$, entre los estados $i$-inicial y $f$-final regidos por un hamiltoniano de interacción $H_{int}$ es:
# 
# $$
# R = (2\pi) \, |M_{fi}|^2 \rho(E),
# $$
# 
# donde:
# 
#    * $M_{if}$ es **el elemento de la matríz** de transición $M_{fi} = \langle f | H | i \rangle$
#    
#    * $\rho(E)$ es **la densidad de estados disponibles** $\rho(E)$, que depende de las posibilidades de momento que pueden tener los estados finales.
#    
# Notar que:
#    
#    * la física inherente a la interacción aparece en el elemento de matríz,
# 
#    * la densidad de estados disponibles depende sólo de la cinemática del evento.

# 
# En mecánica cuántica se calculan dichos factores en la aproximación no-relativista.
# 
# En Física de Partículas se precisa su versión relativista donde los factores $M_{fi}$ y $\rho(E)$ deben ser invariantes Lorentz.
# 
# La versión relativista pasa por dar un tratamiento relativista a la normalización de función de ondas. 

# #### normalización de la función de ondas relativista
# 
# En mecánica cuántica normalizamos la función de ondas, $\psi$ en un cubo de lados $L$, con volumen $V = L^3$, a la unidad:
# 
# $$
# \int_V \psi^* \psi \, \mathrm{d}^3x = 1 
# $$
# 
# Esto es, la partícula está contenida en $V$.
# 
# La densidad densidad de estados disnponible asociada a ese volumen es:
# $$
# \frac{\mathrm{d}^3{\bf p}}{(2\pi)},
# $$
# 
# que viene de expresar $\frac{\mathrm{d}^3 {\bf p}}{h}$ en NU, recordando que en NU: $h = 2\pi$.

# 
# En tratamiento relativista, el volumen no es un invariante Lorentz.
# 
# Un cubo de lado $L$ en un sistema con un factor de Lorentz, $\gamma$ es
# $$
# V ' = \gamma L^3 = \gamma V
# $$
# 
# Si en ese cubo estaba contenida la partícula  $E_0 = m$, en el sistema con factor $\gamma$, su energía es:
# 
# $$
# E' = \gamma E_0
# $$

# 
# Vemos que $V, E$ se tranforman Lorentz de igual forma, via un factor $\gamma$. 
# 
# En mecánica cuántica relativista, normalizamos la función de ondas, $\Psi$, 
# en un volumen $V$ a $2E$:
# 
# $$
# \int_V \Psi^* \Psi \, \mathrm{d}^3 x = 2E,
# $$
# El factor $2$ es por conveniencia.

# #### elemento de matriz relativista
# 
# Como consecuencia de la normalización relativista de la función de ondas $\Psi$, el elemento de matriz relativista $|M_{fi}| = \langle \Psi_f| H_{int}| \Psi_i \rangle$ se relaciona con el no-relativista $T_{fi} = \langle \psi_f | H_{int}| \psi_i \rangle$ por un factor que proviene de la normalización. 
# 
# $$
# M_{fi} = \langle \Psi_f | H_{int}| \Psi_i \rangle = \left(\prod_i \sqrt{2E_i} \right) \langle  \psi_f | H_{int}| \psi_i \rangle =  \left(\prod_i \sqrt{2E_i} \right)  T_{fi}
# $$
# donde índice $i$ corre en las $n$ partículas iniciales y finales.
# 
# Pare recuperar las expresiones de la amplitud de transición o la sección eficaz en el tratamiento no relativista debemos introducir el inverso de ese factor al cuadrado, esto es:
# 
# $$
# \frac{1}{\prod_i (2E_i)}
# $$

# La amplitud de desintegración de una partícula de $E_a$ a $m$ partículas finales.  
# 
# Pasamos de la expresión no-relativista
# 
# $$
# \Gamma  = (2\pi)^4 \int |T_{fi}|^2 \delta^4 \left(\sum_k p_k - p_a \right) \, \prod_i \frac{\mathrm{d}^3{\bf p}_i}{(2\pi)^3}
# $$
# donde $p_a$ es el cuadrimomento de la partícula inicial y $p_i$ los de las finales, $i = 1, \dots, m$. Recordar que las deltas de Dirac imponen la conservación de energía y momento entre la partícula inicial y las finales. 
# 
# A la relativista:
# 
# $$
# \Gamma = \frac{(2\pi)^4}{2 E_a} \int |M_{fi}|^2 \delta^4 \left(\sum_k p_k - p_a \right) \, \prod_i \frac{\mathrm{d}^3{\bf p}_i}{(2\pi)^3 (2E_i)}
# $$

# Para la sección eficaz de la interacción de dos partículas $ a + b$ en $m$ partículas finales.
# 
# La expresión no relativista:
# $$
# \sigma = \frac{(2\pi)^4}{|v_a + v_b|} \int |T_{fi}|^2 \delta^4 \left(\sum_k p_k - p_a - p_b \right) \, \prod_i \frac{\mathrm{d}^3{\bf p}_i}{(2\pi)^3},
# $$
# 
# donde $v_a, v_b$ son el módulo de las velocidades, considerando que las partículas colisionan en una misma dirección y tienen sentidos opuestos. Las deltas de Dirac imponen la condición de conservación de la energía y momento entre los estados inicial y final.
# 
# La expresión relativista es:
# $$
# \sigma = \frac{(2\pi)^4}{(2 E_a) (2 E_b) (\beta_a + \beta_b)} \int |M_{fi}|^2 \delta^4 \left(\sum_k p_k - p_a -p_b\right) \, \prod_i \frac{\mathrm{d}^3{\bf p}_i}{(2\pi)^3 (2E_i)}
# $$
# 
# 

# En la versión relativista, cada uno de los factores de la sección eficaz es invariante Lorentz:
# 
#    * el elemento de matriz al cuadrado: $|M_{fi}|^2 = | \langle \Psi_f| H_{int} | \Psi i\rangle |^2$
#    
#    * la densidad de estados: $\frac{\mathrm{d}^3 {\bf p}}{(2\pi)^3 (2E)}$
#    
#    * el término asociado al flujo: $4 E_a E_b (\beta_a + \beta_b)$
#    
# Notar que a partir de la propiedad de la función delta de Dirac, $\int \delta(E^2-{\bf p}^2 - m^2) \, \mathrm{d}E = \frac{1}{2E}$, podemos reescribir:
# 
# $$
# \frac{\mathrm{d}^3{\bf p}}{(2\pi)^3 (2E)} = \int \delta(p^2-m^2) \frac{\mathrm{d}^4p}{(2\pi)^3}
# $$

# *Cuestión*: Demostrar que $4 E_a E_b (\beta_a + \beta_b) = 16 \sqrt{(p_a \cdot p_b)^2 - m^2_a m^2_b}$

# ## Anchura de desintegración
# 
# Un sistema de N partículas en un volumen V que se desintegran en el tiempo a una frecuencia constante:
# 
# $$
# \frac{\mathrm{d}N}{\mathrm{d}t} = - \Gamma N
# $$
# 
# Cumple:
# $$
# N(t) = N \, e^{-\Gamma t}
# $$
# 
# A $\Gamma$ la llamamos **anchura de desintegración** y a su inverso, $\tau$ **tiempo de vida media**:
# 
# $$
# \tau = \frac{1}{\Gamma}, \;\; N = N \, e^{-t/\tau}
# $$

# En el caso de que la partícula se desintegra a $n$ varios canales, $i = 0, \dots, n$.
# 
# Para cada uno habrá una **anchura parcial de desintegración**, $\Gamma_i$
# 
# La anchura total será la suma de las parciales:
# 
# $$
# \Gamma = \sum_i \Gamma_i
# $$
# 
# Y llamaremos fracción de desintegración, $\mathcal{Br}_i$, de un canal al porcentaje de veces que una partícula se desintegra en ese canal, que corresponde a:
# 
# $$
# \mathcal{Br}_i = \frac{\Gamma_i}{\Gamma}
# $$

# En mecánica cuántica la función de ondas se normaliza a 1 un un volumen unidad
# 
# La relación entre la anchura de desintegración, $\Gamma$ y la frecuencia de transición es simplemente:
# 
# $$
# \Gamma = R
# $$
# 
# En la formulación relativista, tenemos que introducir los factores de normalización $\frac{1}{(2E)}$, para cada partícula.
# 
# $$
# \Gamma = \frac{(2\pi)^4}{2 E_a} \int |M_{fi}|^2 \delta^4 \left(\sum_k p_k - p_a \right) \, \prod_i \frac{\mathrm{d}^3{\bf p}_i}{(2\pi)^3 (2E_i)}
# $$

# ### Anchura de desintegración a dos partículas
# 
# Consideremos la desintegración de una partícula $a$ en dos $b, d$, con masas $m_a, m_b, m_c$.
# 
# | |
# |:--:|
# |<img src="./imgs/intro_drawing_decay2.png" width = 400 align="center">|
# | Esquema de desintegración de la partícula $a$ a dos $b, c$ en el CM|
# 
# El sistema más conveniente para tratar el proceso es el CM, donde la partícula $a$ está en reposo, y tiene $E = m_a$.
# 
# Las partículas $b, c$ salen con momentos opuestos ${\bf p}^* = {\bf p}_b = - {\bf p}_c$. 
# 
# El módulo del momento de las partículas salientes, $p^*$ es (ver sección de los invariantes de Mendelstam): 
# 
# $$
# p^*= \frac{1}{2m_a} \sqrt{[m^2_a - (m_b+m_c)^2] \, [m^2_a - (m_b - m_c)^2] }
# $$
# 
# Notar que a falta de un espín polarizado de la partícula $a$ (que definiría un eje) no hay ninguna dirección privilegiada.

# La anchura de desintegración, $\Gamma$, es la razón de transición de la regla de oro de Fermi.
# 
# Integramos usando $\delta({\bf p}_b + {\bf p}_c)$ en  $\mathrm{d}^3p_{c}$ directamente.
# 
# $$
# \Gamma = \frac{1}{(2 E) (2\pi)^2} \int |M_{fi}|^2 \delta \left(E - E_b - E_c \right) \frac{p^2_b }{4 E_b E_c}\,  \mathrm{d}p_b \mathrm{d}\Omega^*,
# $$
# 
# donde $E_b = \sqrt{m^2_b + p^2_b}, \; E_c = \sqrt{m^2_c + p^2_b}$ y $\mathrm{d}\Omega^*$ el diferencial de ángulo sólido.
# 

# Si aplicamos la siguiente propiedad de la delta de Dirac: $\int \delta(f(p))\,  \mathrm{d} p = \left|\frac{\mathrm{d}f(p)}{\mathrm{d}p} \right|_{p'}^{-1}$, donde $p'$ cumple $f(p') = 0$
# 
# con $f(p_b) = E -  \sqrt{m^2_b + p^2_b}- \sqrt{m^2_c + p^2_b}$ respecto $p = p_b$, y $p' = p^*$:
# 
# $$
# \frac{\mathrm{d}f(p_b)}{\mathrm{d}p_b} = \frac{p_b}{\sqrt{m^2_b + p^2_b}} + \frac{p_b}{\sqrt{m^2_c + p^2_c}} = p_b \frac{E_c + E_d}{E_c E_d} = p_b \frac{E}{E_c E_d}
# $$
# 
# La anchura de desintegración queda:
# 
# $$
# \Gamma = \frac{1}{8 \pi^2 E} \frac{(p^*)^2}{4 E_b E_c} \frac{E_b E_c}{p^* E} \int |M_{fi}|^2 \, \mathrm{d}\Omega^* = 
# \frac{1}{8\pi^3}\frac{p^*}{4 E^2} \int |M_{fi}|^2 \, \mathrm{d}\Omega^*
# $$

# La anchura de desintegracion, en este caso, solo depende de $p^*$, si colocamos $E = E_a = m_a$ en el CM:
# 
# $$
# \Gamma = \frac{p^*}{32 \pi^2 m^2_a} \int_{\Omega} |M_{fi}|^2 \mathrm{d}\Omega^*
# $$

# ### Sección eficaz de una interacción de dos cuerpos
# 
# Sea la interacción entres dos partículas $a, b$ que da lugar a $m$ particulas finales.
# 
# Sea un flujo de partículas $a$, con densidad de $N_a$ partículas en un volumen $V$ que se mueve contra un blanco de $N_b$ partículas en un volumen $V$ que se mueven en sentido opuesto con $v_b$.
# 
# Asignamos una sección eficaz $\sigma$ de interacción a la partículas blanco $b$. La frecuencia de interacciónes corresponderá al número de partículas $a$ que atraviesa un cilindro de base $\sigma$ y altura $v_a + v_b$ para cada partícula $b$.
# 
# | |
# |:--|
# |<img src="./imgs/fun_drawing_xsection.png" width = 400 align="center">|
# | (a) una partícula $a$ recorre en $\mathrm{d}t$ un cilindro de longitud $v_a+v_b$ [MT3.4] |
# | (b) visión transversal del cilindro, las partículas $b$ en el mismo con el disco de la sección eficaz|
# 

# 
# Esto es, en número de interacciones en un volumen $V$ y un tiempo $\mathrm{d}t$ es:
# 
# $$
# \mathrm{d}N = (v_a + v_b) \sigma \, n_a N_b \, \mathrm{d}t
# $$
# 
# El factor $(v_a + v_b) n_a$ es el flujo de partículas $a$, $\phi_a$, con velocidad $v_a$ que atraviesan una superficie perpendicular a su velocidad, cuando ésta se mueve contra ellas con velocidad $v_b$.
# 
# Luego, la frecuencia de interacciones, $R$, en un volumen, V, donde hay $N_b$, partículas blanco, al que llega un flujo $\phi_a$ de partículas $a$ es:
# 
# $$
# R = \phi_a \, \sigma \, N_b
# $$
# 
# 

# En mecánica cuántica se adopta la normalización de una partícula en una unidad de volumen, $V = 1, n_a = 1, n_b = 1$, en ese caso:
# 
# $$
# R = (v_a + v_b) \, \sigma, \;\;\; \sigma = \frac{R}{(v_a+v_b)}
# $$
# 
# En la formulación relativista, tenemos que introdocir in factor de normalización $\frac{1}{(2E)}$ para cada partícula.
# 
# $$
# \sigma = \frac{(2\pi)^4}{(2 E_a) (2 E_b) (\beta_a + \beta_b)} \int |M_{fi}|^2 \delta^4 \left(\sum_k p_k - p_a -p_b\right) \, \prod_i \frac{\mathrm{d}^3{\bf p}_i}{(2\pi)^3 (2E_i)}
# $$

# ### Sección eficaz de interacción de dos cuerpos
# 
# Sea la interacción entre dos $a, b$ partículas incidentes que da lugar a dos partículas finales $c, d$.
# 
# | |
# |:--:|
# |<img src="./imgs/intro_drawing_int2.png" width = 400 align="center">|
# | Esquema de interacción de las partículs $a, b$ a $c, d$ en el CM|
# 
# El sistema más conviente para calcula la sección eficaz es el CM. En este sistema, se conservan los momentos:
# 
# $$
# {\bf p}^*_i = {\bf p}_a = - {\bf p}_b, \;\; {\bf p}^*_f = {\bf p}_c = - {\bf p}_d
# $$
# 
# Notar que las partículas están en un plano, y el único parámetro libre es el ángulo, $\theta^*$ entre ${\bf p}^*_f$ y ${\bf p}^*_i$, y este último define el eje.
# 

# La sección eficaz se obtiene a partir de la expresión general donde $E = E_a + E_b$, $p_b = p_a = p^*_i$ y $p_d = p_c = p^*_f$.
# 
# El término del espacio fásico se calcula igual que en el apartado de la amplitud de desintegración a dos cuerpos, solo que ahora $E = E_a + E_b$:
# 
# $$
# \frac{1}{(2\pi)^2}\frac{p^*_f}{4 E} \int  |M_{fi}|^2 \, \mathrm{d}\Omega^*
# $$
# 
# 
# El término asociado al flujo:
# 
# $$
# 4 E_a E_b (\beta_b + \beta_a) = 4 E_a E_b \left(\frac{p_a}{E_a} + \frac{p_b}{E_b}\right) = 8 E p^*_i
# $$

# La sección queda:
# 
# $$
# \sigma = \frac{1}{64 \pi^2 E^2} \frac{p^*_f}{p^*_i} \int_\Omega |M_{fi}|^2 \, \mathrm{d}\Omega^*
# $$
# 
# si sustituimos $E^2 =  s$ donde $s = \left[(E_a, {\bf p}_a) + (E_b, {\bf p}_b)\right]^2 = [(E_a + E_b, {\bf 0})]^2$, es el cuadrimomento transferido al cuadrado.
# 
# $$
# \sigma = \frac{1}{64 \pi^2 s} \frac{p^*_f}{p^*_i} \int_\Omega |M_{fi}|^2 \, \mathrm{d}\Omega
# $$

# ## Invariables de Mandelstam y cinemática
# 
# | | 
# | :-- |
# | <img src="./imgs/feynman_stchannels.png" width = 400 align="center">|
# | Diagramas asociados a los invariantes de Mandelstam (izda) dispersión (derecha) aniquilación|
# 
# 
# Los cuadrimomentos transferidos, $q^2$, entre las corrientes, de los diagramas de la figura, se denota con:
# 
# $$
# t = (p_c - p_a)^2 = (p_d - p_b)^2, \;\;\; s = (p_a +p_b)^2 = (p_c + p_d)^2,
# $$
# 
# donde $p_\alpha$, $\alpha = a, b, c, d$ son los cuadrimomentos de las partículas, 
# 
# y se denominan **invariantes de Mandelstam** y que corresponden a los **canales** $t$ de dispersión y $s$ de aniquilación respectivamente.
# 

# ### Cinemática con el invariante $s$
# 
# La cantidad $\sqrt{s}$ es la energía en el centro de masas, (CM), de una aniquilación
# $$
# s = (p_a + p_b) = (E^*_a + E^*_b) - ({\bf p}^*_a +{\bf p}^*_b) = (E^*_a + E^*_b)^2,
# $$
# 
# dado que ${\bf p}^*_i = {\bf p}^*_a = -{\bf p}^*_b$ (ver figura de la dispersión de dos cuerpos arriba).
# 
# En la literatura se denota $\sqrt{s}$ para indicar la **energía en el centro de masas**.
# 

# Podemos calcular $E^*_a, E^*_b$ a partir de $s$ y las masas $m_a, m_b$.
# 
# Si calculamos, $E^{*2}_a$ teniendo en cuenta que $E^*_a = (\sqrt{s} - E^*_b)$, y $p^*_a = p^*_b = p^*_i$, obtenemos:
# 
# $$
# E^{*2}_a = m^2_a + p_i^{*2} = E^{*2}_b - 2 \sqrt{s}E^*_b + s = m^2_b + p_i^{*2} - 2 \sqrt{s}E^*_b + s \\
# m^2_a = m^2_b + s - 2 \sqrt{s} E^*_b
# $$
# 
# 
# y como el tratamiento para $a$ es idéntico al de $b$, obtenemos:
# 
# $$
# E^*_a = \frac{s + m^2_a - m^2_b}{2 \sqrt{s}}, \;\; E^*_b = \frac{s + m^2_b - m^2_a}{2 \sqrt{s}}.
# $$
# 
# 

# El momento será:
#     
# $$
# p^{*}_i = \sqrt{E^{*2}_a - m^2_a} = p^* = \sqrt{E^{*2}_b - m^2_b}, 2 p^{2*} = E^{*2}_a + E^{*2}_b - m^2_a - m^2_b \\
# $$
# 
# por lo tanto
# 
# $$
# p^{2*}_i = \frac{E^{*2}_a + E^{*2}_b - m^2_a - m^2_b}{2} = \\ 
# \frac{\left[s + (m^2_a - m^2_b) \right]^2 + \left[s - (m^2_a - m^2_b)\right]^2 - 4 s (m^2_a + m^2_b)}{8 s} = \\
# \frac{s^2 + (m^2_a - m^2_b)^2 - 2 s (m^2_a + m^2_b)}{4s}
# $$
# 
# 

# como: 
# 
# $$
# m^2_a - m^2_b = (m_a + m_b) (m_a - m_b), \\ 2(m^2_a + m^2_b) = (m_a+m_b)^2 + (m_a-m_b)^2
# $$
# 
# tenemos:
# 
# $$
# p^{*2}_i = \frac{s^2 + (m_a+m_b)^2(m_a-m_b)^2 - s \left[ (m_a+m_b)^2 + (m_a - m_b)^2 \right] }{4s} = \\
#        = \frac{\left[s- (m_a+m_b)^2\right] \, \left[s - (m_a-m_b)^2\right]}{4s}
# $$
# 
# por lo que el momento inicial es:
# 
# $$
# p^*_i = \frac{\sqrt{\left[s- (m_a+m_b)^2\right] \, \left[s - (m_a-m_b)^2\right]}}{2\sqrt{s}}
# $$
# 

# Como la situación es idéntica para las partícula finales tenemos:
# 
# $$
# E^*_c = \frac{s + m^2_c - m^2_d}{2 \sqrt{s}}, \;\; E^*_d = \frac{s + m^2_d - m^2_c}{2 \sqrt{s}} \\
# p^*_f = \frac{\sqrt{\left[s- (m_c+m_d)^2\right] \, \left[s - (m_c-m_d)^2\right]}}{2 \sqrt{s}}
# $$

# En el caso de la desintegración a dos cuerpos, $a \to c + d$, la energía en el CM es $ \sqrt{s} = m_a$.
# 
# Los momentos y energía finales son en ese caso:
# 
# $$
# E^*_c = \frac{m^2_a + m^2_c - m^2_d}{2 m_a}, \;\; E^*_d = \frac{m^2_a + m^2_d - m^2_c}{2 m_a} \\
# p^*_f = \frac{1}{2m_a}\sqrt{\left[m^2_a- (m_c+m_d)^2\right] \, \left[m^2_a - (m_c-m_d)^2\right]}
# $$

# ### Cinemática con el invariante $t$
# 
# 
# El invariante t, $t = (p_c - p_a)^2 = (p_d - p_b)^2$, es:
# 
# $$
# t = (p_c - p_a)^2 = p^2_c + p^2_a - 2 p_c \cdot p_a = m^2_a + m^2_c + 2 |{\bf p}_a| |{\bf p}_c| \cos \theta_{ab} - 2 E_a E_c \\
# t = (p_d - p_b)^2 = p^2_d + p^2_b - 2 p_d \cdot p_b = m^2_b + m^2_d + 2 |{\bf p}_b| |{\bf p}_d| \cos \theta_{bd} - 2 E_b E_d
# $$
# 
# donde $\theta_{ac}, \theta_{bd}$ son los ángulos entre los momentos de $ac$ y $bd$ respectivamente.
# 
# 

# En el sistema de laboratorio $p_b = (m_b, {\bf 0})$, tenemos :
# 
# $$
#  t = m^2_b + m^2_d - 2m_b E_d, \;\; E_d = \frac{m^2_b + m^2_d - t}{2m_b}
# $$
# 
# 
# En el sistema CM (ver figura de la interacción a dos cuerpos), $|{\bf p}_a| = p^*_i, \; |{\bf p}_c| = p^*_f$ y $\theta_{ac} = \theta^*$
# 
# $$
# t = m^2_a + m^2_c + 2 p^*_i p^*_f \cos \theta^* - 2E_a E_c, \\
# \cos \theta^* = \frac{t - m^2_a - m^2_b + 2E^*_aE^*_c}{2 p^*_i p^*_f}
# $$
# 
# 
# 

# ## Bibliografía
# 
#  * [AB] Alessandro Bettini, "*Introduction to Elementary Particle Physics*", Cambridge U. press. Tema 1
# 
#  * [MT] Mark Tomson, "*Modern Particle Physics*", Cambridge U. press. Tema 2 y 3.
# 
