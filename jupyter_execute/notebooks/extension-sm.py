#!/usr/bin/env python
# coding: utf-8

# # Introducción a Física de Partículas
# 
# 
# ## Extensión -  Modelo Estandard
# 
# 
# Jose A. Hernando
# 
# *Departamento de Física de Partículas. Universidade de Santiago de Compostela*
# 
# Noviembre 2021
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


# ### Objetivos
#   
# Conocer:
#   
#   * El lagrangiano para campos escalares, fermiónicos y vectoriales.
#   
#   * La invariance Gauge
#   
#   * La unificación electro-débil
#   
#   * El mecanismo de Higgs y la dotación de masas a los bosones vectoriales débiles y los fermiones
#   

# ## Índice
# 
#   * Lagrangiano y simetrías gauge
#   
#   * Simetrías del SM 
#   
#   * Mecanismo de Higgs 

# ## El Lagrangiado del SM
# 
# ### De la Física Clásica a la Teoría Cuántica de Campos
# 
# La dinámica de un sistema en Física Clásica queda determinada por el lagrangiano: 
# 
# $$
# L(q, \dot{q}) = T -V
# $$
# 
# que depende de las coordenadas generalizadas $q_i$, con $i=1, \dots, n$ y sus derivadas temporales $\dot{q}$
# 
# las ecuaciones del movimiento viene dadas por las escuaciones de Euler-Lagrange:
# 
# $$
# \frac{\mathrm{d}}{\mathrm{dt}} \left( \frac{\partial L}{\partial \dot{q}_i}\right) = \frac{\partial L}{\partial q_i}
# $$

# En el caso de una partícula en un potencial, el lagrangiano es su energía cinética menos su potencial:
# 
# $$
# L = \frac{1}{2} m \dot{x}^2 - V(x) 
# $$
# 
# las ecuaciones del movimiento dan:
# $$
# m \ddot{x} = -\frac{\partial V(x)}{\partial x}
# $$
# 
# que es la ecuación de Newton.

# En teoría cuántica de campos, TCC, introducimos campo $\phi(x)$ definido en el espacio $x$ con un lagrangiano:
# 
# $$
# L = \int \mathcal{L}(\phi, \partial_\mu \phi) \, \mathrm{d}^3x 
# $$
# 
# donde $\mathcal{L}$ es la densidad lagrangiana, o por abuso del lenguaje, el lagrangiano.
# 
# La ecuaciones del movimiento son:
# 
# $$
# \partial_\mu \left( \frac{\partial \mathcal{L}}{\partial (\partial_\mu \phi)}\right) = \frac{\partial \mathcal{L}}{\partial \phi}
# $$
# 
# En TCC entendemos las partículas como excitaciones del campo $\phi$.

# ### Lagrangiano del campo escalar
# 
# Sea $\phi(x)$ un campo escalar asociado a partículas de masa $m$ su lagrangiano viene dado por:
# 
# $$
# \mathcal{L}(\phi) = \frac{1}{2} (\partial_\mu \phi) \, (\partial^\mu \phi) - \frac{1}{2} m^2 \phi^2
# $$
# 
# Y su ecuación de movimiento:
# 
# $$
# \partial_\mu \partial^\mu \phi + m^2 \phi = 0
# $$
# 
# Es la ecuación Klein-Gordon

# ### Lagrangiano del los fermiones (espín 1/2)
# 
# Sea $\Psi$ un campo espinorial de Dirac, su langrangiano es:
# 
# $$
# \mathcal{L}(\Psi) = i \bar{\Psi} \gamma^\mu \partial_\mu \Psi - m \bar{\Psi} \Psi
# $$
# 
# Donde:
# $$
# \Psi = \begin{pmatrix} \psi_1 \\ \psi_2 \\ \psi_3 \\ \psi_4\end{pmatrix},
# $$
# 
# cada $\psi_i$ con $i=1, \dots, 4$ es un campo complejo. Y las $\gamma$'s son matrices complejas $4 \times 4$, con reglas de conmutación determinadas (ver [apendice-Dirac])
# 

# La ecuación del Euler-Lagrange con respecto a $\bar{\Psi}$ nos da:
# 
# $$
# (i \gamma^\mu \partial_\mu - m)  \, \Psi = 0 
# $$
# 
# que es la ecuación de Dirac

# ### Lagrangiano del campo vectorial
# 
# Sea el campo vectorial $A^\mu(x)$, por ejemplo electromagnético, $A^\mu = (\phi, {\bf A})$, donde $\Phi$ es el potencial eléctrico y ${\bf A}$ el potencial vector magnético.
# 
# El tensor de fuerza es:
# 
# $$
# F^{\mu \nu} = \partial^\mu A^\nu - \partial^\nu A^\mu
# $$
# 
# y la corriente $j^\mu = (\rho, {\bf j})$, donde $\rho$ es la densidad eléctrica y ${\bf j}$ la corriente eléctrica.
# 
# La ecuación del movimiento:
# 
# $$
# \partial_\mu F^{\mu\nu} = j^\nu
# $$
# 
# nos da las ecuaciones de Maxwell.

# La ecuación se obtiene del lagrangiano:
# 
# $$
# \mathcal{L}(A) = - \frac{1}{4} F_{\mu\nu} F^{\mu\nu} - j^\mu A_\mu
# $$
# 
# donde el término $j^\mu A_\mu$ representa la interacción de una corriente eléctroca con el foton.
# 
# Para el fotón aislado:
# 
# $$
# \mathcal{L}(A) = - \frac{1}{4} F_{\mu\nu} F^{\mu\nu}
# $$
# 
# Y para un fotón masivo, con masa, $m_A$:
# 
# $$
# \mathcal{L}(A) = - \frac{1}{4}  F_{\mu\nu} F^{\mu\nu} + \frac{1}{4} m_A^2 A^\mu A_\mu
# $$
# 
# que se conoce como lagrangiano de Proca.
# 

# ## Simetría gauge
# 
# ### QED
# 
# El lagrangiano de un fermión es invariante respecto a una tranformación global de U(1), esto es respecto una fase global, $q\theta$:
# 
# $$
# \Psi(x) \to \Psi'(x) = e^{iq\theta} \Psi(x)
# $$
# 
# donde $q$ es el valor de la carga conservada asociada al campo y la simetría conservada.
# 
# que es inmediato de comprobar, dado que $\bar{\Psi'} = \bar{\Psi} \, e^{-i\theta}$
# 

# 
# Llamamos **invariancia gauge local** si el lagrangiano es invariante bajo una tranformación local de un grupo de simetría.
# 
# En el caso del campo del fermion, sea una tranformación local bajo el grupo $U(1)$
# 
# $$
# \Psi(x) \to \Psi'(x) = e^{i q \theta(x)} \Psi
# $$
# 
# donde $\theta(x)$ depende ahora del punto local $x$, y $q$ en el caso del electromagnetismo es el acoplo electromagnético, en este caso la carga eléctrica del fermión, por ejemplo $-e$ para el electrón.
# 
# El lagrangiano bajo esta tranformación queda:
# 
# $$
# \mathcal{L'}(\Psi') = i \bar{\Psi} \gamma^\mu \partial_\mu \Psi - m \bar{\Psi} \Psi - q \bar{\Psi} \gamma^\mu \Psi (\partial_\mu \theta)
# $$
# 

# El siguiente término rompe la invariancia:
# 
# $$
# - q \bar{\Psi} \gamma^\mu \Psi \, (\partial_\mu \theta(x))
# $$
# 
# que interpretamos como el acoplo de una corriente $j^\mu = q \bar{\Psi} \gamma^\mu \Psi$, la electromagnética, con un campo vectorial $A_\mu = \partial_\mu \theta(x)$:
# 
# $$
#  - j^\mu \, A_{\mu}
# $$
# 
# Si consideramos el lagrangiano donde hemos añadido el término del campo vectorial:
# 
# $$
# \mathcal{L}_{EL}(\Psi, A) = i \bar{\Psi} \gamma^\mu \partial_\mu \Psi - m \bar{\Psi}\Psi -  j^\mu  A_\mu - \frac{1}{4} F^{\mu\nu}F_{\mu\nu}
# $$
# 
# Este es el lagrangiano del electromagnetico si $q$ es la carga eléctrica.
# 

# El lagrangiano del EM es ahora invariante bajo la tranformación gauge local de U(1) del campo del fermión, $\Psi(x)$, si el campo vectorial, $A^\mu(x)$, cambia de forma acorde:
# 
# $$
# \Psi'(x) = e^{iq \theta(x)} \Psi(x), \;\;\; A'_\mu(x) = A_\mu(x) - \partial_\mu \theta(x)
# $$
# 
# que se puede expresar más elegantemente sustituyendo:
# 
# $$
# \partial_\mu \to D_\mu = \partial_\mu + i q A_\mu
# $$
# 
# $D_\mu$ se denomina **derivada covariante generalizada**.
# 
# El Lagrangiano del electromagnetismo, que describe el fermión, el fotón y su interacción, queda:
# 
# $$
# \mathcal{L}_{EM}(\Psi, A) = i \bar{\Psi} \gamma^\mu D_\mu \Psi - m \bar{\Psi}\Psi - \frac{1}{4} F^{\mu\nu}F_{\mu\nu}
# $$
# 

# #### No invariancia del término de masas del bosón
# 
# Sin embargo el término de masas del bosón vectorial $A$, no es invariante gauge:
# 
# $$
# \frac{1}{4} m_A^2 A'^\mu A'_\mu  = \frac{1}{4}m_A^2 (A^\mu - \partial^\mu \theta) \, (A_\mu - \partial_\mu \theta)
# $$
# 
# por eso decimos que los terminos de masa de los bosones vectoriales rompen la simetría gauge.

# ### Clase acelerada sobre el Higgs en el SM:
# 
# | | | | |
# | :--: |:--: | :--: | :--: |
# | ----- Tipo ----- | --------------- Escalar ---------------- | ---------- Espinor ------------ | ------ Bosón vectorial ---------- |
# | | Klein-Gordon | Dirac | Maxwell |
# |Ecuación | $(\partial^\mu\partial_\mu + m^2) \, \phi = 0$ | $(i\gamma^\mu \partial_\mu - m) \, \Psi = 0$ | $\partial_\mu F^{\mu\nu} = j^\nu$ |
# |Lagrangiano| $\frac{1}{2} (D_\mu \phi)^\dagger (D^\mu \phi) - \frac{1}{2} m^2 \phi^\dagger \phi$ | $i\bar{\Psi}\gamma^\mu D_\mu \Psi - m \bar{\Psi}\Psi$|  $-\frac{1}{4}F^{\mu\nu}F_{\mu\nu} \left[- \frac{1}{4} m^2 A_\mu A^\mu \right]$|
# 
# Donde:
# 
# $$
# D_\mu = \partial_\mu + i g A_\mu, \;\;\; F^{\mu\nu} = \partial^\mu A^\nu - \partial^\nu A^\mu
# $$
# 
# siendo $g$ una constante de acoplo, por ejemplo $g=e$, y si $A_\mu$, el fotón, obtenemos el electromagnetismo.

# El Lagrangiano de espinores y bosones vectoriales:
# 
# $$
# i \bar{\Psi} \gamma^\mu D_\mu \Psi - m \bar{\Psi} \Psi - \frac{1}{4} F^{\mu\nu}F_{\mu\nu}
# $$
# 
# Es invariante bajo la simetría gauge local, dada por $\theta(x)$:
#     
# $$
# \Psi'(x) = e^{i g \theta(x)} \, \Psi(x), \;\;\; A'_\mu(x) = A_\mu(x) -i\partial_\mu \theta(x)
# $$ 
# 
# Pero no el término de masas del bosón vectorial:
# 
# $$
# \frac{1}{2} m_A^2 A'^\mu A'_\mu  = \frac{1}{2}m_A^2 (A^\mu - \partial^\mu \theta) \, (A_\mu - \partial_\mu \theta)
# $$

# Notar que nos en lagrangiano aparece el término:
#     
# $$
# i \bar{\Psi} \gamma^\mu D_\mu \Psi = \dots + (i g \bar{\Psi} \gamma^\mu \Psi) \, A_\mu  
# $$
# 
# que es el acoplo de una corriente fermiónica con un acoplo $g = e$ al fotón que interpretamos como el vértice:
# 
# $$
# e \bar{\Psi} \gamma^\mu \Psi \, A_\mu = j^\mu_{EL} \, A_\mu
# $$

# ### ¿Cómo el higgs da masa a fermiones y bosones?
# 
# Consideremos un campo complejo escalar particular, con un valor $v$ en el **vacío**:
# 
# $$
# \phi(x) = \frac{1}{\sqrt{2}}\left(v + h(x)\right) 
# $$
# 
# Consideremos primero los fermiones:
# 
# La posible interacción en el lagrangiano entre fermiones y este campo, controlada por un acoplo (llamado de **Yukawa**), $\lambda$
# 
# $$
# -\lambda \bar{\Psi} \Psi \, \phi = -\frac{\lambda v}{\sqrt{2}} \bar{\Psi} \Psi - \frac{\lambda}{\sqrt{2}} \bar{\Psi} \, \Psi h(x)
# $$

# El primero nos da la masa del fermión:
# 
# $$
# m= \frac{\lambda v}{\sqrt{2}}
# $$
# 
# Y el segundo la interacción (la desintegración por ejemplo) del Higgs a un par fermión, anti-fermión

# Veamos como el mecanismo dota de masa a los bosones:
# 
# La deriviada covariante es:
# $$
# D_\mu \phi = (\partial_\mu + i g A_\mu) \frac{1}{\sqrt{2}} \left( v + h(x) \right)  = \frac{1}{\sqrt{2}} \left(\partial_\mu h(x) + ig v \, A_\mu + i g h(x) A_\mu \right)
# $$
# 
# Con lo que en el término del lagrangiano aparecen, entre otros, los términos:
# 
# $$
# \frac{1}{4}(D^\mu \phi)^*(D_\mu \phi) = \dots + \frac{1}{4} \frac{q^2 v^2}{2} A^\mu A_\mu + 
# $$
# 

# El primero es el término de masas del bosón vectorial:
# 
# $$
# m_A = \frac{g v}{\sqrt{2}}
# $$
# 
# Y el segundo la interacción del higgs con un par de bosones vectoriales:
# 

# Un simil útil es considerar que el valor esperado en el vacío del campo de Higgs hace que el vacío sea en realidad una "pradera cubierta de nieve", y que las partículas cuando se deslizan sobre ella lo hacen con una inercia (masa) que depende de $\lambda$.

# ### ¿Cómo aparece el campo de higgs?
# 
# ¿Cómo aparece un campo que tenga un valor en el vacío?
# 
# No sabemos por qué, pero se puede obtener si consideramos que un campo escalar $\phi(x)$ interacciona consigo mismo con un determinado potencial, que se llama, del sombrero mexicano:
# $$
# V(\phi) = \frac{\mu^2}{2} (\phi^\dagger \phi) + \frac{\lambda}{4} (\phi^\dagger \phi)^2
# $$
# 
# y desarrollando el campo escalar alrededor del mínimo, $v$, del potencial.
# 
# $$
# \phi(x) = \frac{1}{\sqrt{v}} \left( v + h(x)\right)
# $$

# Las siguientes figuras muestran el potencial del sombrero mexicano para el caso $\mu^2>0, \mu^2<0$ para un campo escalar real $\phi(x)$

# In[3]:


higgs_potential = lambda phi, mu, xlambda : ((mu**2).real * phi**2)/2 + (xlambda/4) * (phi**2)**2
phis = np.linspace(-10, 10, 100)
mu  = 6; xlambda = 1
plt.subplot(1, 2, 1); plt.plot(phis, higgs_potential(phis, mu, xlambda));
plt.subplot(1, 2, 2); plt.plot(phis, higgs_potential(phis, mu*(1j), xlambda));


# El el caso $\mu^2 >0$ el potencial tiene un mínimo en $v = 0$
# 
# Mientras que el caso $\mu^2 <0$ tiene mínimos en:
# 
# $$
# v = \pm  \sqrt{\frac{-\mu^2}{\lambda}}
# $$
# 
# a $v$ denomina **valor esperado en el vacío** (*vev*).
# 
# 

# Si consideramos un campo escalar complejo, obtenemos el potencial del sobrero
# 
# En el caso de que $\phi$ sea un complejo obtenemos la curva del potencial conocida como sobrero mexicano:
# 
# | | 
# | :--: |
# |  <img src="./imgs/sm_mexican_hat.png" width = 300 align="center"> |
# | potencial del campo complejo de Higgs $V(\phi)$ [Wikipedia]|
# 
# 
# La elección de un mínimo de los infinitos se conoce como **rotura espontánea de simetría**

# Para dotar de masa a los bosones $W^\pm, Z$ tendríamos que extender este mecanismo a las interacciones eletro-débiles.
# 
# Pero aún así quedan dudas por responder:
# 
#   * ¿Por qué esa interacción del campo de higgs consigo mismo¿ ¿Por qué el potencial del sobrero mexicano?
#   
#   * ¿Es el Higgs fundamental?

# 
# ## La teoría electrodébil en detalle
# 
# El Modelo Estandard se desarrolló en los 60's por A. Salam, S. Weinberg y S. Glashow.
# 
# Establece la unificación de las fuerzas débil y electromagnética en la fuerza electrodébil.
# 
# El modelo incluye para ser coherente las fuerzas electrodébil y fuerte.
# 
# Está basado en toerías gauge locales.

# El SM establece la invariancia respecto a U(1)$_Y$ SU(2)$_L$ SU(3)$_C$
# 
# Donde $Y$ es la hipercarga débil, $L$ es la quiralidad de los fermiones, y $C$ el color. El SM introduce para el grupo SU(2)$_L$ el isoespín débil, $I_W$. 
# 
# Consideraremos aquí solo la teoría electro-débil, sin el color, pero el tratamiento para éste es similar. 
# 
# Cada fermión tendrá asociada una carga de hypercarga, $Y$, e isospín débil, $I_w$.

# ### Hypercarga e isoespín débil.
# 
# El SM organiza las partículas dependiendo de su quiralidad.
# 
# Recordemos que todo espinor de Dirac se puede descomponer en parte a izquierdad y derechas de quiralidad.
# 
# Cada generación se agrupa en un duplete a izquierdad para leptones, esto es $I_W = 1/2$, y quarks y singletes a derechas, $I_W = 0$:
# 
# $$
# \begin{pmatrix} \nu_{eL} \\ e_L \end{pmatrix}, \, e_R; \;\;\;  
# \begin{pmatrix} u_L \\ d'_L \end{pmatrix}, \, u_R, d'_r. 
# $$
# 
# De igual forma para las otras dos generaciones.
# 
# ¡Notar que no existe el siglete a derechas para los neutrinos!

# Los dupletes tienen tercera componente de isoespín débil, $I^3_W$:
# 
# $$
# I^3_W(\nu_{eL}) = 1/2, \;\; I^3_W(e_L) = -1/2; \\
# I^3_W(u_L) = 1/2, \;\; I^3_W(d'_L) = -1/2 
# $$
# 
# Por ejemplo, para el $\nu_e$:
# 
# $$
# \frac{1}{2}\sigma ^2 \begin{pmatrix} \nu_{eL} \\ 0 \end{pmatrix} = \frac{1}{2} 
# \begin{pmatrix} 1 & 0 \\ 0  & -1 \end{pmatrix}
# \begin{pmatrix} \nu_{eL} \\ 0 \end{pmatrix} = 
# + \frac{1}{2} \begin{pmatrix} \nu_{eL} \\ 0 \end{pmatrix}
# $$
# 
# Mientras que los singletes no tienen tercera componente de isoespín débil:
# 
# $$
# I^3_W (e_R) = I^3_W(u_R) = I^3_W(d'_R) = 0
# $$

# Definimos al hypercarga débil, $Y_W$, como:
#     
# $$
# Y_W \equiv 2 (Q - I^3_W)
# $$
# 
# Lo que nos da
# 
# $$
# Y_W(\nu_{eL}) = -1, \;\; Y_W(e_L) = -1; \\
# Y_W(u_L) = 1/2, \;\; I^3_W(d'_L) = -1/2 
# $$
# 

# ### Interacciones débiles y electromagnéticas
# 
# 
# El campo del fermión debe ser invariante bajo transformaciones locales
# 
#    * una fase local, $\theta(x)$, del grupo U(1), relacionada con la hypercarga débil $Y$, cuyo acoplo denotamos por $g_Y$,
#    
#    * una triple fase local, $\alpha(x)_i$ con $i=1, 2, 3$ del grupo SU(2), asociada al isosespín débil SU(2)$_L$, cuyo acoplo es $g_W$.
# 
# $$
# \Psi'(x) = e^{ig' \frac{\theta(x)}{2}} \, e^{ig \bf{T} \cdot {\bf \alpha}(x)} \,\, \Psi(x)
# $$
# 
# donde ${\bf T}$ son los generadores de SU(2), esto es la matrices de Pauli, $T_i = \frac{\sigma_i}{2}$ con $i=1, 2, 3$. 
# 
# El factor $1/2$ en el primer exponente es por conveniencia.

# 
# Imponer invariance gauge en este grupo equivale a extender la derivada covariante generalizada:
# 
# $$
# D_\mu \equiv \partial_\mu - \frac{g'}{2} B_\mu - g \frac{\sigma}{2} \cdot {\bf W}_\mu
# $$
# 
# Donde:
# 
#   * $B_\mu$ es el campo vectorial asociado al grupo U(1)
# 
#   * ${\bf W}_\mu$ son tres campos vectoriales, $W^i_\mu$, con $i=1, 2, 3$, asociados al grupo SU(2)

# ### Inciso
# 
# La siguiente tabla muestra los generadores, los campos y el acoplo de los grupos de simetría de SM. 
# 
# 
# 
# |      |   hypercarga débil   | ---- isoespín débil ---- |   ----- color  ----- |
# | :--: | :--:                 | :--:           | :--:     |
# | grupo| U$_Y$(1) | SU(2)$_L$ | SU(3) |
# | generadores | $\frac{1}{2}$ | $T^i = \frac{1}{2} \sigma^i$ | $\frac{1}{2}\lambda^j$ |
# | mediadores | $B_\mu$ | $W^i_\mu$| $a^j_\mu$ |
# | acoplos    | $g'$ | $g$| $g_S$ |
# 
# donde $\sigma^i$ son las tres matrices de Pauli y $\lambda^j$ las 8 matrices de Gellman.
# 
# $g_s$ es el acoplo fuerte y $a^j_\mu$ los campos vectoriales de los ocho gluones, $j = 1, \dots, 8$.
# 

# La derivada covariante generalizada nos dara lugar a la interacción de cuatro corrientes que se acopplan con los cuatro campos vectoriales $B_\mu, {\bf W}_\mu$:
# 
# 
# $$
# (j^\mu_{1W} W^1_\mu + j^\mu_{2W} W^2_\mu + j^\mu_{3W} W^3_\mu) +  j^\mu_Y B_\mu
# $$
# 
# Donde (con $i=1, 2, 3$):
# 
# $$
# j^\mu_{iW} = g_W \bar{\Psi} \frac{\sigma_i}{2} \gamma^\mu \Psi, \;\;\; j^\mu_Y = \frac{g_Y}{2} \bar{\Psi} \gamma^\mu \Psi
# $$
# 
# Vamos a relacionar estar corrientes con las corrientes físicas: la del electromagnetismo y la de las corrientes débiles.
# 
# 

# 
# Si reescribimos los campos, y las corrientes:
# 
# $$
# W^\pm_\mu = \frac{1}{\sqrt{2}}(W^1_\mu \mp i W^2_\mu), \;\; j^\mu_{\pm} = \frac{g_W}{\sqrt{2}} (j^\mu_1 \pm i j^\mu_2)
# $$
# 
# Obtenemos:
# 
# $$
#  \left(j^\mu_+ W^+_\mu + j^\mu_- W^-_\mu \right) + j^\mu_3 W^3_\mu +  j^\mu_Y B_\mu
# $$
# 
# 
# Donde los dos primeros sumandos corresponden a la interacción de las corrientes débiles cargadas con los bosones $W^\pm$.

# 
# Podemos reescribir explícitamente las corrientes débiles cargadas:
# 
# $$
# j^\mu_{\pm} = \frac{g_W}{\sqrt{2}} \bar{\Psi} \sigma^\pm \Psi
# $$
# con:
# $$
# \sigma^\pm = \frac{\sigma_1}{2} \pm i \frac{\sigma_2}{2}, \;\;
# \sigma^+ = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \;\;
# \sigma^- = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}
# $$

# Para más claridad escribimos expresamente la corriente positiva, la que se acopla a $W^+$, para los leptones de la primera familia:
# 
# $$
#  j^\mu_+ = \begin{pmatrix} \bar{u}(\nu_e)_{L}, \bar{u}(e)_L \end{pmatrix} \, \gamma^\mu \frac{1}{2}\left(\sigma^1 + i \sigma^2 \right)  \, \begin{pmatrix} \bar{u}(\nu_e)_{L} \\ \bar{u}(e)_L \end{pmatrix} = \\
# \begin{pmatrix} \bar{u}(\nu_e)_{L}, \bar{u}(e)_L \end{pmatrix} \, \gamma^\mu \sigma^+  \, \begin{pmatrix} \bar{u}(\nu_e)_{L} \\ \bar{u}(e)_L \end{pmatrix} = 
#  \begin{pmatrix} \bar{u}(\nu_e)_L, \bar{u}(e)_L \end{pmatrix} \, \gamma^\mu \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} \bar{u}(\nu_e)_L \\ \bar{e}_L \end{pmatrix} = \\
#  \bar{u}(\nu_e)_L \, \gamma^\mu \, u(e)_L = 
#  \bar{u}(\nu_e) \, \gamma^\mu \frac{1}{2} (I - \gamma^5) \, u(e)
# $$
# 
# donde $u(\nu_e), u(e)$ son los espinores de Dirac del $\nu_e$ y $e$ respectivamente.
# 
# Y de igual forma procederíamos para $j^\mu_-$

# También por claridad podemos expresar la corriente $j^\mu_3$:
#     
# $$
# j^\mu_3 =  \begin{pmatrix} \bar{u}(\nu_e)_{L}, \bar{u}(e)_L \end{pmatrix} \, \gamma^\mu \frac{1}{2} \sigma^2  \, \begin{pmatrix} u(\nu_e)_{L} \\ u(e)_L \end{pmatrix} = \\
# \begin{pmatrix} \bar{u}(\nu_e)_L, \bar{u}(e)_L \end{pmatrix} \, \gamma^\mu \frac{1}{2}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \begin{pmatrix} u(\nu_e)_L \\ u(e)_L) \end{pmatrix} = \\
# \frac{1}{2} \bar{u}(\nu_e)_L \gamma^\mu u(\nu_e)_L - \frac{1}{2} \bar{u}(e)_L \gamma^\mu u(e)_L = \\
# I^3_W(\nu_{eL}) \,\,  \bar{u}(\nu_e)_L \gamma^\mu u(\nu_e)_L + I^3_W(e_L) \,\, \bar{u}(e)_L \gamma^\mu u(e)_L
# $$
#     
# Notar que los factores en cada sumando corresponde al valor de su tercera componente de isospín débil, esto es: $I^3_W(\nu_{eL}) = 1/2, \; I^3_W(e_L) = -1/2$.

# #### Las corrientes neutras y electromagnéticas
# 
# Para recuperar la interacción de las corrientes electromagnéticas con el fotón, la teoría electrodébil establece que existe una rotación entre los campos $B_\mu, W^3_\mu$ y los campos físicos $A_\mu, Z_\mu$. Donde $A$ es el fotón y $Z$ el bosón $Z$ neutro.
# 
# $$
# A_\mu = B_\mu \cos \theta_W  + W^3_\mu \sin \theta_W +\\
# Z_\mu = -B_\mu \sin \theta_W + W^3_\mu \cos \theta_W
# $$
# 
# donde $\theta_W$ se denomina ángulo de Weinberg.
# 
# 

# Si reescribimos los dos sumandos anteriores
# 
# $$
# j^\mu_3 W^3_\mu + j^\mu_Y B_\mu =  \\
# ( \sin \theta_W j^\mu_3 +  \cos \theta_W j^\mu_Y) \, A_\mu + 
# (\cos \theta_W j^\mu_3 - \sin \theta_W j^\mu_Y) \, Z_\mu 
# $$
# 
# En este caso, el primer sumando, corresponde a las interacciones electromagnéticas, donde la intensidad del acoplo es $e$ y $Q(f)$, la carga en unidades de $e$ del fermión, por ejemplo $Q(e) = -1$ para el electrón.
# 
# $$
# j^\mu_{EL} = Q(f) \, e \, \Psi \gamma^\mu \Psi
# $$
# 

# En el caso de los neutrinos, que no interacciona electromagnéticamente, su corriente EL es nula:
# 
# $$
# \left[ I^3_W(\nu_{eL}) \, g_W \sin \theta_W + Y_W(\nu_{eL}) \, \frac{g_Y}{2} \cos \theta_W  \right] \, \bar{u}(\nu_{e})_L \gamma^\mu u(\nu_e)_L = 0
# $$
# 
# donde $I^3_W(\nu_{eL})$ es la tercera componente de isospín y $Y_W(\nu_{eL})$ la hipercarga débil del $\nu_{eL}$. Ambos son las cargas del $\nu_{eL}$ respecto a la tercerca componente de isospín e hypercarga débil.
# 
# Como: 
# $$
# I^3_W(\nu_{eL}) = 1/2, \;\;\; Y_W(\nu_{eL}) = 2 \, (Q(\nu_{eL}) - I^3_W(\nu_{eL})) = 2 \, (0 -1/2) = -1
# $$
# 
# dado que la carga del neutrino es nula, $Q(\nu_{eL}) = 0$, obtenemos la relación:
# 
# $$
# -g_W \sin \theta_W + g_Y \cos \theta_W = 0, \;\;\; \tan \theta_W = \frac{g_Y}{g_W}
# $$
# 

# Las corrientes que se acoplan al fotón y al $Z$ se simplifican:
#     
# $$
# \sin \theta_W (j^\mu_3 + \frac{1}{2} j^\mu_Y) \, A_\mu + \frac{1}{\cos \theta_W} (j^\mu_3 \cos^2 \theta_W- \sin^2 \theta_W j^\mu_Y) \, Z_\mu
# $$
# 
# Dada la definición que hemos hecho de hypercarga $Y = 2(Q-I^3_W)$, vemos que efectivamente el primer sumando para un fermión, f, que tiene carga eléctrica Q:
# 
# $$
# Q  \, g_W \sin \theta_W \,\, \bar{\Psi} \gamma^\mu \Psi  = Q \, e \, \bar{\Psi} \, \gamma^\mu \Psi
# $$
# 
# Se cumple la relación:
# 
# $$
# g_W \sin \theta_W = e = \sqrt{4 \pi \alpha}
# $$
# 
# donde $e$ es la carga eléctrica, y $\alpha$ la constante de estructura fina.

# Si atendemos a la interacción con el $Z$, si sustituimos $j^\mu_Y = 2 (j^\mu_Q - j^\mu_3)$, obtenemos:
# 
# $$
# \frac{g_W}{\cos \theta_W} (j^\mu_3 - Q \, \sin^2 \theta_W j^\mu_Q) \, Z_\mu
# $$
# 
# Sea por ejemplo el neutrino como su carga es nula $Q(\nu_{eL}) = 0$ y su tercera componente de isoespín débil, $I^3_W(\nu_{eL}) = 1/2$, obtenemos:
# 
# $$
# \frac{g_W}{\cos \theta_W} \frac{1}{2} \; \bar{u}(\nu_e)_L \gamma^\mu u(\nu_e)_L \, Z_\mu 
# $$
# 
# Podemos identificar la constante de acoplo $g_Z$:
# 
# $$
# g_Z = \frac{g_W}{ \cos \theta_W}
# $$
# 

# De forma general para un fermión obtendremos para sus componentes a izquierdas y derechas:
# 
# $$
# g_Z \left[ \left( I^3_W - Q \sin^2 \theta_W \right) \bar{u}_L \gamma^\mu u_L - Q \sin^2 \theta_W \bar{u}_R \gamma^\mu u_R \right] 
# $$
# 
# Donde $I^3_W$ es la tercera componente de isoespín débil del fermión a izquierdas y $Q$ la carga del fermión en unidades de $e$.

# 
# Que podemos reescribir:
# 
# $$
# g_Z \left[ \left( I^3_W - Q \sin^2 \theta_W \right) \bar{u} \gamma^\mu \frac{1}{2} (I - \gamma^5) u - Q \sin^2 \theta_W \bar{u} \gamma^\mu \frac{1}{2}(I + \gamma^5) u \right] = \\
# g_Z \; \bar{u} \gamma^\mu \left[ \left( I^3_W - 2 Q(f) \sin^2 \theta_W \right) I - I^3_W \gamma^5 \right] u
# $$
# 
# Donde identificamos los factores vectorial y axial de las interacciones neutras de un fermión:
# 
# $$
# c_V = I^3_W - 2 Q \sin^2 \theta_W, \;\;\; c_A = I^3_W
# $$

# La siguiente tabla indica el valor $I^3_W, Q, c_V, c_A$ de los fermiones de la primera generación:
# 
# | fermión | --- $I^3_W$ --- | --- $Q$ --- | --------- $c_V$ --------- | --- $c_A$ --- |
# | :--:    | :--:    | :--: | :--:  | :--: |
# | $\nu_e$ | 1/2     | 0   | 1/2     | 1/2  |
# | $e$     | -1/2    | -1  | $-1/2 + s^2$ | -1/2|
# | $u$     | 1/2     | +2/3  | $1/2 - (2/3) s^2$ | 1/2 |
# | $d$     | -1/2    | -1/3 | $-1/2 + (1/3)s^2$ | -1/2|
# 
# 
# donde $s^2_W = \sin^2 \theta_W$

# y para los antifermiones:
# 
# | antifermión | --- $I^3_W$ --- | --- $Q$ --- | ---------- $c_V$ ---------- | --- $c_A$ --- |
# | :--:    | :--:    | :--: | :--:  | :--: |
# | $e^+$ | 1/2     | +1   | $1/2 - s^2$      | 1/2  |
# | $\bar{\nu}_e$  | -1/2  | 0  | -1/2 | -1/2|
# | $\bar{d}$     | 1/2     | +1/3  | $1/2 - (1/3) s^2$ | 1/2 |
# | $\bar{u}$     | -1/2    | -2/3 | $-1/2 + (2/3) s^2$ | -1/2|

# ### Las corrientes físicas: electromagnéticas, y débiles cargadas y neutras
# 
# Recordemos pues la **relaciones entre las constantes de acoplo**:
#     
# $$
# e = g_W \sin \theta_W = g_Z \sin \theta_W \cos \theta_W
# $$
#     
# Esto es $g_W, g_Z$ vienen dadas por $e$ y el ángulo de Weinberg $\theta_W$.
# 
# Le valor $\theta_W \simeq 33^o$, se ha determinado de numerosas maneras, (ver después)
# 
# $$
# \sin^2 \theta_W = 0.23146 \pm 0.000012
# $$

# La parte electródebil del SM, en particular la interacción con el $Z$, queda determinada por tres parámetros:
#     
# $$
# e, \sin \theta_W, M_W
# $$
#     
# y experimentalmente por:
#     
# $$
# \alpha, G_F, M_Z
# $$
# 
# la constante de estructura fina, $\alpha$, la constante de Fermi, $G_F$ y la masa del bosón, $M_Z$

# #### Las corrientes electromagnéticas y el vértice del fotón
# 
# 
# Las corrientes electromagnéticas no cambian la carga, ni el sabor y conservan paridad.
# 
# | |
# |:--:|
# |<img src="./imgs/sm_emcur.png" width = 250 align="center">|
# | Corriente electromagnética|
# 
# En el vértice con un fotón, introducimos (en la reglas de Feynman) la constante de acoplo $e$ y la carga, $Q$ del fermión:
# 
# $$
# A: \;\;\; Q e \, \bar{u} \gamma^\mu u = Q e \, \left( \bar{u}_L \gamma^\mu u_L + \bar{u}_R \gamma^\mu u_R \right)
# $$

# #### Las corrientes cargadas y el vértice del $W$
# 
# Las corrientes cargadas cambian el sabor, la carga en una unidad, y la tercerca componente de isoespín.
# 
# En las corrientes mediadas por $W^+$ suben una unidad $I^3_W$, y $W^-$ lo bajan.
# 
# Solo intervienen los estados de quiralidad a izquierdas para los fermiones.
# 
# | |
# |:--:|
# |<img src="./imgs/sm_wcur.png" width = 350 align="center">|
# | Corriente cargada|

# 
# En el vértice con un $W^+$ introducimos la constante de acoplo $g_W/\sqrt{2}$ entre dos fermiones de arriba, $f_\uparrow$ y abajo $f_\downarrow$ del duplete de quiralidad a izquierdas:
# 
# $$
# W^+: \;\; \frac{g_W}{\sqrt{2}} \, \bar{u}(f_\uparrow)_L \gamma^\mu u(f_\downarrow)_L = \frac{g_W}{\sqrt{2}} \bar{u}(f_\uparrow) \gamma^\mu \frac{1}{2}(I-\gamma^5) u(f_\downarrow), \\
# W^-: \;\; \frac{g_W}{\sqrt{2}} \, \bar{u}(f_\downarrow)_L \gamma^\mu u(f_\uparrow)_L = \frac{g_W}{\sqrt{2}} \bar{u}(f_\downarrow) \gamma^\mu \frac{1}{2}(I-\gamma^5) u(f_\uparrow),
# $$
# 

# De otra forma podemos decir del vértice de las corrientes cargadas:
#     
#   * tiene la constante de acoplo $g_W/\sqrt{2}$
#   
#   * solo afectan a los espinores de izquierdas de los fermiones, o equivalente, aparece un factor $\frac{1}{2} (I-\gamma^5)$
#   
#   * suben o bajan la tercera componente o aparece como factor la matriz $\sigma^\pm = \frac{1}{2}(\sigma^1 \pm i \sigma^2)$ que media entre los estados de los dupletes

# Veámoslo de forma práctica para  el vértice de un $W^+$ con un $e$ y $\nu_e$:
# 
# $$
# \frac{g_W}{\sqrt{2}} \begin{pmatrix}\bar{u}(\nu_e), & 0 \end{pmatrix} \gamma^\mu \frac{1}{2} (I - \gamma^5) \sigma^+ 
# \begin{pmatrix} 0 \\ u(e)\end{pmatrix} = \frac{g_W}{\sqrt{2}} \bar{u}(\nu_e)_L \gamma^\mu u(e)_L 
# $$

# #### El vértice del Z
# 
# Las interacciones con el $Z$ no cambian el sabor de la partícula, ni tampoco su carga eléctrica, ni su tercerca componente de isoespín débil.
# 
# Pero cambia de forma no trivial la paridad.
# 
# El bosón $Z$ actua diferente para la quiralidad de izquierdas y de derechas, dependiendo del fermión.
# 
# | |
# |:--:|
# |<img src="./imgs/sm_zcur.png" width = 350 align="center">|
# | Corriente neutra|

# La corriente neutra con el bosón $Z$ tiene la forma:
# 
# $$
# Z: \;\;\; g_Z \left( c_L \, \bar{u}_L \gamma^\mu \bar{u}_L + c_R \, \bar{u}_R \gamma^\mu \bar{u}_R \right) = g_Z \,\bar{u} \frac{1}{2}\left(c_V I - c_A \gamma^5\right) u 
# $$
# 
# donde $g_Z$ es la constante de acoplo y $c_L, c_R$ los factores asociados a la parte de quiralidad.
# 
# O equivalente, dos factores $c_V, c_A$, vectorial y axial.

# En el vértice de Feynman de una iteracción con el bosón $Z$ intervendrán los factores:
#     
#    * la intensidad: $g_Z$
#    
#    * las parte de quiralidad a izquierdas: $c_L = I^3_W - Q \sin^2 \theta_W$, donde $Q$ es la carga eléctrica del fermión en unidades de $e$ e $I^3_W$ la tercerca componente de isoespín débil
#    
#    * la parte de quiralidad a derechas: $c_R = -Q \sin^2 \theta_W$

# Por ejemplo para el neutrino
# 
# No tiene parte a derechas, ni tampoco carga $Q(\nu)=0$, solo $I^3_W(\nu) = 1/2$
# 
# Su componente $c_L = 1/2$ y $c_R = 0$
# 
# la corriente neutra con el $Z$ será:
#     
# $$
# \frac{g_Z}{2} \bar{u}(\nu_e)_L \gamma^\mu u(\nu_e)_L 
# $$

# Mientras que para el electrón
# 
# Tiene carga $Q(e)=-1$ y su parte a izquierdas $I^3_W(e_L) = -1/2$. 
# 
# Por lo tanto $c_L = -\frac{1}{2} + \sin^2 \theta_W$ y $c_R = \sin^2 \theta_W$.
# 
# Así la corriente neutra con el $Z$ es:
#     
# $$
# g_Z \left(  \left(-\frac{1}{2} + \sin^2 \theta_W\right) \, \bar{u}_L \gamma^\mu u_L  + \sin^2 \theta_W \, \bar{u}_R \gamma^\mu u_R \right)
# $$

# De forma equivalente podemos dar las componentes vectorial y axial de las corrientes neutras:
#     
# $$
# c_V = I^3_W - Q \sin^2 \theta_W, \;\; c_A = I^3_W 
# $$
# 

# ### Proceso de aniquilación
# 
# En la aniquilación $e + e^+ \to \mu + \mu^+$ intervienen el $\gamma$ y el $Z$.
# 
# | |
# |:--:|
# |<img src="./imgs/sm_gz.png" width = 400 align="center">|
# | aniquilación $e+e^+ \to \mu + \mu^+$ mediada por el fotín (izda) o el $Z$ (derecha) |
# 
# Recordemos los propagadores:
# 
# $$
# -i\frac{g_{\mu\nu}}{q^2}, \;\; -i\frac{g_{\mu\nu}}{q^2 - m_Z^2},
# $$
# donde $m_Z = 91.2$ GeV es la masa del $Z$.
# 
# QED conserva paridad y las corrientes neutras no. 
# 

# La figura muestra la sección eficaz $\sigma(e+e^+ \to \mathrm{hadrons}$ con los datos de diversos experimentos y la curva teórica.
# 
# Se observa que QED, la contribución del fotón, es dominante a baja energía; y la resonancia del $Z$.
#                   
# | |
# |:--:|
# |<img src="./imgs/sm_sigma_eeqq.png" width = 400 align="center">|
# | $\sigma(e+e^+ \to \mathrm{hadrons})$ vs $\sqrt{s}$ de [MT16.2] [LEP-SLD]|
# 
# En la región donde $Z$ domine deben aparecer efectos de violación de paridad que dependerán de $\sin^2 \theta_W$
#                                                                                      

# Experimentalmente calculamos asimetrías. La asimetría *forward-backward*, $A_{FB}, $que depende de forma no trivial de $\sin^2 \theta_W$ y de $\sqrt{s}$, se calcula a partir del número de eventos $N_F$ en la que el $\mu$ va en dirección hacia delante (dada por el $e$) o *forward* y $N_B$ hacia atrás o *backward*:
# 
# $$
# A_{FB} = \frac{N_F - N_B}{N_F + N_B}
# $$
# 
# | |
# |:--:|
# |<img src="./imgs/sm_afb.png" width = 400 align="center">|
# | Asimetría *forward-backward* $e+e^+ \to \mu + \mu^+$ vs $\sqrt{s}$ de [DELPHI]|
# 
# 
# En la figura se muestran la asimetría, $A_{FB}$, medida en experimento DELPHI de LEP para $e+e^+ \to \mu+\mu^+$ y $e+e^+ \to \tau + \tau^+$ en función de $\sqrt{s}$ y la predicción del SM para un valor $\sin^2\theta_W \simeq 0.23$.

# *Nota adicional*
# 
# Notar que la interferencia en el elemento de Matriz, por el efecto de los propagadores, es proporcional a:
# 
# $$
# \frac{Q^2 \alpha }{q^2} + \frac{g^2_Z \beta}{q^2 - m^2_Z}
# $$
# 
# el sugundo sumando cambia de signo al pasar por el polo del $m_Z$. 
# 
# El factor $\beta$ dependerá de las 4 posibles combinaciones de helicidad.

# ## Anchura de desintegración del $Z$
# 
# El colisionador LEP $e+e^+$ que operó durante los 90's verificó con gran detalle las predicciones del SM, especialmente la física del Z.
# 
# Durante un periodo LEP operó a $\sqrt{s} = m_Z = 91$ GeV produciendo millones de $Z$.
# 
# Uno de los estudios más importantes de LEP es la medición de la anchura de desintegración de $Z$.

# ### Anchura de desintegración parcial
# 
# Sea la desintegración: $ Z \to f + \bar{f}$, a un par de fermión y antifermión.
# 
# La anchura de desintegración viene dada por:
# 
# $$
# \Gamma = \frac{p^*}{8 \pi s} \langle |M_{fi}|^2 \rangle
# $$
# 
# donde en este caso $\sqrt{s} = m_Z$ y $p^* = m_z/2$, si despreciamos la masa del fermión en comparación con la del $m_Z$

# El cálculo del $M_{fi}$, aunque no es complicado, escapa al nivel de este curso.
# 
# aunque ya sabemos que debe involucrar a la corriente neutra:
# 
# $$
# g_z \left( c_L \bar{u}_L \gamma^\mu v_L + c_R \bar{u}_R \gamma^\mu v_R \right)
# $$
# 
# Su valor es:
# 
# $$
# \langle |M_{fi}|^2 \rangle = \frac{2}{3} (c^2_L + c^2_R) g^2_Z m^2_Z
# $$
# 
# 

# La anchura parcial de desintegración $Z \to f + \bar{f}$ es:
# 
# $$
# \Gamma(Z \to f + \bar{f}) = \frac{m_Z}{16 \pi m^2_Z} \frac{2}{3} (c^2_L + c^2_R) g^2_Z m^2_Z = \frac{g^2_Z m_Z}{24 \pi} (c^2_L + c^2_R)
# $$

# Por ejemplo para $\nu_e$ como $c^2_L = 1/4$
# 
# $$
# \Gamma (Z \to \nu_e + \bar{\nu}_e) = \frac{g^2_Z m_Z}{96 \pi} = 166 \; \mathrm{MeV}
# $$

# *Cuestión*: Calcula las anchuras de desintegración parciales y totales de $Z$ y sus fracciones de desintegración.

# In[4]:


GF  = units.value("Fermi coupling constant")
s2t = units.value("weak mixing angle")
MW  = 80.34 # GeV 
MZ  = 91.19 # GeV
g2w = 8 * MW**2 * GF /np.sqrt(2)
g2z = g2w/(1-s2t) 
gamma_nu = g2z*MZ/(96*np.pi)
print('gw    {:5.4f}'.format(np.sqrt(g2w)))
print('gz    {:5.4f}'.format(np.sqrt(g2z)))
print('Gamma nue {:5.4f} GeV'.format(gamma_nu))


# In[5]:


nc   = 3 # 3 colors for the quarks
pars = {'e':(-0.5+s2t, s2t), 'nue': (0.5, 0), 'u' : (0.5-2*s2t/3, -2*s2t/3), 'd': (-0.5+s2t/3, s2t/3)}
gamma = lambda cl, cr: g2z*MZ/(24*np.pi)*(cl**2 + cr**2)
gammas = {}
for key in pars.keys():
    cl, cr = pars[key]
    gammai = gamma(cl, cr)
    gammai = nc * gammai if key in ('u', 'd') else gammai
    comment = 'Gamma ' + key + ' {:5.4f} GeV'.format(gammai) 
    print(comment)
    gammas[key] = gammai


# In[6]:


gamma_total = 3 * gammas['nue'] + 3 * gammas['e'] + 2 * gammas['u'] + 3 * gammas['d']
print('Total Gamma {:5.4f} GeV '.format(gamma_total))
for key in gammas.keys():
    #n  = 2 if key == 'u' else 3
    br = 100 * gammas[key]/gamma_total
    print('BR ' + key + ' {:3.1f} %'.format(br))


# ### Sección eficaz
# 
# Aunque no la calcularemos, vamos a presentar ahora los distintos elementos que se necesitan en la sección eficaz de $e+e^+ \to f + \bar{f}$ en el $Z$, donde recordemos la contribución del $\gamma$ es pequeña.
# 
# Sabemos que la sección eficaz viene dada por:
# 
# $$
# \sigma = \frac{1}{64 \pi^2 s} \frac{p^*_f}{p^*_i} \int_\Omega \langle |M_{fi}|^2 \rangle \, \mathrm{d}\Omega
# $$
# 
# Donde en este caso $p^*_i = p^*_f = m_Z/2$ si despreciamos las masas de los fermiones en comparación con $m_Z$.

# Al estar tan próximos del polo del $Z$ debemos incluir en el propagador la parte asociada a su anchura $\Gamma_Z$ que habitualmente despreciamos:
# 
# $$
# \frac{1}{q^2 - m^2_Z} \to \frac{1}{q^2 - (m_Z - i \Gamma_Z/2)^2} \simeq \frac{1}{q^2 - m^2_Z +i m_z\Gamma_Z}
# $$
# 
# que al elevar al cuadrado nos da la expresión:
# 
# $$
# P_Z(s) \equiv \left|\frac{1}{q^2 - m^2_Z +i m_z\Gamma_Z} \right|^2 = \frac{1}{(s-m^2_Z)^2 + m^2_Z \Gamma^2_Z} 
# $$

# #### Inciso: sobre la anchura y masa
# 
# El cálculo correcto se hace en TQC, aquí solo damos una aproximación:
# 
# Si una partícula, con masa $m$, es estable su función de ondas en reposo:
# 
# $$
# \Psi \propto e^{-imt}.
# $$
# 
# Si consideramos que es inestable con una anchura $\Gamma = 1/\tau$, modificamos la función de onda:
# 
# $$
# \Psi \propto e^{-imt} e ^{-\Gamma t/2}
# $$
# 
# Que es equivalente, a reemplazar:
# 
# $$
# m \to m - i \Gamma/2
# $$

# el proceso $e+e^+\to \mu+\mu^+$, mediado solo por el fotón, lo estudiamos ya (ver hadrones).
# 
# Las distintas combinaciones de helicidad son las mismas:
# 
# | | 
# | :--: |
# |  <img src="./imgs/hadrons_eemumu_helicities_2.png" width = 600 align="center"> |
# | Las posibles combinaciones de helicidad de $e^++e \to \mu^+ + \mu$ en el CM en el régimen relativista |
# 
# En el elemento de matriz cuando antes nos apareció $e^2$ ahora aparecerá $g^2_Z$ y los acoplos $c_L, c_R$ correspondientes.
# 
# Los $M_{fi}$ contienen los acoplos de las corrientes y las distribuciones angulares siguientes:
# 
# |      | | | | 
# | :--: | :--: | :--: | :--: |
# | $LR \to LR$ | $LR \to RL$ | $RL \to LR$ | $RL \to RL$ |
# | $g^2_Z s (1+\cos \theta)$| $g^2_z s (1-\cos \theta)$ | $g^2_Z s (1-\cos\theta)$ | $g^2_Z s (1+\cos\theta)$ | 
# | $c_L(e) \, c_L(\mu)$ |$c_L(e) \, c_R(\mu)$ | $c_R(e) \, c_L(\mu)$| $c_R(e) \, c_R(\mu)$ |
# 

# Para dar la sección eficaz calculamos el promedio de los cuatro elementos de matriz e integramos en el ángulo sólido.
# 
# No hacemos el cálculo, pero damos el resultado. 
# 
# La sección eficaz es:  
# 
# $$
# \sigma (e+e^+\to Z \to \mu+\mu^+)= \frac{1}{64\pi^2 s} \frac{1}{4}\frac{g^4_z s^2}{(s-m^2_Z)^2 + m^2_Z\Gamma^2_Z} 
# \frac{16\pi}{3} [c^2_L(e) + c^2_R(e)] [c^2_L(\mu) + c^2_R(\mu)] = \\
#  \frac{1}{48\pi} \frac{g^4_z s}{(s-m^2_Z)^2 + m^2_Z\Gamma^2_Z}  [c^2_L(e) + c^2_R(e)] [c^2_L(\mu) + c^2_R(\mu)]
# $$
# 
# En la segunda igualdad pueden identificarse los distintos factores que intervienen.

# Que podemos reescribir introduciendo las anchuras de desintegración parciales:
# 
# $$
# \Gamma_{ee} = \frac{g^2_z m_z}{24} [c^2_L(e) + c^2_R(e)] \\
# \Gamma_{\mu\mu} = \frac{g^2_z m_z}{24} [c^2_L(\mu) + c^2_R(\mu)] \\
# $$
# 
# Por lo tanto:
# 
# $$
# \sigma (e+e^+\to Z \to \mu+\mu^+)= \frac{12 \pi s}{m^2_Z} \frac{\Gamma_{ee} \Gamma_{\mu\mu}}{(s-m^2_Z)^2 + m^2_Z\Gamma^2_Z} 
# $$
# 

# Observamos:
#     
#   * Se trata de una función de tipo Breit-Wigner
#   
#   * En el polo $\sqrt{s} = m_z$ obtenemos el máximo de la sección eficaz:
#   
#   $$
#   \sigma_{\max} = \frac{12\pi}{m^2_Z} \frac{\Gamma_{ee}\Gamma_{\mu\mu}}{\Gamma^2_Z} 
#   $$
#   
#   * En los valores $\sqrt{s} = m_Z \pm \Gamma_Z/2$ la sección eficaz:
#   
#   $$
#   \sigma \left(\sqrt{s} = m_Z \pm \Gamma_Z/2 \right) = \frac{\sigma_{\max}}{2}
#   $$
#   
#   O lo que es lo mismo la anchura a mitad de altura, FWHM, es $\Gamma_Z$

# *Cuestión* Calcula el máximo de la sección eficaz para $e+e^+\to \mu+\mu^+$ y $e+e^+\to \mathrm{hadrones}$ y la curva se la sección eficaz para hadrones en el pico del $Z$.

# In[7]:


Gee = 0.0838 # GeV 
Gmm = 0.0838 # GeV
Guu = 0.2894 # GeV
Gdd = 0.3713 # GeV
Ghad = 2 * Guu + 3 * Gdd
mz  = 91.19  # GeV
Gz  =  2.45  # GeV
sigma_mumu = (12 * np.pi)/(mz**2) * (Gee * Gmm)/Gz**2
sigma_had  = (12 * np.pi)/(mz**2) * (Gee * Ghad)/Gz**2
hbarc  = 0.197 * units.femto # GeV m
barn =  1e-28 # m^2
sigma_to_barn = (hbarc**2) / barn
print('sigma max (ee->mumu) {:3.1f} nbarns '.format(sigma_to_barn * sigma_mumu / units.nano))
print('sigma max (ee->had) {:3.1f} nbarns '.format(sigma_to_barn * sigma_had /units.nano))


# In[8]:


def sigma(s, Gff = Gmm):
    return (12 * np.pi *s)/(mz**2) * (Gee*Gff/((s-mz**2)**2 + (mz*Gz)**2))
ss = np.linspace(mz-8, mz+8, 100)
plt.plot(ss, sigma_to_barn * sigma(ss*ss, Ghad) / units.nano); 
plt.grid(); plt.xlabel(r'$\sqrt{s}$ (GeV)'); plt.ylabel(r'$\sigma$ (nbarn)');


# ### Tres familias de neutrinos
# 
# Uno de los resultados más importantes de la era LEP es la determinación del número de neutrinos.
# 
# Se obtiene a partir de la anchura de desintegración, $\gamma_Z$, del $Z$.

# 
# | | 
# | :--: |
# |  <img src="./imgs/sm_zresonance.png" width = 350 align="center"> |
# | $\sigma(e+e^+ \to \mathrm{hadrons})$ vs $\sqrt{s}$ de los experimentos de LEP [PDG]|
# 
# La siguiente figura muestra la sección eficaz $\sigma(e+e^+ \to \mathrm{hadrons})$  en función de $\sqrt{s}$ obtenida por los experimentos de LEP: ALEPH, DELPHI, L3, OPAL. 

# 
# Se muestra sobreimpuesta la curva teórica cuando consideramos $N_\nu$ neutrinos.
# 
# Recordemos que el número de neutrinos, $N_\nu$, entra en la anchura de desintegración:
# 
# $$
# \Gamma_Z = 3 \Gamma_{ee} + 2 \Gamma_{uu} + 3 \Gamma_{dd} + N_\nu \, \Gamma_{\nu\nu}
# $$
# 
# A parti de la medida de $\Gamma_Z$ y del valor esperado $\Gamma_{\nu\nu}$ obtenemos:
# 
# $$
# N_\nu = 2.9840\pm 0.0082
# $$
# 
# por lo tanto, **existen tres familias de neutrinos**.

# En la figura, la vista transversal de varios eventos en ALEP, $e+e^+ \to Z$ y $Z \to$ hadrons, $e+^+, \mu+\mu^+, \tau+\tau^+$:
# 
# | | 
# | :--: |
# |  <img src="./imgs/sm_Zevents_ALEPH.png" width = 400 align="center"> |
# | $Z\to$ hadrons, $e+e^+$, $\mu+\mu^+$, $\tau + \tau^+$ [ALEPH]|

# ## El bosón de Higgs
# 
# La teoría electrodébil necesita para ser coherente del mecanismo de Higgs.
# 
# El modelo de Higgs da respuesta (quizás la única posible) a los siguientes problemas graves:
# 
#   * Los bosones débiles son masivos, sin embargo las teorías gauge existen bosones sin masa. ¿Cómo dotar de masa a los bosones $W^\pm, Z$?
# 
# 
#   * La simetría electrodébil, U(1)$_Y$ SU(2)$_L$, separa fermiones a derechas e izquierdas, $f_L, f_R$, y exige que no tengan masas.
# 
# 
#   * otras sección eficaz, $W^+ + W^- \to W^+ + W^-$, crece indefinidamente a no ser que un escalar con los acoplos apropiados la controle.
#   

# El mecanismo de Higgs permite resolver los tres problemas de una forma económica.
# 
# Es imposible discutir sobre el mecanismo de Higgs sin recurrir a TQC y el lagrangiano del SM.
# 
# Vamos a dar solo unas ideas que nos permitan entender su comportamiento y descubrimiento.

# ### El bosón Higgs y sus implicaciones
# 
# La forma más económica del mecanismo de Higgs es considerar la existencia de un campo escalar complejo, $\phi$, que interacciona consigo mismo con un potencial $V(\phi)$.
# 
# El campo es un doblete de U(1)$_Y$SU(2)$_L$:
# 
# $$
# \phi = \begin{pmatrix} \phi^+ \\ \phi^o \end{pmatrix}
# $$
# donde $\phi^0, \phi^+$ son campos complejos
# 
# El potencial $V(\Phi)$ se conoce como el potencial del sombrero mexicano
# 
# $$
# V(\phi) = \frac{\mu^2}{2} (\phi^\dagger \phi) + \frac{\lambda}{4} (\phi^\dagger \phi)^2
# $$

# La elección de unos de sus mínimos se denomina **rotura espontánea de simetría**.
# 
# Expresamos el campo de Higgs, $h(x)$ como una perturbación respecto el valor en el vacío: 
# 
# $$
# \langle \phi(x) \rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}  0 \\ v + h(x) \end{pmatrix}
# $$
# 
# Este elección se conoce como **gauge unitaria**.
# 
# La elección no cambia la física subyacente pero nos da lugar a los campos físicos conocidos.
# 
# *Nota adicional:* Matemáticamente de los 4 grados de libertad iniciales, $\phi$, quedan en uno, el bosón de Higgs, $h(x)$, los otros tres se trasladarán a las componentes logitudinales de los bosones $W^\pm, Z$.

# Al establecer invariancia gauge, el campo de Higgs da lugar a las interacciones:
# 
#  * entre los bosones débiles, $W^\pm, Z$ y el valor $v$ del vacío, que interpretamos como la **masa de los bosones $W^\pm, Z$**.
#  
#  * entre las dos componentes de quiralidad de los fermiones y el *vev*, lo que da lugar a las **masas de los fermiones**.
#  
#  * a interacciones entre el higgs, $h(x)$, y los bosones débiles, $W, Z$, y dar lugar a **desintegraciones del Higgs a los bosones vectoriales**
#  
#  * a interacciones entre el higgs, $h(x)$, y los fermiones, que a su vez da lugar a **desintegraciones a pares fermión-antifermión**
#  
# *nota adicional* Además el higgs se acopla consigo mismo en vértices de tres y cuatro ramas, y con los bosones vectorales débiles en vértices de cuatro ramas.

# ### Las masas
# 
# La fricción de los bosones y los fermiones con el *vev* da lugar a las masas:
# 
# | | 
# | :--: |
# |  <img src="./imgs/sm_vev_mass.png" width = 400 align="center"> |
# | acción del *vev* sobre los fermiones y los bosones débiles |
# 
# En la tabla se muestra la relación entre las masas se relacionan con el *vev* y las costantes de acoplo:
# 
# |  ---- fermión ---- |  ---- bosones $W^\pm$  -----  | ------ bosón $Z$ ------ | ----- Higgs ----- |
# | :--: | :--: | :--: | :--: |
# | $m_f = \frac{1}{\sqrt{2}}\lambda_f v $ | $m_W = \frac{1}{2} g_W v$| $ m_Z = \frac{1}{2} \frac{g_W}{\cos \theta_W} v$ | $m_H = 2 \lambda v^2$|

# Observamos:
#     
#    * Hay un parámetro del modelo $\lambda_f$, que se llama **acoplo de Yukawa**, para dar la masa de cada fermión y del vev $v$. ¡Hay tantos acoplos de Yukawa como fermiones cargados!
#     
#    * Que la masa de los bosones vectoriales, $m_W, m_Z$ está fijada por $v$ y los acoplos, $g_W, g_Z$.
#    
#    * Que la masa del Higgs depende de $v^2$ y $\lambda$. El parámetro $\lambda$ solo interviene en la masa del Higgs.
#     
# El mecanismo de Higgs establece una relación entre la masa de los bosones vectoriales:
# 
# $$
#  \cos \theta_W = \frac{m_W}{m_Z}
# $$

# A partir de los datos experimentales, $M_z, \sin^2 \theta_W$, el SM predice:
# 
# $$
# m_W \simeq 80.34 \;\; \mathrm{GeV}
# $$
# 
# Que se aproxima mucho al valor experimental.
# 
# Esta relación dio credibilidad al mecanismo de Higgs antes de que se el Higgs se descubriera.
# 
# *Nota adicional*: En la determinación de la masa debemos considerar también diagramas de más dimensiones, las correcciones que introducen son de tipo logaritmico, cuya dependencia principal está relacionada con la masa del $t$.
# 
# Estar correcciones predijeron que la masa del $t$ ~175 GeV. El top se descubrió con el Tevatron de Fermilab en 1995.

# In[9]:


s2t = units.value("weak mixing angle")
MW  = 80.378 # GeV 
MZ  = 91.1876 # GeV
cw = np.sqrt(1 - s2t)
mw_pred = MZ * cw
print(' W mass esperada a primer orde {:4.3f} GeV, medida {:4.3f} GeV'.format(mw_pred, MW))


# Finalmente el valor del $vev$ se puede determinar a partir de $g_W, m_W$, o simplemente de $G_F$:
# 
# $$
# v = \frac{2m_W}{g_W} = \sqrt{\frac{1}{\sqrt{2} G_F}} = 246 \; \mathrm{GeV}
# $$
# 

# ### La desintegración
# 
# La interacción del Higgs con los fermiones y bosones vectoriales determina sus fracciones de desintegración.
# 
# | | 
# | :--: |
# |  <img src="./imgs/sm_higgs_decay.png" width = 400 align="center"> |
# | Canales de desintegración principales del Higgs |
# 
# Notar que el Higgs se acopla a sus canales de desintegración proporcionalmente a la masa de éstos:
# 
# 
# |  ---- fermión ---- |  ---- bosones $W^\pm$  -----  | ------ bosón $Z$ ------ |
# | :--: | :--: | :--: |
# | $\frac{\lambda_f}{\sqrt{2}} = \frac{m_f}{v} $ | $g_Z m_W $| $ g_Z m_Z$ |
# 
# *Nota adicional*: El macanimos introduce vértices adicionales tripes, $HHH$, y cuadrupes, $HHVV, HHHH$, donde $H$ es el Higgs y $V$ el bosón vectorial débil. 

# Las fracciones de desintegracción de Higgs dependeran a qué canales pueda desintegrarse respecto a su masa.
# 
# Como el acoplo del Higgs es proporcional a la masa, aquellos canales permitidos con mayor masa son los favorecidos.
# 
# | --------- canal ------- | ----- $Br$ ----- |
# | :--          | --: | 
# | $H\to b + \bar{b}$ | 57.8 % |
# | $H \to W^+ + W^-$  | 21.6 % |
# | $H \to \tau + \tau^+$| 6.4 % |
# | $H \to g + g$         |  8.6 % |
# | $H \to c + \bar{c}$   | 2.9 % |
# | $H \to Z + Z^*$       | 2.7 % |
# | $H \to \gamma + \gamma$| 0.2 % |
# 
# En la tabla se muestran las fracciones de desintegración del Higgs para diversos canales en función de su masa.
# 
# *Nota adicional* Las desintegraciones $H\to \gamma + \gamma, H \to g + g$ ocurren a través de diagramas de lazo, por ejemplo, con el intercambio en triángulo de un $t$.

# ## Descubrimiento del bosón de Higgs
# 
# El bosón de Higgs se descubrió en 2012 en los experimentos ATLAS y CMS del LHC del CERN.
# 
# Con anterioridad los experimentos de LEP y los experimentos CDF y D0 de Fermilab habían constreñido el valor de su masa más ligera entre: $115-150$ GeV.
# 
# Los experimentos y el acelerador LHC se diseñaron en los 90's y su operación se inición a finales de los 2000's.
# 
# En el LHC colisionan $p+p$ a una energía $\sqrt{s}= 7- 14$ TeV y con una luminosidad $\mathcal{L}(t) = 10^{34}$ cm$^{-2}$. Hay $10^{11}$ protones por paquete y una frecuencia de colisión de 40 MHz (cada 25 ns).
# 
# El número de colisiones $10^{9}$ por segundo y por cada cruce hay como máximo $35$ colisiones, lo que de denomila *pile-up* (apilamiento).

# Las sección eficaz $p+p$ is del ~100 mbars en el LHC.
# 
# Las sección eficaz de producción del $H$ entre 20 pb y 60 pbs!
# 
# Los canales de desintegración dónde la señal es más fácil distingible del fondo son $H\to Z+Z^+$ y $H \to \gamma \gamma$.
# 
# Eso equivale a buscar una desintegración relevante entre $10^{13}$.

# @check NUMBERS

# ### Sobre ATLAS y CMS

# ### El descubrimiento del Higgs

# El Higgs se descubrión en dos canales limpios (donde los sucesos de contaminación son o afectan menos).
# 
#   * $H \to \gamma + \gamma$, buscando dos deposiciones en el calorímetro electromagnético, $\gamma$s.
#   
#   Se calcula la masa invariante de los $\gamma$s que provienen del punto de interacción.
# 
# 
#   * En el canal $H \to Z + Z^* \to (l+l^+) + (l'+l'^+)$, en cuatro leptones ($4l$), agrupados en dos parejas del mismo sabor y carga opuesta.
#   
#   Una pareja proviene de un $Z$ real, *on-shell*, y su masa invariantes es próxima al $Z$, y otra del $Z^*$ *off-shell*, que tiene menor masa invariante que el $Z$. 
# 

# La siguiente figura muestra la presenta la distribución de la masa invariante de $2 \gamma$ (izda) para el caso de CMS y de $(4 l)$ para el caso de ATLAS con la estadística de Run-I del LHC 
# 
# | | 
# | :--: |
# |  <img src="./imgs/sm_higgs_CMS_ATLAS.png" width = 600 align="center"> |
# | masa invariante para $H \to \gamma \gamma$ (izda, [CMS]) y $H \to Z+ Z^* \to 4 l$ (derecha, [ATLAS])|
# 

# Observamos:
#     
#    * La presencia de un pico en la masa invariante en ambas distribuciones a 125 GeV.
#    
#    * Ese pico es incompatible con el fondo, en combinación a ~6 desviaciones estandars. 
#   
# Con los datos del run-I del LHC se pudo concluir además:
#    
#    * que se trataba de una partícula escalar a partir de las distribuciones angulares de $(4l)$ en el canal $H\to 4 l$.
#    
#    * que el acoplo a otros canales $\tau\tau^+, b \bar{b}$ es el esperado por el SM

# En la figura dos eventos de ATLAS, identificados como $H \to \gamma\gamma$ (izda) y $H \to e+e^++\mu+\mu^+$
# 
# | | |
# | :--: | :--: |
# |  <img src="./imgs/sm_atlas_h2gammas_event.png" width = 320 align="center"> | <img src="./imgs/sm_atlas_h4leptons_event.png" width = 320 align="center"> |
# | $H \to \gamma + \gamma$ event | $H \to 4l$ event [ATLAS events](https://twiki.cern.ch/twiki/bin/view/AtlasPublic/EventDisplaysFromHiggsSearches) |

# ## Conclusions
# 
# El SM:
# 
#   * El SM describe con gran precisión las partículas y las interacciones que conocemos.
#   
#   * El SM se ha verificado en una gran cantidad de precesos físicos.
#   
#   * No obstante sabemos que los neutrinos tienen masa, lo que implica que debemos extender el SM.
#   
# Pero:
#   
#   * No incluye otra física conocida, como la materia oscura, ¿qué partículas la forman?
#   
#   * El modelo tiene cinco parámetros fundamentales $e, g_W, g_S, v, \lambda$, dos de ellos corresponden al Higgs. Experimentalmente $\alpha, G_L, \alpha_S, m_W, M_H$.
# 
#   * Tiene 9 parámetros adcionales, los acoplos de Yukawa; 4 más para acomodar la matriz CKM de los quarks, y 3 o 4 más para la PNMS de los leptones. 
# 

# ### cuestiones abiertas
# 
#  * ¿Es el neutrino su propia antipartícula? ¿Cómo se acopla al Higgs?
# 
#  * ¿Cuál es el origen de los valores de masas? ¿Por qué es tan pequeña la del neutrino?
# 
#  * ¿Por qué hay 3 familias? ¿Qué releación guardan, si la hay, la matriz CKM y PNMS?
# 
#  * ¿El el Higgs único? ¿Es el Higgs compuesto?
#  
#  * ¿Cuál es el origen del potecial del bosón de Higgs, $V(\Phi)$?
#  
#  * ¿Hasta qué escala de energía, $\Lambda$, es válido el SM? ¿A qué energía aparecerá nueva física?
# 

# ## Bibliografía
# 
# 
#   * [MT] Mark Tomsom, "Modern Particle Physics", Cambridge U. press, Temas 15, 16 y 17.
#   
#   * [AB] Alessandro Bettini, "Introduction to Elementary Particle Physcs", Cambridge U. press, Tema 9. 
#   
#   * [PK] M. Peskik, Lectures on the Theory of the Weak Interaction, SLAC–PUB–17142, (2017), [arXiv:1708.0943v1](https://arxiv.org/abs/1708.09043)
#   
#   * [PDG](https://pdg.lbl.gov/) Particle Data Group.
#   
#   * [LEP-SLD] LEP and SLD Collaborations. 2006. Phys. Rept., 427, 257–454.
#   
#   * [DELPHI] P. Abreu et al., DELPHI Collaboration, *Eur. Phys. J.* **C 11**, 383 (1999).
#   
#   * [ALEPH] D. Decamp et al., ALEPH Collaboration, *Z. Phys.* **C 48**, 365 (1990)
# 
#   * [CMS] V. Khachatryan et al., CMS Collaboration, *Eur. Phys. J.* **C 74**, no. 10, 3076 (2014) [arXiv:1407.0558](https://arxiv.org/abs/1407.0558) 
#   
#   * [ATLAS] G. Aad et al., ATLAS Collaboration, *Phys. Lett.* **B 726**, 88 (2013) Erratum: [Phys. Lett. B 734, 406 (2014)] [arXiv:1307.1427](https://arxiv.org/abs/1307.1427)

# In[10]:


Ejercicio


# In[7]:


sigma   = 60 * units.nano * 1e-24
lumi    = 1e32
t       = 1e7
Gz      = 2.45 # GeV
Gl      = 0.084 # GeV
Br      = 0.17
epsilon = 0.2
Ntau    = sigma * lumi * t * Gl /Gz
Nmu     = Ntau * (Br * epsilon)**2
print('Expected tau, tau+ events ', Ntau)
print('Observed mu, mu+   events ', Nmu)


# #### Partículas y Cosmología
# 
# La Física de Partículas también nos permite entender la evolución primigenia del Universo (Cosmología).
# 
#   A partir de los $10^{-32}$ s después del Big Bang el Universo se rigue por el Modelo Estardad, y depués de 100 s tiene lugar la Nucleoséntisis (Nuclear).
# 
# 
# <img src="./imgs/intro_particles_cosmology.jpg" width = 600 align="center">
# 

# Existe una relación de continuidad entre ramas de la física:
# 
# <center>
# Astropartículas $\Leftrightarrow$ Partículas $\Leftrightarrow$ Nuclear
# </center>
#  
# El *Instituto Galego de Fíxica de Altas Enerxías*, [IGFAE](https://igfae.usc.es/igfae/gl/), cubre las tres ramas y sus posibles aplicaciones.

# #### Teoría, experimentos, detectores
# 
# La Física de Partículas es una rama que involucra a la física teórica, experimental y la de detectores.
# 
#    * Los avances en detección hicieron posible nuevos discubrimientos (i.e detectores de silicio para el LHC)
# 
#    * Los descubrimientos marcan las líneas permitidas de las teorías (i.e. violación de Paridad)
# 
#    * La teoría indica experimentos de interés (i.e. descubrimiento de las corrientes neutras, el bosón de Higgs)
#    
#    
#  <center>
#    Detectores $\Leftrightarrow$ Experimentos $\Leftrightarrow$ Teorías
#  </center>
#  
#  y se relaciona con otras áreas:
#  
#  <center>
#     Ingeniería | Química | Computación
#  </center>

# #### La gran ciencia
# 
# Algunos de los experimentos y de los acelerados son grandes construcciones únicas, que requieren de la participación de centenares o miles de científicos y una gran financiación.
# 
# Las estructuras de los experimentos son complejos, su planificación, construcción y explotación duran décadas.
# 
# <img src="./imgs/intro_atlas.jpg" width = 500 align="center">
# 
# El experimento [ATLAS](https://atlas.cern/) del LHC es uno de los más grandes experimentos construidos. 
# 

# #### Grandes Laboratorios
# 
# Los experimentos de Física de Partículas tienen lugar en grandes laboratorios internacionales.
#   
# Basados en aceleradores:
#   
#    * [CERN](https://home.cern) (Ginebra, Suiza).
#      
#    * [Fermilab](https://www.fnal.gov/) (cerca de Chicago, Illinois), [SLAC](https://www6.slac.stanford.edu/) (Stanford, California), [Brookhaven](https://www.bnl.gov/world/) (New York)
#      
#    * [KEK](https://www.kek.jp/en/) (Japón)
#      
# Subterráneos:
#   
#    * [LNGS](https://www.lngs.infn.it/en) (Italia), [Kamioka](http://www-sk.icrr.u-tokyo.ac.jp/index-e.html) (Japon), [SNOLAB](https://www.snolab.ca/) (Canada), [Canfranc](https://lsc-canfranc.es/) (Spain).

# #### Aplicaciones prácticas
# 
# Las técnicas desarrolladas para los detectores y el volumen de tratamiento de datos han dado lugar a aplicaciones fundamentales:
# 
#    * Física Médica (rayos-X, PEP, terapia hadrónica)
#    
#    * World Wide Wed ([WWW](https://home.cern/science/computing/birth-web/short-history-web)) 

# ### History
# 
# La historia de Partículas es una carrera por alcanzar haces de más energía y detectores más precisos.
# 
# La Física de Partículas es una rama que involucra a la física teórica, experimental y la de detectores.
# 
# <img src="./imgs/intro_detector_experiment_theory.jpg" width = 500 align="center">

# <img src="./imgs/intro_particles_history.jpg" width = 800 align="center">

# @TODO: add fechas 
# 
# Los grandes desarrollos de detetectores y aceleradores se traducen en descubrimientos científicos.
# 
#   * Las emusionen permitieron la observación de partículas, $\mu, \pi$
# 
#   * Las centelleadores y los reactores nucleares el descubrimiento del neutrino $\nu_e$. 
#   
#   * Las cámaras de hilos el descubrimiento de los bosones $W$
#   
#   * Los detectores de silicio la determinación de partículas con vida corta $\tau, B$

# Los experimentos revelan una Naturaleza inesperada:
# 
#   * La violación de Paridad y CP en corrientes débiles.
#   
#   * Tres familias de leptones y de quarks.
#   
#   * La matrix de mezcla, CKM, que rigue los corrientes cargadas entre los quarks.
#   
#   * Oscilaciones de neutrinos.

# La construcción de la teoría guía a los descubrimientos:
# 
#   * Pauli postula la existencia del neutrino.
# 
#   * Gellman propone el modelo de quarks y las ordenación de 'zoo' de hadrones.
#     
#   * La unificaficación electro-debil predice las corrientes neutras de neutrinos.
#   
#   * La rotura espontánea de simetría postula la existencia del bosón de Higgs.

# Los avacen experimentales más relevantes:
#     
#   * El descubrimento del positron (antimateria)
#   
#   * El descubrimiento de partículas inesperadass: $\mu, ...$
#   
#   * El descubrimiento del neutrino y sus propiedades.
#   
#   * La violación de Paridad y de CP en corrientes débiles
#   
#   * La existencia de los quarks y los gluones
#   
#   * La existencia de las corrientes neutras y la verificación del SM
#   
#   * La verificación de la matrix de mezclas CKM
#   
#   * El descubrimiento de oscilaciones de neutrinos
#   
#   * La descubrimiento del bosón de Higgs

# Los avences tóricos más importantes:
# 
#   * La ecuación de Dirac y la existancia de la antimateria
#   
#   * El potencial de Yukawa
#   
#   * La teoría de Fermi, interacciones entre corrientes.
#   
#   * Los diagramas de Feymann (QED)
#   
#   * El modelo de quarks de Gellman and Zweig.
#   
#   * La unificación electro-débil, la modelo estandard (MS) de Glashow, Salam y Weinberg.
#   
#   * El mecanismo de Higgs y Englert.

# ### El modelo estandar
# 
# EL modelo estandar (SM) de Física de Partículas que clasifica las partículas elementales y establece sus interacciones a través de las fuerzas electromagnética, débil y fuerte.
# 
# El *Particle Data Group* ([PDG](https://pdg.lbl.gov/)) recopila toda la información relevantes sobre física de partículas. 

# <img src="./imgs/intro_SM_table.png" width = 600 align="center">

# Recordemos que la fuerzas son responsables entre otras:
#     
#    * electromagnética: de la estabilidad y complejidad de los átomos
#         
#    * fuerte: de la estabilidad y complejidad de los núcleos (protones y neutrones)
#         
#    * débil: de las desintegraciones $\beta$ nucleares.

# #### Fermiones
# 
# Los **fermiones** son las partículas fundamentales, tienen spin 1/2, y son los constituyentes básicos de la materia. 
# 
# Se dividen en:
# 
#   * **quarks** : si interaccional fuertemente
#   
#   * **leptons**: si no interaccionan fuertemente
#     
# | fuerte | electromagnética | débil | 
# |---      |:--       | :--          |
# | quarks | quarks y leptones cargados  | quarks y leptones |
# 
# Los neutrinos sólo sienten la fuerza débil.

# #### anti-partículas
# 
# Los fermiones se riguen por la ecuación de Dirac. Son espinores de Dirac. Por cada fermión existe un anti-fermión.
# 
# Una **antipartícula** tiene las cargas y los números cuánticos opuestos a su partícula (a excepción del spín y la masa que no cambian)
# 
# Si $f$ es un fermión, denotamos $\bar{f}$ como su anti-fermión,
# excepto para los leptones cargados, donde indicamos su anti-leptón con la carga positivos: $e^+, \mu^+, \tau^+$

# #### las tres generaciones de fermiones
# 
# Los fermiones (tanto leptones como quarks) se agrupan en **tres generaciones** o familias. 
# 
# Las tres generaciones de leptones se comportan con **universalidad** (de igual manera) frente a las interacciones si se tiene en cuenta que sus masas son diferentes. 
# 
# 
# En cada generación, tanto leptones como quarks, se agrupan en **dupletes** con posiciones **arriba** y **abajo**.
# 
# Su posición está asociada a la carga débil que se denomina **isoespín débil**.

# |         | posición |             | mass        |              | Q (e)           | fuerte | electromagnética | débil | 
# |---      |:--       | :--         |:--          | :--          | :--             | :--    |:--     |:--   |
# |         |          |      I       |    II      |       III            |           |  |  | | 
# |quarks   | up       |  u  (2 MeV) | c (1.2 GeV) | t (170 GeV)  | $\frac{2}{3}$   | sí | sí | sí |
# |         | down     |  d  (5 MeV)  | s (93 MeV)  | b (4.2 GeV)  |  $\frac{-1}{3}$ | sí | sí|  sí |
# |         |
# | leptons | up       | $\nu_e$ (< eV)| $\nu_\mu$ (< eV) | $\nu_\tau$ (<eV)   | 0  | no | no | sí |
# |         | down     | $e$   (511 keV)| $\mu$  (106 MeV)| $\tau$  (1.85 GeV) | -1 | no | sí | sí |
# 
# 
# La carga eléctrica depende de su posición en el duplete. 
# 
# Las masas son muy diferentes entre generaciones. La mayor diferencia se da entre el $e$ y su neutrino $\nu_e$.
# 
# El electrón es 2000 veces más ligero que un protón. Del neutrino desconocemos su masa, pero es al menos $5 \times 10^5$ veces más pequeña que el electrón.
# 
# @todo hacer plot de scala de masas y puntos con labels

# #### Los hadrones.
# 
# Los leptones se presentan aislados en la naturaleza.
# 
# Los quarks no. Siempre están en particulas compuestas con carga eléctrica entera.
# 
# La fuerza fuerze describe esta realidad con la propiedad de **confinamiento**: los quarks no peuden ser libres.
# 
# La carga asociada a la fuerza fuerte es el **color** que tiene tres posibilidades: RGB (red, green, blue). Y por supuesto no tiene nada que ver con los colores habituales. Los hadrones no tiene color global (son *blancos*).
# 

# 
# Las partículas con varios quarks se llaman **hadrones**.
# 
# Los hadrones formados con tres quarks se llaman **bariones**. 
# 
# Los hadrones formados por un quark y un anti-quark se llaman **mesones**.
# 
# 
# <img src="./imgs/intro_hadrons.png" width = 300 align="center">
# 

# El protón y el neutrón son bariones.
# 
# El protón está formado por el trío $uud$ y el neutrón por $udd$.
# 
# Existe un zoo de hadrones, con propiedades varias, dependiendo de sus masas, carga y composición. 
# 
# los piones, $\pi^\pm, \pi^0$, son los mesones más ligeros y están formado por combinaciones de quarks y antiquarks $ud$
# 
# los kaones, $K^{\pm}, K^0$, son mesones que contienen un quark extraño $s$ y otro quark $u$ o $d$.
# 

# #### Estabilidad.
# 
# Las generaciones II y III son inestables, y se desintegran via la interación fuertes, débiles o electromagnéticas, dependiendo de la partícula y sus propiedades, hasta partículas compuestas por la generación I, neutrinos o fotones.
# 
# Los leptones cargados se desintegran débilmente, sus vidas son (s):
# 
# 
# | $e$ | $\mu$ | $\tau$|
# | :------------| :------------- | :--------------- |
# | estable | $2.6 \, 10^{-6}$| $2.9 \, 10^{-13}$|
# 
# 
# Los leptones estables son el $e$ y los neutrinos.
# 

# Los hadrones libres se desintegran, dependiendo de la interacción que tienen permitida.
# 
# Vidas de algunos hadrones (s):
# 
# 
# | $p$ | $n$ | $\pi^0$| $\pi^\pm$ | $K^\pm$| 
# | :--- | :--- | :--- | :-- | :-- | 
# | estable | $900$| $8.4 \, 10^{-17}$| $2.6 \, 10^{-8}$| $1.23 \, 10^{-8}$ |
# 
# La única partícula hadrónica libre estable es el protón.
# 
# El Universo está formado casi en totalidad por protones, neutrones, electrones y neutrinos. Esto es (𝑢,𝑑),(𝜈,𝑒)
# 
# 
# Por eso decimos que el Universo es de 'baja energía'.
# 
# El átomo está compuesto de un núcleo con protones (que a su vez tiene un trio de quarks 𝑢𝑢𝑑
# ) y neutrones (𝑢𝑑𝑑); y electrones.
# 
# @todo plot con las vidas medias de algunas partículas.

# #### Creación de partículas
# 
# Para crear una partícula necesitamos al menos una energía equivalente a la masa de la nueva partícula. $E = m c^2$.
# 
# Un par fermión-antifermión se puede aniquilar (mediante el intercambio de bosones neutros) para producir otro par de fermión-antifermión. $(e^- + e^- \to \mu^- + \mu^+)$.
# 
# Siempre que producimos una anti-fermión producimos también su fermión. Siempre que creamos anti-materia creamos materia.
# 
# Existen dos tipos principales de experimentos de creacción de partículas:
# 
#   * colisionador: haces de $e, e^+, p, \bar{p}$ colisionan entre sí, por ejemplo [LEP](https://es.wikipedia.org/wiki/Large_Electron-Positron_collider) ($e^+e^-$)
# 
#   * blanco fijo (*fixed target*): haces de $e, p, \mu, \nu$ contra un material (átomos, núcleos). Experimentos de neutrinos, por ejemplo el experimento [Gargamelle](https://en.wikipedia.org/wiki/Gargamelle).
# 
#  

# #### Colisionadores
# 
# Colisionadores recientes:
# 
# | collider | Laboratory | Type       | Period    | Energy, $\sqrt{s}$, (GeV) | Luminosity, $\mathcal{L}$, ($cm^{-2}s^{-1})$ |
# | :--      | :--        | :--        | :--       | :--              | :-- |
# | PEP-II   | SLAC       | $e^+e^-$   | 1999-2009 | 10.5             | $1.2 \, 10^{34}$|
# | LEP      | CERN       | $e^+e^-$   | 1989-2000 | 90-209           |  $10^{32}$ |
# | Tevatron | Fermilab   | $p\bar{p}$ | 1987-2012 | 1960             | $4 \, 10^{32}$ |
# | LHC      | CERN       | $p p$      | 2009-     | 14000            | $10^{34}$| 

# Los dos parámetros fundamentales de un colisionador son:
#     
#  * **Energía** en el centro de masas, $\sqrt{s}$, que limita la masa de las partículas que podemos crear.
# 
#  * **Luminosidad**, o el número de partículas que lanzamos entre ellas por unidad de área y tiempo.
# Delimita el número de interacciones interesantes en cada cruze. 
# 
# 

# 
# Para dos vagones con $n_1, n_2$ partículas que se cruzan entre sí en la dirección $z$ con una frecuencia $f$ y que tienen una sección transversal gausiana $\sigma_x, \sigma_y$, la luminosidad es:
#  
#  $$
#  \mathcal{L} = f \frac{n_1 n_2}{ 4 \pi \sigma_x \sigma_y}
#  $$
#  
#  Si un tipo de interacción tiene una sección eficaz $\sigma$, esperamos $n = \sigma \mathcal{L}$ interaciones por cruze. Recordar que la sección eficaz $\sigma$ se mide en $m^2$ o en unidades naturales barns (1 barn = $10^{-24} cm^2$.
# 
#  Y a lo largo de un tiempo (típicamente años), $N = \int_t \sigma \mathcal{L}(t) \, \mathrm{d}t$. 
#  
#  Donde $\mathcal{L}(t)$ es la luminosidad instantanea. A la canditad $\int_t \mathcal{L}(t) \, \mathrm{d}t$ la llamamos luminosidad integrada.
# 
#  A la interación por cruce se las conoce como **evento**.
# 

# #### Alguna preguntas sin respuesta:
# 
# * ¿Son los quarks y leptones elementales?
# 
# * ¿Por qué hay 3 generaciones?
# 
# * ¿Hay un patrón de masas de los fermiones?
# 
# * ¿Por qué hay tanta disparidad de masas entre el neutrino y el resto de fermiones?
# 
# * ¿Por qué el Universo está formado muy primordialmente por materia y no anti-materia?
# 
# * ¿Por qué si el protón es compuesto su carga eléctrica es exactamente igual y opuesta a la del electrón?

# ##### Propiedades de los portadores
# 
# La interacción se describe entre dos fermiones mediante el intercambia de otra partícula (virtual) que es el portador.
# 
# Cada portador interviene según la carga (eléctrica, isospín débil, color) del fermión al que se acopla.
# 
# Los portadores son bosones vectoriales (spín 1), son los trasmisores de la fuerza.
# 
# 
# La intensidad approximada para dos partículas a una distancia de $10^{-15}$ m. 
# 
# | fuerza            | intensidad | bosón            | Q (e) | mass (GeV) |
# | :--               | :--        | :--              | :--   | :--     | 
# | fuerte            | 1          | $g$ (gluón)      | 0     | 0       | 
# | electromagnetismo | $10^{-3}$  | $\gamma$ (fotón) | 0     | 0       | 
# | débil             | $10^{-8}$  | $W^{\pm}$        | $\pm1$  | 80.4 | 
# |                   |           | $Z^0$              | 0  | 91.2 |
# | gravedad          | $10^{-37}$| 

# La masa del bosón determina el rango (en distancia) de la interacción. El rango de la interacciones débiles es de $10^{-15}$ m. 
# 
# El rango de $100$ GeV, aproximadamente la masa del bosón $Z$, se conoce como **escala electrodébil.**
# 
# Los bosones $W^\pm$ son cargados. Sus interacciones se conocen como **corrientes cargadas** (CC).
# 
# Las interacciones del bosón $Z^0$ se denominan también **corrientes neutras** (NC). 
# 
# Existen 8 typos de gluones, que cambian el color de los quarks, son electricamente neutros y de masa nula. 
# 
# Los quarks y gluones exiben la propiedad de confinamiento y no pueden observarse aislados. Es por ello que a pesar de su masa nula, el rango de acción de la fuerza fuerte es también de $10^{-15}$ m.

# ### Observables
# 
# En física de partículas habitualmente estimamos:
# 
#   * La sección eficaz, $\sigma$, de una interacción (barns).
#   
#   * El tiempo de vida, $\tau$, (s) o su inverso, la anchura de desintegración, $\Gamma$, (GeV).
#   
#   * La fracción de desintegración, $\mathcal{BR}$, en un canal (%) (por ejemplo: $\tau \to \mu \bar{\nu}_\mu \bar{\nu}_\tau$)
#   
#  
# 
# 
# 

# #### Elementos de la interacción.
# 
# La ideas fundamentales sobre la interacción entre partículas:
# 
#   * se produce mediante un portador o mediador (Yukawa).
#   
#   * tiene lugar entre dos corrientes de fermiones (Fermi).
#   
#   * Regla de Oro de Fermi. La probabilidad de la interacción es proporcional a $|M_{if}|^2$, donde $M_{if}$ es la matriz de transición entre los estados iniciales y finales.
#   

# #### Diagrammas de Feynman
# 
# Son una representación gráfica de la interacción y proporcional una reglas de cálculo de la matriz de transición.
# 
# La derivación se obtiene de teoría cuántica de campos (QFT)
# 

# 
# En los diagramas de Feynman:
#     
# <img src="./imgs/feynman_default.png" width = 300 align="center">
# 
#    * partículas iniciales (a, b) y finales (c,d), el portador de la interacción (X), vértices de interacción, con itensidad dada por la **constante de acoplo** $g$. El eje temporal es el horizontal.
# 
#    * Dado que no observamos $X$ lo colocamos verticalmente porque no sabemos su ordenamiento en el tiempo.

# #### Vértices con las interacciones
# 
# <img src="./imgs/feynman_forces.png" width = 500 align="center">
# 
# 
# En los vértices se conserva la carga eléctrica y el número total de partículas + antipartículas.
# 
# Las interacciones no cambian la identidad del fermión a excepción de la corriente débil cargada (mediada por $W^\pm$) que cambia la identidad, cambia el **sabor**.
# 
# Cada interacción tiene una intensidad relacionada con (un scalar) asociado al vértice: $e, g_s, g_w, q_z$

# El bosón $W$ se acopla con un leptón de arriba y abajo pero siempre de la misma generación.
# 
# $$
# \begin{pmatrix}\nu_e \\ e \end{pmatrix}, \; \begin{pmatrix}\nu_\mu \\ \mu \end{pmatrix}, \;
# \begin{pmatrix}\nu_\tau \\ \tau \end{pmatrix}
# $$
# 
# La constante de acoplo es la misma para las tres generaciones $g_W$ (universalidad).
# 
# <img src="./imgs/feynman_wleptons.png" width = 500 align="center">
# 

# O entre un quark de arriba y de abajo:
# $$
# \begin{pmatrix}u \\ d \end{pmatrix}, \; \begin{pmatrix} u \\ s \end{pmatrix}, \; \begin{pmatrix}u \\ b \end{pmatrix} \\
# \begin{pmatrix}c \\ d \end{pmatrix}, \; \begin{pmatrix} c \\ s \end{pmatrix}, \; \begin{pmatrix}c \\ b \end{pmatrix} \\
# \begin{pmatrix}t \\ d \end{pmatrix}, \; \begin{pmatrix} t \\ s \end{pmatrix}, \; \begin{pmatrix}t \\ b \end{pmatrix}
# $$
# 
# La constante de acoplo depende del par arriba y abajo que se acopla, $g_W V_{ij}$, (donde $i = \{u, c, t\}, j = \{d, s, b\}$).
# 
# Los elementos $V_{ij}$ forman una matriz unitaria llamada de **matrix CKM** (Cabibbo, Kobayashi, Maskawa). 
# 
# El acoplo es más intenso entre pares de la misma generación.
# 
# <img src="./imgs/feynman_wquarks.png" width = 500 align="center">

# ##### Dispersión y aniquilación
# 
# Existen tres tipos de diagramas principales: **dispersión**, **desintegración** y **aniquilación**
# 
# <img src="./imgs/feynman_annhilation.png" width = 500 align="center">

# Los líneas de los antifermiones llevan una fecha en sentido contratio al tiempo. De tal forma que una corriente no se crea, es una continuidad
# 
# En los procesos de aniquilación un par fermión-antifermión se aniquilan y se median mediante ($Z, \gamma$) y producen otro par fermión-antifermión.

# 
# ejemplos de dispersión de electrones, desintegración de un muón, y de aniquilación $e^+e^- \to \mu^+\mu^-$
# 
# <img src="./imgs/feynman_examples.png" width = 450 align="center">
# 
# *cuestión*: verifica la conservación de la carga eléctrica y número de partículas y anti-partículas en cada vértice y en los estados iniciales y finales.

# El diagrama de Feynman para la desintegración $\beta$ de un neutrón sería:
# 
# <img src="./imgs/feynman_neutron_decay.png" width = 250 align="center">
# 
# Los quarks (ud) de arriba actuan como meros 'espectadores'.

# #### Diagramas de árbol y de lazo
# 
# En un diagrama de Feynman cada vértice introduce una constante de acoplo, $g$. 
# 
# La matriz de transición, $M_{if}$, de un diagrama árbol será proporcional a $g^2$. Y la probabilidad de una interacción a $|M_{if}|^2 \propto g^4$.
# 
# En algunas ocasiones se utiliza la constante sin dimensiones, $\alpha \propto g^2$, que caracteriza la intensidad de un diagrama. Para el electromagneticos asociamos $\alpha$ a la constante de estructura fina:
# $$
# \alpha = \frac{e^2}{4 \pi \epsilon_0 \hbar c}
# $$
# 
# Las constantes de las fuerzas serían aproximadamente:
# 
# | electromagnetismo | débil | fuerte |
# | :--  | :-- | :-- |  
# 1/137 | 1/30 | 1 |
# 

# Los siguientes diagramas corresponden a un diagrama de árbol y de lazo.
# 
# 
# <img src="./imgs/feynman_tree_loop.png" width = 400 align="center">
# 
# La matriz de transición del primer diagrama es $\propto \alpha^2$, mientras que la del segundo $\propto \alpha^4$.
# 
# Si aplicamos aproximaciones perturbativas, podemos considerar solo el primer nivel (*leading order*) o nivel árbol es relevante, pero dependera de cada situación.
# 
# 

# #### Nomenclatura:
# 
# | nombres          | definición |
# | :---          | :--- |
# | *partícula*  | corpúsculo que puede sentir la fuerzas electromagnética, débil o fuerte |
# | *antipartícula* | partícula con las cargas y números cuánticos (excepto masa y spín) opuestos a la partícula|
# | *fermión*  | partícula de spin 1/2 |
# | *leptón*  (liguero) | fermión que no siente la fuerza fuerte |
# | *quark*    | fermión que siente la fuerta fuerte |
# | *bosón*    | particula de spín entero |
# | *gauge boson*| bosones portadores de la fuerza $W, Z, \gamma$ tienen spín 1 (vectoriales) |
# | *hadrón*  (fuerte) | partícula compuesta que siente la fuerza fuerte|
# | *barión*   | hadrón compuesto de tres quarks |
# | *mesón*   (mediano)| hadrón compuesto de quark y antiquark |
# 
# 

# In[11]:


labels = (r'$\rho$', r'$J/\psi$')
forces = (3, 1)
times  = (1e-24, 1e-21)

plt.scatter(np.log(times), forces)
ax  = plt.gca();
for i, name in enumerate(labels):
    ax.annotate(name, (np.log(times[i]), forces[i]), fontsize = 20)


# Las partículas se clasifican de acuerdo con sus 'cargas' frente a las fuerzas y determinados números cuánticos (spín).
# 

# In[ ]:


Un poco de léxico:
    
    


# ### Tema I 
# 
# 

# ### Extras
# 

# La Física de Partículas está ligada al desarrollo tecnológico de aceleradores.
#    
# ![](./imgs/intro_markII_colider.jpg) ![](./imgs/intro_slac_aerial.jpg "title-2")
#    
# <img src="./imgs/intro_markII_colider.jpg" width = 100 /> <img src="./imgs/intro_slac_aerial.jpg"    width = 100 /> 
# 
# <p float='center'>
# <img src="./imgs/intro_markII_colider.jpg" width = 20 />
# <img src="./imgs/intro_slac_aerial.jpg"    width = 20 />
# </p>
#    
#   * <font color= 'blue'> El acelarador actual más potentes el LHC (Large Hadron Collider) en el CERN </font>
#     
# Pero también hay aceleradores naturales:
# 
#   * Rayos Cósmicos (protons, ...) pero también neutrinos
#   
#   

# ##### Propiedades de los fermiones
# 
# | lepton | Q (e) | mass (GeV) | lifetime (s) | | quark | Q (e) | mass (GeV) |
# | :--    | :--   | :--       | :--            | | :--   | ---   | :--        |
# | $\nu_e$  | 0     | $<10^{-9}$ | $\infty$   | | u     | $\frac{-1}{3}$ | $\sim0.002$         |
# | $e$      | -1    | 0.000511     | $\infty$    | | d     |  $\frac{2}{3}$  |$\sim0.005$         |
# | --- |
# | $\nu_\mu$  | 0     | $<10^{-9}$ | $\infty$  | | c     | $\frac{-1}{3}$ | $\sim1.2$         |
# | $\mu$      | -1    | 0.106      |   $2.2 \times 10^{-6}$        | | s     |  $\frac{2}{3}$ | $\sim0.093$         |
# | --- |
# | $\nu_\tau$ | 0     | $<10^{-9}$ | $\infty$  | | t     | $\frac{-1}{3}$ | $\sim 170$         |
# | $\tau$      | -1    | 1.850      |    $2.9 \times 10^{-13}$         | | b     |  $\frac{2}{3}$ | $\sim $4.2$         |
# 
# 
# @todo hacer plot de scala de masas y puntos con labels

# ### Tema III 

# ### Bibliografía
# 
# * "Elementary Particle Physics", S. Bettini, Cambridge.
# 
# * "Modern Particle Physics", Mark Thomson, Cambridge.
# 
# * "Lepton and Quark", Francis Halzen, Alan D. Martin, Josh Wiley & Sons
# 

# In[ ]:




