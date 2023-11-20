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


# # Indice
# 
#  * Ecuación de Dirac
#     * Ecuación de Dirac
#     * Matrices $\gamma$
#  * Soluciones de la partícula libre - espinores
#  * Helicidad y quiralidad

# ## Ecuación de Dirac
# 
# 
# ### Formulación de la ecuación de Dirac
# 
# Para dar una versión relativista de la mecánica cuántica, Dirac propuso una ecuación lineal en la primera derivada temporal y en las derivadas espaciales a los que multiplicó por los factores $\alpha, \beta$.
# 
# La **ecuación de Dirac** es:
# 
# $$
# \hat{H} \,\Psi = \left( \bf{\alpha} \cdot {\bf \hat p} + \beta \, m \right) \, \Psi,
# $$
# 
# donde $\hat{H} = i \frac{\partial}{\partial t}$ es el hamiltoniano de la partícula libre, ${\hat {\bf p}} = -i \nabla$ el operador momento lineal y $\Psi$ la función de ondas. 

# 
# #### Las matrices $\alpha, \beta$
# 
# Al elevar al cuadrado la ecuación se obtiene la relación de Einstein:
# $$
#  E^2 = p^2 + m^2
# $$
# 
# siempre que los factores ${\bf \alpha}, \beta$ sean matrices y cumplan las condiciones:
# 
# $$
# \alpha_i^2 = \beta^2 = I, \;\; \alpha_i \alpha_j + \alpha_j \alpha_i = 0 \; (i \neq j), \;\; \alpha_i \beta + \beta \alpha_i = 0,
# $$
# con $i= 1, 2, 3$. 
# 
# Deben ser además matrices hermíticas, $\alpha_i = \alpha_i^\dagger, \; \beta = \beta^\dagger$, para que el hamiltonianto tenga valores reales.
# 
# 

# Las soluciones más sencillas son matrices $4 \times 4$.
# 
# Una **representación** de las matrices conveniente es la **Pauli-Dirac** que usa las **matrices de Dirac** $2 \times 2$, $\sigma_i$ con $i=1, 2, 3$.
# 
# $$
# \beta = \begin{pmatrix} I & 0 \\ 0 & I \end{pmatrix}, \;\;
# \alpha_i = \begin{pmatrix} 0 & \sigma_i \\ \sigma_i & 0 \end{pmatrix}
# $$
# 
# Donde:
# 
# $$
# I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \;\;
# \sigma_1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \;\;
# \sigma_2 = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \;\;
# \sigma_3 = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}, \;\;
# $$
# 
# Existen otras representaciones útiles, por ejemplo la quiral, pero la física de la ecuación de Dirac no depende de la representación.

# #### El espinor de Dirac
# 
# La función de onda, solución de la ecuación de Dirac, $\Psi$, es un **cuadri-spinor de Dirac**:
# 
# $$
# \Psi = \begin{pmatrix} \psi_1 \\ \psi_2 \\ \psi_3 \\ \psi_4\end{pmatrix},
# $$
# 
# que tiene cuatro componentes, $\psi_i$ con $i=1, 4$, complejas en forma vector columna.
# 

# ### Representación covariante de la ecuación de Dirac 
# 
# La representación más común de la ecuación de Dirac es la covariante. 
# 
# Introducimos primero las  **matrices-$\gamma$** que se definen como:
# 
# $$
# \gamma^0 = \beta, \;\; \gamma^1 = \beta \alpha_1, \; \gamma^2 = \beta \alpha_2, \;\; \gamma^3 = \beta \alpha_3
# $$
# 
# y la **derivada covariante**:
# 
# $$\partial_\mu = \left( \frac{\partial}{\partial t}, \frac{\partial}{\partial x},  
#                 \frac{\partial}{\partial y}, \frac{\partial}{\partial z}  \right),$$
# 
# La ecuación de Dirac se escribe de forma **covariante** como:
# 
# $$
# (i \gamma^\mu \partial_\mu  - m) \, \Psi = 0.
# $$

# #### Las matrices $\gamma$
# 
# Las matrices-$\gamma$ cumplen:
# $$
# (\gamma^0)^2 = I, \; (\gamma^k)^2 = - I, \;\; \gamma^\mu \gamma^\nu = - \gamma^\nu \gamma^\mu \, (\mu \neq \nu),
# $$
# con $k=1, 2, 3$, y $\mu = 0, 1, 2, 3$.
# 
# De forma equivalente, cumplen las siguientes relaciones de anti-conmutación: 
# 
# $$
# \{ \gamma^\mu, \gamma^\nu \} \equiv \gamma^\mu \gamma^\nu + \gamma^\nu \gamma^\mu = 2 g^{\mu\nu}
# $$
# 
# La matrix $\gamma^0$ es hermítica y las $\gamma^k$, con $k=1, 2, 3$, anti-hermíticas:
# 
# $$
# \gamma^{0\dagger} = \gamma^0, \;\; \gamma^{k \dagger} = - \gamma^{k}
# $$

# #### Matrix $\gamma^5$
# 
# Definimos la matrix $\gamma^5$ a partir del producto de las otras matrices gamma:
# 
# $$
# \gamma^5 \equiv i \gamma^0 \gamma^1 \gamma^2 \gamma^3,
# $$
# 
# La matriz $\gamma^5$ juega un papel fundamental en la interacciones débiles.
# 
# Tiene las siguientes propiedades:
# 
# $$
# (\gamma^5)^2 = I, \;\; \gamma^{5\dagger} = \gamma^5, \;\; \gamma^5\gamma^\mu = - \gamma^\mu \gamma^5
# $$

# En la representación de Pauli-Dirac, la matrices $\gamma$ son:
# 
# $$
# \gamma^0 = \begin{pmatrix} I & 0 \\ 0 & -I \end{pmatrix}, \;\;
# \gamma^k = \begin{pmatrix} 0 & \sigma_k \\ -\sigma_k & 0 \end{pmatrix}, \;\;
# \gamma^5 = \begin{pmatrix} 0 & I \\ I & 0 \end{pmatrix},
# $$
# 
# donde $I$ es la matriz identidad $2\times2$.

# #### El espinor de Dirac adjunto
# 
# Es conveniente también definir el **spinor adjunto** como:
# 
# $$
# \bar{\Psi} = \Psi^\dagger \gamma^0
# $$
# 
# En la representación de Pauli-Dirac:
# $$\bar{\Psi} = (\psi^*_1, \psi^*_2, -\psi^*_3, -\psi^*_4),$$
# que tiene forma de vector fila.

# #### Densidad y corriente de probabilidad
# 
# La densidad, $\rho$, y la corriente, $j^k$  con $k=1, 2, 3$, de probabilidad del spinor de Dirac son:
# 
# $$
# \rho = \Psi^\dagger \Psi = |\psi_1|^2 + |\psi_2|^2 + |\psi_3|^2 + |\psi_4|^2, \;\; j^k  = \Psi^\dagger \alpha_k \Psi
# $$
# 
# que cumplen:
# $$
# \frac{\partial \rho}{\partial t} + \nabla \cdot {\bf j} = 0.
# $$
# 
# Podemos introducir el cuadrivector corriente de probabilidad:
# 
# $$
# j^\mu = \Psi^\dagger \gamma^0 \gamma^\mu \Psi = \bar{\Psi} \gamma^\mu \Psi,
# $$
# 
# Y la conservación de probabilidad puede expresarse:
# 
# $$
# \partial_\mu j^\mu = 0
# $$
# 

# ### Spin en la ecuación de Dirac
# 
# En mecánica cuántica la evolución temporal de un observable, O, con operador, $\hat O$, y hamiltoniano $\hat H$, viene dada por:
# 
# $$\frac{d O}{dt} = i \langle | \Psi | [\hat H, \hat O] | \Psi \rangle $$ 
# 
# En el caso del operador momento angular: ${\bf \hat L} = {\bf \hat r} \times {\bf \hat p}$, y el hamiltoniano de Dirac: $\hat H = {\bf \alpha } \cdot {\bf \hat p} + \beta m$, obtenemos:
# 
# $$[\hat H, {\bf \hat L}] = - i {\bf \alpha} \times {\bf \hat p}$$
# 
# No se conserva en el tiempo.
# 
# (Para los detalles del cálculo ver [MT4.4])

# Pero si consideramos el operador spín, ${\bf \hat S}$, definido por:
# 
# $$
# {\bf \hat S } = \frac{1}{2} {\bf \hat \Sigma} = 
# \frac{1}{2} \begin{pmatrix} {\bf \sigma} & 0 \\ 0 & {\bf \sigma} \end{pmatrix} 
# $$
# 
# obtenemos que:
# 
# $$
# [\hat H, {\bf \hat S}] = i {\bf \alpha} \times {\bf \hat p}
# $$

# Luego el momento angular total: ${\bf \hat J} = {\bf \hat L} + {\bf \hat S}$, se conserva.
#     
# Podemos interpretar ${\bf \hat S}$ como un momento angular intrínsico, el spín, de la partícula.
# 
# ${\bf \hat S}$ tiene las mismas relaciones de conmutación que ${\bf \hat L}$:
# 
# $$
# [\hat S_x, \hat S_y] = i \hat S_z, \;\; 
# [\hat S_y, \hat S_z] = i \hat S_z, \;\,
# [\hat S_z, \hat S_x] = i \hat S_y, \\
# [\hat S^2, \hat S_y] =  [\hat S^2, \hat S_z] = [\hat S^2, \hat S_x] = 0. 
# $$

# 
# 
# ## Soluciones de la ecuación de Dirac
# 
# Vamos a obtener la soluciones de la ecuación de Dirac a partir de las funciones de la onda plana de una partícula libre:
# 
# $$
# \Psi = u(E, {\bf p}) \, e^{i ({\bf p} \cdot {\bf x} - E t)}
# $$
# 
# Donde $u(E,{\bf p})$ es un espinor de Dirac que no depende ahora de ${\bf x}, t$.
# 
# Al aplicar la ecuación de Dirac obtenemos:
# 
# $$
# (i\gamma^\mu \partial_\mu - m) \, \Psi = (\gamma^0 E - \gamma^1 p_1 -\gamma^2 p_2 - \gamma^3 p_3) \, u(E, {\bf p}) \, e^{i({\bf p \cdot x} - E t)}
# $$
# 
# luego $u(E, {\bf p})$ debe cumplir la ecuación:
# 
# $$
# (\gamma^\mu p_\mu - m) \, u(E, {\bf p}) = 0,
# $$
# donde $p_\mu = (E, -p_x, -p_y, -p_z)$ es el cuadrimomento *covariante*.
# 
# Notar que ecuación ahora no contiene derivadas.
# 

# ### Soluciones de la partícula en reposo
# 
# Para una partícula en reposo ${\bf p} = {\bf 0}$, la onda de la partícula libre es: $\Psi = u(E, {\bf 0})\,e^{-iEt}$ y la ecuación para el espinor $u(E, {\bf 0})$:
# 
# $$
# \left(E \gamma^0 - m  \right) \, u = 0
# $$
# 
# En la representación de Pauli-Dirac, hay cuatro soluciones:
# 
# $$
# u_1 = N \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix} \;
# u_2 = N \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix} \;
# u_3 = N \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix} \;
# u_4 = N \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix} . 
# $$
# Donde $N$ es un factor de normalización.
# 

# La ecuación de Dirac, $(E\gamma^0 - m) \, u_i$, aplicada a:
# 
#    * $u_1, u_2$ queda, $E - m = 0$. Esto es, $u_1, u_2$ tienen energía positiva, $E = m$
# 
#    * $u_3, u_4$ queda $-E -m = 0$. Esto es $u_3, u_4$ tiene energía negativa, $E = -m$.
# 
# Respecto al spín ${\hat S_z}$
# 
#    * $u_1, u_3$ tiene spín $1/2$, arriba. 
#    
#    * $u_2, u_4$ tiene spin $-1/2$, abajo.
#    
# Asociamos estas soluciones de energía positiva con una partícula de spín $1/2$, un **fermión**, y las soluciones de energía negativa con una antipartícula de spín $1/2$, **antifermión**.

# #### ¿Cuál es el significado de la solución de energía negativa?
# 
# Dirac propuso la teoría del "mar de Dirac". El *vacío* estaba lleno con todos los estados de energía negativos ocupados. Un fotón con $E > 2 m_e c^2$ podía hacer saltar un electrón del vacío a la zona de energía positiva produciendo un electrón y un hueco en el mar que se interpreta como el positrón.
# 
# Feynman y Stückelberg propusieron que las soluciones de energía negativa eran en realidad partículas que se propagaban hacía atrás en el tiempo o equivalente antipartículas con cargas opuestas que se propagan hacia delante en el tiempo con energía positiva. 
# 
# Experimentalmente las antipartículas se comportan como las partículas solo que tienen sus cargas opuestas.

# En la interpretación de Feynman-Stückerlberg $t \to -t$, lo que cambia el momento ${\bf p} \to -{\bf p}$.
# 
# Las funciones de ondas de las partículas, $\Psi$, son:
# $$
# \Psi = u(E, {\bf p}) \, e^{+i({\bf p \cdot x} - Et)},
# $$
# 
# y las de las antipartículas, $\Psi^v$, quedan:
# 
# $$
# \Psi^v = v(E, {\bf p}) \, e^{-i({\bf p \cdot x} - Et)}
# $$
# 
# donde $v(E, {\bf p})$ es el espinor de las antipartículas.
# 
# Darse cuenta del cambio de signo en el exponente en la función de ondas.
# 
# *Nota adicional* Veremos que en los diagramas de Feynman los anti-fermiones se dibujan con una flecha en sentido opuesto al tiempo.

# #### Ecuación de Dirac para los espinores $v$
# 
# Si aplicamos la ecuación de Dirac sobre la función de onda de las antipartículas, $\Psi^v$
# 
# $$
# \psi^v ({\bf x}, t) = v(E, {\bf p}) \, e^{-i ({\bf p \cdot x} - E t)}
# $$
# 
# obtenemos:
# 
# $$
# (-\gamma^0E + \gamma^1p_1 + \gamma^2 p_2 + \gamma^3 p_3 -m) \, v(E, {\bf p}) = 0
# $$
# 
# que reescribimos para dar la ecuación de Dirac en los espinores $v$:
# 
# $$
# (\gamma^\mu p_\mu + m) \, v(E, {\bf p}) = 0
# $$
# 
# Notar que cambia de signo el término de masa.

# ### Soluciones generales de la partícula libre
# 
# Vamos a calcular primero las soluciones generales de la ecuación de Dirac para los espinores $u$ y $v$.
# 
# En cada caso aparecen cuatro soluciones, dos con energía positiva y dos con negativa.
# 
# Pero resulta más conveniente trabajar con las dos soluciones de energía positiva de $u$, las partículas, y las dos de $v$, las antipartículas. 
# 
# Recordemos la ecuación de Dirac para ambos es:
# 
# $$
# (\gamma^\mu p_\mu- m) \, u = 0, \;\; (\gamma^\mu p_\mu + m) \, v = 0
# $$

# #### los espinores de las partículas $u$
# 
# La ecuación de Dirac sobre el espinor $u$, esto es $(\gamma^\mu p_\mu -m ) \,u = 0$, queda:
# 
# $$
# \left[ \begin{pmatrix} I & 0 \\ 0 & - I\end{pmatrix} E - 
#         \begin{pmatrix} 0 & \sigma \cdot {\bf p} \\ - \sigma \cdot {\bf p} & 0 \end{pmatrix} - 
#         m \begin{pmatrix} I & 0 \\ 0 & I \end{pmatrix}\right] u = 0
# $$
# 
# si expresamos el spinor a partir de dos vectores columna con dos componentes cada uno, $u_A, u_B$: 
# 
# $$
# u = \begin{pmatrix} u_A \\ u_B \end{pmatrix}
# $$
# Obtenemos:
# 
# $$
# \begin{pmatrix} (E-m) I & - \sigma \cdot {\bf p} \\ \sigma \cdot {\bf p} & - (E + m) I\end{pmatrix}  
# \begin{pmatrix} u_A \\ u_B \end{pmatrix} = 0
# $$
# 

# que se reduce a dos equaciones acopladas:
# $$
# u_A = \frac{\sigma \cdot {\bf p}}{E - m} u_B \\
# u_B = \frac{\sigma \cdot {\bf p}}{E + m} u_A \\
# $$
# 
# Notar que la parte arriba está relacionada con la de abajo.
# 
# Para más claridad, reescribimos:
# 
# $$
# \sigma \cdot {\bf p } =  \begin{pmatrix} p_z & p_x -  i p_y \\ p_x +i p_y & - p_z  \end{pmatrix}  
# $$
# 
# Si consideramos estas cuatro posibilidades para las componentes $u_A, u_B$:
# 
# $$
# u^1_A = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \;\; 
# u^2_A = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \;\;
# u^3_B = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \; 
# u^4_B = \begin{pmatrix} 0 \\ 1 \end{pmatrix} 
# $$
# 

# 
# Obtenemos las cuatro soluciones:
# 
# $$
# u_1 = N \begin{pmatrix} 1 \\ 0 \\ \frac{p_z}{E+m} \\ \frac{p_x+ip_y}{E+m} \end{pmatrix}, \;\;
# u_2 = N \begin{pmatrix} 0 \\ 1 \\ \frac{p_x-ip_y}{E+m} \\ \frac{-p_z}{E+m} \end{pmatrix}, \;\;
# u_3 = N \begin{pmatrix} \frac{p_z}{E-m} \\ \frac{p_x+ip_y}{E-m} \\ 1 \\ 0 \end{pmatrix}, \;\;
# u_4 = N \begin{pmatrix} \frac{p_x-ip_y}{E-m} \\ \frac{-p_z}{E-m} \\ 0 \\ 1 \end{pmatrix}. 
# $$
# 
# Donde $N$ son factores de normalización.
# 
# Que en el límite ${\bf p} = {\bf 0}$, se reducen a los casos con energía positiva y negativa.
# 
# $u_1, u_2$ están asociados a las partículas de energía positiva: $E = + \sqrt{p^2 + m^2}$
# 
# $u_3, u_4$ están asociados a la partículas de energía negativa: $E = - \sqrt{p^2 + m^2}$

# #### espinores de las antipartículas $v$
# 
# De marera similar, la ecuación para los spinores $v$
# 
# $$ 
# \left(\gamma^\mu p_\mu + m \right) v = 0
# $$
# 
# Si consideramos un espinor como dos vectores columna, arriba y abajo, $v_A, v_B$, con dos componentes cada uno:
# 
# $$
# v = \begin{pmatrix} v_A \\ v_B\end{pmatrix}
# $$
# Obtenemos de nuevos dos ecuaciones acopladas:
# 
# $$
# v_A = \frac{{\bf \sigma} \cdot {\bf p}}{E + m} v_B, \;\; v_B = \frac{{\bf \sigma} \cdot {\bf p}}{E - m} v_A,
# $$

# Que da lugar a a cuatro soluciones para $v$:
# 
# $$
# v_1 = N \begin{pmatrix} \frac{p_x - ip_y}{E+m} \\ \frac{-p_z}{E+m} \\ 0 \\ 1 \end{pmatrix}, \;\;
# v_2 = N \begin{pmatrix} \frac{p_z}{E+m} \\ \frac{p_x + ip_y}{E+m} \\ 1 \\ 0 \end{pmatrix}, \;\;
# v_3 = N \begin{pmatrix} 1 \\ 0 \\ \frac{p_z}{E-m} \\ \frac{p_x + ip_y}{E-m}  \end{pmatrix}, \;\;
# v_4 = N \begin{pmatrix} 0 \\ 1 \\ \frac{p_x - ip_y}{E-m} \\ \frac{-p_z}{E-m} \end{pmatrix}, \;\;
# $$
# 
# Los espinores $v_1, v_2$ tienen ahora energía positiva, $E = \sqrt{{\bf p}^2 + m^2}$
# 
# Los espinores $v_3, v_4$ tienen ahora energía negativa, $E = - \sqrt{{\bf p}^2 + m^2}$

# #### Los espinores $u$ de las partículas y $v$ de las antipartículas
# 
# Resulta conviente utilizar los espinores $u_i$ y $v_i$, con $i=1, 2$, asociados a partículas y antipartículas con energía positiva.
# 
# $$
# \Psi_i = u_i(E, {\bf p}) \, e^{+i({\bf p} \cdot {\bf x} - E t)}, \;\; 
# \Psi^v_i = v_i(E, {\bf p}) \, e^{-i({\bf p} \cdot {\bf x} - E t)},
# $$
# 
# donde:
# $$
# u_1 = N \begin{pmatrix} 1 \\ 0 \\ \frac{p_z}{E+m} \\ \frac{p_x+ip_y}{E+m} \end{pmatrix}, \;
# u_2 = N \begin{pmatrix} 0 \\ 1 \\ \frac{p_x-ip_y}{E+m} \\ \frac{-p_z}{E+m} \end{pmatrix}, \;
# v_1 = N \begin{pmatrix} \frac{p_x-ip_y}{E+m} \\ \frac{-p_z}{E+m} \\ 0 \\ 1 \end{pmatrix}, \;
# v_2 = N \begin{pmatrix} \frac{p_z}{E+m} \\ \frac{p_x + ip_y}{E+m} \\ 1 \\ 0 \end{pmatrix}. 
# $$
# 
# 

# ### Operadores sobre los spinores de anti-partículas
# 
# Los operadores ${\hat H} = i \frac{\partial}{\partial t}, \; {\hat {\bf p}} = - i \nabla$. Al aplicarlos sobre las soluciones de las anti-partículas nos dan:
# 
# $$
# \hat{H} \, \Psi^v = - E \, \Psi^v, \;\; {\bf \hat p} \, \Psi^v = - {\bf p} \, \Psi^v,
# $$
# 
# que son las soluciones de energía negativa yendo hacia atrás en el tiempo.
# 
# A los operadores de las antipartículas afectados por la interpretación de Feynman-Stückelber los denotaremos con $\hat{O}^v$. 
# 
# Estos operadores, para que den las cantidades *física* adecuadas, son:
# 
# $$
# {\hat H}^v = - i \frac{\partial}{\partial t}, \;\;\;
# {\hat {\bf p}}^v = i \nabla, \;\;\; {\hat S}_z^v = - {\hat S}_z
# $$

# ### Conjungación de carga
# 
# La conjungación de carga cambia partículas por antipartículas.
# 
# El **operador cojungación de carga** viene dado por:
# 
# $$
# \Psi' = \hat{C} \, \Psi = i \gamma^2 \Psi^*
# $$
# 
# Este operador cambia $\Psi_1 = u_1 \, e^{+i({\bf p \cdot x} - Et)}$  en $\Psi^v_1 = v_1 \, e^{-i({\bf p \cdot x} - Et)}$
# 
# Veámoslo:
# 
# $$
# \Psi' = \hat{C} \, \Psi_1 = i \gamma^2 \left[u_i e^{+i ({\bf p} \cdot {\bf x} - Et)} \, \right]^* = i \gamma^2 u^*_1 \, e^{-i ({\bf p} \cdot {\bf x} - Et)} 
# $$

# Queda comprobar que $i\gamma^2 u^*_1 = v_1$
# 
# En la representación de Pauli-Dirac: 
# $$
# i \gamma^2 u^*_1 = 
# i 
# \begin{pmatrix} 0 & 0 & 0 & -i  \\ 0 & 0 & i & 0 \\ 
#                 0 & i & 0 & 0   \\ -i & 0 & 0 & 0  \end{pmatrix}
# \begin{pmatrix} 1 \\ 0 \\ \frac{p_z}{E+m} \\ \frac{p_x - i p_y}{E+m}\end{pmatrix} = 
# \begin{pmatrix} \frac{p_x - ip_y}{E+m} \\ \frac{-p_z}{E+m} \\ 0 \\ 1 \end{pmatrix} = v_1
# $$
# 
# Verificamos que no la conjungación de carga no cambia el espín:
# $$
# \hat{S}_z \, \Psi_1 = \hat{S}^v_z \Psi^v_1
# $$
# 
# Se comprueba de la misma forma para $u_2, v_1, v_2$.

# ### Paridad
# 
# La operación de paridad, ${\hat P}$, cambia ${\bf x} \to -{\bf x}$. Lo que tiene como consecuencia que también cambie ${\bf p} \to - {\bf p}$, pero no el momento angular.
# 
# Definimos el operador paridad como: 
# $$
# \hat{P} \equiv \gamma^0,
# $$
# 
# De tal forma que los fermiones en reposo, tienen autovalor +1 bajo paridad y las antifermiones -1:
# 
# $$
# \hat{P} \, u_i(m, {\bf 0}) = \gamma^0 u_i(m, {\bf 0})  = + u_i (m, {\bf 0}), \;\; \hat{P} \, v_i(m, {\bf 0}) = \gamma^0 v_i(m, {\bf 0}) = - v_i(m, {\bf 0}),
# $$
# con $i = 1, 2$.
# 

# Si aplicamos paridad, $\hat{P}$ sobre $\Psi_1$
# 
# El exponente de la función de ondas, $\Psi$, no cambia por paridad ya que si ${\bf x \to -x}$ entonces ${\bf p} \to -{\bf p}$.
# 
# Y al aplicar $\hat{P} \, u_1(E, {\bf p})$ obtenemos:
# 
# $$
#  N \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix} 
# \begin{pmatrix} 1 \\ 0 \\ \frac{p_x +i p_y}{E+m} \\ \frac{p_z}{E+m} \end{pmatrix} = 
# N \begin{pmatrix} 1 \\ 0 \\ \frac{-p_x - i p_y}{E+m} \\ \frac{-p_z}{E+m} \end{pmatrix} = u_1(E, - {\bf p}) 
# $$
# 
# comprobamos que en el espinor cambia ${\bf p} \to - {\bf p}$
# 
# Se comprueba de forma simular para $u_2, v_1, v_2$.

# ### Factor de normalización
# 
# La función de ondas en la versión relativista se normaliza a $2E$ partículas en una unidad de volumen $V$ (ver [apéndice-fundamentos])
# 
# Si normalizamos por ejemplo para $\Psi_1$ obtenemos:
# 
# $$
# \Psi_1^\dagger \Psi_1 = u_1^\dagger u_1 = |N|^2 \, 
# \left(1 + \frac{p_z^2}{(E+m)^2} + \frac{p^2_x + p^2_y}{(E+m)^2} \right) = |N|^2 \frac{2E}{E+m} = 2E
# $$
# 
# El factor de normalización es:
# $$
# N = \sqrt{E + m}
# $$

# ## Helicidad y Quiralidad
# 
# #### Tercera componente de spin
# 
# Los espinores, $u, v$, son autoestados del operador tercera componente de espín, ${\hat S}_z$.
# 
# Se comprueba fácil para el caso en reposo, $u_i(m, {\bf 0}), \, v_i(m, {\bf 0})$, con $i=1, 2$.
# 
# Para las partículas en movimiento siempre podemos definar la dirección $z$ como la de momento, ${\bf p} = p {\hat k}$. En ese caso:
# 
# $$
# u_1 = N \begin{pmatrix} 1 \\ 0 \\ \frac{p}{E+m} \\ 0 \end{pmatrix}, \;
# u_2 = N \begin{pmatrix} 0 \\ 1 \\ 0 \\ \frac{-p_z}{E+m} \end{pmatrix}, \;
# v_1 = N \begin{pmatrix} 0 \\ \frac{-p_z}{E+m} \\ 0 \\ 1 \end{pmatrix}, \;
# v_2 = N \begin{pmatrix} \frac{p_z}{E+m} \\ 0 \\ 1 \\ 0 \end{pmatrix} 
# $$
#  
# se cumple:
# $$
# \hat{S}_z \, u_i(E, p \hat{k}) = \pm \frac{1}{2} u_i(E, p \hat{k}), \;\;\;
# \hat{S}^v_z \, v_i(E, p \hat{k}) = \pm \frac{1}{2} v_i(E, p \hat{k}),
# $$
# 
# Los espinores con $i=1$ tiene autovalores $+1/2$ (up) y los de $i=2$, $-1/2$ (down) de la tercera componente de spin.

# ### Helicidad
# 
# Definimos la helicidad, $h$, como la proyección normalizada del spin sobre el momento:
# 
# $$
# h \equiv \frac{{\bf S} \cdot {\bf p}}{p}.
# $$
# 
# Para un espinor de Dirac el operador de helicidad, $\hat{h}$, en la representación Pauli-Dirac:
# 
# $$
# {\hat h} = \frac{{\bf \Sigma} \cdot {\hat {\bf p}}}{ 2 p} = \frac{1}{2p} 
# \begin{pmatrix} {\bf \sigma} \cdot {\bf {\hat p}} & 0 \\ 0 & {\bf \sigma} \cdot {\bf {\hat p}} \end{pmatrix}
# $$
# 

# El operador helicidad conmuta con el hamiltoniano: $[{\hat H}, {\hat h}] = 0 $ y por lo tanto la helicidad se conserva.
# 
# Pero no es un invariante Lorentz, para una partícula siempre podemos encontrar un sistema de referencia (con una velocidad mayor) que revierta el momento y por lo tanto la helicidad.
# 
# Vamos a calcular la representación de los espinores, $u, v$ que sean autoestados de helicidad.
# 
# Estos espinores son de gran utilidad cuando se calculan los elementos de matriz de las desintegraciones o interacciones de partículas.

# ### Espinores de helicidad
# 
# Recordemos que la parde de arriba y abajo, $u_A, u_B$, del espinor $u = \begin{pmatrix} u_A \\ u_B \end{pmatrix}$ están relacionadas:
# 
# $$
# u_B = \frac{{\bf \sigma \cdot p}}{E+m} \, u_A
# $$
# 
# Solo necesitamos calcular las componentes, $a, b$, de $u_A = \begin{pmatrix} a \\ b \end{pmatrix}$
# 
# Siendo $u$ autoestado de helicidad
# 

# 
# #### Autovalores de helicidad
# 
# la condición de autoestado de helicidad, $\hat{h}\, u = \lambda \, u$:
# 
# $$
# \frac{1}{2p} \begin{pmatrix} \sigma \cdot {\bf p} & 0 \\ 0  & \sigma \cdot {\bf p}\end{pmatrix} \, 
# \begin{pmatrix} u_A \\ u_B \end{pmatrix} = \lambda \, \begin{pmatrix} u_A \\ u_B \end{pmatrix},
# $$
# se traduce en dos condiciones:
# $$
# (\sigma \cdot {\bf p}) \, u_A = (2p) \lambda u_A, \;\; (\sigma \cdot {\bf p}) \, u_B = (2p) \lambda u_B
# $$
# 
# si multiplicamos la primera ecuación a ambos lados por $(\sigma \cdot {\bf p})$ y teniendo en cuenta que $(\sigma \cdot {\bf p})^2 = p^2$, obtenemos:
# 
# $$
# p^2 u_A = (2p) \lambda (\sigma \cdot {\bf p}) u_A = 4p^2 \lambda^2 u_A 
# $$
# 
# de donde obtenemos los autovalores de helicidad:
# 
# $$
# \lambda = \pm \frac{1}{2}
# $$

# #### Autoestados de helicidad
# 
# Si expresamos el momento en coordenadas esféricas ${\bf p} = p \, (\sin \theta \cos \phi, \sin \theta, \cos \theta)$, obtenemos:
# 
# $$
# \frac{{\bf \sigma} \cdot {\bf p}}{p} = \frac{1}{p}\begin{pmatrix} p_z & p_x - ip_y \\ p_x+ip_y & -p_z \end{pmatrix} = 
# \begin{pmatrix} \cos\theta & \sin\theta e^{-i\phi} \\ \sin\theta e^{i\phi} & -\cos\theta \end{pmatrix}
# $$
# 
# La condición de autoestado de $u_A$ es:
# 
# $$
# \begin{pmatrix} \cos\theta & \sin\theta e^{-i\phi} \\ \sin\theta e^{i\phi} & -\cos\theta \end{pmatrix}
# \begin{pmatrix} a \\ b \end{pmatrix} = 
# 2 \lambda \begin{pmatrix} a \\ b \end{pmatrix}
# $$
# 

# 
# Las componentes $a, b$ cumplen:
# $$
# \frac{b}{a} = \frac{2\lambda -\cos \theta}{\sin\theta} e^{i\phi}
# $$
# 
# Para el caso de helicidad positiva $\lambda = 1/2$, obtenemos:
# 
# $$
# \frac{b}{a} = \frac{1 -\cos \theta}{\sin\theta} e^{i\phi} = e^{i\phi} \tan \frac{\theta}{2}
# $$
# 
# De tal forma:
# $$
# u_A = \begin{pmatrix} \cos \frac{\theta}{2} \\ e^{i\phi}\sin \frac{\theta}{2}\end{pmatrix}
# $$
# 

# Por la relación entre las partes $u_B$ y $u_A$ que impone la ecuación de Dirac, $(E+m) \, u_B = ({\bf \sigma \cdot p \cdot}) \, u_A$ obtenemos y por ser autoestado de helicidad, $({\bf \sigma \cdot p}) \, u_A = (2p) \lambda u_A$, obtenemos el espinor $u_+$ con $\lambda = 1/2$
# 
# $$
# u_{+} = N \begin{pmatrix} c \\ s e^{i\theta} \\ \frac{p}{E+m} c \\ \frac{p}{E+m} s e^{i\theta} \end{pmatrix},
# $$
# donde $c = \cos \theta/2, s = \sin \theta/2$, y $N$ el factor de normalización.
# 
# 
# que es autovector del operador helicidad, $\hat{h}$ con autovalor $1/2$, helicidad positiva:
# 
# $$
# \hat{h} \, u_+ = \frac{1}{2} \, u_+
# $$

# Procediendo de forma similar para el espinor $u$ con $\lambda = -1/2$ y para los espinores $v$ con $\lambda = \pm 1/2$ obtenemos los **espinores de helicidad**:
# 
# 
# $$
# u_{+} = N \begin{pmatrix} c \\ s e^{i\theta} \\ \kappa c \\ \kappa s e^{i\theta} \end{pmatrix}, \;
# u_{-} = N \begin{pmatrix} -s \\ c e^{i\theta} \\ \kappa s \\ -\kappa c e^{i\theta}\end{pmatrix}, \;
# v_{+} = N \begin{pmatrix} \kappa s \\ -\kappa c e^{i\theta}\\ -s \\ ce^{i\theta} \end{pmatrix}, \;
# v_{-} = N \begin{pmatrix} \kappa c \\ \kappa s e^{i\theta}\\ c \\ s e^{i\theta} \end{pmatrix}. 
# $$
# 
# donde $\kappa = \frac{p}{E+m}, c = \cos \theta/2, s = \sin \theta/2$ y $N$ el factor normalización.
# 
# que cumplen:
# 
# $$
# \hat{h} \, u_\pm = \pm \frac{1}{2} \, u_\pm, \;\;\; \hat{h}^v \, v_\pm = \pm \frac{1}{2} \, v_\pm
# $$
# 

# Los espinores de quiralidad se suelen representar de forma gráfica con la siguientes figuras:
# 
# | |
# |:--:|
# |<img src="./imgs/dirac_spinors_helicities.png" width = 400 align="center">|
# | spín (flecha azul) y momento (negra) de los spinores de helicidad|
# 
# $u_\pm, v_\pm$ son los espinores de las partículas y antipartículas respectivamente.
# 
# $u_{+}, v_{+}$ son espinores de **helicidad positiva** (el espín tiene el mismo sentido que el momento).
# 
# $u_{-}, v_{-}$ tienen **helicidad negativa** (el spín y momento tienen sentido opuesto).
# 
# 

# ##### Caso de partículas ultrarelativistas
# 
# Si consideramos elcaso de partículas ultrarelativistas, $E \gg m$, o sin masa $m=0$, como es el caso del neutrino, tenemos $E = p$ y por lo tanto $\kappa = 1$.
# 
# Los espinores de helicidad para el caso ultrarelativista son:
# 
# $$
# u_{+} = N \begin{pmatrix} c \\ s e^{i\theta} \\ c \\  s e^{i\theta} \end{pmatrix}, \;
# u_{-} = N \begin{pmatrix} -s \\ c e^{i\theta} \\ s \\ - c e^{i\theta}\end{pmatrix}, \;
# v_{+} = N \begin{pmatrix}  s \\ -c e^{i\theta}\\ -s \\ ce^{i\theta} \end{pmatrix}, \;
# v_{-} = N \begin{pmatrix} c \\ s e^{i\theta}\\ c \\ s e^{i\theta} \end{pmatrix}. 
# $$
# 
# 
# 

# ### Quiralidad
# 
# La quiralidad juega un papel fundamental en la interacciones débiles.
# 
# La quiralidad se corresponde con la matriz $\gamma^5$. Los autoestados de $\gamma^5$ son autoestados de quiralidad. 
# 
# Recordemos que $\gamma^5$ en la representación de Pauli-Dirac es:
# 
# $$
# \gamma^5 = \begin{pmatrix} 0 & I \\ I & 0 \end{pmatrix}
# $$
# 
# 

# En esta representación, los espinores $u = \begin{pmatrix} u_A \\ u_B \end{pmatrix}$ son autoestados de quiralidad si sus componentes $u_A, u_B$ cumplen:  
# 
# $$
# u_B = \pm u_A
# $$
# 
# Los espinores
# 
# $$
# u_R = \begin{pmatrix} u_A \\ u_A \end{pmatrix}, \;\;\; u_L = \begin{pmatrix} u_A \\ -u_A \end{pmatrix},
# $$
# 
# cumplen:
# 
# $$
# \gamma^5 \, u_R = + u_R; \;\;\; \gamma^5 \, u_L = - u_L
# $$
# 
# Decimos que $u_R$ tiene quiralidad a derechas y $u_L$ a izquierdas.

# Para los espinores $v$ de las antipartículas, la situación se invierte.
# 
# Los espinores:
# 
# $$
# v_R = \begin{pmatrix} -v_B \\ v_B \end{pmatrix}, \;\;\;
# v_L = \begin{pmatrix} v_B \\ v_B \end{pmatrix}
# $$
# 
# Son autoestados de quiralidad $v_R$ a derechas y $v_L$ a izquierdas, y cumplen:
# 
# $$
# \gamma^5 \, v_R = - v_R, \;\;\; \gamma^5 v_L = +v_L
# $$

# Si definimos $\gamma^{5v} = - \gamma^5$, para los espinores $v$, tenemos:
#     
# $$
# \gamma^{5v} v_R = + v_R, \;\; \gamma^{5v} v_L = - v_L 
# $$

# #### quiralidad de los espinores de helicidad de las partículas ultra-relativistas
# 
# Si consideramos los **estados de helicidad de partículas ultra-relativistas**,
# 
# $$
# u_{+} = N \begin{pmatrix} c \\ s e^{i\theta} \\ c \\  s e^{i\theta} \end{pmatrix}, \;
# u_{-} = N \begin{pmatrix} -s \\ c e^{i\theta} \\ s \\ - c e^{i\theta}\end{pmatrix}, \;
# v_{+} = N \begin{pmatrix}  s \\ -c e^{i\theta}\\ -s \\ ce^{i\theta} \end{pmatrix}, \;
# v_{-} = N \begin{pmatrix} c \\ s e^{i\theta}\\ c \\ s e^{i\theta} \end{pmatrix}. 
# $$
# 
# observamos que **son autoestados de quiralidad**:
# 
# 
# $$
# \gamma^5 \, u_+ = +u_+, \;\;\; \gamma^5 \, u_- = -u_-, \;\;\; 
# \gamma^{5} \, v_+ = -v_+, \;\;\; \gamma^{5} \, v_- = +v_-
# $$
# 
# Esto es:
# $$
# u_R = u_+, \;\;\; u_L = u_-, \;\;\; v_R = v_+, \;\;\; v_L = v_-
# $$
# 
# *Cuidado*: ¡sólo es válido para partículas ultra-relativistas!

# #### Proyectores de quiralidad
# 
# Los proyectores de quiralidad a derechas, $P_R$, y a izquierdad, $P_L$ son:
# 
# $$
# P_R = \frac{1}{2} (I + \gamma^5), \;\;\, P_L = \frac{1}{2} (I - \gamma^5)
# $$
# 
# Es fácil comprobar a partir de las proiedades de la $\gamma^5$ que $P_R, \, P_L$ cumplen las condiciones de proyectores:
# 
# $$
# P_R + P_L = I, \;\; P^2_R = P_R, \;\;, P^2_L = P_L, \;\; P_L P_R = P_R P_L = 0
# $$
# 
# En la representación de Pauli-Dirac son:
# 
# $$
# P_R = \frac{1}{2}\begin{pmatrix} I & I \\ I & I \end{pmatrix}, \;\;\;
# P_L = \frac{1}{2}\begin{pmatrix} I & -I \\ -I & I \end{pmatrix},
# $$
# 

# Los proyectores actuan sobre los espinores $u$ de la siguiente forma:
# 
# $$
# P_R \, u_R = u_R, \;\; P_R \, u_L = 0, \;\;\; P_L \, u_R = 0, \;\; P_L \, u_L = u_L.
# $$
# 
# De tal forma que un espinor $u$ se puede descomponer en su parte a derechas e izquierdas:
# 
# $$
# P_R \, u = \alpha \, u_R, \;\;\; P_L \, u = \beta \, u_L, \;\;\; u = \alpha \, u_R + \beta \,u_L
# $$
# donde $\alpha, \beta$ son dos coeficientes complejos.
# 
# Mientras que sobre los espinores $v$, actuan:
# 
# $$
# P_R \, v_R = 0, \;\; P_R \, v_L = v_L, \;\;\; P_L \, v_R = v_R, \;\; P_L \, v_L = 0.
# $$
# 
# Y un espinor $v$ puede descomponerse en partes a izquierdas y derechas:
# 
# $$
# P_L \, v = \alpha' \, v_R, \;\; P_R \, v = \beta' \, v_L, \;\; v = \alpha' \, v_L + \beta' \, v_R
# $$
# 

# Si definimos los operadores proyección para los espinores $v$ como:
# 
# $$
#  P^{v}_R = \frac{1}{2} (1 + \gamma^{5v}), \;\; P^{v}_L = \frac{1}{2} (1 - \gamma^{5v})
# $$
# 
# Recuperamos la notación simétrica:
# 
# $$
# P^v_R \, v_R = v_R, \;\; P^v_R \, v_L = 0, \;\;\; P^v_L \, v_R = 0, \;\; P^v_L \, v_L = v_L.
# $$

# 
# Resulta muy útil reconocer que las siguientes corrientes entre proyecciones de quiralidas son nulas:
# 
# $$
# {\bar u}_L \gamma^\mu u_R = {\bar u}_R \gamma^\mu u_L = {\bar v}_L \gamma^\mu u_R = {\bar v}_R \gamma^\mu u_L = 
# {\bar v}_L \gamma^\mu v_R = {\bar v}_R \gamma^\mu v_L = 0
# $$
# 
# Por ejemplo:
# 
# $$
# \bar{u}_L \gamma^\mu u_R = u^\dagger \frac{1}{2}(1 - \gamma^5)^\dagger \gamma^0 \gamma^\mu \frac{1}{2} (1 + \gamma^5) = u^\dagger \, \gamma^0 \gamma^\mu \frac{1}{2}(1-\gamma^5) \frac{1}{2}(1+\gamma^5) u = \bar{u} \gamma^\mu P_L P_R u = 0
# $$
# 
# por las propiedades de las $\gamma$'s: $\gamma^{5\dagger} = \gamma^5, \, \gamma^\mu \gamma^5 = - \gamma^\mu \gamma^5$.
# 
# 
# De las posibles corrientes fermiónicas, solo pueden ser no nulas las que tienen fermiones con la misma quiralidad en ambos lados de la corriente. Estas son:
# 
# $$
# {\bar u}_L \gamma^\mu u_L, \;\; {\bar u}_R \gamma^\mu u_R, \;\;  {\bar v}_L \gamma^\mu u_L, \;\; {\bar v}_R \gamma^\mu v_R 
# $$
# 
# *Nota adicional*: En las corrientes débiles cargadas intervienen solamente, $u_L$, espinores de izquierdas de los fermiones, y $v_R$, espinores de derechas de los antifermiones.

# ### Relación entre helicidad y quiralidad
# 
# Ya vimos que los espinores de helicidad para las partículas ultrarelativistas $E >> m$ o sin masa $m = 0$ eran
# también autoestados de quiralidad.
# 
# Si aplicamos los proyectores de quiralidad sobre el autoestado de helicidad $u_+$, obtenemos:
# 
# $$
# P_R \, u_+ = N \frac{1}{2} (1 + \kappa) \, \begin{pmatrix} c \\ s e^{i\phi} \\ c \\ se^{i\phi} \end{pmatrix}, \;\; 
# P_L \, u_+ = N \frac{1}{2} (1 - \kappa) \, \begin{pmatrix} c \\ -s e^{i\phi} \\ -c \\ -se^{i\phi} \end{pmatrix}, 
# $$
# 
# La componente de quiralidad a izquierdas de un espinor con helicidad positiva es proporcional a:
# 
# $$
# \frac{1}{2} (1-\kappa)
# $$
# 
# De forma similar obtendríamos la descomposición de $u_-$ y $v_\pm$.
# 
# Luego sólo cuando $\kappa = 1$, la autoestados de quiralidad y helicidad coinciden, esto es solo el régimen ultrarelativista $E \gg m$ o sin masa $m=0$.

# ### Paridad y conjugación de carga en espinores de quiralidad
# 
# Si aplicamos el operador **paridad**, $\hat{P} = \gamma^0$, y los proyectores de quiralidad, $P_{R/L}$:
#     
# $$
# \gamma^0 \frac{1}{2} (1 \pm \gamma^5) = \frac{1}{2}(1 \mp \gamma^5) \gamma^0 
# $$
# 
# **intercambia las componentes de izquierdas y las de derechas**.
# 
# Notar que si una teoría es invariante bajo paridad debe tratar por igual a las componentes de quiralidad a izquierdas que a derechas.

# Si aplicamos conjugación de carga, $\hat{C} \, \Psi = i \gamma^2 \Psi^*$, a los proyectores de quiralidad, $P_{R/L}$:
#     
# $$
# i \gamma^2 \frac{1}{2} (1 \pm \gamma^5) =  (1 \mp \gamma^5) i \gamma^2
# $$
#   
# Como los proyectores de quiralidad sobre los espores $v$ tienen el comportamiento complementario que sobre los $u$. Vemos que **conjugación de carga no cambia la quiralidad**.

# ## Bibliografía
# 
#  * [MT] Mark Tomson, "Modern Particle Physics", Cambridge U. press. Tema 4 y 6.4
# 
