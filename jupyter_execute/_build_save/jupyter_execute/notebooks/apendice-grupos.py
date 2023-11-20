#!/usr/bin/env python
# coding: utf-8

# # Introducción a Física de Partículas
# 
# 
# ## Apéndice: Grupos
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
#  * Grupos, definitions
#  * SU(2)
#     * Generadores y álgebra
#     * producto
#  * SU(3)
#     * Generadores y álgebra
#     * productos

# ## Grupos
# 
# Definimos una partícula a partir de su masa, su espín y sus **cargas** respecto a las fuerzas. 
# 
# Todos esos valores son constantes y no dependen (y no pueden depender) del sistema en el que definamos la partículas. Son **invariantes respecto a un determinado grupo de simetría**.
# 
# Sabemos por el teorema de Nöether que una simetría del sistema físico representa una cantidad conservada.
# 
# Los grupos de simetría son una de las herramientas matemáticas principales de la Física de Partículas:
# 
#   * están relacionadas con las cantidades conservadas, las cargas, que definen la partícula.
#  
#   * sirven para representar los estados de las partículas que los asignamos a los elementos de la base de las representaciones irreducibles de los grupos.
# 
# Vamos a recordar el concepto de grupo, de representación, para después revisar los grupos principales en Partículas

# Los grupos más comunes en física de partículas son:
# 
#  * El **grupo de Lorentz-Poincaré**, asociado a las rotaciones, traslaciones y cambios de sistemas de referencia inerciales en el espacio-tiempo.
#    
#      Las representaciones más comunes que encontramos son: **escalar, vectorial, espinorial**
#      
#      
#  * Los **grupos unitarios** (cuya representación son matrices unitarias) de dimensiones pequeñas:
#    
#    * **U(1)** rotación de una fase de un número complejo, que asociamos con la conservación de la carga eléctrica. 
#       
#    * **SU(2)** matrices unitarias de dimensión 2 con determinante positivo, que asociamos al espín, y a otras contidades físicas como el iso-espín débil. Su base fundamental es un duplete con posiciones arriba y abajo.
#       
#    * **SU(3)** matrices unitarias de dimensión 3 con determinante positivo, que asociamos al color. Su base fundamental es un triplete, que asociamos en el caso del color con los tres colores: rojo, verde y azul.

# ### Definión de grupo
# 
# Un grupo es una colección de elementos ${g_i}$, finito o infinito, que complen las siguientre propiedades:
# 
#    * completitud: $g_i \cdot g_j = g_k$ (la multiplicación de dos elementos del grupo da otro elemento del grupo)
#    
#    * asociación: $g_i \cdot (g_j \cdot g_k) = (g_i \cdot g_j) \cdot g_k$
#    
#    * existencia de la identidad, $1$: $ g_i \cdot 1 = 1 \cdot g_i = g_i$
#    
#    * existencia de la inversa, $g_i \cdot g^{-1}_i = 1$ para todo $g_i$ del grupo.
#    
# Decimos que el grupo es abeliano si es conmutativo: $g_i \cdot g_j = g_j \cdot g_i$, si no, es no-abeliano
# 

# 
# #### Clasificación de grupos
# 
# Los grupos principales son:
# 
# * espacio-temporales, lo que incluye el grupo de Lorentz
# 
# * simetrías internas es una espacio interno, llamado *isotópico*, que pueden ser globales o locales (que depende del punto espacio-temporal).
# 
# Tambien se pueden calsificar los grupos en:
# 
# * continuos, si el grupo depende de un parámetro $\theta$ continuo, que puede ser compacto o no dependiendo si el intervalo del valor que toma $\theta$ es abierto o cerrado. Poe ejemplo el grupo de Lorentz que depende del parámetro $\beta = v/c$ no es compacto porque no puede tomar el valor $\beta = 1$.
# 
# * discretos, donde hay un grupo finito de elementos. Por ejemplo la inversión por paridad forma un grupo discreto de un elemento.
# 
# En Física de Partículas juegan un papel fundamental los **grupos de Lie** que son aquellos cuya elementos se relacionan de forma continua y diferenciable con respecto a un conjunto de parámetros $\theta$.
# 

# ### Representaciones
# 
# Llamamos **representación $D(g)$** lineal al conjunto de matrices asociadas con los elementos $g$ de un grupo $G$ que cumplen **las mismas reglas de multiplicación** que los elementos de $G$.
# 
# Esto es:
# 
# $$
#  g_i \cdot g_j = g_k \; \Rightarrow D(g_i) \, \cdot D(g_j) = D(g_k)
# $$
# 
# La **dimensión de la representación**, $n$, es la dimensión de las matrices.
# 
# Los vectores de dimensión $n$ sobre los que actuan las matrices forman la **base de la representación**.
# 
# Llamamos **representación fundamental** aquella representación con la dimensión menor. 

# #### representación irreducible
# 
# Decimos que una representación $D(g)$ de un grupo $G$ es reducible si puede descomponerse en representaciones más pequeñas.
# 
# Sea la representación $D(g)$ de matrices de dimensión $n\times n$ de un grupo, la representación es redudible si la matrix $D(g)$ se pueden descomponer en diagonal con matrices de menor dimensión, $D_1(g), D_2(g)$, de la forma:
# 
# $$
# D(g) = \begin{pmatrix} D_1(g) & 0 \\ 0 & D_2(g) \end{pmatrix}
# $$
# 
# También podemos decir que la base de la representación $D(g)$ se puede descomponer en varias bases, de tal forma que los elementos de cada sub-base forma su propio grupo. Los elementos de la base $D_1(g)$ no se mezclan con los elementos de la base de $D_2(g)$.
# 
# Como dijimos antes los estados de las partículas se representan en las bases de la representación irreducible.

# **Ejemplo**
# 
# El ejemplo conocido, y que veremos luego, es la combinación de espíns 1/2:
# 
# Sea el duplete:
# $$
# \uparrow = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \;\; \downarrow = \begin{pmatrix} 0 \\ 1 \end{pmatrix}.
# $$
# 
# Es la base de la representación fundamental de $SU(2)$ que tiene $2$ elementos.
# 
# Asociamos los elementos de la base a los estados de espín $|j, m\rangle$, donde $j=1/2, m = \pm 1/2$.

# El producto de dos dupletes, que se denota por ${\bf 2} \otimes {\bf 2}$, es: $\uparrow\uparrow, \uparrow\downarrow,
# \downarrow\uparrow, \downarrow\downarrow$.
# 
# La representación del la composición ${\bf 2} \otimes {\bf 2}$ tienen dimensión 4, cuya representación irreducible es está formada por un triplete y un singlete, que se denota como ${\bf 3} \oplus {\bf 1}$.
# 
# El triplete es:
# $$
# |1, 1 \rangle = \; \uparrow\uparrow, \;\;\;
# |1, 0 \rangle = \frac{1}{\sqrt{2}} (\uparrow\downarrow + \downarrow\uparrow),  \;\;\; |1, -1 \rangle, = \; \downarrow\downarrow, 
# $$
# y el singlete:
# $$
# |0, 0 \rangle = \frac{1}{\sqrt{2}}(\uparrow\downarrow - \downarrow\uparrow)
# $$
# 
# Que asociamos con los estados, $|j, m\rangle$, donde $j$ es el espín total $j = |j_1-j_2| \le j \le j_1 + j_2$ y $m = -j, \dots, j$, su tercera componente de espín.
# 
# Veremos abajo cómo se contruye esta representación irredudible.

# ## El grupo U(1)
# 
# Consideremos el grupo de las rotaciones de fase $\theta$ en el espacio complejo, $x  = a \, e^{i\phi}$, con $a>0$.
# 
# La rotación de un ángulo $\theta$ corresponde a un factor $D(\theta) = e^{i \theta} $:
# 
# $$
# x' = D(\theta) \, x = e ^{i\theta} \, x
# $$
# 
# La cantidad conservada es el módulo del número complejo: $a$.
# 
# Este grupo se denomina **grupo unitario** U(1).
# 

# La multiplicación de los elementos del grupo es:
# 
# $$
# D(\theta_1) \cdot D(\theta_2) = D(\theta_1 + \theta_2), \;\;\; D(\theta + 2 \pi) = D(\theta)
# $$
# 
# Nos dice que es un grupo abeliano. No importa el orden en que se realizan dos rotaciones. Esto es el orden de multiplicación.
# 
# La inversa es simplemente:
# 
# $$
# D(\theta) = e^{i\theta}, \;\;\; D^{-1}(\theta) = D(-\theta) = e^{-i\theta}
# $$
# 
# Vemos pues que la representación fundamental tiene dimensión 1.

# ### Relación con SO(2)
# 
# El grupo U(1) está relacionado con el grupo SO(2)
# 
# El grupo SO(2) está formado por la matrices de las rotaciones en dos dimensiones, $U(\theta)$, propias, esto es con el determinante positivo, $\mathrm{det}(U(\theta)) > 0$
# 
# Sea $(x, y)$ un punto del espacio bi-dimensional, la rotación es:
# 
# $$
# \begin{pmatrix} x' \\ y'\end{pmatrix} =
# \begin{pmatrix} \cos \theta & \sin \theta \\ - \sin \theta & \cos \theta \end{pmatrix} \, 
# \begin{pmatrix} x \\ y \end{pmatrix}
# $$
# 
# Donde 
# $$
# U(\theta) = \begin{pmatrix} \cos \theta & \sin \theta \\ - \sin \theta & \cos \theta \end{pmatrix}, \;\;\; 
# \mathrm{det}(D(\theta)) = 1
# $$
# 
# El grupo SO(2) es continuo y compacto, con $\theta \in [0, 2 \pi]$.
# 
# Esta simetría preserva la distancia, o el modulo del vector:
# 
# $$
# x'^2 + y'^2 = x^2 + y^2
# $$
# 

# Las reglas de multiplicación son:
# 
# $$
# U(\theta_1) \cdot U(\theta_2) = U(\theta_1 + \theta_2), \;\;\; U(\theta + 2 \pi) = U(\theta)
# $$
# 
# Si hacemos una rotación de $\theta_2$ después de hacer una rotación previa de $\theta_1$ es equivalente a rotar desde el principio con $\theta_1+\theta_2$
#  
# Es por lo tanto un grupo abeliano.
# 
# La inversa de una rotación con $\theta$ es simplemente rotar con $-\theta$:
# 
# $$
# U^{-1}(\theta) = U(-\theta)
# $$
# 
# En la representación matricial $2\times2$, la inversa es la transpuesta:
# 
# $$
# U(\theta)  = \begin{pmatrix} \cos \theta & \sin \theta \\ - \sin \theta & \cos \theta \end{pmatrix}, \;\;\;
# U^{-1}(\theta)  = U(-\theta) = U^T(\theta) = \begin{pmatrix} \cos \theta & -\sin \theta \\  \sin \theta & \cos \theta \end{pmatrix}
# $$

# ### Generadores
# 
# Podemos definir las rotaciones abusando de la notación de la exponencial y su desarrollo de Taylor:
# 
# $$
# U(\theta) = e ^{i \theta T} = \sum_n \frac{(i\theta T)^n}{n!} = 1 + i \theta T - \frac{\theta^2}{2} T^2 + \dots  = 
# I \cos \theta + T \sin \theta = 
# \begin{pmatrix} \cos \theta & \sin \theta \\ - \sin \theta & \cos \theta \end{pmatrix},
# $$
# 
# si  definimos:
# 
# $$
# T = -i \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}
# $$
# 
# $T$ es el **generador** de la rotación infinitesimal $\delta \theta$ del grupo SO(2):
# 
# $$
# U(\delta \theta) \simeq 1 + i T \, \delta \theta
# $$
# 
# Notar que el generador es hermítico: $T^\dagger = T^T = T$

# Las reglas de multiplicación del grupo SO(2) y $U(1)$ son las mismas:
# 
# $$
# D(\theta_1) \cdot D(\theta_2) = D(\theta_1 + \theta_2), \;\;\; D(\theta + 2 \pi) = D(\theta)
# $$
# 
# Decimos que dos grupos son **isomorfos** si comparten las mismas reglas de multiplación.
# 
# En este caso SO(2) y U(1) son isomorfos.
# 
# $$
# SO(2) \sim U(1)
# $$
# 
# La base de uno y otro se relacionan por:
# 
# $$
# |a| e^{i\phi} \leftrightarrow (x, y) = (|a| \cos \theta, |a| \sin \theta)
# $$
# 
# Notar que el generador de U(1) es simplemente 1.

# ## El grupo SU(2)
# 
# **SU(2)** es el grupo que forman las **matrices complejas unitarias $2 \times 2$ con determinante unidad**.
# 
# $$
# U^\dagger U = I, \;\; \mathrm{det}(U) = 1
# $$
# 
# Las matrices de SU(2) se definen con 3 parámetros, o ángulos: $\alpha_i, \, i = 1, 2, 3$.
# 
# Notar que hay 8 parámetros reales en $U$, la condición de unitariedad impone 4 ligaduras, y la del determinante una más. 
# 
# Las matrices pueden construirse a partir de los ángulos $\alpha_i$ y 3 **generadores** $T_i$, con $i=1, 2, 3$:
# 
# $$
# U({\bf \alpha}) = e^{i \alpha \cdot {\bf T}} = e^{i \sum_i \alpha_i T^i}
# $$

# Los generadores son las matrices de Pauli:
# 
# $$
# T^i = \frac{1}{2} \sigma^i
# $$
# 
# $$
# \sigma^1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \;\;
# \sigma^2 = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \;\;
# \sigma^3 = \begin{pmatrix} 1 & 0 \\ 0 & -1\end{pmatrix}.
# $$
# 
# que cumplen la siguientes relaciones:
# 
# $$
# [T^1, T^2] = i T^3, \;\; [T^2, T^3] = i T^1, \;\; [T^3, T^1] = i T^2,
# $$
# 
# donde $[T^1, T^2] = T^1 T^2 - T^2 T^1$ es el conmutador.
# 

# 
# esto es:
# $$
# [T^i, T^j] = i \epsilon_{ijk} T^k,
# $$
# donde $\epsilon_{ijk}$ es el tensor de Levi-Civita de 3 dimensiones.
# 
# La relación anterior es la **algebra de Lie** de grupo. Todos los grupos que tengas esta álgebra son isomorfos.
# 
# Reconocemos en ellas las reglas de conmutación del momento angular y del espín ${\bf S}$:
# 
# $$
# [S^i, S^j] = i \epsilon_{ijk} S^k
# $$
# 

# Por similitud con el espín, sabemos que existen dos operadores que conmutan, ${\bf T}^2$, que llamaremos **isoespín** para distinguirlo del espín, y la tercera componente de isospin, $T_3$:
# 
# $$
# {\bf T}^2 = (T^1)^2 + (T^2)^2 + (T^3)^3, \;\; T_3; \;\; [{\bf T}^2, T_3] = 0
# $$
# 
# Sabmos igualmente que podemos dar los estados con los autovalores del ${\bf T}^2, T_3$, $|I, I_3 \rangle$.
# 
# $$
# {\bf T}^2 \, |I, I_3 \rangle = I (I+1) \, | I, I_3 \rangle, \\ T_3 \, |I, I_3 \rangle = I_3 \, |I, I_3 \rangle.
# $$
# 
# La base de la representación fundamental de SU(2) será, por similitud con el espín, $|1/2, \pm 1/2 \rangle$:
# 
# $$
# \uparrow = \begin{pmatrix} 1 \\ 0 \end{pmatrix} = |1/2, 1/2 \rangle, \;\; 
# \downarrow =  \begin{pmatrix} 0 \\ 1 \end{pmatrix} = |1/2, -1/2 \rangle.
# $$

# ### representaciones 
# 
# Podemos construir representaciones de SU(2) de mayor dimensión a partir del producto (o combinación) de los estados de la base de la representación fundamental: $\uparrow, \downarrow$.
# 
# El producto de dos dupletes, ${\bf 2} \otimes {\bf 2}$, es: $\uparrow\uparrow, \uparrow\downarrow,
# \downarrow\uparrow, \downarrow\downarrow$, tiene dimensión 4.
# 
# Y su representación irreducible es un triplete y un singlete: $\bf{3} \oplus {\bf 1}$
# 
# Para construirla utilizamos los operadores escalera:
# 
# $$
# T^\pm \equiv T^1 \pm i T^2 
# $$
# 
# que en la representación fundamental son simplemente:
# 
# $$
# T^+ = \sigma^+ = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \;\;\;
# T^- = \sigma^- = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}.
# $$
# 

# Aplicando los operadores escalera obtenemos los estados del triplete, $I = 1$:
# 
# $$
# \uparrow\uparrow = |1, 1 \rangle, \\
# T_- \, \uparrow\uparrow = (T_-\uparrow)\uparrow + \uparrow (T_- \uparrow), \\
# T_- \left(\uparrow\downarrow + \downarrow\uparrow \right) = 2 \downarrow \downarrow
# $$
# 
# Que si normalizamos:
# $$
# |1, 1 \rangle =  \uparrow\uparrow, \;\; 
# |1, 0 \rangle = \frac{1}{\sqrt{2}} (\uparrow\downarrow + \downarrow\uparrow), \;\;
# |1, -1 \rangle = \downarrow\downarrow.
# $$
# 
# y finalmente, el miembro del singlete $I= 0$ es el ortogonal a $|1, 0 \rangle$:
# 
# $$
# |0, 0 \rangle = \frac{1}{\sqrt{2}}(\uparrow\downarrow - \downarrow\uparrow)
# $$
# 
# Los miembros del triplete son simétricos bajo el intercambio $\uparrow \leftrightarrow \downarrow$ y el singlete anti-simétrico.

# ### reglas de combinación de espinores:
# 
# 
# Para construir las representaciones de SU(2) de mayor dimensión recurrimos a las reglas que conocemos del espín. O a partir de primeros principios con los operadores escalera.
# 
# Recordemos que los operadores escalrera actuan sobre la base $|I, I_3 \rangle$ de la siguiente forma:
# 
# $$
# T_+ \, |I, I_3 \rangle = \sqrt{I (I+1) - I_3 (I_3+1)} \, |I, I_3 \rangle, \\
# T_- \, |I, I_3 \rangle = \sqrt{I (I+1) - I_3 (I_3-1)} \; |I, I_3 \rangle. \\
# $$
# 
# suben arriba y abajo en los elementos de la base.

# Una representación irreducible tiene como base $|I, I_3 \rangle$ donde $I$ puede tomar los valores, $I = 0, \frac{1}{2}, 1, \frac{3}{2}, \dots$, e $I_3 = -I, \dots, I$, con incrementos de 1. 
# 
# La dimesión de la representación irreducible es $2I + 1$.
# 
# La representación irreducible de la combinación de dos representaciones de menor dimension, cuyas bases son, $|I^1, I^1_3 \rangle, \; | I^2, I^2_3 \rangle$, tendrá una dimensión $(2I^1+1) (2I*2+1)$, y se compondrá de varios multipletes $| I, I_3 \rangle$, con valores de $I, I_3$ que vienen dados por:
# 
#    * $I = |I^1 - I^2|, \dots, I^1 + I^2$, en incrementos de 1.
#    
#    * $I_3 = -I , \dots, I $, en incrementos de 1.
#       
# Los elementos de la base $|I, I_3 \rangle$ se relacionan con los iniciales $|I^1, I^1_3\rangle |I^2, I^2_3 \rangle$ mediante los coeficientes de [Clebsch-Gordan](https://pdg.lbl.gov/2018/reviews/rpp2018-rev-clebsch-gordan-coefs.pdf).

# ### Relación con el grupo SO(3)
# 
# Llamamos SO(3) al grupo de las rotaciones en el espacio vectorial de tres dimensiones, cuya matriz de rotación tiene determinate positivo.
# 
# $$
# x' = U({\bf \theta}) \, x, \;\;\; \mathrm{det}(U(\theta)) = 1
# $$
# 
# $U(\theta)$ una matriz real $3 \times 3$, y $x = (x, y, z)$ un vector tridimensional.
# 
# Las matrices quedan definidas con tres ángulos, $\theta_i$, con $i=1, 2, 3$, 

# La matrix $U(\theta)$ puede obtenerse a partir de los 3 generadores $T^i$ del grupo:
# 
# $$
# U(\theta) = e^{i \theta \cdot T} = e^{i \sum_{i=1}^3 \theta_i T^i}
# $$
# 
# Notar que $\theta \cdot T$ es el producto de los tres angulos $\theta_i$ con tres generadores $T^i$, con $i=1, 2, 3$
# 
#     
# $$
# T^1 = -i \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0\end{pmatrix}, \;\;
# T^2 = -i \begin{pmatrix} 0 & 0 & -1 \\ 0 & 0 & 0 \\ 1 & 0 & 0\end{pmatrix}, \;\;
# T^3 = -i \begin{pmatrix} 0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 0\end{pmatrix}. 
# $$
# 

# 
# Las reglas de conmutación de los generadores son:
# 
# $$
# [T^1, T^2] = iT^3, \;\;\; [T^2, T^3] = i T^1, \;\;\; [T^3, T^1] = i T^2
# $$
# 
# Esto es:
# 
# $$
# [T^i, T^j] = i \, \epsilon_{ijk} \, T^k
# $$
# 
# donde $\epsilon_{ijk}$ es el tensor Levi-Civita.
# 
# Notar que las reglas de conmutación coinciden con las del grupo $SU(2)$ y son las del momento angular y el espín.

# El grupo SU(2) es por lo tanto isomorfo a SO(3)
# 
# $$
# SU(2) \sim SO(3)
# $$
# 
# En el caso $SO(2)$ y $U(1)$, habíamos visto que sus elementos se relacionaban:
# 
# $$
# (x, y) \sim a e^{i\phi}
# $$
# 
# En este caso la relación viene de:
# 
# $$
# (x, y, z) \sim \sigma \cdot x = \begin{pmatrix} z & x - i y \\ x + iy & z \end{pmatrix}
# $$
# 

# ### Rotación de espín
# 
# Es común que a la hora de calcular elementos de matriz utilizando los espinores de helicidad de Dirac en secciones eficaces nos aparezca una configuración de espín de salida $|1, m' \rangle_{\theta}$ en una dirección ${\bf n} = (\sin \theta, 0, \cos \theta)$ que queremos relacionar con configuración de espín inicial en $|1, m \rangle$ en el eje $z$.
# 
# Es el caso de la interacción $e + e^+ \to \mu + \mu^+$ donde hay cuatro combinaciones posibles de helicidad:
# 
# | |
# | :--: |
# |  <img src="./imgs/hadrons_eemumu_helicities_2.png" width = 700 align="center"> |
# | Las posibles combinaciones de helicidad de $e^++e \to \mu^+ + \mu$ en el CM, negro (momento), azul (espín) |
# 
# En la figura representamos el plano $(x, z)$ donde el eje $z$ viene dado por la dirección del electrón incidente, la dirección de salida del muón, es por lo tanto ${\bf n} = (\sin \theta, 0, \cos \theta)$
# 
# Por ejemplo, en la interacción $RL\to RL$ (derecha) la composición de espín de entrada es $|1, 1 \rangle$ y la de salida $|1, 1 \rangle_{\theta}$.
# 

# Los generadores de las rotaciones que se corresponden a la representación de dimensión 1 del grupo SU(2) se pueden obtener a partir de los operadores escalera.
# 
# La matriz de rotación de un ángulo $\theta$ en torno al eje $y$ es:
# 
# $$
# U(\theta) = e^{i \theta T^2} = \sum_n \frac{(i\theta T^2)^n}{n!}
# $$
# 
# Donde 
# $$
# T^2 = - \frac{i}{2} (T^+ - T^-)
# $$
# 
# Teniendo en cuenta el funcionamiento de los operadores escalera para $I = 1$ (ver la definición anterior) construimos su representación matricial:
# 
# $$
# T^+ = \begin{pmatrix} 0 & \sqrt{2} & 0 \\ 0 & 0 & \sqrt{2} \\ 0 & 0 & 0 \end{pmatrix}, \;\;\;
# T^- = \begin{pmatrix} 0 & 0 & 0 \\ \sqrt{2} & 0 & 0\\ 0 & \sqrt{2} & 0\end{pmatrix}
# $$

# Y por lo tanto:
# $$
# T^2 = -\frac{i}{\sqrt{2}} \begin{pmatrix} 0 & 1 & 0 \\ -1 & 0 & 1 \\ 0 & -1 & 0\end{pmatrix}
# $$
# 
# Se puede comprobar fácilmente que: $(T^2)^{2n+1} = T^2$, y:
# 
# $$
# (T^2)^{2n} = \frac{1}{2} \begin{pmatrix}1  & 0 & -1 \\ 0 & 2 & 0 \\ -1 & 0 & 1 \end{pmatrix}
# $$
# con $n=1, 2, \dots$

# Del desarrollo:
# 
# $$
# U(\theta) = I + i \theta T^2 - \frac{\theta^2}{2} (T^2)^2 + \dots = I + i T^2 \sin \theta + (T^2)^2 (\cos \theta -1)
# $$
# 
# Obtenemos:
# 
# $$
# U(\theta) = \begin{pmatrix} 
# \frac{1}{2} (1+\cos \theta) & \frac{1}{\sqrt{2}} \sin \theta &\frac{1}{2}(1-\cos \theta) \\
# -\frac{1}{\sqrt{2}} \sin \theta & \cos \theta &  \frac{1}{\sqrt{2}}\sin \theta  \\
# \frac{1}{2} (1-\cos \theta) & -\frac{1}{\sqrt{2}} \sin \theta &\frac{1}{2}(1+\cos \theta) \\
# \end{pmatrix}
# $$
# 

# El estado de espín total de salida, $|1, 1 \rangle_{\theta}$, se relaciona con los posibles estados de espín total de entrada, $|1, m \rangle$ con $m=1, 2, 3$, por:
# 
# $$
# | 1, 1 \rangle_{\theta} = \frac{1}{2}(1+\cos \theta) \, |1, 1 \rangle + \frac{1}{\sqrt{2}} \sin \theta \, |1, 0 \rangle + \frac{1}{2}(1-\cos \theta) \, |1, -1 \rangle
# $$
# 
# En el caso de $RL\to RL$ (derecha), el factor asociado al espín que entra en el elemento de Matriz, $M_{fi}$, es:
# 
# $$
# \langle 1, 1 | 1, 1 \rangle_\theta = \frac{1}{2}(1+\cos \theta)
# $$
# 
# Puede observarse por el dibujo (derecha) que para $\theta = 0$ el factor es 1 (el espín total de entrada y salida están alineados) y para $\theta = \pi$, el factor es nulo (el espín total de entrada y salida son opuestos)
# 

# 
# La matriz anterior $U_{m', m} (\theta)$, donde $m$ corresponde a los índices $m = -1, 0, 1$ del espín total de entrada y $m' = -1, 0, 1$ del de salida, es la (transpuesta) de la matriz "*Darkstellung*" $d^j_{m, m'}(\theta)$ de Wigner con $j=1$.
# 
# Los elementos $d^j_{m, m'}(\theta)$ de la matrices $d$ se definen:
# 
# $$
# d^j_{m', m} (\theta) = \langle m', j | e^{-i\theta T^2} | j, m \rangle
# $$
# 
# Los valores $d^j_{m', m}(\theta)$ de las configuraciones de $j$ menores se recogen en la tabla del PDG de los [Clebsch-Gordan](https://pdg.lbl.gov/2018/reviews/rpp2018-rev-clebsch-gordan-coefs.pdf).
# 
# En nuestro caso, donde $\theta$ es la rotación del sistema de referencia, utilizaríamos:
# 
# $$
# |j, m' \rangle_{\theta} = \sum_{m=-j}^j d^j_{m', m}(-\theta) \, |j, m \rangle = \sum_{m=-j}^j d^j_{m, m'}(\theta) \, |j, m \rangle
# $$
# 
# 

# ## SU(3)
# 
# SU(3) es el grupo de las **matrices unitarias $3 \times 3$ con determinante unidad**. Cumplen:
# 
# $$
# U^\dagger U = I, \;\; \mathrm{det}(U) = 1
# $$
# 
# El número de parámetros reales libres de SU(3) es 8. Hay $18 = 3 \times 3 \times 2$ parámetros reales en $U$, 9 ligaduras de la condición de unitariedad y 1 por ser el determinante unidad.
# 
# Las matrices de SU(3) pueden construirse a partir 8 generadores hemíticos, $T_i$, y $\alpha_i$ con $i = 1, \dots, 8$
# 
# $$
# U(\alpha) = e^{i \alpha \cdot {\bf T}}
# $$
# 
# 
# 
# 

# Notar que los generadores de los grupos unitarios son hermíticos.
# 
# Si hacemos una rotación infinitesimal $\delta \alpha_i$ tenemos:
# 
# $$
# I = U(\delta \alpha_i)^\dagger U(\delta \alpha_i) \simeq (I - i \alpha_i T^{i\dagger})(I + i \alpha_i T^i) \simeq I + i \alpha_i (T^i - T^{i\dagger}), \;\;\; T^{i\dagger} = T
# $$
# 
# para cada generador $T^i$.

# Una opción conveniente para los generadores es construirlos a partir de las matrices $\lambda$ de Gell-Mann.
# 
# $$
# T_i = \frac{1}{2} \lambda_i
# $$
# 
# Donde:
# 
# $$
# \lambda_1 = \begin{pmatrix} 0 & 1 & 0  \\  1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \;
# \lambda_2 = \begin{pmatrix} 0 & -i & 0 \\  i & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \;
# \lambda_3 = \begin{pmatrix} 1 & 0 & 0  \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \;
# \lambda_4 = \begin{pmatrix} 0 & 0 & 1  \\ 0 & 0 & 0  \\ 1 & 0 & 0 \end{pmatrix}, \\
# \lambda_5 = \begin{pmatrix} 0 & 0 & -i \\ 0 & 0 & 0  \\ i & 0 & 0 \end{pmatrix}, \;
# \lambda_6 = \begin{pmatrix} 0 & 0 & 0  \\ 0 & 0 & 1  \\ 0 & 1 & 0 \end{pmatrix}, \;
# \lambda_7 = \begin{pmatrix} 0 & 0 & 0  \\ 0 & 0 & -i \\ 0 & i & 0 \end{pmatrix}, \;
# \lambda_8 = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 & 0 & 0  \\ 0 & 1 & 0 \\ 0 & 0 & -2 \end{pmatrix}.
# $$
# 

# Solo existen tres generadores que conmutan: $T^2, T_3, Y_8$, que asociaremos con los números cuánticos de *tercera componente del espinor*, $I_3$, e **hypercarga**, $Y$.
# 
# $$
# T^2 = \sum_{i=1}^8 T^2_i = \frac{4}{3}    \begin{pmatrix} 1 & 0 & 0 \\ 0 &  1 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \;
# T_3 = \frac{1}{2} \lambda_3 = \frac{1}{2} \begin{pmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \;
# \;\; Y = \frac{1}{\sqrt{3}} \lambda_8   = \frac{1}{3} \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -2 \end{pmatrix}, 
# $$
# 
# Si definimos como los spinors: 
# $$
# r = \begin{pmatrix} 1 \\ 0 \\ 0\end{pmatrix}, \;
# g = \begin{pmatrix} 0 \\ 1 \\ 0\end{pmatrix}, \;
# b = \begin{pmatrix} 0 \\ 0 \\ 1\end{pmatrix}.
# $$
# 
# Tienen como tercera componente e hypercarga:
# 
# $$
# T_3 r = 1/2 r, \; Y r = 1/3 r; \;\;\;
# T_3 g = -1/2 g, \; Y g = 1/3 g; \;\;\;
# T_3 b = 0, \; Y b = -2/3 b
# $$

# El grupo SU(3) se asocia a los quarks, que tendrán carga $r, g, b$, los anti-quarks tendrán cargas opuestas (opuesta $T_3 ,Y$). 
# 
# La figura muestra los dos tripletes de color ${\bf 3}$ de los quarks y anti-quarks ${\bf \bar{3}}$ en el plano $T_3, Y$. Notar la oposición de cargas.
# 
# | | 
# | :--: |
# |  <img src="./imgs/groups_su3_color.png" width = 500 align="center"> |
# | Posición de $r, g, b$ en $(T_3, Y)$, ${\bf 3}$, y la dirección de los operadores $T_\pm, V_\pm, U_\pm$ (izda) y de los $\bar{r}, \bar{g}, \bar{b}$, ${\bf \bar{3}}$, (derecha)|
# 
# 
# Existen ahora tres operadores escalera:
# $$
# T_{\pm} = \frac{1}{2}(\lambda_1 \pm i \lambda_2), \; V_{\pm} = \frac{1}{2} (\lambda_4 \pm i \lambda_5), \; 
# U_{\pm} = \frac{1}{2}(\lambda_6 \pm 9 \lambda_7)
# $$
# 
# que suben y bajan los spinores a lo largo de la direcciones del triángulo (ver figura):
# 
# $$
# T_+ g = r, \; T_- r = g; \;\;\; V_+ b = r, \; V_-r = b; \;\; U_+b = g, \; U_-g = b
# $$
# 

# La combinación de quark y antiquark (mesón), ${\bf 3} \otimes {\bf \bar{3}}$, se efectua colocando el triplete ${\bf \bar{3}}$ sobre los vértices del ${\bf 3}$. Lo que da lugar a un octete y un singlete en su representación irreducible.
# 
# | | 
# | :--: |
# |  <img src="./imgs/groups_su3_33b.png" width = 550 align="center"> |
# | Combinación ${\bf 3} \otimes {\bf \bar{3}} = {\bf 8} \oplus {\bf 1}$ [MT]|
# 
# El único singlete ($T_3, Y = 0$), sin color, es:
# 
# $$
# \frac{1}{\sqrt{3}} (r\bar{r} + g\bar{g} + b \bar{b})
# $$

# La combinación de dos quarks, ${\bf 3} \otimes {\bf 3}$, no da un singlete de color pero sí la de tres quarks, ${\bf 3} \otimes {\bf 3} \otimes {\bf 3}$ (ver figura).
# 
# | | 
# | :--: |
# |  <img src="./imgs/groups_su3_333.png" width = 550 align="center"> |
# | Combinación ${\bf 3} \otimes {\bf \bar{3}} = {\bf 10} \oplus {\bf 8} \otimes {\bf 8} \otimes {\bf 1}$ [MT]|
# 
# El singlete ($T_3, Y = 0$), sin color, es:
# 
# $$
# \frac{1}{\sqrt{6}} (rgb - rbg + gbr - grb + brg - brg)
# $$
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


plt.figure(figsize = (12, 5));
plt.subplot(1, 2, 1)
plt.xlim((-1, 1)); plt.ylim((-1, 1)); plt.gca().set_aspect('equal');
plt.plot((-1, 1), (0, 0), c = 'grey'); plt.plot((0, 0), (-1, 1), c = 'grey');
opts = {'length_includes_head' : True, 'head_width' : 0.1, 'head_length' : 0.1}
plt.arrow(0, -2/3, 1/2, 1, color = 'magenta', width = 0.01, **opts)
plt.arrow(1/2, 1/3, -1/2, -1, color = 'magenta', width = 0.01, **opts)
plt.arrow(0, -2/3, -1/2, 1, color = 'cyan', width = 0.01,  **opts)
plt.arrow(-1/2, 1/3, 1/2, -1, color = 'cyan', width = 0.01,  **opts)
plt.arrow(-1/2, 1/3, 1, 0, color = 'yellow', width = 0.01,  **opts)
plt.arrow(1/2, 1/3, -1, 0, color = 'yellow', width = 0.01,  **opts)
plt.scatter((1/2), (1/3), c = 'r', label = 'r', s = 200 , alpha = 1 );
plt.scatter((-1/2), (1/3), c = 'g', label = 'g', s = 200, alpha = 1 );
plt.scatter((0), (-2/3), c = 'b', label = 'b', s = 200, alpha = 1 );
epsilon = 1/6
plt.annotate('r', (0, 0), (1/2, 1/3 + epsilon), fontsize = 15)
plt.annotate('g', (0, 0), (-1/2, 1/3 + epsilon), fontsize = 15)
plt.annotate('b', (0, 0), (0, -2/3 - epsilon), fontsize = 15)
plt.annotate('$T_{\pm}$', (0, 0), (0, 1/3 + epsilon), fontsize = 15)
plt.annotate('$V_{\pm}$', (0, 0), ( 1/4  + epsilon/2, -1/4), fontsize = 15)
plt.annotate('$U_{\pm}$', (0, 0), (-1/4  - epsilon, -1/4), fontsize = 15);
plt.xlabel('$T_3$'); plt.ylabel('Y');

plt.subplot(1, 2, 2)
plt.scatter((-1/2), (-1/3), c = 'r', label = 'r', s = 200 , alpha = 1 );
plt.scatter((1/2), (-1/3), c = 'g', label = 'g', s = 200, alpha = 1 );
plt.scatter((0), (2/3), c = 'b', label = 'b', s = 200, alpha = 1 );
#plt.annotate(s = 'r', xy =  (0, 0), xytext = (1/2, 1/3), fontsize = 16)
plt.xlim((-1, 1)); plt.ylim((-1, 1)); plt.gca().set_aspect('equal');
plt.plot((-1, 1), (0, 0), c = 'grey'); plt.plot((0, 0), (-1, 1), c = 'grey');
plt.plot((-1/2, 1/2), (-1/3, -1/3), c = 'black', ls = '--' )
plt.plot((-1/2, 0)  , (-1/3,  2/3), c = 'black', ls = '--' )
plt.plot((0  , 1/2) , (2/3, -1/3), c = 'black', ls = '--' )
epsilon = 1/6
plt.annotate(r'$\bar{r}$', (0, 0), (-1/2, -1/3 - 1.5 * epsilon), fontsize = 15)
plt.annotate(r'$\bar{g}$', (0, 0), (1/2, -1/3 - 1.5 * epsilon), fontsize = 15)
plt.annotate(r'$\bar{b}$', (0, 0), (0, 2/3 + epsilon), fontsize = 15)
plt.xlabel('$T_3$'); plt.ylabel('Y');


# In[4]:


r = 4.1
hbarc = 0.197e-15
r_si = r * hbarc
r_si


# ## Relatividad special
# 
# ### Transformación de Lorentz
# 
# La transformación de Lorentz nos relaciona el espacio-tiempo $(t, {\bf r})$ en un sistema inercial $\Sigma$ con el espacio-tiempo $(t' {\bf r}')$ en otros sistema inercial $\Sigma'$ que se desplaza respecto del primero con velocidad $v$ en la dirección $z$.
# 
# Einstein postuló que la velocidad de la luz, $c$, es la misma en los dos sistemas y nada puede viajar más rápido que la luz $v \lt c$. 
# 
# La luz producida en $t = t' = 0$, cuando los dos sistemas coinciden, cumple: $c^2t^2 - {\bf r}^2 = c^2 t'^2 - {\bf r}'^2$. 
# 
# Si ${\bf r} = (x, y, z)$, y de igual manera para ${\bf r}' = (x', y', z')$.
# 

# La condición se cumple si las coordenadas en los dos sistemas están relacionadas por la transformación de Lorentz.
# 
# $$
# t' = \gamma \left( t - \frac{v}{c^2} z \right), \; x' = x, \; y' = y, \, z' = \gamma (z - vt) 
# $$
# 
# donde introducimos, el **factor de Lorentz**, $\gamma$
# 
# $$
# \gamma = (1 - \beta^2)^{-1/2}, \; \; \beta = v/c 
# $$
# 

# En UN:
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
# El primero, $x^\mu$, se llama cuadri-vector **contra-variante** y el segundo, $x_\mu$, **co-variante**.
# 
# Están relacionados: $x_\mu = g_{\mu\nu} x^\nu$. 
# 
# Donde:
# 
# $$
# g_{\mu\nu } = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}
# $$
# 
# El producto entre ambos se puede denotar:
# 
# $$a^\nu b_\nu = g_{\nu\mu} a^\nu b^\mu = a_\mu b^\mu$$

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
# La energía y el momento de una partícula relativista de masa $m$ y velocidad ${bf v}$ son $E = \gamma m c^2, \, {\bf p } = \gamma m {\bf v}$, en UN: $E = \gamma m, \; {\bf p} = \gamma m \bf{\beta}$.
# 
# Definimos el **cuadrimomento** (como vector contra-variante); $p^\mu = (E, p_x, p_y, p_z)$
# 
# El producto escalar es un invariante Lorentz:
# $$
# p^\mu p_\mu = E^2 - {\bf p}^2 = m^2,
# $$ 
# dado que para el caso ${\bf p} = {\bf 0}$, tenemos $p_\mu = (m, 0, 0, 0)$.
# 
# Esta expresion es *la relación de Einstein entre energía y momento*.
# 
# En general usaremos $p^2 = p^\mu p_\mu$, para distinguirlo del producto escalar en 3D, ${\bf p}^2  = |{\bf p}|^2$

# En el caso de un sistema de n partículas, $i = 1, \dots, n$, el cuadrimomento total,
# 
# $$
# p^\mu = \sum_i p^\mu_i,
# $$
# cumple:
# $$
# p^\mu p_\mu = \left(\sum_i E_i \right)^2 - \left( \sum_i {\bf p}_i \right)^2
# $$
# 
# En el caso en el que una partícula se desintegre $a \to c + d$, se cumple:
# 
# $$
# (p_b + p_c)^\mu (p_b + p_c)_\mu = p_a^\mu p_{a\mu} = m^2_a,
# $$
# que se demonima la *masa invariante*.

# En general usaremos $p^2 = p^\mu p_\mu$, para distinguirlo del producto escalar en 3D, ${\bf p}^2  = |{\bf p}|^2$
# 
# Para indicar que el evento está definido en el caso especial del *sistema del centro de masas*, dode se cumple $\sum_i {\bf p}_i = {\bf 0}$, usamos la notación, $p^*$ para el cuadri-momento, y $E^*, {\bf p}^*$ para la energía y el momento.

# ### Derivadas co-variantes y contra-variantes
# 
# La derivadas primeras en el sistema $\Sigma'$ se relactionan con las del sistema $\Sigma$ por:
# 
# 
# $$
# \begin{pmatrix} \partial/\partial t' \\ \partial/\partial x' \\  
#                 \partial/\partial y'  \\ \partial/\partial z' \end{pmatrix} = 
# \begin{pmatrix} \gamma         & 0 & 0 & +\gamma \beta \\
#                  0             & 1 & 0 & 0 \\
#                  0             & 0 & 1 & 0 \\
#                  +\gamma \beta & 0 & 0 & \gamma 
# \end{pmatrix}
# \begin{pmatrix} \partial/\partial t \\ \partial/\partial x \\  
#                 \partial/\partial y  \\ \partial/\partial z \end{pmatrix}
# $$
# 
# 

# Luego:
# $$\partial_\mu = \left( \frac{\partial}{\partial t}, \frac{\partial}{\partial x},  
#                 \frac{\partial}{\partial y}, \frac{\partial}{\partial z}  \right),$$
# transforma como un cuadri-vector *co-variante*. Es **derivada co-variante**
# 
# Y
# $$\partial^\mu = \left( \frac{\partial}{\partial t}, -\frac{\partial}{\partial x},  
#                 -\frac{\partial}{\partial y}, -\frac{\partial}{\partial z}  \right),$$
# lo hace como *contra-variante*. Es la *derivada contra-variante*.
# 
# El equivalente para el laplaciando para cuadri-vectores:
# 
# $$
# \Box^2 = \partial^\mu \partial_\mu = \frac{\partial}{\partial t^2} -\frac{\partial}{\partial x^2}   
#                 -\frac{\partial}{\partial y^2} -\frac{\partial}{\partial z^2}
# $$
# 

# ### Invariables de Mandelstam
# 
# 
# Si consideramos los siguientes diagramas de Feynman,
# 
# | | 
# | :-- |
# | <img src="./imgs/feynman_channels.png" width = 500 align="center">|
# | Diagramas asociados a los invariantes de Mandelstam |
# 
# 
# El cuadrimomento transferido, $q^2$, entre las corrientes, se denota con:
# 
# $$
#  s = (p_a +p_b)^2 = (p_c + p_d)^2, \;\; t = (p_c - p_a)^2 = (p_d - p_b)^2, \;\; u = (p_d-p_a)^2 = (p_c - p_b)^2
# $$
# 
# que corresponden a los **canales $s, t, u$**. 
# 

# 
# El canal $s$ es relevante en los procesos de aniquilación, y correponde a la energía en el centro de masas.
# 
# $$s = (p_a + p_b) = (E^*_a + E^*_b) - (p^*_a + p^*_b) = (E^*_a + E^*_b)^2.$$
# 
# En la literatura se denota $\sqrt{s}$ para indicar la **energía en el centro de masas** de los colisionadores.
# 
# El canal $t$ es relevante en los procesos de dispersión (*scattering*), y el $u$ en los procesos con partículas finales indistiguibles.

# ## Ecuación de Dirac
# 
# La ecuación de Klein-Gordon se deriva de la equación de momento-energía: $E^2 = {\bf p}^2 + m^2$, considerando los operadores cuánticos de momento y energía. 
# 
# Dirac propuso una solución lineal en espacio-tiempo, que cumpliese también la equación momento-energía. La **ecuación de Dirac** es:
# 
# $$
# \hat{E} \Psi = \left( \bf{\alpha} \cdot {\bf \hat p} + \beta m \right) \Psi
# $$
# 
# Las condiciones de ${\bf \alpha}, \beta$, se obtienen al elevar 'al cuadrado' la ecuación.
# 
# $$
# \alpha_i^2 = \beta^2 = I, \;\; \alpha_i \alpha_j + \alpha_j \alpha_i = 0 \; (i \neq j), \;\; \alpha_i \beta + \beta \alpha_i = 0,
# $$
# con $i= 1, 2, 3$. 
# 
# Y deben ser además hermíticas $\alpha_i = \alpha_i^\dagger, \; \beta = \beta^\dagger$, para que el hamiltonianto tenga valores reales.
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

# ### Representación covariante de la ecuación de Dirac 
# 
# La **matrices-$\gamma$** se definien:
# 
# $$
# \gamma^0 = \beta, \;\; \gamma^1 = \beta \alpha_1, \; \gamma^2 = \beta \alpha_2, \;\; \gamma^3 = \beta \alpha_3
# $$
# 
# y usamos la derivada covariante:
# 
# $$\partial_\mu = \left( \frac{\partial}{\partial t}, \frac{\partial}{\partial x},  
#                 \frac{\partial}{\partial y}, \frac{\partial}{\partial z}  \right),$$
# 
# podemos reexpresar la ecuación de Dirac de forma covariante:
# 
# $$
# (i \gamma^\mu \partial_\mu  - m) \Psi = 0.
# $$

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
# La matrix $\gamma^0$ es hermítica y las otras anti-hermíticas:
# 
# $$
# \gamma^{0\dagger} = \gamma^0, \;\; \gamma^{k \dagger} = - \gamma^{k}
# $$

# #### Matrix $\gamma^5$
# 
# Definimos la matrix $\gamma^5$ a partir del producto del resto de matrices gamma:
# 
# $$
# \gamma^5 \equiv i \gamma^0 \gamma^1 \gamma^2 \gamma^3,
# $$
# 
# La matrix $\gamma^5$ tiene las siguientes propiedades:
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

# #### El espinor de Dirac
# 
# El Hamiltoniano de Dirac actua en un función de onda, $\Psi$ de cuatro componentes, quer se llama **cuadri-spinor de Dirac**:
# 
# $$
# \Psi = \begin{pmatrix} \psi_1 \\ \psi_2 \\ \psi_3 \\ \psi_4\end{pmatrix}.
# $$
# 
# Es conveniente definir el **spinor adjunto** como:
# 
# $$
# \bar{\Psi} = \Psi^\dagger \gamma^0
# $$
# 
# Esto es, en la representación de Pauli-Dirac:
# $$\bar{\Psi} = (\psi^*_1, \psi^*_2, -\psi^*_3, -\psi^*_4).$$

# #### Densidad y corriente de probabilidad
# 
# La densidad y la corriente de probabilidad del spinor de Dirac son:
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
# Podemos introducir el cuadri-vector covariante de las corriente de probabilidad:
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

# ### Soluciones de la ecuación de Dirac
# 
# Las soluciones de onda plana de una partícula libre serán de la forma:
# 
# $$
# \Psi = u(E, {\bf p}) \, e^{i ({\bf p} \cdot {\bf x} - E t)}
# $$
# 
# Donde $u(E,{\bf p})$ es un espinor de Dirac que no depende de ${\bf x}, t$, que ahora debe cumplir la ecuación:
# 
# $$
# (\gamma^\mu p_\mu - m) \, u = 0,
# $$
# 
# que no contiene derivadas.
# 

# ### Soluciones de la partícula en reposo
# 
# Para una partícula en reposo ${\bf p} = {\bf 0}$, la onda de la partícula libre es: $\Psi = u(E)\,e^{-iEt}$
# y la ecuación para el espinor $u$:
# 
# $$
# \left(E \gamma^0 - m \right) \, u = 0
# $$
# 
# En la representación de Pauli-Dirac nos da cuatro soluciones:
# $$
# u_1 = N \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix} \;
# u_2 = N \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix} \;
# u_3 = N \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix} \;
# u_4 = N \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix} . 
# $$
# Donde $N$ es un factor de normalización.
# 
# $u_1, u_2$ tiene energía positiva $+m$. $u_3, u_4$ negativa, $-m$. $u_1, u_3$ tiene spín arriba, $u_2, u_4$ spín abajo.

# Respecto al hamiltoniano $\hat H$:
# 
#    * $u_1, u_2$ tiene energía positiva $E = +m$. 
# 
#    * $u_3, u_4$ negativa, $E = -m$.
# 
# Respecto al spín ${\hat S_z}$
# 
#    * $u_1, u_3$ tiene spín $1/2$, arriba. 
#    
#    * $u_2, u_4$ tiene spin $-1/2$, abajo.
#    
# Asociamos estas soluciones de energía positica con una partícula de spín $1/2$ y las soluciones de energía negativa con una antipartícla de spín $1/2$.

# #### ¿Cuál es el significado de la solución de energía negativa?
# 
# Dirac propuso la teoría del "mar de Dirac". El *vacio* estaba lleno con todos los estados de energía negativos ocupados. Un fotón con $E > 2 m_e c^2$ podía hacer salat un electrón del vacio a la zona de energía positiva produciendo un electrón y un hueco en vacio que se interpreta como el positrón.
# 
# Feynman y Stückelberg propusieron que en las soluciones de energía negativa se propagaban hacía atrás en el tiempo. De esta forma el factor $e^{-iEt}$ no cambia $E \to - E, \; t \to -t$. 
# 
# Pero sí cambia el signo del momento ${\bf p} \to {\bf p}$. De ahí que en los diagramas de Feynman los anti-fermiones tengan una flecha hacia atrás en el tiempo.
# 
# Podemos obtener los spinores de las antipartículas $v_1, v_2$, que experimentalmente se comportan como partículas y van hacia delante en el tiempo con la inversión: 
# 
# @check (-E)(-t)
# $$
# v_1(E, {\bf p}) \, e^{-i ({\bf p} \cdot {\bf x} - Et)} = u_3(-E, -{\bf p}) \, e^{i(-{\bf p} \cdot {\bf x} + Et)} \\
# v_2(E, {\bf p}) \, e^{-i ({\bf p} \cdot {\bf x} - Et)} = u_4(-E, -{\bf p}) \; e^{i(-{\bf p} \cdot {\bf x} + Et)} \\
# $$ 
# 
# 

# ### Soluciones generales de la partícula libre
# 
# En el caso $\Psi = u(E, {\bf p}) \, e^{i({\bf p} \cdot x - E t)}$, la ecuación sobre el spinor $u$ queda:
# 
# $$
# \left[ \begin{pmatrix} I & 0 \\ 0 & - I\end{pmatrix} E - 
#         \begin{pmatrix} 0 & \sigma \cdot {\bf p} \\ - \sigma \cdot {\bf p} & 0 \end{pmatrix} - 
#         m \begin{pmatrix} I & 0 \\ 0 & I \end{pmatrix}\right] u = 0
# $$
# 
# si expresamos el spinor a partir de dos espinores de dos componentes: 
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
# y cuatro soluciones:
# 
# $$
# u_1 = N \begin{pmatrix} 1 \\ 0 \\ \frac{p_z}{E+m} \\ \frac{p_x+ip_y}{E+m} \end{pmatrix}, \;
# u_2 = N \begin{pmatrix} 0 \\ 1 \\ \frac{p_x-ip_y}{E+m} \\ \frac{-p_z}{E+m} \end{pmatrix}, \;
# u_3 = N \begin{pmatrix} \frac{p_z}{E-m} \\ \frac{p_x+ip_y}{E-m} \\ 1 \\ 0 \end{pmatrix}, \;
# u_4 = N \begin{pmatrix} \frac{p_x-ip_y}{E-m} \\ \frac{-p_z}{E-m} \\ 0 \\ 1 \end{pmatrix}. 
# $$
# 
# Donde $N$ son factores de normalización.

# Que en el límite ${\bf p} = {\bf 0}$, se reducen a los casos con energía positiva y negativa.
# 
# $u_1, u_2$ están asociados a la energía positiva: $E = + \sqrt{p^2 + m^2}$
# 
# $u_3, u_4$ están asociados a la energía negativa: $E = - \sqrt{p^2 + m^2}$

# ### Spinores de las anti-partículas
# 
# En la Naturaleza las anti-partículas se comportan como las partículas y van hacia delante en el tiempo. 
# Es mejor trabajar con spinores de anti-partículas, $v_1(E, {\bf p}), v_2(E, {\bf p})$, en vez de con las soluciones de energía negativa $u_3, u_4$.
# 
# Podemos obtenerlos simplemente con los cambios:
# 
# @check (-E)(-t)
# $$
# v_1(E, {\bf p}) \, e^{-i ({\bf p} \cdot {\bf x} - Et)} = u_3(-E, -{\bf p}) \, e^{i(-{\bf p} \cdot {\bf x} + Et)} \\
# v_2(E, {\bf p}) \, e^{-i ({\bf p} \cdot {\bf x} - Et)} = u_4(-E, -{\bf p}) \; e^{i(-{\bf p} \cdot {\bf x} + Et)} \\
# $$ 
# 
# 

# O formalmente a partir de las ondas planas de las antipartículas:
# 
# $$
# \Psi({\bf x}, t) = v(E, {\bf p}) \, e^{-i ({\bf p} \cdot {\bf x} - E t)}
# $$
# 
# que dan lugar a la ecuación para los spinores de las antipartículas:
# 
# $$ 
# \left(\gamma^\mu p_\mu + m \right) v = 0
# $$
# Si consideramos un espinor compuesto de dos espinores de dos componentes: 
# 
# $$
# v = \begin{pmatrix} v_A \\ v_B\end{pmatrix}
# $$
# Obtenemos de nuevos dos ecuaciones acopladas:
# 
# $$
# v_A = \frac{{\bf \sigma} \cdot {\bf p}}{E + m} v_B, \;\; v_B = \frac{{\bf \sigma} \cdot {\bf p}}{E - m} v_A,
# $$

# Así obtenemos cuatro soluciones, las dos primeras, son las de las antipartículas, ahora con energía positiva:
# 
# $$
# v_1 = N \begin{pmatrix} \frac{p_x - ip_y}{E+m} \\ \frac{-p_z}{E+m} \\ 0 \\ 1 \end{pmatrix}, \;
# v_2 = N \begin{pmatrix} \frac{p_z}{E+m} \\ \frac{p_x + ip_y}{+m} \\ 1 \\ 0 \end{pmatrix}. 
# $$
#     
#     

# Las funciones de ondas solución de la ecuación de Dirac para partículas y antipartículas son: 
# 
# $$
# \Psi_i = u_i(E, {\bf p}) \, e^{+i({\bf p} \cdot {\bf x} - E t)}, \;\; 
# \Psi_{2+i} = v_i(E, {\bf p}) \, e^{-i({\bf p} \cdot {\bf x} - E t)}
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

# ### Conjungación de carga
# 
# La conjungación de carga cambia partículas por antipartículas pero manteniendo el spin, esto es los espinores $u_i \to v_i$ con $i=1, 2$.
# 
# En la representación de Pauli-Dirac, el **operador cojungación de carga** viene dado por:
# 
# $$
# \Psi' = \hat C \Psi = i \gamma^2 \Psi^*
# $$
# 
# Podemos comprobarlo, al aplicarlo sobre $\Psi_1$, lo cambia a $\Psi_3$
# 
# $$
# \Psi' = \hat C \Psi_1 = i \gamma^2 u^*_1 \, e^{-i ({\bf p} \cdot {\bf x} - Et)} = v_1 e^{-i ({\bf p} \cdot {\bf x} - Et)} = \Psi_3
# $$
# 
# Dado que: 
# $$
# i 
# \begin{pmatrix} 0 & 0 & 0 & -i  \\ 0 & 0 & i & 0 \\ 
#                 0 & i & 0 & 0   \\ -i & 0 & 0 & 0  \end{pmatrix}
# \begin{pmatrix} 1 \\ 0 \\ \frac{p_z}{E+m} \\ \frac{p_x - i p_y}{E+m}\end{pmatrix} = 
# \begin{pmatrix} \frac{p_x - ip_y}{E+m} \\ \frac{-p_z}{E+m} \\ 0 \\ 1 \end{pmatrix}
# $$

# #### Operadores sobre los spinores de anti-partículas
# 
# Recordemos que los operadores ${\hat H} = i \frac{\partial}{\partial t}, \; {\hat {\bf p}} = - i \nabla$. Al aplicarlos sobre las soluciones de las anti-partículas nos dan:
# 
# $$
# \hat{H} \Psi = - E \Psi, \;\; {\bf \hat p} \Psi = - {\bf p} \Psi,
# $$
# 
# que son las soluciones de energía negativa yendo hacia atrás en el tiempo.
# 
# por lo que los operadores que dan las canditades *físicas* de las anti-partículas (denotados con superíndice $v$) deben ser:
# 
# $${\hat H}^v = - i \frac{\partial}{\partial t}, \; {\hat {\bf p}}^v = i \nabla, \; {\hat S}^v = - {\hat S}$$

# #### Factor de normalización
# 
# Si normalizamos la densidad por unidad de volumen a $2E$. Obtenemos, por ejemplo para $\Psi_1$:
# 
# $$
# \rho = \Psi^\dagger \Psi = u_1^\dagger u_1 =
# |N|^2 \left(1 + \frac{p_z^2}{(E+m)^2} + \frac{p^2_x + p^2_y}{(E+m)^2} \right) = |N|^2 \frac{2E}{E+m}
# $$
# 
# Obtenemos:
# $$
# N = \sqrt{E + m}
# $$

# ### Paridad
# 
# La operación de paridad, ${\hat P}$, cambia ${\bf x} to -{\bf x}$. 
# 
# Si exigimos que la función de onda transformada bajo paridad, $\Psi' = \hat{P} \Psi$, satisfaga la ecuación de Dirac, y dado que ${\hat P}^2 = I$, encontramos que: $\hat{P} = \pm \gamma^0$.
# 
# Definimos el operador paridad: 
# $$
# \hat{P} \equiv \gamma^0,
# $$
# 
# de tal forma que las partículas tienen autoestad +1 bajo paridad y las anti-partículas negativo:
# 
# $$
# \hat{P} \, u_i = \gamma^0 u_i = + u_i, \;\; \hat{P} \, v_i = \gamma^0 v_i = - v_i
# $$
# 
# 
# 

# ### Helicidad y Quiralidad
# 
# #### Tercera componente de spin
# 
# Los espinores en reposo, $u_i(m, 0), v_i(m, 0)$, con $i=1, 2$, son autoestados del operador spín, ${\hat S}_z$.
# 
# En ausencia de un campo magnético o un sistema de referencia externo, la dirección $z$ viene definida por la dirección del momento ${\bf p} = p {\hat k}$. 
# 
# En ese caso los espinores, $u_i(E, p \hat{k}), v_i(E, p \hat{k})$ también son autoestados del operador spín.
# 
# $$
# u_1 = N \begin{pmatrix} 1 \\ 0 \\ \frac{p}{E+m} \\ 0 \end{pmatrix}, \;
# u_2 = N \begin{pmatrix} 0 \\ 1 \\ 0 \\ \frac{-p_z}{E+m} \end{pmatrix}, \;
# v_1 = N \begin{pmatrix} 0 \\ \frac{-p_z}{E+m} \\ 0 \\ 1 \end{pmatrix}, \;
# v_2 = N \begin{pmatrix} \frac{p_z}{E-m} \\ 0 \\ 1 \\ 0 \end{pmatrix} 
# $$
#  
# se cumple:
# $$
# \hat{S}_z u_i(E, p \hat{k}) = \pm \frac{1}{2} u_i(E, p \hat{k}), \;\;\;
# \hat{S}^v_z v_i(E, p \hat{k}) = \pm \frac{1}{2} v_i(E, p \hat{k}),
# $$
# 
# donde los espinores con $i=1$ tiene autoestade $+1/2$ (up) y los de $i=2$, $-1/2$ (down)

# ### Helicidad
# 
# Definimos la helicidad como la proyección normalizada del spin sobre el momento:
# 
# $$
# h \equiv \frac{{\bf S} \cdot {\bf p}}{p}.
# $$
# 
# Para un espinor de Dirac el operador de helicidad en la representación Pauli-Dirac:
# 
# $$
# {\hat h} = \frac{{\bf \Sigma} \cdot {\hat {\bf p}}}{ 2 p} = \frac{1}{2p} 
# \begin{pmatrix} {\bf \sigma} \cdot {\bf {\hat p}} & 0 \\ 0 & {\bf \sigma} \cdot {\bf {\hat p}} \end{pmatrix}
# $$
# 
# Las interacciones se analizan en muchas ocasiones respecto a los espines de las partículas, y en ese caso la valided de los espinores $u_1, u_2, v_1, v_2$ es limitada.
# 
# El operador spín no comunta con el hamiltoniano libre: $[{\hat H}, {\hat S}_z] \neq 0$
# 
# 

# El operador spín, ${\hat S}_z$ no comunta con el hamiltoniano libre: $[{\hat H}, {\hat S}_z] \neq 0$ pero sí el operador helicidad: $[H, {\hat h}] = 0 $
# 
# Por lo tanto existen soluciones a la ecuación de Dirac que son también autoestados de helicidad.
# 
# Si expresamos ${\bf p} = p ( \sin \theta \cos \phi, \sin \theta \sin \phi, \cos \theta)$ en esféricas y $c = \cos \theta/2, \, s = \sin \theta/2$. Esos espinores son:
# 
# $$
# u_{+} = N \begin{pmatrix} c \\ s e^{i\theta} \\ \frac{p}{E+m} c \\ \frac{p}{E+m} s e^{i\theta} \end{pmatrix}, \;
# u_{-} = N \begin{pmatrix} -s \\ c e^{i\theta} \\ \frac{p}{E+m} s \\ -\frac{p}{E+m} c e^{i\theta}\end{pmatrix}, \;
# v_{+} = N \begin{pmatrix} \frac{p}{E+m} s \\ -\frac{-p}{E+m} c e^{i\theta}\\ -s \\ ce^{i\theta} \end{pmatrix}, \;
# v_{-} = N \begin{pmatrix} \frac{p}{E+m} c \\ \frac{p}{E+m} s e^{i\theta}\\ c \\ s e^{i\theta} \end{pmatrix}. 
# $$
# 
# Donde $u_{+}, v_{+}$ son espinores de helicidad positiva (el espín va en la misma dirección que el momento) y $u_{-}, v_{-}$ negativa, esto es:
# 
# $$
# {\hat h} \, u_{\pm} = \pm \frac{1}{2} u_{\pm}, \;\;\; {\hat h}^v \, v_{\pm} = \mp \frac{1}{2} v_{\pm}
# $$
# 
# @todo: check $v_{\pm}$
# 

# El elemento:
# 
# $$
# \frac{{\bf \sigma} \cdot {\bf p}}{p} = \begin{pmatrix} p_z & px - ipy \\ px+ipy & -pz \end{pmatrix} = 
# \begin{pmatrix} \cos\theta & \sin\theta e^{-i\phi} \\ \sin\theta e^{i\phi} & -\cos\theta \end{pmatrix}
# $$

# Las interacciones se analizan en muchas ocasiones respecto a los espines de las partículas, y en ese caso la valided de los espinores $u_1, u_2, v_1, v_2$ es limitada.
# 
# El operador spín no conmunta con el hamiltoniano libre: $[{\hat H}, {\hat S}_z] \neq 0$.
# 
# **La helicidad no es un invariante Lorentz**, para una partícula siempre podemos encontrar un sistema de referencia (con una velocidad mayor) que revierta el momento y por lo tanto la helicidad!

# ### Quiralidad
# 
# La quiralidad corresponde a los autoestados del operador $\gamma^5$. 
# 
# Podemos dar dos operadores de proyección de quiralidad a derechas, $P_R$, y a izquierdad, $P_L$:
# 
# $$
# P_R = \frac{1}{2} (I + \gamma^5), \;\;\, P_L = \frac{1}{2} (I - \gamma^5)
# $$
# 
# cumples las condiciones de proyección:
# $$
# P_R + P_L = I, \;\; P^2_R = P_R, \;\;, P^2_L = P_L, \;\; P_L P_R = P_R P_L = 0
# $$
# 
# De tal forma que los espinores se pueden descomponer en una parte de quiralidad a izquierdas y derechas:
# 
# $$
# u = P_R u + P_L u, \;\; u_R = P_R u, \;\; u_L = P_L u
# $$
# 
# 

# Las proyecciones complen:
# 
# $$
# P_R u_R = u_R, \,\, P_R u_L = 0, \;\; P_R v_R = 0, \;\; P_R v_L = v_L, \\
# P_L u_R = 0, \,\, P_L u_L = u_L, \;\; P_L v_R = v_L, \;\; P_L v_L = 0.
# $$
# 
# Las siguientes corrientes entre las proyecciones de chiralidad son nulas:
# 
# $$
# {\bar u}_L \gamma^\mu u_R = {\bar u}_R \gamma^\mu u_L = {\bar v}_L \gamma^\mu u_R = {\bar v}_R \gamma^\mu u_L = 
# {\bar v}_L \gamma^\mu v_R = {\bar v}_R \gamma^\mu v_L = 0
# $$
# 
# La quiralidad es fundamental en las interacciones débiles.
# 
# En las corrientes cargadas intervendrán solamente los espinores a izquierdad de las partículas, $u_L$, y a derechas de las antipartículas, $u_R$.

# Si aplicamos paridad a los espinores de quiralidad:
#     
# $$
# \hat{P} \, u_R = u_L, \;\; \hat{P} \, u_L = u_R \\
# \hat{P} \, v_R = v_L, \;\; \hat{P} \, v_L = v_R
# $$
#  
# cambia su quiralidad de izquierdas a derechas y viceversa.

# @CHECK
# 
# Si aplicamos conjugación de carga sobre los espinores de quiralidad:
# 
# $$
# \hat{C} \, u_R = v_L, \;\; \hat{C} \, u_L = v_R \\
# \hat{C} \, v_R = u_L, \;\; \hat{C} \, v_L = u_R
# $$
# cambia partícula por antipartícula, e izquierda por derecha.

# ### Relación entre helicidad y quiralidad
# 
# Los espinores de helicidad para partículas sin masa, $m = 0$, o relativistas ($E >> m, \, E = p$) son:
# 
# $$
# u_{+} = N \begin{pmatrix} c \\ s e^{i\theta} \\ c \\  s e^{i\theta} \end{pmatrix}, \;
# u_{-} = N \begin{pmatrix} -s \\ c e^{i\theta} \\ s \\ - c e^{i\theta}\end{pmatrix}, \;
# v_{+} = N \begin{pmatrix}  s \\ - c e^{i\theta}\\ -s \\ ce^{i\theta} \end{pmatrix}, \;
# v_{-} = N \begin{pmatrix}  c \\ s e^{i\theta}\\ c \\ s e^{i\theta} \end{pmatrix}. 
# $$
# también sus proyecciones de quiralidad: 
# 
# $$
# u_R \equiv  P_R u_{+} = u_{+} , \;\; u_L \equiv P_R u_{-} = u_{-}, \;\;
# v_R \equiv  P_R v_{+} = v_{+}, \;\; v_L \equiv P_L v_{-} = v_{-} 
# $$
# 
# 

# Si definimos:
# 
# $$
# \kappa = \frac{p}{E+m}
# $$
# 
# podemos expresar los espinores de helicidad en función de los de quiralidad:
# 
# $$
# u_{+} \propto \frac{1}{2} (1 + \kappa) \, u_R + \frac{1}{2} (1 - \kappa) \, u_L
# $$
# 
# En el caso de $E >> m$ ($\kappa \to 1$), los espinores de helicidad positiva convergen a los de quiralidad positiva, y solo tiene una pequeña fracción $(1-\kappa)/2$ de la quiralidad negativa. 

# In[ ]:




