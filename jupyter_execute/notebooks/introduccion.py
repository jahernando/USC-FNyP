#!/usr/bin/env python
# coding: utf-8

# # Introducción a Física de Partículas
# 
# 
# ## Introducción
# 
# 
# *Jose A. Hernando Morata*
# 
# *Departamento de Física de Partículas. Universidade de Santiago de Compostela*
# 
# *primera versión: Septiembre 2021; última versión: Octubre 2023*
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
# * Dar una visión del entorno de la Física de Partículas:  
#   - Historia, relaciones con otras ramas, los laboratorios. 
#   
# * Dar una introducción general del Modelo Estándar.
#   - La materia (fermiones),  las fuerzas (bosones), y el aglutador (Higgs)
#   

# ## Primeros pasos en Física de Partículas
# 
# ### ¿Qué es la Física de Partículas?
# 
# 
#   Física de Partículas es la rama de la Física que estudia los **corpúsculos** más pequeños (elementales?) de la Naturaleza y las **interacciones** que median entre ellos.
# 
# | |
# |:--:|
# | <img src="./imgs/intro_distances_scale.jpg" width = 800 align="center">|
# 
# 
# Las escalas de **distancia** (m) y **energía** (eV):
#   
#   || Atómica | Nuclear | Partículas|
#   | -- | --       | --     | --        |
#   distancia (m)|  $10^{-10}$ | $10^{-14}$ | $10^{-18}$ |
#   Energía (eV) |  $1$ | $10^{6}$ | $10^{9-11}$ |
#   
# Si queremos explorar distancias muy cortas necesitamos, por la relación de Broglie, $\lambda = h /p$, energías más altas.

# ### Metodología
# 
# Estudiamos la naturaleza mediante pruebas con dispersión (*scattering*) o colisiones.
# 
# | | | 
# |:--:| :--: |
# | <img src="./imgs/intro_dispersion.png" width = 400 align="center">| <img src="./imgs/intro_colision.png" width = 400 align="center">|
# | Esquema de dispersión | Esquema de  colisión |
#     
# Literalmente lanzamos unas partículas ($e, p, \nu$) contra otras a muy altas energías (GeV-TeV) y observamos sus productos.

# ### Partículas y Energía
# 
# Se conoce también a Física de Partículas como **Física de Altas Energías**, ordenes típicos $\mathcal{0}(1-10^3) $GeV.
# 
# La Física de Partículas está asociada con el desarrollo de aceleradores de partículas y detectores.
# 
#    * El acelerador más potente es el LHC [[>]](https://www.youtube.com/watch?v=pQhbhpU9Wrg)
#    
#    * Existen aceleradores cósmicos naturales que producen rayos cósmicos que alcanzan la Tierra. Ver el detector IceCube [[>]](https://www.youtube.com/watch?v=D50LNnioXQc)

# ### Partículas y Cosmología
# 
# | |
# | :--: |
# | <img src="./imgs/intro_universe_history_2.jpg" width = 600 align="center"> |
# | Esquema de la evolución del Universo [PDG]|
# 

# La Física de Partículas nos permite entender la evolución primigenia del Universo (Cosmología).
# 
# A partir de los $10^{-32}$ s después del Big Bang el Universo se rigue por el Modelo Estándar, (SM) y depués de 100 s tiene lugar la Nucleosíntesis (Nuclear).
# 
# Existe una relación de continuidad entre ramas de la física:
# 
# <center>
# Astropartículas $\Leftrightarrow$ Partículas $\Leftrightarrow$ Nuclear
# </center>
#  
# El *Instituto Galego de Fíxica de Altas Enerxías*, [IGFAE](https://igfae.usc.es/igfae/gl/), cubre las tres ramas y sus posibles aplicaciones.

# ### Teoría, experimentos, detectores
# 
# La Física de Partículas es una rama que involucra a la física teórica, experimental y la de detectores.
# 
#    * Los avances en detección hacen posible nuevos discubrimientos
# 
#    * Los descubrimientos marcan las líneas permitidas de las teorías 
# 
#    * La teoría indica qué experimentos son de interés
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

# ### La gran ciencia
# 
# Los experimentos y los acelerados son grandes construcciones, únicas, que requieren de la participación de miles de científicos y una gran financiación.
# 
# Las estructuras de los experimentos son complejos, su planificación, construcción y explotación duran décadas. Un ejemplo: [ATLAS](https://atlas.cern/)
# 
# | |
# | :--: |
# | <img src="./imgs/intro_atlas.jpg" width = 600 align="center"> |
# | Imagen del detector ATLAS durante su construcción [ATLAS]|
# 
# El experimento [ATLAS](https://atlas.cern/) del LHC es uno de los más grandes experimentos construidos. 
# 

# ### Los grandes Laboratorios
# 
# Los experimentos de Física de Partículas tienen lugar en grandes laboratorios internacionales.
# 
# | |
# | :--: |
# | <img src="./imgs/Dr_Fabiola_Gianotti.jpg" width = 300 align="center"> |
# | F. Gianotti, directora del [CERN](https://home.cern)|
# 

# 
# Basados en aceleradores:
#   
#    * [CERN](https://home.cern) (Ginebra, Suiza).
#      
#    * [Fermilab](https://www.fnal.gov/) (Chicago, Illinois), [SLAC](https://www6.slac.stanford.edu/) (Stanford, California), [Brookhaven](https://www.bnl.gov/world/) (New York)
#      
#    * [KEK](https://www.kek.jp/en/) (Japón)
#      
# Subterráneos:
#   
#    * [LNGS](https://www.lngs.infn.it/en) (Italia), [Kamioka](http://www-sk.icrr.u-tokyo.ac.jp/index-e.html) (Japon), [SNOLAB](https://www.snolab.ca/) (Canada), [Canfranc](https://lsc-canfranc.es/) (Spain).

# #### Aplicaciones prácticas
# 
# Las técnicas desarrolladas para los detectores y los algoritmos para el tratamiento del gran volumen de tratamiento de datos han generados aplicaciones fundamentales en otros campos:
# 
#    * Física Médica (rayos-X, PEP, terapia hadrónica)
#    
#    * World Wide Wed ([WWW](https://home.cern/science/computing/birth-web/short-history-web)) 

# ### Historia
# 
# Una vista a vuelo de pájaro de la historia de la Física de Partículas:
# 
# | |
# | :--: | 
# | <img src="./imgs/intro_particles_history.jpg" width = 800 align="center">|
# | Hitos de la Historia de Física de Partículas|

# Los grandes desarrollos de detetectores y aceleradores se traducen en descubrimientos científicos.
# 
#   * Las emulsiones fotográficas permitieron la observación de partículas, $\mu, \pi$
# 
#   * Las centelleadores y los reactores nucleares el descubrimiento del neutrino $\nu_e$. 
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
#   * Gellman propone el modelo de quarks y la ordenación de 'zoo' de hadrones.
#     
#   * La unificación electro-débil predice las corrientes neutras de neutrinos.
#   
#   * La rotura espontánea de simetría postula la existencia del bosón de Higgs.

# Los avacen experimentales más relevantes:
#     
#   * El descubrimento del positron (antimateria)
#   
#   * El descubrimiento de partículas inesperadas: $\mu, ...$
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

# Los avences teóricos más importantes:
# 
#   * La ecuación de Dirac y la existancia de la antimateria
#   
#   * El potencial de Yukawa
#   
#   * La teoría de Fermi de interacciones entre corrientes.
#   
#   * Los diagramas de Feynman (QED)
#   
#   * El modelo de quarks de Gellman and Zweig.
#   
#   * La unificación electro-débil, el modelo estandard (SM) de Glashow, Salam y Weinberg.
#   
#   * El mecanismo de Higgs y Englert.

# ## Introducción al modelo estandar
# 
# EL modelo estandar (SM) de Física de Partículas clasifica las partículas elementales y establece sus interacciones a través de las fuerzas electromagnética, débil y fuerte.
# 
# El *Particle Data Group* ([PDG](https://pdg.lbl.gov/)) recopila toda la información relevante sobre física de partículas. 

# ### El SM a simple vista
# 
# 
# | |
# |:--:|
# |<img src="./imgs/intro_SM_table.png" width = 600 align="center">|
# |Las partículas del modelo estandar|
# 

# La materia está formada por los **fermiones**, que son partículas elementales de spín $1/2$.
# 
# Los **bosones vectoriales**, con spín $1$, son los mediadores de las fuerzas fuerte, débil y electromagnética.
# 
# El **bosón scalar de Higgs**, que tiene spín 0, dota de masas a los fermiones y a los bosones vectoriales de la fuerza débil.
# 

# ### Fermiones
# 
# Los fermiones se dividen en:
# 
#   * **quarks** : si interaccionan fuertemente
#   
#   * **leptons**: si no interaccionan fuertemente
#     
# | fuerte | electromagnética | débil | 
# |---      |:--       | :--          |
# | quarks | quarks y leptones cargados  | quarks y leptones |
# 
# 

# #### anti-fermiones
# 
# Los fermiones se riguen por la ecuación de Dirac. Son espinores de Dirac. (ver [Apéndice-Dirac])
# 
# Por cada fermión existe un antifermión.
# 
# Una **antipartícula** tiene las cargas y los números cuánticos opuestos a su partícula, a excepción del espín y la masa que no cambian.
# 
# Si $f$ es un fermión, denotamos $\bar{f}$ como su antifermión.
# 
# Indicamos los antileptones con carga (+): $e^+, \mu^+, \tau^+$

# #### generaciones
# 
# Los fermiones (tanto leptones como quarks) aparecen en **tres generaciones** o familias. 
# 
# $$
# \begin{pmatrix} \nu_e \\ e \end{pmatrix}, \begin{pmatrix} \nu_\mu \\ \mu \end{pmatrix}, 
# \begin{pmatrix} \nu_\tau \\ \tau \end{pmatrix}; \;\;\;
# \begin{pmatrix} u \\ d \end{pmatrix}, \begin{pmatrix} c \\ s \end{pmatrix}, 
# \begin{pmatrix} t \\ b \end{pmatrix} \\
# $$
# 
# Decimos que las partículas se presentan en tres **sabores**. Por ejemplo neutrino se presenta en sabor electrónico, muónico y tauónico.
# 
# Las tres generaciones de leptones se comportan con **universalidad** (de igual manera) frente a las interacciones si se tiene en cuenta que sus masas son diferentes. 
# 
# No hay una motivación de por qué hay tres generaciones.
# 
# En cada generación, tanto leptones como quarks, se agrupan en **dupletes** con posiciones **arriba** y **abajo**.

# La siguiente tabla muestra los fermiones agrupados por familias, su carga eléctrica y las fuerzas que sienten: 
# 
# |         | posición |             | generación        |              | Q (e)           | fuerte | electromagnética | débil | 
# |---      |:--       | :--         |:--          | :--          | :--             | :--    |:--     |:--   |
# |         |          |      I       |    II      |       III            |           |  |  | | 
# |quarks   | up       |  u  (2 MeV) | c (1.2 GeV) | t (170 GeV)  | $\frac{2}{3}$   | sí | sí | sí |
# |         | down     |  d  (5 MeV)  | s (93 MeV)  | b (4.2 GeV)  |  $\frac{-1}{3}$ | sí | sí|  sí |
# |         |
# | leptons | up       | $\nu_e$ (< eV)| $\nu_\mu$ (< eV) | $\nu_\tau$ (<eV)   | 0  | no | no | sí |
# |         | down     | $e$   (511 keV)| $\mu$  (106 MeV)| $\tau$  (1.77 GeV) | -1 | no | sí | sí |
# 
# 
# La carga eléctrica depende de su posición en el duplete. El fermión de arriba tiene una carga $+1$ comparada con la del de abajo.
# 

# La siguiente figura muestra de forma gráfica la disparidad de masas entre los fermiones:
# 
# 
# | |
# |:--:|
# |<img src="./imgs/intro_masas.png" width = 500 align="center">|
# |Masas de los fermiones [MT1.1]|
# 
# El electrón es 2000 veces más ligero que el protón. 
# 
# Del neutrino desconocemos su masa, pero ésta es al menos $5 \times 10^5$ veces más pequeña que el electrón. 
# 
# Como veremos en el [Tema-SM] la masa de los fermiones aparece de su interacción con el bosón de Higgs. En el SM aparecen las masas como parámetros libres del modelo.

# ### Leptones
# 
# Las tres familias de leptones son:
# 
# $$
# \begin{pmatrix} \nu_e \\ e \end{pmatrix}, \begin{pmatrix} \nu_\mu \\ \mu \end{pmatrix}, 
# \begin{pmatrix} \nu_\tau \\ \tau \end{pmatrix}
# $$
# 
# Los leptones cargados $\mu, \tau$ son una copia con más masa del $e$.
# 
# Las interacciones conservan el **número leptónico**, $L$, que es el número de leptones menos el de anti-leptones. 
# 

# 
# Por ejemplo en la desintegración $\beta$:
# 
# $$
# n \to p + e + \bar{\nu}_e
# $$
# 
# El número leptónico antes y después de de la desintegración es $L = 0$. Dado que $e$ es un leptón, $L = +1$  y $\bar{\nu}_e$ un anti-lepton, $L=-1$.
# 
# Notar que en la desintegración $\beta$ **se conserva la carga eléctrica.**
# 
# **El electrón**, que es la leptón cargado más ligero, **es estable**. 

# Los leptones cargados se desintegran débilmente:
# 
# Sus vidas medias son:
# 
# | $e$  |  --------- $\mu$ --------- | ------- $\tau$ ----|
# | :--: | :--: | :--: | 
# | estable | $2.6 \, 10^{-6}$ s| $2.9 \, 10^{-13}$ s|
# 
# Los electrones interaccionan con los átomos de la materia, ionizando y radiando fotones por *Bremsstrahlung*. 
# 
# Los muones son partículas penetrantes, del orden de $\mathcal{0}(10)$m (depende de su momento) porque su radiación de *Bremsstrahlung* está suprimida por $1/m^2_\mu$ comparada con los electrones.
# 
# Los tauones se desintegran al poco de ser producidos. Pueden llegar a recorrer $\mathcal{0}$(cm).
# 

# Existen tres neutrinos de sabor, $\nu_e, \nu_\mu, \nu_\tau$ asociados a sus parejas leptónicas $e, \mu, \tau$ respectivamente.
# 
# **El SM postula que no tienen masa**, pero gracias a los experimentos de oscilaciones de neutrinos sabemos que tienen masa.
# 
# Su masa se desconoce pero es muy pequeña $\lt 1$ eV comparada con la del $e$.
# 
# **Los neutrinos** no tiene carga eléctrica, **tienen una masa muy pequeña**, **son estables**, y **sólo interactuan débilmente**.  ¡Son muy *cool*!
# 
# Son los leptones más peculiares del SM y que tengan masa es **la única prueba hasta el momento de que el SM es incompleto o incorrecto**.
# 
# Las características de los leptones y las peculiaridades de la desintegración débil y de los neutrinos las estudiaremos en el [Tema-leptones].

# ### Los quarks
# 
# Las tres generaciones de quarks son:
# 
# $$
# \begin{pmatrix} u \\ d \end{pmatrix}, \begin{pmatrix} c \\ s \end{pmatrix}, 
# \begin{pmatrix} t \\ b \end{pmatrix} \\
# $$
# 
# Los quarks $(c, s)$, encanto y extrañeza respectivamente, y $(t, b)$, top y bottom (o belleza), son copias con más masa del par $(u, d)$, arriba y abajo.
# 
# Los quarks $u, d, s$ son ligeros, los quark $c, b$ son pesados. El top, $t$, es incluso más pesado que los bosones vectoriales $W^\pm, Z$ y el Higgs, $H$, y no llega a producir hadrones.
# 
# **Los quarks tienen carga fraccionaria**: los de arriba $+2/3$ y los de abajo $-1/3$ en unidades de $e$.
# 
# **Los quarks no son libres**, sólo aparecen en partíclas compuestas, los **hadrones**. Este fenómeno se conoce como **confinamiento**.
# 

# 
# Los quarks aparecen con tres cargas fuertes o de **color**, que se denominan: **rojo, verde y azul**, $r, g, b$. Y por supuesto no tiene nada que ver con los colores habituales. 
# 
# Un mismo quark existe en tres colores, por ejemplo el quark belleza, $b$, puede ser rojo, verde o azul. Hay por lo tanto 18 quarks diferentes.
# 
# Los únicos hadrones que existen son **neutros al color**, digamos que son *descoloridos* o *blancos*, de forma similar al hecho de que los átomos sean eléctricamente neutros.
# 
# Estudiaremos los quarks, los hadrones y el color en el [Tema-quarks].

# Los **hadrones** se clasifican en.
# 
#   * **bariones**, formados por tres quarks.
#   
#   * **mesones**, formados por un par quark y antiquark.
# 
# | |
# | :--| 
# | <img src="./imgs/intro_hadrons.png" width = 500 align="center"> |
# | Imagen esquemática de un mesón y un barión|
# 
# que como veremos ([Tema-quarks]) son las únicas combinaciones de singletes de color y por lo tanto los únicos hadrones blancos.
# 
# Notar que la carga eléctrica de los hadrones es entera.

# El protón y el neutrón son bariones.
# 
# El protón está formado por el trío de quarks de valencia ($uud$), cuya carga es $+e$,  y el neutrón por ($udd$), cuya carga es nula.
# 
# En las interacciones se converva **el número bariónico**, $B$, (el número de bariones menos el de antibariones).
# 
# Por ejemplo, de nuevo, en la desintegración $\beta$:
# 
# $$
# n \to p + e + \bar{\nu}_e.
# $$
# 
# A ambos lados de la interacción el número bariónico es 1. El $p$ y el $n$ tienen los dos $B = 1$
# 
# **El protón**, que es el barión más ligero, **es estable**.
# 

# Remarquemos que la desintegración $\beta$ 
#     
# $$
# n \to p + e + \bar{\nu}_e.
# $$
# 
# y el resto de **interacciones conservan**:
# 
#     
#   * la **carga eléctrica**, $Q$,
#   
#   * el **número leptónico**, $L$,
#   
#   * el **número bariónico**, $B$.

# 
# A excepción del protón, el resto de hadrones, así como los leptones cargados $\mu, \tau$, se desintegran:
# 
# La vida media de la desintegración depende:
# 
#   * del tipo de desintegración permitida, ya sea ésta fuerte, electromagnética (via un fotón) o débil (vía bosones $W^\pm$).
#   
#   * del espacio fásico permitido. A mayor diferencia de masas entre la partícula madre y las hijas más espacio fásico.
#   

# 
# La figura muestra la vida media de varias partículas y su tipo de desintegración. 
# 
# | |
# |:--|
# | <img src="./imgs/intro_lifetimes.png" width = 900 align="center">|
# | Vida media de algunas partículas dependiendo del tipo de desintegración [MT1.1]
# 
# La partículas que se desintegran débilmente son aquellas con una vida media "mayor".

# ### Los mediadores de las fuerzas
# 
# Recordemos que la fuerzas son responsables entre otros fenómenos de:
#     
#    * la **electromagnética**, de la estabilidad y complejidad de los átomos
#    
#    * la **débil**, de las desintegraciones $\beta$ nucleares.
#         
#    * la **fuerte**, de la estabilidad y complejidad de los núcleos (protones y neutrones) y de los hadrones.
#         

# #### Propiedades de los portadores
# 
# La interacción se describe como el intercambio de un mediador o portador de la fuerza entre corrientes de fermiones.
# 
# Cada portador interviene según la carga (por ejemplo eléctrica o color) de los fermiones a los que se acopla.
# 
# Los portadores **son bosones vectoriales** (spín 1).
# 
# En la tabla se muestra intensidad approximada de las distintas fuerzas: 
# 
# | fuerza            | intensidad | bosón            | Q (e) | mass (GeV) |
# | :--               | :--        | :--              | :--   | :--     | 
# | fuerte            | 1          | $g$ (gluón)      | 0     | 0       | 
# | electromagnetismo | $10^{-3}$  | $\gamma$ (fotón) | 0     | 0       | 
# | débil             | $10^{-8}$  | $W^{\pm}$        | $\pm1$  | 80.4 | 
# |                   |           | $Z^0$              | 0  | 91.2 |
# | gravedad          | $10^{-37}$| 

# La masa del bosón determina el rango (en distancia) de la interacción. El rango de la interacciones débiles es de $<10^{-18}$ m. 
# 
# El rango de $100$ GeV, aproximadamente la masa del bosón $Z$, se conoce como **escala electrodébil.**
# 
# Los bosones $W^\pm$ son cargados. Sus interacciones se conocen como **corrientes cargadas** (CC).
# 
# Las interacciones del bosón $Z^0$ se denominan también **corrientes neutras** (NC). 
# 
# Existen 8 tipos de gluones, que cambian el color de los quarks, son electricamente neutros y de masa nula. Los gluones tienen carga de color e interaccionan entre sí, generando "lazos" entre ellos y reduciendo la interacción fuerte al rango de fm.
# 
# El rango de los fotones en infinito mientras que el de los bosones débiles y los gluones es puntual.

# #### Propiedades de las fuerzas
# 
# Adelantamos ahora algunas propiedades de las fuerzas que se discutirán en los temas siguientes:
# 
# **Las interacciones electromagnética y fuerte conservan**:
#    * **el sabor**,
# 
#    * **la paridad**.
#    
# Recordemos que paridad es la inversión de ${\bf x} \to - {\bf x}$, cambio de signo en las coordenadas espaciales; y la inversión de carga cambia todas las cargas de las partículas (a excepción del espín y la masa).
# 
# La conservación del sabor implica que el número de partículas menos el antipartículas de un sabor, por ejemplo del quark belleza, se conserva antes y después de una interacción.

# 
# En los temas siguientes veremos también que:
# 
# **las interacciones débiles violan**:
#     
#    * **paridad**, (las cargadas de forma máxima),
#    
#    * **carga y paridad**, CP, mínimamente y solo en corrientes cargadas en hadrones, (no sabemos si también se viola en leptones).
#     
# **las corrientes cargadas cambian el sabor, pero las neutras lo conservan**.
#    
# **La interacción fuerte conserva** además:
# 
#    * las cargas de **color**.
#    
# *Las interacciones débiles son "peculiares", especialmente la cargada.*

# ### El bosón de Higgs
# 
# El bosón de Higgs es un **scalar**, su espín es $0$.
# 
# Juega un papel crucial en el SM:
# 
#    * **dota de masas a las partículas**.
#    
#    * dota de masas **a los bosones véctoriales débiles $W^\pm, Z$**, **manteniendo las simetrías** de la teoría.
#    
# Sin embargo el SM introduce las masas de los fermiones como parámetros libres del modelo.
#    
# Estudiaremos las características del SM y del bosón de Higgs en el [Tema-SM]

# ### El Universo está formado de materia
# 
# No hay evidencia de partes del Universo formadas por antimateria.
# 
# Los átomos están formados por electrones y núcleos, éstos a su vez por protones (que son un trío de quarks ($uud$) y neutrones ($udd$)).
# 
# El Universo está compuesto pues **por electrones, neutrinos y quarks $(u, d)$.**
# Decimos que el Universo es de "baja" energía.
# 
# El Universo primigenio estaba a más energía, y existían leptones y quarks (hadrones) de las otras generaciones. 
# 
# Conforme el Universo se enfrió solo los elementos estables sobrevivieron.
# 
# No sabemos por qué en el Universo solo sobrevivió la materia dado que inicialmente debió existir la misma cantidad de partículas que anti-partículas.

# ### Algunas preguntas sin respuesta
# 
# * ¿Son los quarks y leptones elementales? ¿Y el bosón de Higgs?
# 
# * ¿Por qué hay 3 generaciones?
# 
# * ¿Hay un patrón de masas de los fermiones?
# 
# * ¿Por qué hay tanta disparidad de masas entre el neutrino y el resto de fermiones?
# 
# * ¿Es el neutrino su propia anti-partícula?
# 
# * ¿Por qué el Universo está formado por materia y no por anti-materia? ¿Qué paso con la anti-materia del Universo?
# 
# * ¿Qué forma la materia oscura?
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
# | *quark*    | fermión que siente la fuerza fuerte |
# | *bosón*    | particula de spín entero |
# | *gauge boson*| bosones portadores de la fuerza $W, Z, \gamma$ tienen spín 1 (vectoriales) |
# | *hadrón*  (fuerte) | partícula compuesta que siente la fuerza fuerte|
# | *barión*   | hadrón compuesto de tres quarks |
# | *mesón*   (mediano)| hadrón compuesto de quark y antiquark |
# 
# 

# 
# 
# ## Bibliografía
# 
#  * [AB] Alessandro Bettini, "Introduction to Elementary Particle Physcs", Cambridge U. press. Tema 1
# 
#  * [MT] Mark Tomsom, "Modern Particle Physics", Cambridge U. press. Tema 1
#     
#  * [PDG](https://pdg.lbl.gov/) Particle Data Group.
