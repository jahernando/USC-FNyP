#!/usr/bin/env python
# coding: utf-8

# # Sobre el Modelo Estándar
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


# ### Objetivos
#   
# Conocer:
#   
#   * La unificación electrodébil.
#   
#   * El bosón $Z$. La anchura del $Z$ y la existencia de tres neutrinos.
#   
#   * El bosón de Higgs y la masa de los bosones $W, Z$ y de los fermiones. Descubrimiento del Higgs.

# ## Los elementos del SM
# 
# Varios son los elementos que modelan el SM:
# 
#   * La **unificación electrodébil** previene que la seccioón eficaz $e+e^+ \to W^+ + W^-$ ¡crezca infinitimente!
#   
#   * El **mecanismo de Higgs y la rotura espontánea de simetría** dotan de masas a los bosones $W^\pm, Z$ y a los firmiones.
#   
#   * Las teorías deben prservar **la simetría gauge local** para ser renormalizables (y evitar que en los cálculos de las amplitudes aparezcan infinitos)
# 

# ### Hitos de la creacción del SM
# 
#   * 1964 Teoría del bosón de Higgs, Englert, Brout. 
# 
#   * 1961-1964-1967 Teoría electrodébil de S. Glashow, A. Salam and S. Weinberg.
#   
#   * 1973 Descubrimiento de las corrientes neutras en Gargamelle (CERN)
#   
#   * 1973 G.'t Hooft et al, renormalización de la teoría gauge.
#   
#   * 1983 Rubbia et al, descubrimiento de los bosones $W^\pm, Z$ en el CERN.
#   
#   * 1990's Los experimentos del CERN confirman el SM, la física del Z y la existencia de tres familias de neutrinos.
#   
#   * 2014 Descubrimiento del Higgs en los experimmentos ATLAS y CMS del LHC. 
#   

# ## Corrientes neutras, el bosón $Z^0$.
# 
# La interacción $e^+ + e \to W^+ + W^-$, mediada por un fotón, presentaba un problema: crecía indefinidamente con la energía, a no ser que existía un bosón neutro, el $Z^0$, que mediase en un nuevo diagrama de Feynman que a su vez interfiriese negativamente con el fotón moderando así la sección eficaz.
# 
# Esto bosón implicaba también la existencia de **corrientes débiles neutras**, en particular que un haz de nuetrinos podría interaccionar neutramente con otras partículas, el neutrino se dispersaría, y si el neutrino tuviese la sufienciente energía podría romper un nucleón.
# 
# Las corrientes neutras son del tipo: (todo diagramas dispersión y aniquilación!)
# 

# ### El descubrimiento de las corrientes neutras
# 
# [Gargamelle](https://home.cern/science/experiments/gargamelle), en el CERN, en los 70's, fue el experimento que observó las corrientes neutras. Mediante un haz estaba compuesto principalmente por $\nu_\mu$. 
# 
# Las corrientes cargadas de neutrinos con los nucleones producían un $\mu$, una partícula altamente penetrante.
# 
# $$
# \nu_\mu + N \to \mu + X
# $$
# 
# Mientras que las corrientes neutras se esperaba que el neutrino escapase indetectado pero dejase trazas debidas a la ruptura de nucleón.
# 
# $$
# \nu_\mu + N \to \nu_\mu + X
# $$
# 
# Gargamel era una gran cámara de burbujas, en un campo magnético, donde se tomaban fotografías pautadas con la llegada del haz de neutrinos

# | |
# |:--:|
# |<img src="./imgs/sm_gargamelle_event.jpeg" width = 400 align="center">|
# | Evento de corriente neutra observado en [Gargamelle](https://en.wikipedia.org/wiki/Gargamelle) [CERN]|
# 
# ¡El neutrino interacciona con un nucleón sin producir su leptón asociado!
# 
# Las corrientes neutras observadas en Gargamelle estaban de acuerdo con la teoría de unificación electrodébil desarrollada unos años antes.

# ### Isoespín débil
# 
# Recordemos que en las corrientes cargadas solo intervienen espinores a izquierdas de los fermiones (y de derechas de los antifermiones).
# 
# Y que las corrientes cargadas tiene lugar en parejas (i.e $\nu_e$, e)
# 
# Para representar esta realidad, el SM introduce el grupo de simetría SU(2)$_L$, donde $L$ es por quiralidad a izquierdas, asociado al isoespín debil.
# 
# Cada pareja de fermiones a izquierdas, Por ejemplo el $\nu_e$ y el $s$ formarán un dublete de isopín débil, con tercera componente $\pm 1/2$ respectivamente, mientras que el electrón un singlete a derechas. Sea $u^{(q)_L}(p)$ es espinor de Dirac a izquierdas del fermión $q = \nu_e, e$ con cuadrimomento $p$ se denota habitualmente de forma "compacta":
# 
# $$
# \begin{pmatrix} \nu_{eL} \\ 0 \end{pmatrix} \equiv \chi^\uparrow_W u^{(\nu_e)}_L(p) = \begin{pmatrix} 1 \\ 0 \end{pmatrix}_W u^{(\nu_e)}_L(p)
# $$
# $$
# \begin{pmatrix} 0 \\ e _L\end{pmatrix} \equiv \chi^\downarrow_W u^{(e)}_L(p) = \begin{pmatrix} 0 \\ 1 \end{pmatrix}_W u^{(e)}_L(p) 
# $$
# 
# E introducimos los espinores a derechas:
# 
# $$
# e_R \equiv u^{(e)}_R(p)
# $$
# 
# Notar que el neutrino no tiene componente a derechas

# Para cada una de las generaciones obtenemos:
# 
# $$
# \begin{pmatrix} \nu_{eL} \\ e_L \end{pmatrix}, \, e_R; \;\;\;  
# \begin{pmatrix} u_L \\ d'_L \end{pmatrix}, \, u_R, d'_R. 
# $$
# 

# 
# Por ejemplo, para el $\nu_{eL}$, la tercera componente de isospín débil viene dada por:
# 
# $$
# \tau^3_W \begin{pmatrix} \nu_{eL} \\ 0 \end{pmatrix} = \frac{1}{2} 
# \begin{pmatrix} 1 & 0 \\ 0  & -1 \end{pmatrix}
# \begin{pmatrix} \nu_{eL} \\ 0 \end{pmatrix} = 
# + \frac{1}{2} \begin{pmatrix} \nu_{eL} \\ 0 \end{pmatrix}
# $$
# 

# ### Las corrientes y el vértice
# 
# #### Electromagnetimo 
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

# #### Las corrientes cargadas
# 
# Las corrientes cargadas acoplan dos fermiones a izquierdas en un duplete de isospín débil con un acoplo $g_W/\sqrt{2}$. 
# Y en el vértice introducimos las matrices de subida o bajada de isospín débil, $\sigma^\pm_W = \frac{1}{2}(\sigma^1_W \pm i \sigma^2_W)$
# 
# | |
# |:--:|
# |<img src="./imgs/sm_wcur.png" width = 350 align="center">|
# | Corriente cargada|
# 

# Por ejemplo la corriente cargada de $e \to \nu_e$ con $W^+$ como portador quedaría:
# 
# $$
# \frac{g_W}{\sqrt{2}} \bar{u}^{(\nu_e)}_L \gamma^\mu u^{(e)}_L  = 
# \frac{g_W}{\sqrt{2}} \begin{pmatrix}\bar{u}^{(\nu_e)}_L, & 0 \end{pmatrix} \gamma^\mu  \sigma^+_W 
# \begin{pmatrix} 0 \\ u^{(e)}_L\end{pmatrix} = \\ \frac{g_W}{\sqrt{2}} \bar{u}^{(\nu_e)} \left[ (\chi^\uparrow)_W^\dagger \sigma^+_W \chi^\downarrow_W \right] \left[ \, \frac{1}{2} \gamma^\mu (I - \gamma^5) \right] u^{(e)} = \frac{g_W}{\sqrt{2}} \bar{u}^{(\nu_e)} \left[ \gamma^\mu \frac{1}{2}(I - \gamma^5) \right] u^{(e)}
# $$
# 
# donde hemos omitido el momento $p_e, p_{\nu_e}$ de los espinores, $u^{(e, {\nu_e})}$.
# 
# De otra forma podemos decir del vértice de las corrientes cargadas tiene tres factores:
#     
#   * la constante de acoplo $g_W/\sqrt{2}$
#   
#   * la proyeciuón de quiralidad a izquierdas  $\frac{1}{2} (I-\gamma^5)$.
#   
#   * la matriz de subida o bajada de tercera componente de isospín, $\sigma^\pm_W$, que medialos estados de los dupletes de isospín débil.
# 
# Habitualmente se ignora la matriz $\sigma^+_W$ en el vértice y simplemente se crean las corrientes con los espinoresde arriba y abajo del duplete. 
# 

# #### La unificación electrodébil
# 
# El SM establece la unificación entre fuerza electromagnética y débil. 
# 
# Ambos procesos, electromagnetismo y débil, están ligados.  El parámetro que determina su "mezcla" es el **ángulo de Weinberg**, $\sin \theta_W$.
# 
# La intensidad de los acoplos entre las interacciones electromagnética $e$, débil cargada, $g_W$, y débil neutra, $g_Z$, viene dada por:
# 
# $$
# e = g_W \sin \theta_W = g_Z \sin \theta_W \cos \theta_W
# $$
# 
# Le valor $\theta_W \simeq 33^o$, se ha determinado experimentalmente en varios procesos físicos, (ver después):
# 
# $$
# \sin^2 \theta_W = 0.23146 \pm 0.000012
# $$
# 
# Eso es, determinamos $g_W, g_Z$ a partir de la carga del electron $e$ (o de la constante de estructura fina $\alpha$) y del ángulo de Weinberg $\theta_W$.
# 

# #### Las corrientes neutras
# 
# Las interacciones con el $Z$ no cambian el sabor de la partícula, ni tampoco su carga eléctrica, ni su tercerca componente de isoespín débil.
# 
# Pero cambia de forma no trivial la paridad, porque el bosón $Z$ tiene un acoplo diferente para la quiralidad de izquierdas y de derechas, dependiendo del fermión.
# 
# | |
# |:--:|
# |<img src="./imgs/sm_zcur.png" width = 350 align="center">|
# | Corriente neutra|

# La corriente neutra con el bosón $Z$ tiene la forma:
# 
# $$
# g_Z \left( c_L \, \bar{u}_L \gamma^\mu \bar{u}_L + c_R \, \bar{u}_R \gamma^\mu \bar{u}_R \right) 
# $$
# 
# donde $g_Z$ es la constante de acoplo del $Z$ y $c_L, c_R$ los factores asociados a la parte de quiralidad del fermión.
# 
# El modelo estándard establece los coeficientes de acoplo del vértice con el $Z$:
# 
# 
#    | --- $c_L$ --- | --- $c_R$  --- | 
#    | :--:  | :--: |
#    | $I^3_W - Q \sin^2 \theta_W$  | $-Q \sin^2 \theta_W$| 
# 
# Donde $Q$ es la carga en unidades de electrón $e$, y $I^3_W$ es la tercerca componente de isospín débil

# Por ejemplo para el $\nu_e$ y el $e$
# 
#  | | --- $c_L$ --- | --- $c_R$  --- | 
#  | :--:  | :--:  | :--: |
#  | $\nu_e$  | $1/2$  | $0$|  
#  | $e$ | $-1/2 +s^2_W$ | $s^2_W$ |
# 
# Donde $s_W \equiv \sin \theta_W$
# 
# la corriente neutra del neutrino con el $Z$ será:
#     
# $$
# \frac{g_Z}{2} \bar{u}^{(\nu_e)}_L \gamma^\mu u^{(\nu_e)}_L 
# $$
# 
# Mientras que la del electrón:
# 
# $$
# g_Z \left(-\frac{1}{2} + s^2_W \right) \, \bar{u}^{(e)}_L \gamma^\mu u^{(e)}_L  + g_Z s^2_W \, \bar{u}^{(e)}_R \gamma^\mu u^{(e)}_R
# $$

# Para los fermiones y antifermiones
# 
# | ---- fermión ---- | -------- $c_L$ -------- | -------- $c_R$ ------ | --- antifermión --- | ---------- $c_L$ ----------- | --------- $c_R$ --------- |
# | :--:    | :--:  | :--: | :--:  | :--: | :--: |
# | $\nu_e$ | $1/2$     | 0   | $e^+$  | $s^2_W$  | $-1/2 + s^2_W$|
# | $e$    | $-1/2 + s^2_W$  | $s^2_W$  | ${\bar \nu}_e$ | 0 | $-1/2$ |
# | $u$     | $1/2 -2/3 s^2_W$ | $-2/3s^2_W$  | ${\bar d}$ | $-1/3s^2_W$| $1/2 - 1/3s^2_W$|
# | $d$     | $-1/2+1/3s^2_W$  | $1/3s^2_W$ | ${\bar u}$ |  $2/3 s^2_W$ | $-1/2 + (2/3)s^2_W$ |

# ### Aniquilación a muones, $e+e^+ \to \mu+\mu^+$ y la asimetría adelante y atrás.
# 
# En la aniquilación $e + e^+ \to \mu + \mu^+$ intervienen el $\gamma$ y el $Z$. Nota que el portador debe ser necesariamente neutro.
# 
# | |
# |:--:|
# |<img src="./imgs/sm_gz.png" width = 400 align="center">|
# | aniquilación $e+e^+ \to \mu + \mu^+$ mediada por el fotín (izda) o el $Z$ (derecha) |
# 
# Recordemos los propagadores respectivos son proporcionales a:
# 
# $$
# \frac{g_{\mu\nu}}{q^2}, \;\; \frac{g_{\mu\nu}}{q^2 - m_Z^2},
# $$
# donde $m_Z = 91.2$ GeV es la masa del $Z$, y $q^2$ el cuadrado del cuadrimomento transferido al bosón.
# 

# Tenemos tres regímenes de $q^2$ diferentes:
# 
# * $q^2 \ll m^2_Z$, en ese caso domina el propagador del fotón, la interacción es mayoritariamente electromagnética, y preserva paridad.
# 
# * $q^2 \sim m^2_Z$ domina el propagador del $Z$, la interacción es mayoritariamente débil neutra, y cambia al pasar por el valor $q^2 = m^2_Z$
# 
# * $q^2 \gg m^2_Z$ los dos propagadores contribuyen con una fracción cada uno.

# La figura muestra la sección eficaz $\sigma(e+e^+ \to \mathrm{hadrons})$ con los datos de diversos experimentos y la curva teórica.
# 
# En la figura se muestra también la sección eficaz debida solo al fotón (curva punteada).
# 
# Se observa que la contribución del fotón, es dominante a baja energía. que luego domina la resonancia del $Z$, y finalmente hay un equilibrio.
# 
# La sección eficaz no se explicaría sin la interferencia del $Z$.
#                   
# | |
# |:--:|
# |<img src="./imgs/sm_sigma_eeqq.png" width = 400 align="center">|
# | $\sigma(e+e^+ \to \mathrm{hadrons})$ vs $\sqrt{s}$ de [MT16.2] [LEP-SLD]|
#                                                                                      

# En la región donde $Z$ domine o sea relevante aparecen efectos de violación de paridad que dependerán de $s^2_W$.
# 
# Experimentalmente calculamos asimetrías. Consideramos el eje dado por el choque $e+e^+$ y la dirección hacia delante dada por el $e$ y contamos cuantos $\mu$ salen hacia delante, $N_F$, *forward* y hacia atrás, $N_B$, *backwards*, y calculamos la *asimetría adelante-atrás*, $A_{FB}$:
# 
# $$
# A_{FB} = \frac{N_F - N_B}{N_F + N_B}
# $$
# 
# Si no hay violación de paridad $A_{FB} = 0$
# 
# Estos son los valores experimentales obtenidos por el experimento DELPHI y el la predicción (línea) para $s^2_W \simeq 0.23$.
# 
# | |
# |:--|
# |<img src="./imgs/sm_afb.png" width = 400 align="center">|
# | Asimetría *forward-backward* $e+e^+ \to \mu + \mu^+$ vs $\sqrt{s}$|
# | datos (puntos) del experimento [DELPHI] y predicción del SM para $s^2_W \simeq 0.23$ (línea)|
# 

# ### Anchura de desintegración del $Z$
# 
# El colisionador LEP $e+e^+$ que operó durante los 90's verificó con gran detalle las predicciones del SM, especialmente la física del Z.
# 
# Durante un periodo LEP operó a $\sqrt{s} = m_Z = 91$ GeV produciendo millones de $Z$.
# 
# Uno de los estudios más importantes de LEP es la medición de la anchura de desintegración de $Z$.

# #### Anchura de desintegración parcial
# 
# 
# La anchura de desintegración viene dada por:
# 
# $$
# \Gamma = \frac{p^*}{8 \pi s} \langle |M_{fi}|^2 \rangle
# $$
# 
# En el caso de  $Z \to f + \bar{f}$, $\sqrt{s} = m_Z$ y $p^* = m_z/2$, (en el CM), si despreciamos la masa del fermión en comparación con la del $m_Z$.

# La anchura parcial de desintegración $Z \to f + \bar{f}$ es:
# 
# $$
# \Gamma(Z \to f + \bar{f}) = \frac{m_Z}{16 \pi m^2_Z} \frac{2}{3} (c^2_L + c^2_R) g^2_Z m^2_Z = \frac{g^2_Z m_Z}{24 \pi} (c^2_L + c^2_R)
# $$

# Por ejemplo para $\nu_e$ como $c^{(\nu_e)}_L = 1/2, \; c^{(\nu_e)}_R = 0$:
# 
# $$
# \Gamma (Z \to \nu_e + \bar{\nu}_e) = \frac{g^2_Z m_Z}{96 \pi} = 166 \; \mathrm{MeV}
# $$

# *Cuestión*: Calcula las anchuras de desintegración parciales y totales de $Z$ y sus fracciones de desintegración.

# In[3]:


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


# In[4]:


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


# La anchura total del $Z \to f + \bar{f}$ será la suma de las parciales:
# 
# $$
# \Gamma_Z = 3 \Gamma_{ll} +  \Gamma_{\mathrm{hadrons}} + N_\nu \Gamma_{\nu\nu}
# $$
# 
# donde $\Gamma_{ll} = \Gamma_{ee} + \Gamma_{\mu\mu} + \Gamma_{\tau\tau}$, $\Gamma_{\mathrm{hadrons}}$, la anchura de desintegración a hadrones (la total a $qq$) y $N_\nu$ es el número de neutrinos, y $\Gamma_{\nu\nu}$, la achura a un tipo de neutrino.
# 
# Los neutrinos no se observan, pero si medimos la anchura total $\Gamma_Z$ podemos estimar el número de neutrinos, $N_\nu$.

# In[5]:


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
# Donde en este caso $p^*_i = p^*_f = m_Z/2$, en el CM, podemos despreciar las masas de los fermiones en comparación con $m_Z$.

# La sección eficaz expresada en función de las anchuras de desintegración parciales, $\Gamma_{ee}, \Gamma_{\mu\mu}$ y total, $\Gamma_Z$, es:
# 
# $$
# \sigma (e+e^+\to Z \to \mu+\mu^+)= \frac{12 \pi s}{m^2_Z} \frac{\Gamma_{ee} \Gamma_{\mu\mu}}{(s-m^2_Z)^2 + m^2_Z\Gamma^2_Z} 
# $$
# 
# Las datos experimentales confirmaron que las anchuras $\Gamma_{ee} = \Gamma_{\mu\mu}$ son iguales, como predecía el modelo.

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

# In[6]:


Gee = 0.0838 # GeV 
Gmm = 0.0838 # GeV
Gnn = 0.1657 # GeV
Guu = 0.2894 # GeV
Gdd = 0.3713 # GeV
Ghad = 2 * Guu + 3 * Gdd
Gtot = Ghad + 3 * Gee + 3 * Gnn
#print(Gtot)
mz  = 91.19  # GeV
Gz  =  2.45  # GeV
sigma_mumu = (12 * np.pi)/(mz**2) * (Gee * Gmm)/Gz**2
sigma_had  = (12 * np.pi)/(mz**2) * (Gee * Ghad)/Gz**2
sigma_tot  = (12 * np.pi)/(mz**2) * (Gee * Gz)/(Gz**2)
hbarc  = 0.197 * units.femto # GeV m
barn =  1e-28 # m^2
sigma_to_barn = (hbarc**2) / barn
print('sigma max (ee->mumu) {:3.1f} nbarns '.format(sigma_to_barn * sigma_mumu / units.nano))
print('sigma max (ee->had)  {:3.1f} nbarns '.format(sigma_to_barn * sigma_had /units.nano))
print('sigma max (ee->Z)    {:3.1f} nbarns '.format(sigma_to_barn * sigma_tot /units.nano))


# In[7]:


def sigma(s, Gff = Gmm):
    return (12 * np.pi *s)/(mz**2) * (Gee*Gff/((s-mz**2)**2 + (mz*Gz)**2))
ss = np.linspace(mz-8, mz+8, 100)
plt.plot(ss, sigma_to_barn * sigma(ss*ss, Ghad) / units.nano); 
plt.grid(); plt.xlabel(r'$\sqrt{s}$ (GeV)'); plt.ylabel(r'$\sigma$ (nbarn)');


# ### Tres familias de neutrinos
# 
# Uno de los resultados más importantes de la era LEP es la determinación del número de neutrinos.
# 
# Se obtiene a partir de la anchura de desintegración, $\Gamma_Z$, del $Z$, obtenida por la anchura de la resonancia en la sección eficas $\gamma(e+e^+ \to hadrons)$.

# En la figura, la vista transversal de varios eventos en ALEP, $e+e^+ \to Z$ y $Z \to$ hadrons, $e+e^+, \mu+\mu^+, \tau+\tau^+$:
# 
# | | 
# | :--: |
# |  <img src="./imgs/sm_Zevents_ALEPH.png" width = 400 align="center"> |
# | $Z\to$ hadrons, $e+e^+$, $\mu+\mu^+$, $\tau + \tau^+$ [ALEPH]|

# 
# | | 
# | :--: |
# |  <img src="./imgs/sm_zresonance.png" width = 350 align="center"> |
# | $\sigma(e+e^+ \to \mathrm{hadrons})$ vs $\sqrt{s}$ de los experimentos de LEP [PDG]|
# 
# La siguiente figura muestra la sección eficaz $\sigma(e+e^+ \to \mathrm{hadrons})$  en función de $\sqrt{s}$ obtenida por los experimentos de LEP: ALEPH, DELPHI, L3, OPAL. 
# 
# 
# Se muestra sobreimpuesta la curva teórica cuando consideramos $N_\nu$ neutrinos.

# 
# Recordemos que el número de neutrinos, $N_\nu$, entra en la anchura de desintegración:
# 
# $$
# \Gamma_Z = 3 \Gamma_{ll} +  \Gamma_{\mathrm{hadrons}} + N_\nu \, \Gamma_{\nu\nu}
# $$
# 
# Dado que para cada leptón cargadro, $l = e, \mu, \tau$, $\Gamma_{ll} = \Gamma_{ee}$.
# 
# A parti de la medida de $\Gamma_Z = 2.4952 \pm 0.0023$, y de las anchuras medidas $\Gamma_{ll}, \, \Gamma_{\mathrm{hadrons}}$ y del valor teórico de $\Gamma_{\nu\nu}$:
# 
# $$
# \Gamma (\nu_e + \bar{\nu}_e) = \frac{g^2_Z m_Z}{96 \pi} = 166 \; \mathrm{MeV}
# $$
# 
# Obtenemos el número de neutrinos:
# 
# $$
# N_\nu = 2.9840\pm 0.0082
# $$
# 
# Existen por lo tanto **tres familias de neutrinos**.

# ## El bosón de Higgs
# 
# El modelo de Higgs da respuesta (quizás la única posible) a los siguientes problemas graves que presentabal la teoría:
# 
#   * Los bosones débiles son masivos, sin embargo las teorías gauge exigen bosones sin masa. ¿Cómo dotar entonces de masa a los bosones $W^\pm, Z$?
# 
#   * La simetría electrodébil separa fermiones a derechas e izquierdas, $u_L, u_R$, en dupletes y singletes, y exige que no tengan masas. ¡Pero sabemos que los fermiones tienen masa!
# 
#   * La sección eficaz, $W^+ + W^- \to W^+ + W^-$, crece indefinidamente a no ser que un escalar, el Higgs, con los acoplos apropiados la controle.
#   

# El mecanismo de Higgs permite resolver los tres problemas de una forma económica.
# 
# Es imposible discutir sobre el mecanismo de Higgs sin recurrir de forma breve al TCQ y al Lagrangiano del Modelo Estándad.
# 
# Vamos a dar solo unas ideas que nos permitan entender su comportamiento y descubrimiento.

# Los tres campos que componen las partículas elementales son: escalar (bosón de Higgs), espinorial (fermiones y anti-fermiones) y vectorial (bosones de las fuerzas), cuyas ecuaciones y lagrangianos son:
# 
# | | | | |
# | :--: |:--: | :--: | :--: |
# | ----- Tipo ----- | --------------- Escalar ---------------- | ---------- Espinor ------------ | ------ Bosón vectorial ---------- |
# | | Klein-Gordon | Dirac | Maxwell |
# |Ecuación | $(\partial^\mu\partial_\mu + m^2) \, \phi = 0$ | $(i\gamma^\mu \partial_\mu - m) \, \Psi = 0$ | $\partial_\mu F^{\mu\nu} = j^\nu$ |
# |Lagrangiano| $(D_\mu \phi)^\dagger (D^\mu \phi) - m^2 \phi^\dagger \phi$ | $i\bar{\Psi}\gamma^\mu D_\mu \Psi - m \bar{\Psi}\Psi$|  $-\frac{1}{4}F^{\mu\nu}F_{\mu\nu} \left[+ \frac{1}{2} m^2 A_\mu A^\mu \right]$|
# 
# Donde:
# 
# $$
# D_\mu = \partial_\mu + i g A_\mu, \;\;\; F^{\mu\nu} = \partial^\mu A^\nu - \partial^\nu A^\mu
# $$
# 
# $D_\mu$ es la *derivada covariante*, $g$ es el acoplo, por ejemplo $g=-e$ para el electrón en el electromagnetismo, y en ese caso $A_\mu$ correspondería al campo del fotón.
# 

# El Lagrangiano de espinores y bosones vectoriales (sin masa), por ejemplo del electromagnetismo es:
# 
# $$
# i \bar{\Psi} \gamma^\mu D_\mu \Psi - m \bar{\Psi} \Psi - \frac{1}{4} F^{\mu\nu}F_{\mu\nu}
# $$
# 
# que invariante bajo la simetría gauge local, dada por un cambio continua arbitrario $g\theta(x)$ en cada punto, local, $x$:
#     
# $$
# \Psi'(x) = e^{i g \theta(x)} \, \Psi(x), \;\;\; A'_\mu(x) = A_\mu(x) - \partial_\mu \theta(x)
# $$ 
# 
# ¡Pero el término de masas del bosón vectorial no es invariante gauge local!:
# 
# $$
# \frac{1}{2} m_A^2 A'^\mu A'_\mu  = \frac{1}{2}m_A^2 (A^\mu - \partial^\mu \theta) \, (A_\mu - \partial_\mu \theta) \neq \frac{1}{2} m_A^2 A^\mu A_\mu 
# $$
# 

# Notar que nos en lagrangiano aparece el término:
#     
# $$
# i \bar{\Psi} \gamma^\mu D_\mu \Psi = i \bar{\Psi} \gamma^\mu  (\partial_\mu + i q A_\mu) \Psi = \dots - g \bar{\Psi} \gamma^\mu \Psi \, A_\mu  
# $$
# 
# que es el acoplo de una corriente fermiónica, si $g = -e$ es el interacción de la corriente electromagnética del electrón con el fotón. E interpretamos el término como un vértice de tres cuerpos con acoplo $e$.
# 
# $$
#  e \bar{\Psi} \gamma^\mu \Psi \, A_\mu \equiv e j^\mu_{(e)} \, A_\mu
# $$
# 

# 
# ### Cómo el higgs otorga la masa a fermiones y bosones.
# 
# #### Fermiones
# 
# El término de masa asociado a los fermiones que aparece en el Lagrangiano es:
# 
# $$
# m \bar{\Psi} \Psi
# $$
# 
# que podemos obtener si acoplamos los fermiones con un campo escalar $\phi$, de la forma:
# 
# $$
# \lambda \bar{\Psi} \Psi \phi
# $$
# 
# que corresponde a un vértice de tres ramas con un acoplo $\lambda$ que llamamos de **acoplo de Yukawa**. 
# 
# Consideremos ahora un campo complejo escalar particular, $\phi$, con un **valor esperado en el vacío**, $v$:
# 
# $$
# \phi(x) = \frac{1}{\sqrt{2}}\left(v + h(x)\right) 
# $$
# 
# La interacción que hemos escrito anteriormente entre fermiones y el campo $\phi$ queda:
# 
# $$
# \lambda \bar{\Psi} \Psi \, \phi = \frac{\lambda v}{\sqrt{2}} \bar{\Psi} \Psi + \frac{\lambda}{\sqrt{2}} \bar{\Psi} \, \Psi h(x)
# $$
# 

# El primero término podemos asociarlo a la masa del fermión:
# 
# $$
# m \bar{\Psi} \Psi = \frac{\lambda v}{\sqrt{2}} \bar{\Psi} \Psi \Rightarrow m= \frac{\lambda v}{\sqrt{2}}.
# $$
# 
# 
# La masa aparece como la fricción, via el acoplo de Yukawa, $\lambda$, con el valor esperado del campo escalar (de Higgs) en el vacío, $v$.
# 
# El segundo términdo $\bar{\Psi} \Psi h$ veremos más adelante que lo interpretaremos como la interacción, desintegracion, del Higgs a un par fermión, anti-fermión.
# 

# #### Bosones
# 
# El mecanismo dota de masa a los bosones es más complejo.
# 
# La masa de un bosón vectorial tendría un término (ver el lagrangiano de la tabla anterior)
# 
# $$
# \left[+ \frac{1}{2} m^2 A_\mu A^\mu \right]
# $$
# 
# A partir de la deriviada covariante aplicada al campo escalar obtenemos:
# $$
# D_\mu \phi = (\partial_\mu + i g A_\mu) \frac{1}{\sqrt{2}} \left( v + h(x) \right)  = \frac{1}{\sqrt{2}} \left(\partial_\mu h(x) + ig v \, A_\mu + i g h(x) A_\mu \right)
# $$
# 
# En el lagrangiano del campo escalar aparece el producto escalar de la derivada covariante consigo misma, que si lo desarrollamos:
# 
# $$
# (D^\mu \phi)^*(D_\mu \phi) = \dots + \frac{1}{2} g^2 v^2 A^\mu A_\mu + \dots 
# $$
# 
# el término que hemos dejado explícito lo asociamos ahora con el término de masas del bosón. 
# 
# $$
# m = g v
# $$

# En al Modelo Estándar, el cálculo es similar aunque más elaborado, por ejemplo $\phi$ es un duplete y no un singlete, y obtenemos como masa de los bosones:
# 
# $$
# m_W = \frac{g_W v}{2}, \;\; m_Z = \frac{g_z v}{2} = \frac{g_W v}{2 \cos \theta_W}
# $$
# 
# Esto es, la unificación electrodébil y el mecanismo de Higgs predicen:
# 
# $$
# \frac{m_W}{m_Z} = \cos \theta_W
# $$
# 
# A partir de los datos experimentales, $M_z, \sin^2 \theta_W$, el SM predice,
# 
# $$
# m_W \simeq 80.34 \;\; \mathrm{GeV}
# $$
# 
# ¡que se aproxima mucho al valor experimental! 
# 
# Esta relación dio credibilidad al mecanismo de Higgs antes de que se el Higgs se descubriera.

# In[8]:


s2t = units.value("weak mixing angle")
MW  = 80.378 # GeV 
MZ  = 91.1876 # GeV
cw = np.sqrt(1 - s2t)
mw_pred = MZ * cw
print(' W mass esperada a primer orde {:4.3f} GeV, medida {:4.3f} GeV'.format(mw_pred, MW))


# ### El campo del bosón de Higgs
# 
# Vemos que podemos dotar de masa a los fermiones y bosones vectoriales débiles con un campo con un valor especial en el vacío, $v$, ¿pero de donde sale ese valor?
# 
# A partir de asociar -ad hoc- al campo de Higgs un potencial particular, el del sombrero mexicano.
# 
# El término del Lagrangiano de un camplo escalar complejo, $\phi$:
# 
# $$
# (D_\mu \phi)^\dagger (D^\mu \phi) - V(\phi)
# $$
# 
# Con:
# 
# $$
# V(\phi) = \frac{\mu^2}{2} (\phi^\dagger \phi) + \frac{\lambda}{4} (\phi^\dagger \phi)^2
# $$
# 
# y desarrollando el campo escalar alrededor del mínimo, $v$, del potencial.
# 
# $$
# \phi(x) = \frac{1}{\sqrt{v}} \left( v + h(x)\right)
# $$

# Las siguientes figuras muestran el potencial del sombrero mexicano para el caso $\mu^2>0, \mu^2<0$

# In[9]:


higgs_potential = lambda phi, mu, xlambda : ((mu**2).real * phi**2)/2 + (xlambda/4) * (phi**2)**2
phis = np.linspace(-10, 10, 100)
mu  = 6; xlambda = 1
plt.subplot(1, 2, 1); plt.plot(phis, higgs_potential(phis, mu, xlambda));
plt.subplot(1, 2, 2); plt.plot(phis, higgs_potential(phis, mu*(1j), xlambda));


# El estado del vacío es el estado de mínima energía: 
# 
# * El el caso $\mu^2 >0$ el potencial tiene un mínimo en $v = 0$
# 
# 
# * Mientras que el caso $\mu^2 <0$ tiene mínimos en:
# 
# $$
# v = \pm  \sqrt{\frac{-\mu^2}{\lambda}}
# $$
# 
# donde a  $v$ denomina **valor esperado en el vacío** (*vev*).
# 
# El hecho que el estado en el vacío se situe en uno de los mínimos de potencial de los posibles equivalentes se conoce como **rotura espontánea de simetría**.
# 

# En el casso de que $\phi$ sea un complejo obtenemos la curva del potencial conocida como sobrero mexicano:
# 
# | | 
# | :--: |
# |  <img src="./imgs/sm_mexican_hat.png" width = 300 align="center"> |
# | potencial del campo complejo de Higgs $V(\phi)$ [Wikipedia]|
# 
# En el modelo estándar es más complejo porque $\phi$ es un duplete.

# Al establecer invariancia gauge, el campo de Higgs da lugar a las interacciones:
# 
#  * entre los bosones débiles, $W^\pm, Z$ y el valor $v$ del vacío, que interpretamos como la **masa de los bosones $W^\pm, Z$**.
#  
#  * entre las dos componentes de quiralidad de los fermiones y $v$, lo que da lugar a las **masas de los fermiones**.
#  
#  * a interacciones entre el higgs, $h(x)$, y los bosones débiles, $W, Z$, y dar lugar a **desintegraciones del Higgs a los bosones vectoriales**
#  
#  * a interacciones entre el higgs, $h(x)$, y los fermiones, que a su vez da lugar a **desintegraciones a pares fermión-antifermión**
#  
# Además el higgs se acopla consigo mismo en vértices de tres y cuatro ramas, y con los bosones vectorales débiles en vértices de cuatro ramas. Esto es, genera nuevas interacciones entre sí, y con los bosones vectoriales.

# #### Las masas
# 
# La fricción de los bosones y los fermiones con el *vev* del Higgs da lugar a las masas:
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
# El mecanismo de Higgs establece **una relación entre la masa de los bosones vectoriales**:
# 
# $$
#  \cos \theta_W = \frac{m_W}{m_Z}
# $$

# El valor del $v$ se puede determinar a partir de $g_W, m_W$, o simplemente de $G_F$:
# 
# $$
# m_W = \frac{1}{2} g_W v \Rightarrow v = \frac{2m_W}{g_W} = \sqrt{\frac{1}{\sqrt{2} G_F}} = 246 \; \mathrm{GeV}
# $$
# 

# In[10]:


GF = units.value("Fermi coupling constant") # 1/GeV^2
v  = np.sqrt(1/(np.sqrt(2)*GF))
print('valor esperado del vacío {:4.1f} GeV'.format(v))


# #### la desintegraciones del bosón de Higgs
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
# *Nota adicional*: El mecanismo de Higgs introduce vértices adicionales tripes entre tres higgs; y cuadruples: entre cuatro higgs; y dos higgs y dos bosones vectoriales débiles.

# Las fracciones de desintegracción de Higgs dependeran a qué canales pueda desintegrarse dependiendo de su masa.
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
# En la tabla se muestran las fracciones de desintegración del Higgs para $m_H = 125$ GeV.
# 
# *Nota adicional* Las desintegraciones $H\to \gamma + \gamma, H \to g + g$ ocurren a través de diagramas de lazo, por ejemplo, con el intercambio en triángulo de un $t$.

# ### Descubrimiento del bosón de Higgs
# 
# El bosón de Higgs se descubrió en 2012 en los experimentos ATLAS y CMS del LHC del CERN.
# 
# Con anterioridad los experimentos de LEP y los experimentos CDF y D0 de Fermilab habían constreñido el valor de su masa más ligera entre: $115-150$ GeV.
# 
# Los experimentos y el acelerador LHC se diseñaron en los 90's y su operación se inición a finales de los 2000's.
# 
# En el LHC colisionan $p+p$ a una energía $\sqrt{s}= 7- 13$ TeV y con una luminosidad $\mathcal{L}(t) = 10^{34}$ cm$^{-2}$. Hay $10^{11}$ protones por paquete y una frecuencia de colisión de 40 MHz (cada 25 ns).
# 
# El número de colisiones $10^{9}$ por segundo y por cada cruce hay como máximo $35$ colisiones, lo que de denomila *pile-up* (apilamiento).

# Las sección eficaz $p+p$ is del ~100 mbars en el LHC.
# 
# Las sección eficaz medida de producción del $H$ para $\sqrt{s}=13$ TeV es 54 pbarns.
# 
# Los canales de desintegración dónde la señal es más fácil distinguible del fondo son $H\to Z+Z^*$ y $H \to \gamma \gamma$.
# 
# Eso equivale a buscar una desintegración relevante entre $10^{13}$.

# #### El descubrimiento del Higgs en ATLAS y CMS
# 
# 
# En la figura se muestra el esquema del experimento CMS del LHC:
# 
# | |
# |:-- |
# | <img src="./imgs/det_CMS_subdetectors.png" width = 500 align = "center"> |
# | Esquema de un sector del *Compact Muon Selenoid* ([CMS](https://cms.cern/detector)) del LHC|
# 

# El Higgs se descubrió en dos canales limpios (donde los sucesos de contaminación son o afectan menos).
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
#    * que se trataba de una partícula escalar a partir de las distribuciones angulares de los $(4l)$ en el canal $H\to 4 l$.
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
#   * El SM clasifica las partículas conocidas y describe con precisión las interacciones electrodébiles y fuertes entre ellas.
#   
#   * El SM se ha verificado en una gran cantidad de procesos físicos.
#   
#   * No obstante sabemos que los neutrinos tienen masa, lo que implica que debemos extender el SM.
#   
# Pero:
#   
#   * No incluye otra Física conocida, como la materia oscura, ¿qué partículas la forman?
#   
#   * El modelo tiene cinco parámetros fundamentales $e, g_W, g_S, v, \lambda$, dos de ellos corresponden al Higgs. Experimentalmente $\alpha, G_L, \alpha_S, m_Z, M_H$.
# 
#   * Tiene 9 parámetros adcionales, los acoplos de Yukawa (las masas de los leptones cargados y de los quarks); 4 más para acomodar la matriz CKM de los quarks, y 3 o 4 más para la PNMS de los leptones (la que rige las ocilaciones de neutrinos). 
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
