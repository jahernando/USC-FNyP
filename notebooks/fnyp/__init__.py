"""
fnyp — código auxiliar de los apuntes de Física Nuclear y de Partículas (USC).

Criterio: **aquí solo va el código que el alumno no necesita ver**. Los cálculos
que resuelven una cuestión del texto, y las fórmulas de física escritas en código,
se quedan visibles en la celda del notebook.

Lo que se extrae a este paquete:

* el bloque de imports y configuración que se repite en todos los temas
  (:mod:`fnyp.common`);
* las constantes y masas del PDG, para no repetirlas —con valores distintos— en
  cada notebook (:mod:`fnyp.common`);
* la cosmética de las figuras (:func:`fnyp.common.plot_xy`);
* definiciones de función largas cuya expresión ya está escrita en LaTeX en el
  texto, como las secciones eficaces de Rutherford y de Mott
  (:mod:`fnyp.introduccion`).

Uso en un notebook::

    from fnyp import common as fn
    from fnyp import introduccion as t1

    t1.plot_secciones_eficaces(E_alpha=5e6, E_electron=5e6, M_nucleus=197)
"""

from . import common      # noqa: F401
from . import introduccion  # noqa: F401

__all__ = ['common', 'introduccion']
