"""
Configuración común de los notebooks, en una sola línea::

    from fnyp.inicio import *

Activa las figuras en línea y la recarga automática del paquete ``fnyp`` (si se
edita un módulo, el notebook ve el cambio sin reiniciar el kernel), importa
``np``, ``plt``, ``units`` y los módulos ``fn`` (común), ``t1`` (introducción),
``t2`` (perspectivas) y ``sim`` (simulaciones), e imprime la fecha de ejecución.
"""

from IPython import get_ipython

_ip = get_ipython()
if _ip is not None:                       # fuera de un notebook no hay magics
    _ip.run_line_magic('matplotlib', 'inline')
    _ip.run_line_magic('reload_ext', 'autoreload')
    _ip.run_line_magic('autoreload', '2')

import numpy as np                        # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402
import scipy.constants as units           # noqa: E402

from fnyp import common as fn             # noqa: E402
from fnyp import introduccion as t1       # noqa: E402
from fnyp import perspectivas as t2       # noqa: E402
from fnyp import simula as sim            # noqa: E402

fn.version()

__all__ = ['np', 'plt', 'units', 'fn', 't1', 't2', 'sim']
