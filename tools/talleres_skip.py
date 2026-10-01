"""
Talleres en *skip*: lista y, con ``--fix``, marca como ``skip`` todas las celdas
de los talleres de un notebook.

Un taller es la celda markdown cuyo título lleva ``[>] Taller`` y las que la
siguen hasta la siguiente celda markdown que no sea su ``<details>`` de
**Solución**: el enunciado, sus celdas de código y la solución. El texto
``[>] Taller`` es la marca visible; el ``slide_type`` de la metadata es lo que
lee RISE. Este script mantiene las dos cosas de acuerdo.

Uso::

    python tools/talleres_skip.py notebooks/perspectivas.ipynb          # lista
    python tools/talleres_skip.py notebooks/perspectivas.ipynb --fix    # marca skip

Cerrar el notebook en Jupyter antes de usar ``--fix``: si está abierto, al
guardar lo sobrescribe.
"""

import argparse
import json
import re
import sys

MARCA = '[>] Taller'


def _texto(celda):
    return ''.join(celda['source'])


def _es_solucion(celda):
    s = _texto(celda).lstrip()
    return (celda['cell_type'] == 'markdown' and s.startswith('<details>')
            and 'Solución' in s.split('\n', 2)[1])


def talleres(nb):
    """Lista de (título, [índices de celda]) de cada taller del notebook."""
    res, celdas = [], nb['cells']
    i = 0
    while i < len(celdas):
        s = _texto(celdas[i])
        if celdas[i]['cell_type'] == 'markdown' and MARCA in s:
            titulo = re.search(r'\[>\] Taller\s*—?\s*(.*?)</b>', s)
            titulo = titulo.group(1) if titulo else s.split('\n')[0]
            bloque = [i]
            j = i + 1
            while j < len(celdas) and (celdas[j]['cell_type'] == 'code'
                                       or _es_solucion(celdas[j])):
                bloque.append(j)
                j += 1
            res.append((titulo, bloque))
            i = j
        else:
            i += 1
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('notebook')
    ap.add_argument('--fix', action='store_true', help='marca skip las celdas de los talleres')
    args = ap.parse_args()

    with open(args.notebook, encoding='utf-8') as f:
        nb = json.load(f)

    cambios = 0
    for titulo, bloque in talleres(nb):
        tipos = [nb['cells'][k]['metadata'].get('slideshow', {}).get('slide_type', '-')
                 for k in bloque]
        estado = 'ok  ' if all(t == 'skip' for t in tipos) else 'NO  '
        print(f'{estado} celdas {bloque[0]:3d}-{bloque[-1]:3d}  {titulo}   {tipos}')
        if args.fix:
            for k in bloque:
                md = nb['cells'][k]['metadata']
                if md.get('slideshow', {}).get('slide_type') != 'skip':
                    md.setdefault('slideshow', {})['slide_type'] = 'skip'
                    cambios += 1

    if args.fix and cambios:
        with open(args.notebook, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')
        print(f'\n{cambios} celdas marcadas como skip.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
