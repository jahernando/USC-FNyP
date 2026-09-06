# Universidade de Santiago de Compostela
## Facultade de Física
## Curso Física Nuclear y de Partículas
### author: J. A. Hernando
### date  : September 2021


This repository contains Python-Notebooks and Python code for the lectures
on "Introduction to Particle Physics" of the "Nuclear and Particle Physics" introductory course of the University of Santiago de Compostela.

Index and links to the material of the course at *indice.ipynb*

Clich here to start your interactive session (be patient!):

Google: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jahernando/USC-FNyP/blob/main/notebooks/introduccion.ipynb)

Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jahernando/USC-FNyP/main)


## Entorno

Para **ejecutar los notebooks y compilar el Book** (entorno del autor):

```bash
conda env create -f environment.yml
conda activate fnyp
jupyter-book build .
```

Para **solo ejecutar los notebooks** basta con `requirements.txt`, que es lo que usa
Binder:

```bash
pip install -r requirements.txt
```

Las versiones están acotadas a propósito en los dos ficheros, y deben mantenerse en
paralelo: sin cotas, Binder resuelve lo último de cada día y las figuras que ve el
alumno pueden no coincidir con las que se compilan en local.

> **jupyter-book está fijado a la serie 0.15.** La 2.x es una reescritura sobre el
> motor MyST y no lee este `_config.yml` ni este `_toc.yml`. Actualizar exige migrar
> los dos ficheros y las directivas `admonition` de los talleres.
