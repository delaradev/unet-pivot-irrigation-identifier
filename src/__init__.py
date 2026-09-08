"""CPIC Brazil — pacote raiz.

Sem re-exports: importar `src` ou qualquer subpacote não deve forçar o carregamento
de TensorFlow, necessário só aos módulos de modelagem/treino. Os scripts de dados e
geoprocessamento (src/data/amaja.py, tiles.py, review_tiles.py) rodam
sem TensorFlow instalado. Importe sempre do submódulo específico, ex.:
`from src.models.unet import build_unet`.
"""
