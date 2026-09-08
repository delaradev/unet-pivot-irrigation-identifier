"""Módulos de dados. Sem re-exports (ver src/__init__.py) — alguns módulos aqui
(dataset_balancer.py) dependem de TensorFlow, outros (amaja.py, tiles.py,
review_tiles.py, gee_export_amaja.py) dependem só de geopandas/rasterio/PIL e devem
poder ser importados isoladamente sem puxar TensorFlow."""
