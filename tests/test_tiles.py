"""Testes para src/data/tiles.py: filtro de tiles nodata e alinhamento pixel a pixel
entre tile de imagem e tile de máscara. Pula automaticamente se geopandas/rasterio
não estiverem instalados."""
import numpy as np
import pytest

gpd = pytest.importorskip('geopandas')
rasterio = pytest.importorskip('rasterio')

from rasterio.transform import from_origin
from shapely.geometry import box

from src.data.tiles import TileGenerator


def _write_raster(path, width, height, nodata=None, fill_left_zero=False):
    data = (np.random.rand(3, height, width) * 200 + 20).astype(np.uint8)
    if fill_left_zero:
        data[:, :, :width // 2] = 0
    transform = from_origin(0, height * 30, 30, 30)
    with rasterio.open(
        path, 'w', driver='GTiff', height=height, width=width, count=3,
        dtype='uint8', crs='EPSG:31982', transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data)
    return transform


def test_nodata_tiles_are_skipped(tmp_path):
    raster_path = tmp_path / 'source.tif'
    _write_raster(raster_path, width=1200, height=600, nodata=0, fill_left_zero=True)

    out_dir = tmp_path / 'tiles'
    gen = TileGenerator(
        source_path=raster_path, output_dir=out_dir,
        tile_size=512, overlap=32, normalize_method='percentile',
        max_nodata_fraction=0.95,
    )
    saved, meta = gen.generate()

    assert meta['skipped_nodata'] > 0
    assert saved == meta['saved_tiles']
    assert saved < meta['total_tiles']
    assert saved == len(list(out_dir.glob('tile_*.png')))


def test_mask_tile_aligns_with_known_geometry(tmp_path):
    raster_path = tmp_path / 'source.tif'
    _write_raster(raster_path, width=1024, height=1024)

    # Um quadrado conhecido no canto superior-esquerdo do raster (mundo -> pixel:
    # x=0..300, y=0..300 a partir do transform from_origin(0, height*30, 30, 30)).
    known_square = box(0, 1024 * 30 - 300, 300, 1024 * 30)
    mask_gdf = gpd.GeoDataFrame(geometry=[known_square], crs='EPSG:31982')

    img_dir = tmp_path / 'images'
    msk_dir = tmp_path / 'masks'
    gen = TileGenerator(
        source_path=raster_path, output_dir=img_dir, tile_size=512, overlap=32,
        normalize_method='percentile', mask_gdf=mask_gdf, mask_output_dir=msk_dir,
    )
    gen.generate()

    from PIL import Image
    first_mask = np.array(Image.open(sorted(msk_dir.glob('tile_*.png'))[0]))
    # O quadrado conhecido cobre pixels (0..10, 0..10) do primeiro tile (300m / 30m/px).
    assert first_mask[5, 5] == 255
    assert first_mask[500, 500] == 0
