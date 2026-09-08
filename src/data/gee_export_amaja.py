"""
Exportação da composição anual Landsat (2023) para a região da AMAJA via Google Earth
Engine, replicando as 3 bandas de Liu et al. (2023): EVI_max, BSI_max, green_median.

Depende da API do Earth Engine e de uma conta Google autorizada com um projeto GEE.
`ee.Authenticate()` abre um fluxo OAuth interativo em navegador, então este script não
funciona como script standalone (`python gee_export_amaja.py`) nem em pipelines
não-interativos — precisa ser executado célula a célula em um notebook Jupyter/Colab:

    !pip install -q earthengine-api geemap
    import ee
    ee.Authenticate()
    ee.Initialize(project='SEU_PROJETO_GEE')

    from src.data.gee_export_amaja import export_amaja_composite
    export_amaja_composite(
        aoi_gpkg='data/raw/amaja/amaja_aoi.gpkg',
        year=2023,
        drive_folder='cpic_amaja',
    )

Resultado: GeoTIFF de 3 bandas (EVI_max, BSI_max, green_median) exportado para o Google
Drive, em EPSG:31982 e 30 m de resolução — pronto para tiling com src/data/tiles.py.

Índices espectrais (Liu et al., 2023, Tabela 2):
    EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)          [Huete et al., 1997]
    BSI = ((SWIR2+RED) - (NIR+BLUE)) / ((SWIR2+RED) + (NIR+BLUE))   [Diek et al., 2017]

BSI usa SWIR2 (SR_B7), exatamente como impresso na Tabela 2 do artigo — não a variante
mais comum na literatura (que usa SWIR1). A prioridade aqui é bater com a banda que
gerou o dataset de treino de Liu et al. (train_images/train_masks do Zenodo), já que o
teste de generalização na AMAJA só é válido se a composição da AMAJA usar a mesma
receita de bandas do treino; usar SWIR1 introduziria uma diferença de distribuição de
entrada que se confundiria com a diferença geográfica que o experimento quer medir.
"""
from typing import Optional


def _load_aoi_geojson(aoi_gpkg: str) -> dict:
    import geopandas as gpd
    gdf = gpd.read_file(aoi_gpkg).to_crs('EPSG:4326')
    return gdf.geometry.iloc[0].__geo_interface__


def _mask_landsat_c2l2(image):
    qa = image.select('QA_PIXEL')
    dilated_cloud = 1 << 1
    cirrus = 1 << 2
    cloud = 1 << 3
    cloud_shadow = 1 << 4
    mask = (qa.bitwiseAnd(dilated_cloud).eq(0)
            .And(qa.bitwiseAnd(cirrus).eq(0))
            .And(qa.bitwiseAnd(cloud).eq(0))
            .And(qa.bitwiseAnd(cloud_shadow).eq(0)))
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True).updateMask(mask)


def _add_indices(image):
    blue = image.select('SR_B2')
    green = image.select('SR_B3')
    red = image.select('SR_B4')
    nir = image.select('SR_B5')
    swir2 = image.select('SR_B7')

    evi = image.expression(
        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
        {'NIR': nir, 'RED': red, 'BLUE': blue}
    ).rename('EVI')

    bsi = image.expression(
        '((SWIR2 + RED) - (NIR + BLUE)) / ((SWIR2 + RED) + (NIR + BLUE) + 1e-9)',
        {'SWIR2': swir2, 'RED': red, 'NIR': nir, 'BLUE': blue}
    ).rename('BSI')

    return image.addBands([evi, bsi]).addBands(green.rename('GREEN'))


def export_amaja_composite(
    aoi_gpkg: str = 'data/raw/amaja/amaja_aoi.gpkg',
    year: int = 2023,
    output_crs: str = 'EPSG:31982',
    scale: float = 30.0,
    drive_folder: str = 'cpic_amaja',
    file_prefix: Optional[str] = None,
    max_cloud_cover: float = 80.0,
):
    """Gera a composição anual de 3 bandas (EVI_max, BSI_max, green_median) para a AOI
    da AMAJA e inicia a exportação para o Google Drive. Deve ser chamado de dentro do
    Colab, com o Earth Engine já autenticado (ee.Authenticate() + ee.Initialize())."""
    import ee

    aoi_geojson = _load_aoi_geojson(aoi_gpkg)
    aoi = ee.Geometry(aoi_geojson)

    collections = [
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'),
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'),
    ]
    merged = collections[0].merge(collections[1])

    filtered = (merged
                .filterBounds(aoi)
                .filterDate(f'{year}-01-01', f'{year}-12-31')
                .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover)))

    processed = filtered.map(_mask_landsat_c2l2).map(_add_indices)

    evi_max = processed.select('EVI').max().rename('EVI_max')
    bsi_max = processed.select('BSI').max().rename('BSI_max')
    green_median = processed.select('GREEN').median().rename('green_median')

    # Projeção final fica a cargo do Export.image.toDrive (crs/scale/region); chamar
    # .reproject() aqui forçaria a redução inteira na grade de destino antes da hora.
    composite = ee.Image.cat([evi_max, bsi_max, green_median]).clip(aoi)

    n_images = filtered.size().getInfo()
    print(f"Cenas Landsat 8/9 utilizadas para {year}: {n_images}")

    file_prefix = file_prefix or f'landsat_cpic_amaja_{year}'
    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=file_prefix,
        folder=drive_folder,
        fileNamePrefix=file_prefix,
        region=aoi,
        crs=output_crs,
        scale=scale,
        maxPixels=1e10,
        fileFormat='GeoTIFF',
    )
    task.start()
    print(f"Exportação iniciada: task id = {task.id}")
    print("Acompanhe em https://code.earthengine.google.com/tasks")
    print(f"Ao concluir, o arquivo estará em: Google Drive/{drive_folder}/{file_prefix}.tif")
    return task


if __name__ == '__main__':
    print(__doc__)
