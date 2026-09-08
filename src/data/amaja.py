"""
Baixa e prepara os dados de referência da AMAJA (Associação dos Municípios do Alto
Jacuí/RS): limites municipais, área de interesse (AOI) e pivôs centrais da ANA
filtrados para os municípios da associação. Ver README.md para o pipeline completo
(composição Landsat via GEE, tiling e revisão humana das máscaras).

Fontes:
  - Lista de municípios: https://amaja.com.br/site/municipios.php (acesso: 2026-09-06)
  - Limites municipais: API de Malhas do IBGE (servicodados.ibge.gov.br/api/v2/malhas)
  - Pivôs georreferenciados: ANA, "Agricultura Irrigada por Pivôs Centrais no Brasil"
    (mesmo registro citado por Liu et al. 2023), edição PivosCentrais2019_AtlasIrrigacao2021
    (Atlas da Irrigação, 2ª ed. 2021), distribuída via metadados.snirh.gov.br/geonetwork.
"""
import argparse
import json
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd

logger = logging.getLogger(__name__)

# Municípios da AMAJA e respectivos códigos IBGE (7 dígitos).
# Fonte: https://amaja.com.br/site/municipios.php + servicodados.ibge.gov.br/api/v1/localidades
AMAJA_MUNICIPALITIES: Dict[str, str] = {
    "4300471": "Almirante Tamandaré do Sul",
    "4302220": "Boa Vista do Cadeado",
    "4302238": "Boa Vista do Incra",
    "4304705": "Carazinho",
    "4305603": "Colorado",
    "4305850": "Coqueiros do Sul",
    "4306106": "Cruz Alta",
    "4307500": "Espumoso",
    "4308458": "Fortaleza dos Valos",
    "4310009": "Ibirubá",
    "4311270": "Lagoa dos Três Cantos",
    "4312658": "Não-Me-Toque",
    "4315354": "Quinze de Novembro",
    "4316436": "Saldanha Marinho",
    "4316709": "Santa Bárbara do Sul",
    "4317756": "Santo Antônio do Planalto",
    "4320305": "Selbach",
    "4321006": "Tapera",
    "4323200": "Victor Graeff",
    "4316451": "Salto do Jacuí",
}

# Registro do GeoNetwork/SNIRH ("Agricultura Irrigada por Pivôs Centrais no Brasil"),
# o mesmo citado por Liu et al. (2023). Edição mais recente disponível publicamente.
ANA_RECORD_UUID = "e2d38e3f-5e62-41ad-87ab-990490841073"
ANA_ATTACHMENT = "PivosCentrais2019_AtlasIrrigacao2021.zip"
ANA_DOWNLOAD_URL = (
    f"https://metadados.snirh.gov.br/geonetwork/srv/api/records/"
    f"{ANA_RECORD_UUID}/attachments/{ANA_ATTACHMENT}"
)

# CRS métrico oficial (SIRGAS 2000 / UTM 22S) usado em todo o pipeline da AMAJA,
# consistente com o CRS de origem da ANA (EPSG:4674, mesmo datum SIRGAS2000).
TARGET_CRS = "EPSG:31982"


def fetch_municipality_boundaries(
    municipalities: Dict[str, str] = AMAJA_MUNICIPALITIES,
    output_path: Optional[Path] = None,
    quality: str = "intermediaria",
) -> gpd.GeoDataFrame:
    """Baixa o limite de cada município via API de Malhas do IBGE e retorna um único
    GeoDataFrame (uma feature por município, CRS EPSG:4674)."""
    features = []
    for code, name in municipalities.items():
        url = (f"https://servicodados.ibge.gov.br/api/v2/malhas/{code}"
               f"?formato=application/vnd.geo+json&qualidade={quality}")
        logger.info(f"Baixando limite de {name} ({code})")
        req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            if resp.info().get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
        geojson = json.loads(raw.decode("utf-8"))
        for feat in geojson["features"]:
            feat["properties"] = {"CD_GEOCMU": code, "NM_MUNICIP": name}
        features.extend(geojson["features"])

    gdf = gpd.GeoDataFrame.from_features(
        {"type": "FeatureCollection", "features": features}, crs="EPSG:4674")
    logger.info(f"Total de municípios baixados: {len(gdf)}/{len(municipalities)}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GPKG")
        logger.info(f"Limites municipais salvos em: {output_path}")

    return gdf


def build_aoi(boundaries: gpd.GeoDataFrame, buffer_meters: float = 5000.0) -> gpd.GeoDataFrame:
    """Dissolve os municípios em uma única área de interesse (AOI), com um buffer
    padrão de 5 km, retornada em TARGET_CRS. Usada para recortar a exportação do GEE.
    O buffer evita que pivôs próximos à borda administrativa da AMAJA fiquem sem
    contexto de imagem completo quando essa borda cair no meio de um tile 512x512
    (15.360 m de lado a 30 m/px) na etapa de tiling."""
    dissolved = boundaries.to_crs(TARGET_CRS).union_all()
    if buffer_meters:
        dissolved = dissolved.buffer(buffer_meters)
    return gpd.GeoDataFrame(geometry=[dissolved], crs=TARGET_CRS)


def download_ana_pivots(output_dir: Path) -> Path:
    """Baixa e extrai o shapefile nacional de pivôs da ANA. Retorna o caminho do .shp.
    Verifica se um download anterior está de fato íntegro (não só se o arquivo existe)
    antes de pular o download — um .zip corrompido por uma conexão interrompida seria
    tratado como "já pronto" indefinidamente sem essa checagem."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / ANA_ATTACHMENT

    needs_download = True
    if zip_path.exists():
        if zipfile.is_zipfile(zip_path):
            needs_download = False
            logger.info(f"Arquivo já existe e é válido, pulando download: {zip_path}")
        else:
            logger.warning(
                f"Arquivo existente está corrompido/incompleto, baixando novamente: {zip_path}")
            zip_path.unlink()

    if needs_download:
        logger.info(f"Baixando dados da ANA de: {ANA_DOWNLOAD_URL}")
        urllib.request.urlretrieve(ANA_DOWNLOAD_URL, zip_path)
        if not zipfile.is_zipfile(zip_path):
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download falhou ou o arquivo obtido não é um .zip válido: {ANA_DOWNLOAD_URL}")
        logger.info(f"Download concluído: {zip_path} ({zip_path.stat().st_size} bytes)")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)

    shp_files = list(output_dir.glob("*.shp"))
    if not shp_files:
        raise RuntimeError(f"Nenhum .shp encontrado após extração em {output_dir}")
    return shp_files[0]


def filter_ana_pivots_to_amaja(
    shp_path: Path,
    municipalities: Dict[str, str] = AMAJA_MUNICIPALITIES,
    output_path: Optional[Path] = None,
    min_area_ha: float = 0.5,
) -> gpd.GeoDataFrame:
    """Filtra o shapefile nacional de pivôs da ANA para os municípios da AMAJA e aplica
    os filtros de qualidade necessários antes de virar máscara de treino/validação:
      - geometria válida e não vazia (corrige com buffer(0) quando necessário);
      - área mínima (min_area_ha) para descartar polígonos espúrios/artefatos de
        digitalização (pivôs reais têm tipicamente dezenas a centenas de hectares;
        o próprio artigo de Liu et al. reporta que pivôs < 3 ha são < 0,06% da área
        total e já representam o limite inferior de detectabilidade em Landsat).
    """
    gdf = gpd.read_file(shp_path)
    logger.info(f"Total de pivôs no Brasil (ANA): {len(gdf)}")

    codes = list(municipalities.keys())
    amaja = gdf[gdf["CD_GEOCMU"].astype(str).isin(codes)].copy()
    logger.info(f"Pivôs nos municípios da AMAJA: {len(amaja)}")

    amaja["geometry"] = amaja["geometry"].buffer(0)
    amaja = amaja[~amaja.geometry.is_empty & amaja.geometry.notna()]

    amaja_metric = amaja.to_crs(TARGET_CRS)
    area_ha = amaja_metric.geometry.area / 10_000.0
    before = len(amaja_metric)
    amaja_metric = amaja_metric[area_ha >= min_area_ha]
    logger.info(
        f"Descartados {before - len(amaja_metric)} polígonos com área < {min_area_ha} ha")

    counts_by_mun = amaja_metric["NM_MUNICIP"].value_counts()
    logger.info(f"Pivôs por município:\n{counts_by_mun}")
    logger.info(f"Área total: {area_ha[area_ha >= min_area_ha].sum():.1f} ha")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        amaja_metric.to_file(output_path, driver="GPKG")
        logger.info(f"Pivôs da AMAJA salvos em: {output_path}")

    return amaja_metric


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construção da base de dados da região da AMAJA (limites municipais + pivôs da ANA)")
    parser.add_argument('--output_dir', type=str, default='data/raw/amaja',
                        help='Diretório de saída para limites e pivôs filtrados')
    parser.add_argument('--ana_dir', type=str, default='data/raw/ana',
                        help='Diretório para download/extração do shapefile nacional da ANA')
    parser.add_argument('--min_area_ha', type=float, default=0.5,
                        help='Área mínima (ha) para manter um pivô como válido')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)

    boundaries = fetch_municipality_boundaries(
        output_path=output_dir / 'amaja_municipios.gpkg')

    aoi = build_aoi(boundaries)
    aoi.to_file(output_dir / 'amaja_aoi.gpkg', driver='GPKG')
    bounds = aoi.to_crs('EPSG:4326').total_bounds
    logger.info(
        f"AOI da AMAJA (EPSG:4326, para uso no script GEE): "
        f"minLon={bounds[0]:.4f}, minLat={bounds[1]:.4f}, maxLon={bounds[2]:.4f}, maxLat={bounds[3]:.4f}")

    shp_path = download_ana_pivots(Path(args.ana_dir))
    filter_ana_pivots_to_amaja(
        shp_path, output_path=output_dir / 'amaja_pivos.gpkg', min_area_ha=args.min_area_ha)

    logger.info(
        "Próximo passo: rodar o script GEE (src/data/gee_export_amaja.py) no Colab "
        "usando a AOI salva em amaja_aoi.gpkg, depois gerar os tiles pareados com tiles.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main()
