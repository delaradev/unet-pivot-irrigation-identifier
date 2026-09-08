import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio import features
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from shapely import force_2d
from shapely.geometry import box

logger = logging.getLogger(__name__)


class NormalizationStrategy:
    BAND_BOUNDS = {
        0: {'name': 'EVI_max', 'low': 0.0, 'high': 0.7},
        1: {'name': 'BSI_max', 'low': -0.1, 'high': 0.3},
        2: {'name': 'GREEN_median', 'low': 0.0, 'high': 0.15}
    }

    @staticmethod
    def _percentile_normalize(band: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
        valid_pixels = band[~np.isnan(band)]
        if len(valid_pixels) == 0:
            return np.zeros_like(band)
        low_val = np.percentile(valid_pixels, p_low)
        high_val = np.percentile(valid_pixels, p_high)
        if high_val <= low_val:
            return np.zeros_like(band)
        normalized = (band - low_val) / (high_val - low_val)
        return np.clip(normalized, 0.0, 1.0)

    @classmethod
    def apply(cls, stack: np.ndarray, method: str = 'global') -> np.ndarray:
        normalized = np.zeros_like(stack, dtype=np.float32)
        n_bands = stack.shape[2]
        for band_idx in range(n_bands):
            band = stack[:, :, band_idx]
            if method == 'global':
                bounds = cls.BAND_BOUNDS.get(band_idx)
                if bounds is None:
                    band_norm = cls._percentile_normalize(band)
                else:
                    band_norm = (band - bounds['low']) / \
                        (bounds['high'] - bounds['low'])
                    band_norm = np.clip(band_norm, 0.0, 1.0)
            else:
                band_norm = cls._percentile_normalize(band)
            normalized[:, :, band_idx] = band_norm
        return (normalized * 255).astype(np.uint8)


class TileGenerator:
    def __init__(self, source_path: Path, output_dir: Path, tile_size: int = 512,
                 overlap: int = 32, normalize_method: str = 'global',
                 mask_gdf: Optional[gpd.GeoDataFrame] = None,
                 mask_output_dir: Optional[Path] = None,
                 max_nodata_fraction: float = 0.95):
        """
        mask_gdf: geometrias de referência (ex.: pivôs da ANA filtrados para a AMAJA,
            ver src/data/amaja.py) usadas para gerar, para cada tile de imagem, um tile
            de máscara binária correspondente (mesmo nome de arquivo), alinhado pixel a
            pixel via o transform exato daquela janela. Se None, gera apenas imagens.
        mask_output_dir: obrigatório quando mask_gdf é informado.
        max_nodata_fraction: tiles com fração de pixels nodata acima deste limiar são
            ignorados (imagem e máscara). Relevante para AOIs não retangulares (ex.:
            união de municípios da AMAJA): o GeoTIFF exportado do GEE tem nodata fora
            do polígono da AOI mas dentro do bounding box retangular, e sem esse filtro
            tiles inteiramente nessa faixa entrariam no dataset como imagem em branco.
        """
        self.source_path = Path(source_path)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        self.normalize_method = normalize_method
        self.max_nodata_fraction = max_nodata_fraction
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Source file not found: {self.source_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: Dict = {}

        self.mask_gdf = mask_gdf
        self.mask_output_dir = Path(mask_output_dir) if mask_output_dir else None
        if self.mask_gdf is not None and self.mask_output_dir is None:
            raise ValueError("mask_output_dir is required when mask_gdf is provided")
        if self.mask_output_dir:
            self.mask_output_dir.mkdir(parents=True, exist_ok=True)

    def _nodata_fraction(self, tile_data: np.ndarray, nodata_value: Optional[float]) -> float:
        if nodata_value is not None and not np.isnan(nodata_value):
            return float(np.mean(tile_data == nodata_value))
        # Sem nodata explícito no raster: trata pixel com todas as bandas em zero
        # (convenção comum em composições exportadas do GEE fora da área de recorte).
        return float(np.mean(np.all(tile_data == 0, axis=-1)))

    def _compute_tile_grid(self, width: int, height: int) -> Tuple[int, int, int, int]:
        step = self.tile_size - self.overlap
        n_tiles_x = (width - self.tile_size + step - 1) // step + 1
        n_tiles_y = (height - self.tile_size + step - 1) // step + 1
        return n_tiles_x, n_tiles_y, step, step

    def _extract_tile(self, src: rasterio.DatasetReader, x: int, y: int) -> np.ndarray:
        window = Window(x, y, self.tile_size, self.tile_size)
        tile_data = src.read(window=window)
        return np.transpose(tile_data, (1, 2, 0))

    def _rasterize_mask_tile(self, mask_gdf_proj: gpd.GeoDataFrame, window: Window,
                             transform: rasterio.Affine) -> np.ndarray:
        tile_bounds = window_bounds(window, transform)
        candidate_idx = list(mask_gdf_proj.sindex.query(box(*tile_bounds)))
        shapes = []
        if candidate_idx:
            for geom in mask_gdf_proj.geometry.iloc[candidate_idx]:
                if geom is None or geom.is_empty:
                    continue
                clipped = geom.intersection(box(*tile_bounds))
                if not clipped.is_empty:
                    shapes.append((force_2d(clipped), 1))

        tile_transform = rasterio.windows.transform(window, transform)
        if shapes:
            mask = features.rasterize(
                shapes, out_shape=(self.tile_size, self.tile_size),
                transform=tile_transform, fill=0, dtype=np.uint8)
        else:
            mask = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        return mask * 255

    def generate(self) -> Tuple[int, Dict]:
        logger.info(f"Processing source: {self.source_path}")
        with rasterio.open(self.source_path) as src:
            width, height, bands = src.width, src.height, src.count
            logger.info(
                f"Dimensions: {width}x{height}x{bands}, resolution: {src.res} m/px, CRS: {src.crs}")
            n_tiles_x, n_tiles_y, step_x, step_y = self._compute_tile_grid(
                width, height)
            total_tiles = n_tiles_x * n_tiles_y
            logger.info(
                f"Tile grid: {n_tiles_x}x{n_tiles_y} = {total_tiles} tiles, "
                f"step: {step_x}px, overlap: {self.overlap}px")

            mask_gdf_proj = None
            if self.mask_gdf is not None:
                mask_gdf_proj = self.mask_gdf.to_crs(src.crs)
                logger.info(
                    f"Máscara de referência: {len(mask_gdf_proj)} geometrias reprojetadas para {src.crs}")

            nodata_value = src.nodata

            tile_counter = 0
            saved_counter = 0
            skipped_nodata = 0
            for y_idx in range(n_tiles_y):
                y = min(y_idx * step_y, height - self.tile_size)
                for x_idx in range(n_tiles_x):
                    x = min(x_idx * step_x, width - self.tile_size)
                    window = Window(x, y, self.tile_size, self.tile_size)
                    tile_data = self._extract_tile(src, x, y)

                    if self._nodata_fraction(tile_data, nodata_value) >= self.max_nodata_fraction:
                        skipped_nodata += 1
                        tile_counter += 1
                        continue

                    normalized = NormalizationStrategy.apply(
                        tile_data, method=self.normalize_method)
                    tile_name = f"tile_{tile_counter:05d}.png"
                    Image.fromarray(normalized).save(self.output_dir / tile_name)

                    if mask_gdf_proj is not None:
                        mask_tile = self._rasterize_mask_tile(
                            mask_gdf_proj, window, src.transform)
                        Image.fromarray(mask_tile).save(
                            self.mask_output_dir / tile_name)

                    tile_counter += 1
                    saved_counter += 1
                    if tile_counter % 100 == 0:
                        logger.info(
                            f"Generated {tile_counter}/{total_tiles} tiles")

            if skipped_nodata:
                logger.info(
                    f"Tiles ignorados por nodata (>= {self.max_nodata_fraction:.0%} vazio): "
                    f"{skipped_nodata}/{total_tiles}")

            self.metadata = {
                'source': str(self.source_path), 'width': width, 'height': height, 'bands': bands,
                'tile_size': self.tile_size, 'overlap': self.overlap, 'total_tiles': total_tiles,
                'saved_tiles': saved_counter, 'skipped_nodata': skipped_nodata,
                'normalize_method': self.normalize_method, 'resolution_meters': src.res[0], 'crs': str(src.crs),
                'mask_output_dir': str(self.mask_output_dir) if self.mask_output_dir else None,
            }
            logger.info(
                f"Generation complete: {saved_counter}/{total_tiles} tiles created ({skipped_nodata} skipped)")
            return saved_counter, self.metadata

    def validate(self) -> Dict:
        tiles = sorted(self.output_dir.glob("tile_*.png"))
        actual_count = len(tiles)
        # Usa saved_tiles (não total_tiles) como referência: tiles pulados por nodata
        # são uma ausência esperada, não uma falha de geração.
        expected_count = self.metadata.get(
            'saved_tiles', self.metadata.get('total_tiles', 0))
        validation = {
            'expected_tiles': expected_count, 'actual_tiles': actual_count,
            'complete': actual_count == expected_count, 'difference': expected_count - actual_count
        }
        logger.info(
            f"Validation: expected {expected_count}, actual {actual_count}")
        if not validation['complete']:
            logger.warning(f"Missing {validation['difference']} tiles")
        return validation

    def compute_statistics(self, n_samples: int = 100) -> Dict:
        tiles = sorted(self.output_dir.glob("tile_*.png"))[:n_samples]
        if not tiles:
            logger.warning("No tiles found for statistics")
            return {}
        band_means, band_stds = [], []
        for tile_path in tiles:
            arr = np.array(Image.open(tile_path)) / 255.0
            band_means.append(arr.mean(axis=(0, 1)))
            band_stds.append(arr.std(axis=(0, 1)))
        band_means = np.array(band_means)
        band_stds = np.array(band_stds)
        stats = {
            'n_samples': len(tiles),
            'band_means': band_means.mean(axis=0).tolist(),
            'band_stds': band_stds.mean(axis=0).tolist(),
            'band_means_range': [(band_means[:, i].min(), band_means[:, i].max()) for i in range(3)]
        }
        logger.info(f"Statistics from {len(tiles)} tiles:")
        for b, name in enumerate(['EVI_max', 'BSI_max', 'GREEN_median']):
            logger.info(
                f"  {name}: mean={stats['band_means'][b]:.3f}, std={stats['band_stds'][b]:.3f}")
        return stats


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate normalized tiles from Landsat CPIC composites")
    parser.add_argument('--tif_path', type=str,
                        default='ibge/landsat_cpic_2023.tif', help='Path to input GeoTIFF')
    parser.add_argument('--out_dir', type=str, default='tiles/images',
                        help='Output directory for PNG tiles')
    parser.add_argument('--tile_size', type=int,
                        default=512, help='Tile size in pixels')
    parser.add_argument('--overlap', type=int, default=32,
                        help='Overlap between adjacent tiles in pixels')
    parser.add_argument('--normalize', type=str, default='global',
                        choices=['global', 'percentile'], help='Normalization method')
    parser.add_argument('--validate', action='store_true',
                        help='Validate tile generation')
    parser.add_argument('--stats', action='store_true',
                        help='Compute statistics')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of tiles for statistics')
    parser.add_argument('--mask_gpkg', type=str, default=None,
                        help='Optional path to a vector file (e.g. amaja_pivos.gpkg) with '
                             'reference geometries; when set, generates matching mask tiles')
    parser.add_argument('--mask_out_dir', type=str, default=None,
                        help='Output directory for mask tiles (required with --mask_gpkg)')
    parser.add_argument('--max_nodata_fraction', type=float, default=0.95,
                        help='Skip tiles with a nodata/empty fraction at or above this value')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    try:
        mask_gdf = None
        if args.mask_gpkg:
            if not args.mask_out_dir:
                raise ValueError("--mask_out_dir is required when --mask_gpkg is set")
            mask_gdf = gpd.read_file(args.mask_gpkg)

        generator = TileGenerator(
            source_path=Path(args.tif_path), output_dir=Path(args.out_dir),
            tile_size=args.tile_size, overlap=args.overlap, normalize_method=args.normalize,
            mask_gdf=mask_gdf,
            mask_output_dir=Path(args.mask_out_dir) if args.mask_out_dir else None,
            max_nodata_fraction=args.max_nodata_fraction,
        )
        total_tiles, _ = generator.generate()
        if args.validate:
            generator.validate()
        if args.stats:
            generator.compute_statistics(n_samples=args.sample_size)
        logger.info(
            f"Tile generation completed. Output: {args.out_dir}, total tiles: {total_tiles}")
    except Exception as e:
        logger.error(f"Tile generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main()
