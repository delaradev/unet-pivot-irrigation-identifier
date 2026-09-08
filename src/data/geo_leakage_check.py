"""
Verificação de sobreposição geográfica entre o dataset de treino de Liu et al. (2023) e
a área de validação da AMAJA, para a alegação de exclusão de amostras do RS descrita no
TCC (Seção 4.2).

O dataset do Zenodo não traz nenhum metadado geográfico por amostra: os arquivos são só
PNGs (confirmado sem chunks de metadado), e o nome de arquivo (`I-3_2005_000000_1.png`)
codifica um ID de célula de uma grade nacional (~180x180 km, Liu et al. §3.2) sem chave
pública que traduza esse ID para coordenadas reais. Como a grade não pode ser decodificada
diretamente, este módulo usa correspondência por conteúdo: gera, via
`src.data.gee_export_amaja`, uma composição de referência da área da AMAJA nos mesmos
anos presentes no dataset do Liu (2005/2010/2017) e usa correlação cruzada normalizada
(banda a banda) para achar, para cada célula de grade única do dataset, se algum recorte
da composição de referência bate com ela.

Comparar ano-contra-ano (em vez da composição de 2023 da AMAJA direto) evita o principal
fator de confusão: pivôs são instalados e desativados em poucos anos, então a "assinatura"
de pivôs de uma mesma área muda bastante entre 2005/2010/2017 e 2023 — mas relevo, solo e
drenagem (refletidos sobretudo em BSI) permanecem estáveis, então a correlação ano-a-ano
ainda captura corretamente uma sobreposição real.

Este script não decide sozinho quais células excluir: gera um manifesto CSV com o melhor
score de cada célula e uma imagem lado a lado (candidato vs. recorte de referência) para
os casos de score mais alto, para confirmação humana — mesmo padrão de
`src/data/review_tiles.py`. Só depois de revisado, a lista de IDs confirmados alimenta
`data.excluded_grid_ids` em config.yaml (ver `dataset_balancer.create_balanced_dataset`).

Uso (após gerar as composições de referência com gee_export_amaja.export_amaja_composite
para year=2005/2010/2017, salvas como landsat_cpic_amaja_<ano>.tif):
    python -m src.data.geo_leakage_check generate \\
        --dataset D:/data/dataset.zip \\
        --reference_dir data/raw/amaja/reference_composites \\
        --output_dir data/validation/geo_leakage
"""
import argparse
import csv
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

GRID_ID_PATTERN = re.compile(r'^([A-Za-z]+-\d+)_(\d{4})_')

DatasetSource = Union[str, Path]


def parse_grid_id(filename: str) -> Optional[Tuple[str, int]]:
    """Extrai (grid_id, ano) de um nome de arquivo do dataset do Liu, ex.:
    'I-3_2005_000000_1.png' -> ('I-3', 2005). Retorna None se não bater o padrão."""
    match = GRID_ID_PATTERN.match(filename)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def find_grid_representatives(dataset: DatasetSource) -> Dict[Tuple[str, int], str]:
    """Varre train_images/ e valid_data/valid_images/ (dataset extraído ou o .zip
    original do Zenodo) e retorna, para cada (grid_id, ano) único, o caminho/entrada da
    primeira imagem encontrada — só precisamos de UMA amostra por célula de grade, já
    que a exclusão é por célula inteira, não por arquivo individual."""
    representatives: Dict[Tuple[str, int], str] = {}
    dataset = Path(dataset)

    if dataset.is_file() and dataset.suffix.lower() == '.zip':
        with zipfile.ZipFile(dataset) as zf:
            for name in zf.namelist():
                if not name.lower().endswith('.png'):
                    continue
                if '/train_images/' not in name and '/valid_images/' not in name:
                    continue
                parsed = parse_grid_id(Path(name).name)
                if parsed and parsed not in representatives:
                    representatives[parsed] = f"zip://{dataset}!{name}"
    else:
        for subdir in ('train_images', 'valid_data/valid_images'):
            img_dir = dataset / subdir
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob('*.png'):
                parsed = parse_grid_id(img_path.name)
                if parsed and parsed not in representatives:
                    representatives[parsed] = str(img_path)

    logger.info(f"{len(representatives)} células de grade únicas encontradas em {dataset}")
    return representatives


def read_representative_image(ref: str) -> np.ndarray:
    """Carrega uma imagem retornada por find_grid_representatives, seja um caminho
    direto ou uma referência 'zip://<zip>!<entry>'."""
    import io

    from PIL import Image

    if ref.startswith('zip://'):
        zip_path, entry = ref[len('zip://'):].split('!', 1)
        with zipfile.ZipFile(zip_path) as zf:
            data = zf.read(entry)
        return np.array(Image.open(io.BytesIO(data)).convert('RGB'))
    return np.array(Image.open(ref).convert('RGB'))


def load_reference_composite(tif_path: Path) -> np.ndarray:
    """Carrega a composição GeoTIFF (EVI_max, BSI_max, green_median, valores float) e
    aplica a mesma normalização global usada em src/data/tiles.py para o dataset de
    treino/AMAJA, para que a comparação de conteúdo seja like-for-like."""
    import rasterio

    from src.data.tiles import NormalizationStrategy

    with rasterio.open(tif_path) as src:
        stack = np.transpose(src.read(), (1, 2, 0)).astype(np.float32)
    return NormalizationStrategy.apply(stack, method='global')


def match_patch_in_reference(patch: np.ndarray, reference: np.ndarray) -> Tuple[float, int, int]:
    """Desliza `patch` (H x W x 3, uint8) sobre `reference` (maior, mesma composição de
    bandas) via correlação cruzada normalizada banda a banda (cv2.TM_CCOEFF_NORMED),
    somando o score das 3 bandas. Retorna (score, linha, coluna) do melhor casamento.
    score é a soma das 3 correlações (cada uma em [-1, 1]); >~2.4 (média >0.8 por banda)
    indica correspondência forte, mas o limiar final é uma decisão humana no manifesto."""
    import cv2

    ph, pw = patch.shape[:2]
    rh, rw = reference.shape[:2]
    if ph > rh or pw > rw:
        raise ValueError(
            f"Referência ({rh}x{rw}) menor que o patch ({ph}x{pw}) — AOI de referência "
            "precisa cobrir uma área maior que um único tile."
        )

    total_score = 0.0
    best_loc = (0, 0)
    for band in range(3):
        result = cv2.matchTemplate(
            reference[:, :, band], patch[:, :, band], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        total_score += max_val
        if band == 0:
            best_loc = max_loc  # (col, row); bandas subsequentes só contribuem pro score

    col, row = best_loc
    return total_score, row, col


def save_comparison_image(patch: np.ndarray, reference: np.ndarray, row: int, col: int,
                          output_path: Path) -> None:
    from PIL import Image

    ph, pw = patch.shape[:2]
    crop = reference[row:row + ph, col:col + pw]
    side_by_side = np.concatenate([patch, crop], axis=1)
    Image.fromarray(side_by_side).save(output_path)


def run_matching(dataset: DatasetSource, reference_dir: Path, output_dir: Path,
                 file_prefix: str = 'landsat_cpic_amaja') -> List[dict]:
    reference_dir = Path(reference_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    representatives = find_grid_representatives(dataset)
    if not representatives:
        raise ValueError(f"Nenhuma amostra encontrada em {dataset}")

    reference_cache: Dict[int, Optional[np.ndarray]] = {}
    manifest = []

    for (grid_id, year), ref in sorted(representatives.items()):
        if year not in reference_cache:
            tif_path = reference_dir / f"{file_prefix}_{year}.tif"
            if tif_path.exists():
                reference_cache[year] = load_reference_composite(tif_path)
            else:
                logger.warning(f"Sem composição de referência para {year} ({tif_path}), pulando")
                reference_cache[year] = None
        reference = reference_cache[year]
        if reference is None:
            continue

        patch = read_representative_image(ref)
        try:
            score, row, col = match_patch_in_reference(patch, reference)
        except ValueError as e:
            logger.warning(f"{grid_id}/{year}: {e}")
            continue

        comparison_name = f"match_{grid_id}_{year}.png"
        save_comparison_image(patch, reference, row, col, output_dir / comparison_name)

        manifest.append({
            'grid_id': grid_id,
            'year': year,
            'sample': ref,
            'score': round(score, 4),
            'match_row': row,
            'match_col': col,
            'comparison_image': comparison_name,
            'decision': '',  # a preencher: confirm (sobreposição real, excluir do treino) / reject
            'notes': '',
        })

    manifest.sort(key=lambda r: r['score'], reverse=True)
    manifest_path = output_dir / 'geo_leakage_manifest.csv'
    with open(manifest_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'grid_id', 'year', 'sample', 'score', 'match_row', 'match_col',
            'comparison_image', 'decision', 'notes'])
        writer.writeheader()
        writer.writerows(manifest)

    logger.info(f"{len(manifest)} células comparadas. Manifesto: {manifest_path}")
    logger.info(
        "Abra as imagens 'match_*.png' (candidato à esquerda, recorte da referência à "
        "direita) começando pelas de maior score, e preencha 'decision' no CSV: "
        "confirm (sobreposição real) / reject (coincidência)")
    return manifest


def confirmed_grid_ids_from_manifest(manifest_path: Path) -> List[str]:
    """Lê um manifesto já revisado e retorna os grid_ids marcados 'confirm' — essa
    lista alimenta diretamente `data.excluded_grid_ids` em config.yaml."""
    confirmed = []
    with open(manifest_path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['decision'].strip().lower() == 'confirm':
                confirmed.append(row['grid_id'])
    return sorted(set(confirmed))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verifica sobreposição geográfica entre o dataset do Liu et al. e a AMAJA")
    sub = parser.add_subparsers(dest='command', required=True)

    gen = sub.add_parser('generate', help='Roda o matching e gera o manifesto para revisão')
    gen.add_argument('--dataset', type=str, required=True,
                     help='Caminho do dataset.zip do Zenodo ou do diretório já extraído')
    gen.add_argument('--reference_dir', type=str, required=True,
                     help='Diretório com landsat_cpic_amaja_<ano>.tif para 2005/2010/2017')
    gen.add_argument('--output_dir', type=str, required=True)

    ids = sub.add_parser('confirmed_ids', help='Lista os grid_ids confirmados num manifesto revisado')
    ids.add_argument('--manifest', type=str, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.command == 'generate':
        run_matching(Path(args.dataset), Path(args.reference_dir), Path(args.output_dir))
    elif args.command == 'confirmed_ids':
        ids = confirmed_grid_ids_from_manifest(Path(args.manifest))
        print(', '.join(ids) if ids else '(nenhum grid_id confirmado)')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main()
