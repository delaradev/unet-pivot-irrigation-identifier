"""
Apoio à validação humana das máscaras da AMAJA derivadas do cadastro da ANA. O cadastro
é atualizado em intervalos de ~5 anos, então pivôs novos podem faltar e pivôs
desativados podem persistir na máscara gerada — a checagem visual contra a própria
composição Landsat é o que corrige isso.

Este script não decide sozinho quais máscaras estão corretas: gera, para cada tile
imagem+máscara, uma miniatura de comparação lado a lado e um manifesto CSV para que o
revisor humano registre a decisão (mantém, descarta, ou precisa de correção manual).

Uso (após gerar os tiles pareados com tiles.py):
    python src/data/review_tiles.py generate \
        --images_dir data/dataset_amaja/valid_images \
        --masks_dir data/dataset_amaja/valid_masks \
        --output_dir data/validation/amaja_review
"""
import argparse
import csv
import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _read_csv_rows(path: Path) -> List[dict]:
    """Lê o manifesto tentando encodings comuns, nessa ordem: utf-8-sig (o que este
    script escreve), utf-8 puro, e cp1252 (padrão do Excel em Windows PT-BR quando o
    revisor edita e salva o CSV novamente sem manter o BOM utf-8)."""
    last_error = None
    for encoding in ('utf-8-sig', 'utf-8', 'cp1252'):
        try:
            with open(path, encoding=encoding) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as e:
            last_error = e
    raise last_error


def _erode(mask_bin: np.ndarray) -> np.ndarray:
    """Erosão binária 4-conectada usando apenas numpy (evita depender de scipy)."""
    shifted_up = np.zeros_like(mask_bin)
    shifted_up[:-1, :] = mask_bin[1:, :]
    shifted_down = np.zeros_like(mask_bin)
    shifted_down[1:, :] = mask_bin[:-1, :]
    shifted_left = np.zeros_like(mask_bin)
    shifted_left[:, :-1] = mask_bin[:, 1:]
    shifted_right = np.zeros_like(mask_bin)
    shifted_right[:, 1:] = mask_bin[:, :-1]
    return mask_bin & shifted_up & shifted_down & shifted_left & shifted_right


def _make_overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask_bin = mask > 127

    # contorno da máscara (diferença entre a máscara e sua versão erodida em 1px)
    contour = mask_bin & ~_erode(mask_bin)

    overlay = image.copy()
    overlay[contour] = [255, 255, 0]  # amarelo, alto contraste contra composições EVI/BSI

    side_by_side = np.concatenate([image, overlay], axis=1)
    return Image.fromarray(side_by_side.astype(np.uint8))


def generate_review_set(images_dir: Path, masks_dir: Path, output_dir: Path) -> List[dict]:
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles = sorted(images_dir.glob('*.png'))
    if not tiles:
        raise ValueError(f"Nenhum tile encontrado em {images_dir}")

    manifest = []
    for img_path in tiles:
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            logger.warning(f"Máscara ausente para {img_path.name}, pulando")
            continue

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))
        fg_ratio = float((mask > 127).sum()) / mask.size

        overlay_path = output_dir / f"review_{img_path.stem}.png"
        _make_overlay(image, mask).save(overlay_path)

        manifest.append({
            'tile': img_path.name,
            'fg_ratio': round(fg_ratio, 5),
            'review_image': overlay_path.name,
            'decision': '',   # a preencher: keep / discard / fix
            'notes': '',
        })

    manifest.sort(key=lambda r: r['fg_ratio'], reverse=True)
    manifest_path = output_dir / 'review_manifest.csv'
    # utf-8-sig (com BOM) para o Excel no Windows abrir corretamente os acentos ao
    # dar duplo-clique no arquivo, sem precisar importar manualmente como UTF-8.
    with open(manifest_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f, fieldnames=['tile', 'fg_ratio', 'review_image', 'decision', 'notes'])
        writer.writeheader()
        writer.writerows(manifest)

    logger.info(f"{len(manifest)} tiles preparados para revisão em {output_dir}")
    logger.info(f"Manifesto: {manifest_path}")
    logger.info(
        "Abra as imagens 'review_*.png' (composição à esquerda, contorno da máscara em "
        "amarelo à direita) e preencha a coluna 'decision' no CSV: keep / discard / fix")
    return manifest


def apply_review_decisions(
    images_dir: Path, masks_dir: Path, manifest_path: Path,
    output_images_dir: Path, output_masks_dir: Path,
) -> None:
    """Copia para o dataset final apenas os tiles marcados como 'keep' no manifesto
    revisado (tiles 'discard' são excluídos; tiles 'fix' exigem edição manual da máscara
    antes de serem incluídos, e por isso NÃO são copiados automaticamente)."""
    import shutil

    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    output_images_dir, output_masks_dir = Path(output_images_dir), Path(output_masks_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    kept = skipped = needs_fix = 0
    for row in _read_csv_rows(Path(manifest_path)):
        decision = row['decision'].strip().lower()
        tile = row['tile']
        if decision == 'keep':
            shutil.copy2(images_dir / tile, output_images_dir / tile)
            shutil.copy2(masks_dir / tile, output_masks_dir / tile)
            kept += 1
        elif decision == 'fix':
            needs_fix += 1
            logger.warning(
                f"{tile} marcado como 'fix' — edite a máscara manualmente e copie-a "
                f"para {output_masks_dir} antes de treinar")
        else:
            skipped += 1

    logger.info(f"Aplicado: {kept} mantidos, {skipped} descartados, "
                f"{needs_fix} pendentes de correção manual")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apoio à validação humana das máscaras da AMAJA")
    sub = parser.add_subparsers(dest='command', required=True)

    gen = sub.add_parser('generate', help='Gera overlays + manifesto para revisão')
    gen.add_argument('--images_dir', type=str, required=True)
    gen.add_argument('--masks_dir', type=str, required=True)
    gen.add_argument('--output_dir', type=str, required=True)

    apply_p = sub.add_parser('apply', help='Aplica as decisões do manifesto revisado')
    apply_p.add_argument('--images_dir', type=str, required=True)
    apply_p.add_argument('--masks_dir', type=str, required=True)
    apply_p.add_argument('--manifest', type=str, required=True)
    apply_p.add_argument('--output_images_dir', type=str, required=True)
    apply_p.add_argument('--output_masks_dir', type=str, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.command == 'generate':
        generate_review_set(Path(args.images_dir), Path(
            args.masks_dir), Path(args.output_dir))
    elif args.command == 'apply':
        apply_review_decisions(
            Path(args.images_dir), Path(args.masks_dir), Path(args.manifest),
            Path(args.output_images_dir), Path(args.output_masks_dir),
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main()
