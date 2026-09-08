"""Testes para src/data/geo_leakage_check.py: parsing do ID de grade do dataset do
Liu et al. e matching de conteúdo por correlação cruzada. Pula automaticamente se
opencv não estiver instalado."""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from src.data.geo_leakage_check import (
    confirmed_grid_ids_from_manifest,
    match_patch_in_reference,
    parse_grid_id,
)


@pytest.mark.parametrize('filename,expected', [
    ('I-3_2005_000000_1.png', ('I-3', 2005)),
    ('M-15_2017_004006.png', ('M-15', 2017)),
    ('AA-102_2010_009002_0.png', ('AA-102', 2010)),
    ('not_a_grid_id.png', None),
])
def test_parse_grid_id(filename, expected):
    assert parse_grid_id(filename) == expected


def test_match_patch_in_reference_finds_known_offset():
    rng = np.random.default_rng(42)
    reference = (rng.random((300, 300, 3)) * 255).astype(np.uint8)
    row, col = 120, 80
    patch = reference[row:row + 64, col:col + 64].copy()

    score, found_row, found_col = match_patch_in_reference(patch, reference)

    assert (found_row, found_col) == (row, col)
    assert score > 2.9  # patch idêntico embutido: correlação ~1.0 nas 3 bandas


def test_match_patch_in_reference_rejects_unrelated_patch():
    rng = np.random.default_rng(1)
    reference = (rng.random((300, 300, 3)) * 255).astype(np.uint8)
    unrelated_patch = (rng.random((64, 64, 3)) * 255).astype(np.uint8)

    score, _, _ = match_patch_in_reference(unrelated_patch, reference)

    assert score < 1.5  # ruído independente: correlação baixa nas 3 bandas


def test_match_patch_in_reference_raises_when_patch_larger_than_reference():
    reference = np.zeros((32, 32, 3), dtype=np.uint8)
    patch = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        match_patch_in_reference(patch, reference)


def test_confirmed_grid_ids_from_manifest(tmp_path):
    manifest_path = tmp_path / 'manifest.csv'
    manifest_path.write_text(
        'grid_id,year,sample,score,match_row,match_col,comparison_image,decision,notes\n'
        'I-3,2005,x,2.95,0,0,x.png,confirm,\n'
        'M-15,2010,x,1.10,0,0,x.png,reject,\n'
        'M-15,2017,x,1.05,0,0,x.png,,\n'
        'N-13,2017,x,2.80,0,0,x.png,confirm,\n',
        encoding='utf-8-sig',
    )

    assert confirmed_grid_ids_from_manifest(manifest_path) == ['I-3', 'N-13']
