"""Testes para src/data/dataset_balancer.py: checador de completude de dataset e
split treino/validação-interna (ver Trainer.prepare_data). Requer TensorFlow — pula
automaticamente se ausente."""
import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')

from PIL import Image

from src.data.dataset_balancer import CPICDatasetBuilder, pairs_are_ready


def _make_pairs(dataset_path, split_img_dir, split_mask_dir, n):
    img_dir = dataset_path / split_img_dir
    msk_dir = dataset_path / split_mask_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        name = f"tile_{i:05d}.png"
        Image.fromarray((np.random.rand(16, 16, 3) * 255).astype(np.uint8)).save(img_dir / name)
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(msk_dir / name)


def test_pairs_are_ready_false_when_missing(tmp_path):
    assert pairs_are_ready(tmp_path) is False


def test_pairs_are_ready_false_when_incomplete(tmp_path):
    _make_pairs(tmp_path, 'train_images', 'train_masks', 5)
    # Simula uma execução interrompida: apaga uma máscara sem apagar a imagem par.
    (tmp_path / 'train_masks' / 'tile_00000.png').unlink()
    assert pairs_are_ready(tmp_path) is False


def test_pairs_are_ready_true_when_complete(tmp_path):
    _make_pairs(tmp_path, 'train_images', 'train_masks', 5)
    assert pairs_are_ready(tmp_path) is True


def test_train_val_split_covers_all_pairs_without_overlap(tmp_path):
    _make_pairs(tmp_path, 'train_images', 'train_masks', 50)
    builder = CPICDatasetBuilder(dataset_path=str(tmp_path), seed=42)

    train_pairs, val_pairs = builder.load_pairs_train_val_split(val_fraction=0.1)

    assert len(train_pairs) + len(val_pairs) == 50
    assert set(train_pairs).isdisjoint(set(val_pairs))
    assert 3 <= len(val_pairs) <= 7  # ~10% de 50, com folga para arredondamento

    # O split (a mesma instância) não deve embaralhar/mudar de resultado ao repetir.
    train_pairs_2, val_pairs_2 = builder.load_pairs_train_val_split(val_fraction=0.1)
    assert train_pairs_2 == train_pairs
    assert val_pairs_2 == val_pairs
