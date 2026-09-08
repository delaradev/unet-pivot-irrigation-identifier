"""Teste de integração para src/training/train.py::Trainer. Roda um ciclo completo
(prepare_data -> build_and_compile_model -> setup_callbacks -> train ->
post_training_analysis) sobre um dataset sintético minúsculo, só para pegar erros de
wiring entre os módulos — não valida qualidade de treino. Requer TensorFlow — pula
automaticamente se ausente."""
import numpy as np
import pytest
import yaml
from PIL import Image

tf = pytest.importorskip('tensorflow')

from src.training.train import Trainer

IMG_SIZE = 64


def _make_pair(img_dir, mask_dir, name, positive: bool):
    img = (np.random.rand(IMG_SIZE, IMG_SIZE, 3) * 255).astype(np.uint8)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    if positive:
        mask[10:40, 10:40] = 255
    Image.fromarray(img).save(img_dir / name)
    Image.fromarray(mask).save(mask_dir / name)


def _build_raw_dataset(root):
    train_img_dir = root / 'raw' / 'train_images'
    train_msk_dir = root / 'raw' / 'train_masks'
    valid_img_dir = root / 'raw' / 'valid_data' / 'valid_images'
    valid_msk_dir = root / 'raw' / 'valid_data' / 'valid_masks'
    for d in (train_img_dir, train_msk_dir, valid_img_dir, valid_msk_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        _make_pair(train_img_dir, train_msk_dir, f'pos_{i}.png', positive=True)
    for i in range(5):
        _make_pair(train_img_dir, train_msk_dir, f'neg_{i}.png', positive=False)
    for i in range(2):
        _make_pair(valid_img_dir, valid_msk_dir, f'valid_{i}.png', positive=(i == 0))

    return root / 'raw'


def _write_config(path, dataset_path, balanced_path):
    config = {
        'project': {'name': 'test_run', 'version': '0.0.0', 'seed': 42},
        'data': {
            'dataset_path': str(dataset_path),
            'balanced_path': str(balanced_path),
            'min_fg_ratio': 0.01,
            'desired_pos_ratio': 0.7,
            'internal_val_fraction': 0.25,
        },
        'model': {
            'img_size': IMG_SIZE,
            'unet_base_filters': 4,
            'input_channels': 3,
            'output_channels': 1,
        },
        'training': {
            'batch_size': 2,
            'epochs': 1,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'early_stopping_patience': 5,
            'reduce_lr_patience': 5,
            'loss': {'name': 'tversky', 'alpha': 0.7, 'beta': 0.3},
            'randommix': False,
        },
        'gpu': {'memory_growth': False, 'mixed_precision': False},
    }
    with open(path, 'w') as f:
        yaml.dump(config, f)


def test_trainer_runs_end_to_end_on_synthetic_dataset(tmp_path, monkeypatch):
    dataset_path = _build_raw_dataset(tmp_path)
    balanced_path = tmp_path / 'balanced'
    config_path = tmp_path / 'config.yaml'
    _write_config(config_path, dataset_path, balanced_path)

    monkeypatch.setenv('CPIC_OUTPUT_DIR', str(tmp_path / 'runs'))

    trainer = Trainer(str(config_path))
    trainer.run()

    assert (trainer.output_dir / 'best_model.keras').exists()
    assert (trainer.output_dir / 'last_model.keras').exists()
    assert (trainer.output_dir / 'training_log.csv').exists()
    assert (trainer.output_dir / 'history.json').exists()
    assert (trainer.output_dir / 'analysis' / 'ranking_top5_bottom5.json').exists()
