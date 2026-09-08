"""Testes para src/models/unet.py. Requer TensorFlow — pula automaticamente se
ausente."""
import pytest

tf = pytest.importorskip('tensorflow')

import numpy as np

from src.models.unet import build_unet


def test_output_shape_matches_input_resolution():
    model = build_unet(input_shape=(64, 64, 3), base_filters=4, output_channels=1)
    x = np.random.rand(2, 64, 64, 3).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 64, 64, 1)


def test_output_is_sigmoid_probability():
    model = build_unet(input_shape=(64, 64, 3), base_filters=4, output_channels=1)
    x = np.random.rand(1, 64, 64, 3).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


def test_rejects_input_not_divisible_by_16():
    # 4 poolings 2x2 exigem dimensões múltiplas de 16 para o skip-connection
    # concatenar com o mesmo shape do encoder correspondente; o mismatch já
    # aparece na construção do grafo (Concatenate), não só no predict.
    with pytest.raises(Exception):
        build_unet(input_shape=(50, 50, 3), base_filters=4)
