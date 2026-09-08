"""Testes para src/models/losses.py e src/training/metrics.py. Requer TensorFlow —
pula automaticamente se ausente."""
import pytest

tf = pytest.importorskip('tensorflow')

from src.models.losses import (
    build_loss,
    build_loss_custom_objects,
    dice_loss,
    tversky_loss,
)
from src.training.metrics import dice_score, iou_score, precision_score, recall_score

PERFECT_TRUE = tf.constant([[[[1.0], [0.0]], [[0.0], [1.0]]]])
PERFECT_PRED = tf.constant([[[[0.9], [0.4]], [[0.1], [0.8]]]])  # binariza igual a PERFECT_TRUE em threshold 0.5

NO_OVERLAP_TRUE = tf.constant([[[[1.0], [0.0]], [[0.0], [0.0]]]])
NO_OVERLAP_PRED = tf.constant([[[[0.1], [0.9]], [[0.9], [0.9]]]])  # binariza sem nenhum acerto


def test_iou_score_perfect_match():
    assert float(iou_score()(PERFECT_TRUE, PERFECT_PRED)) == pytest.approx(1.0, abs=1e-4)


def test_iou_score_no_overlap():
    assert float(iou_score()(NO_OVERLAP_TRUE, NO_OVERLAP_PRED)) == pytest.approx(0.0, abs=1e-3)


def test_dice_score_perfect_match():
    assert float(dice_score()(PERFECT_TRUE, PERFECT_PRED)) == pytest.approx(1.0, abs=1e-4)


def test_precision_recall_no_overlap():
    assert float(precision_score()(NO_OVERLAP_TRUE, NO_OVERLAP_PRED)) == pytest.approx(0.0, abs=1e-3)
    assert float(recall_score()(NO_OVERLAP_TRUE, NO_OVERLAP_PRED)) == pytest.approx(0.0, abs=1e-3)


def test_tversky_loss_zero_at_perfect_match():
    y = tf.ones((1, 4, 4, 1))
    assert float(tversky_loss()(y, y)) == pytest.approx(0.0, abs=1e-4)


def test_tversky_loss_near_one_at_full_miss():
    y_true = tf.ones((1, 4, 4, 1))
    y_pred = tf.zeros((1, 4, 4, 1))
    assert float(tversky_loss()(y_true, y_pred)) == pytest.approx(1.0, abs=1e-3)


def test_dice_loss_zero_at_perfect_match():
    y = tf.ones((1, 4, 4, 1))
    assert float(dice_loss()(y, y)) == pytest.approx(0.0, abs=1e-4)


def test_build_loss_dispatches_by_name():
    y = tf.ones((1, 4, 4, 1))
    assert float(build_loss({'name': 'tversky', 'alpha': 0.7, 'beta': 0.3})(y, y)) == pytest.approx(0.0, abs=1e-4)
    assert float(build_loss({'name': 'dice'})(y, y)) == pytest.approx(0.0, abs=1e-4)
    assert isinstance(build_loss({'name': 'bce'}), tf.keras.losses.Loss)


def test_build_loss_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_loss({'name': 'not_a_real_loss'})


def test_build_loss_custom_objects_has_expected_keys():
    custom_objects = build_loss_custom_objects({'name': 'tversky', 'alpha': 0.7, 'beta': 0.3})
    assert set(custom_objects) == {'tversky_loss', 'dice_loss', 'loss'}
