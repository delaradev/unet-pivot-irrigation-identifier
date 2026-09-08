"""Testes para src/utils/gpu_utils.py. Requer TensorFlow — pula automaticamente se
ausente."""
import pytest

tf = pytest.importorskip('tensorflow')

from src.utils.gpu_utils import configure_gpu


def test_no_gpu_returns_false(monkeypatch):
    monkeypatch.setattr(tf.config, 'list_physical_devices', lambda kind: [])
    assert configure_gpu(memory_growth=True, mixed_precision=True) is False


def test_conflicting_memory_options_still_enable_mixed_precision(monkeypatch):
    """memory_growth e memory_limit_mb são mutuamente exclusivos por GPU no TensorFlow;
    mixed_precision deve ser aplicado independente do resultado da configuração de memória."""
    fake_gpu = object()
    monkeypatch.setattr(tf.config, 'list_physical_devices', lambda kind: [fake_gpu])
    monkeypatch.setattr(tf.config.experimental, 'set_memory_growth', lambda gpu, enabled: None)

    policy_calls = []
    monkeypatch.setattr(tf.keras.mixed_precision, 'set_global_policy', policy_calls.append)

    configure_gpu(memory_growth=True, memory_limit_mb=40960, mixed_precision=True)

    assert policy_calls == ['mixed_float16']
