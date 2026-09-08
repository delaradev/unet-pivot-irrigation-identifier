from typing import Optional

import tensorflow as tf

from src.utils.logging import get_logger

logger = get_logger(__name__)


def configure_gpu(
    memory_growth: bool = True,
    memory_limit_mb: Optional[int] = None,
    mixed_precision: bool = False
) -> bool:
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        logger.info("No GPU found. Using CPU.")
        return False

    if memory_growth and memory_limit_mb:
        logger.info("memory_growth and memory_limit_mb are mutually exclusive on the same "
                    "GPU; ignoring memory_limit_mb and keeping memory_growth enabled.")
        memory_limit_mb = None

    success = True
    for gpu in gpus:
        logger.info(f"GPU found: {gpu}")
        try:
            if memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("Memory growth enabled")
            elif memory_limit_mb:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit_mb)]
                )
                logger.info(f"Memory limit set to {memory_limit_mb} MB")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
            success = False

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("Mixed precision enabled (float16)")

    return success


def get_gpu_info() -> dict:
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
                '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info = result.stdout.strip().split(', ')
            return {'name': info[0], 'memory_total': info[1], 'memory_used': info[2] if len(info) > 2 else 'N/A'}
    except Exception:
        pass
    return {'name': 'Unknown', 'memory_total': 'N/A', 'memory_used': 'N/A'}
