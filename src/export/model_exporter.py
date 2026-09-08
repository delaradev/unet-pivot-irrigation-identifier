from pathlib import Path
from typing import List, Optional

import tensorflow as tf
import yaml

from src.models.losses import build_loss_custom_objects
from src.training.metrics import dice_score, iou_score, precision_score, recall_score
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CPICExporter:
    def __init__(self, model_path: str, output_path: str, config_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)

        config = None
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        loss_config = (config or {}).get(
            'training', {}).get('loss', {'name': 'tversky', 'alpha': 0.7, 'beta': 0.3})

        custom_objects = {
            **build_loss_custom_objects(loss_config),
            'iou_score': iou_score(),
            'dice_score': dice_score(),
            'precision_score': precision_score(),
            'recall_score': recall_score(),
        }

        logger.info(f"Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(
            str(self.model_path), custom_objects=custom_objects)

    def export(self) -> List[str]:
        self.output_path.mkdir(parents=True, exist_ok=True)
        exported_paths: List[str] = []

        saved_model_dir = self.output_path / 'saved_model'
        self.model.export(str(saved_model_dir))
        exported_paths.append(str(saved_model_dir))
        logger.info(f"SavedModel exported to {saved_model_dir}")

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        tflite_model = converter.convert()
        tflite_path = self.output_path / 'model.tflite'
        tflite_path.write_bytes(tflite_model)
        exported_paths.append(str(tflite_path))
        logger.info(f"TFLite model exported to {tflite_path}")

        return exported_paths
