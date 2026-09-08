from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image

from src.data.dataset_balancer import CPICDatasetBuilder
from src.models.losses import build_loss_custom_objects
from src.training.metrics import dice_score, iou_score, precision_score, recall_score
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CPICPredictor:
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.threshold = 0.5

        self.config = None
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        loss_config = (self.config or {}).get(
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
        self.img_size = self.model.input_shape[1]

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = tf.io.read_file(str(image_path))
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [self.img_size, self.img_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image.numpy()

    def predict_file(self, input_path: Path, output_path: Optional[str] = None) -> Path:
        input_path = Path(input_path)
        image = self._load_image(input_path)
        pred = self.model.predict(image[np.newaxis, ...], verbose=0)[0, :, :, 0]
        pred_bin = (pred >= self.threshold).astype(np.uint8) * 255

        if output_path is None:
            output_path = input_path.with_name(f"{input_path.stem}_pred.png")
        else:
            output_path = Path(output_path)
            if output_path.is_dir() or output_path.suffix == '':
                output_path = output_path / f"{input_path.stem}_pred.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(pred_bin).save(output_path)
        logger.info(f"Prediction saved: {output_path}")
        return output_path

    def predict_directory(self, input_dir: Path, output_dir: Optional[str] = None) -> Path:
        input_dir = Path(input_dir)
        tiles = sorted(input_dir.glob('*.png'))
        if not tiles:
            raise ValueError(f"No PNG tiles found in {input_dir}")

        output_dir = Path(output_dir) if output_dir else input_dir.parent / \
            f"{input_dir.name}_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, tile_path in enumerate(tiles, start=1):
            self.predict_file(tile_path, output_dir / tile_path.name)
            if i % 100 == 0:
                logger.info(f"Predicted {i}/{len(tiles)} tiles")

        logger.info(f"Prediction completed for {len(tiles)} tiles. Output: {output_dir}")
        return output_dir

    def validate(self, split: str = 'valid') -> Dict:
        if self.config is None:
            raise ValueError(
                "A config file is required for validation. Pass --config.")

        data_config = self.config['data']

        # image_size vem do modelo .keras carregado, não de config['model'].
        builder = CPICDatasetBuilder(
            dataset_path=data_config['balanced_path'],
            image_size=self.img_size,
            seed=self.config['project']['seed'],
        )
        pairs = builder.load_pairs(split)
        if not pairs:
            raise ValueError(f"No image-mask pairs found for split '{split}'")

        tp = fp = fn = tn = 0
        for img_path, mask_path in pairs:
            image = self._load_image(Path(img_path))

            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.image.resize(mask, [self.img_size, self.img_size])
            mask = (tf.cast(mask, tf.float32) / 255.0).numpy()[:, :, 0]
            mask_bin = (mask >= 0.5).astype(np.uint8)

            pred = self.model.predict(image[np.newaxis, ...], verbose=0)[0, :, :, 0]
            pred_bin = (pred >= self.threshold).astype(np.uint8)

            tp += int(np.sum((pred_bin == 1) & (mask_bin == 1)))
            fp += int(np.sum((pred_bin == 1) & (mask_bin == 0)))
            fn += int(np.sum((pred_bin == 0) & (mask_bin == 1)))
            tn += int(np.sum((pred_bin == 0) & (mask_bin == 0)))

        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)

        return {
            'iou': iou,
            'dice': f1,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        }
