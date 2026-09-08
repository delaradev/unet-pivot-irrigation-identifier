import json
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class PredictionSaver(tf.keras.callbacks.Callback):
    def __init__(self, validation_ds, save_dir: str, max_samples: int = 4, threshold: float = 0.5):
        super().__init__()
        self.validation_ds = validation_ds
        self.save_dir = Path(save_dir)
        self.max_samples = max_samples
        self.threshold = threshold
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        epoch_dir = self.save_dir / f'epoch_{epoch+1:03d}'
        epoch_dir.mkdir(exist_ok=True)

        for i, (images, masks) in enumerate(self.validation_ds.take(1)):
            predictions = self.model.predict(images, verbose=0)

            for j in range(min(self.max_samples, len(images))):
                self._save_image(images[j], epoch_dir /
                                 f'sample_{j}_image.png')
                self._save_mask(masks[j], epoch_dir /
                                f'sample_{j}_true_mask.png')
                pred_mask = (predictions[j] >=
                             self.threshold).astype(np.float32)
                self._save_mask(pred_mask, epoch_dir /
                                f'sample_{j}_pred_mask.png')

    def _save_image(self, image, path):
        from PIL import Image
        if hasattr(image, 'numpy'):
            img = (image.numpy() * 255).astype(np.uint8)
        else:
            img = (image * 255).astype(np.uint8)
        Image.fromarray(img).save(path)

    def _save_mask(self, mask, path):
        from PIL import Image
        if hasattr(mask, 'numpy'):
            msk = mask.numpy()
        else:
            msk = mask
        if msk.ndim == 3 and msk.shape[-1] == 1:
            msk = msk.squeeze(axis=-1)
        elif msk.ndim == 3 and msk.shape[0] == 1:
            msk = msk.squeeze(axis=0)
        msk = (msk * 255).astype(np.uint8)
        Image.fromarray(msk).save(path)


class EpochVisualizationCallback(tf.keras.callbacks.Callback):
    """
    Salva a cada época previsões e métricas para amostras do dataset de validação.
    Preserva os nomes originais dos arquivos.
    """

    def __init__(self, validation_ds, output_dir: str, val_filenames: List[str],
                 num_samples=9, sample_strategy='fixed', random_seed=42):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.sample_strategy = sample_strategy
        self.random_seed = random_seed
        self.validation_ds = validation_ds
        self.val_filenames = val_filenames
        self.metrics_history = []

        # Converte o dataset para lista uma vez
        self._dataset_list = list(self.validation_ds.unbatch().batch(1))
        self._total = len(self._dataset_list)

        if self.sample_strategy == 'fixed':
            self._select_fixed_samples()
        elif self.sample_strategy == 'random_once':
            self._select_random_samples(epoch=0, force_fixed=True)
        else:
            self.sample_images = []
            self.sample_masks = []
            self.sample_names = []

    def _select_fixed_samples(self):
        self.sample_images = []
        self.sample_masks = []
        self.sample_names = []
        for idx in range(min(self.num_samples, self._total)):
            img, mask = self._dataset_list[idx]
            self.sample_images.append(img)
            self.sample_masks.append(mask)
            self.sample_names.append(self.val_filenames[idx])

    def _select_random_samples(self, epoch=None, force_fixed=False):
        if force_fixed or self.sample_strategy == 'random_once':
            rng = random.Random(self.random_seed)
            indices = rng.sample(range(self._total), min(
                self.num_samples, self._total))
        else:
            epoch_seed = self.random_seed + epoch
            rng = random.Random(epoch_seed)
            indices = rng.sample(range(self._total), min(
                self.num_samples, self._total))

        self.sample_images = []
        self.sample_masks = []
        self.sample_names = []
        for idx in indices:
            img, mask = self._dataset_list[idx]
            self.sample_images.append(img)
            self.sample_masks.append(mask)
            self.sample_names.append(self.val_filenames[idx])

    def on_epoch_end(self, epoch, logs=None):
        if self.sample_strategy == 'random_each_epoch':
            self._select_random_samples(epoch=epoch)

        epoch_dir = self.output_dir / f'epoch_{epoch+1:04d}'
        epoch_dir.mkdir(exist_ok=True)

        epoch_metrics = {'epoch': epoch+1, 'samples': []}

        for i, (img, mask) in enumerate(zip(self.sample_images, self.sample_masks)):
            pred = self.model.predict(img, verbose=0)[0, :, :, 0]
            pred_bin = (pred >= 0.5).astype(np.float32)
            mask_np = mask.numpy()[0, :, :, 0]

            intersection = np.sum((pred_bin == 1) & (mask_np == 1))
            union = np.sum((pred_bin == 1) | (mask_np == 1))
            iou = intersection / (union + 1e-8)
            dice = 2 * intersection / \
                (np.sum(pred_bin) + np.sum(mask_np) + 1e-8)
            precision = intersection / (np.sum(pred_bin) + 1e-8)
            recall = intersection / (np.sum(mask_np) + 1e-8)

            epoch_metrics['samples'].append({
                'name': self.sample_names[i],
                'iou': float(iou),
                'dice': float(dice),
                'precision': float(precision),
                'recall': float(recall)
            })

            # Visualização (grid 2x2)
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            axes[0, 0].imshow(img.numpy()[0])
            axes[0, 0].set_title('Image')
            axes[0, 0].axis('off')
            axes[0, 1].imshow(mask_np, cmap='gray')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            axes[1, 0].imshow(pred_bin, cmap='gray')
            axes[1, 0].set_title(f'Prediction (IoU={iou:.3f})')
            axes[1, 0].axis('off')
            overlay = np.zeros((*mask_np.shape, 3), dtype=np.float32)
            tp = (mask_np == 1) & (pred_bin == 1)
            fp = (mask_np == 0) & (pred_bin == 1)
            fn = (mask_np == 1) & (pred_bin == 0)
            overlay[tp, 1] = 1.0
            overlay[fp, 0] = 1.0
            overlay[fn, 2] = 1.0
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('Overlay (TP, FP, FN)')
            axes[1, 1].axis('off')
            plt.suptitle(f"{self.sample_names[i]} - Epoch {epoch+1}")
            plt.tight_layout()
            plt.savefig(
                epoch_dir / f"{self.sample_names[i]}_epoch_{epoch+1:04d}.png", dpi=100)
            plt.close()

        with open(epoch_dir / 'metrics.json', 'w') as f:
            json.dump(epoch_metrics, f, indent=2)

        self.metrics_history.append(epoch_metrics)
        with open(self.output_dir / 'all_epochs_metrics.json', 'w') as f:
            json.dump({'epochs': self.metrics_history}, f, indent=2)

        if (epoch + 1) % 5 == 0:
            self.plot_iou_trend(self.output_dir / 'iou_trend.png')

    def plot_iou_trend(self, save_path):
        if not self.metrics_history:
            return
        epochs = [m['epoch'] for m in self.metrics_history]
        num_samples = len(self.metrics_history[0]['samples'])
        plt.figure(figsize=(10, 6))
        for i in range(num_samples):
            ious = [m['samples'][i]['iou'] for m in self.metrics_history]
            label = self.metrics_history[0]['samples'][i]['name']
            plt.plot(epochs, ious, marker='o', label=label)
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('IoU Evolution per Sample')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
