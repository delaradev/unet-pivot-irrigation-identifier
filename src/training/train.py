import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from src.data.dataset_balancer import (
    CPICDatasetBuilder,
    create_balanced_dataset,
    pairs_are_ready,
)
from src.data.randommix import generate_randommix_dataset
from src.models.losses import build_loss, build_loss_custom_objects
from src.models.unet import build_unet, get_model_summary
from src.training.callbacks import EpochVisualizationCallback, PredictionSaver
from src.training.metrics import dice_score, iou_score, precision_score, recall_score
from src.utils.gpu_utils import configure_gpu, get_gpu_info
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, config_path: str, resume_from: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.run_name = f"{self.config['project']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_output = os.environ.get('CPIC_OUTPUT_DIR', 'runs')
        self.output_dir = Path(base_output) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        gpu_config = self.config['gpu']
        configure_gpu(
            memory_growth=gpu_config['memory_growth'],
            memory_limit_mb=gpu_config.get('memory_limit_mb'),
            mixed_precision=gpu_config['mixed_precision']
        )

        logger.info(f"Run: {self.run_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"GPU: {get_gpu_info()}")

        self.resume_from = resume_from
        self.initial_epoch = 0

    def prepare_data(self):
        data_config = self.config['data']
        model_config = self.config['model']
        train_config = self.config['training']

        max_samples = data_config.get('max_train_samples', None)
        seed = self.config['project']['seed']

        balanced_path = Path(data_config['balanced_path'])
        if not pairs_are_ready(balanced_path, 'train_images', 'train_masks'):
            logger.info(
                f"Balanced dataset not found/incomplete at {balanced_path}, "
                f"building it from {data_config['dataset_path']}")
            create_balanced_dataset(
                src_root=data_config['dataset_path'],
                dst_root=str(balanced_path),
                min_fg_ratio=data_config['min_fg_ratio'],
                desired_pos_ratio=data_config['desired_pos_ratio'],
                seed=seed,
                excluded_grid_ids=data_config.get('excluded_grid_ids'),
            )

        randommix_enabled = train_config.get('randommix', False)
        randommix_source = None
        if randommix_enabled:
            randommix_path = Path(data_config.get(
                'randommix_path', f"{data_config['balanced_path']}_randommix"))
            if not pairs_are_ready(randommix_path, 'train_images', 'train_masks'):
                logger.info(f"Generating RandomMix dataset at {randommix_path}")
                generate_randommix_dataset(
                    original_train_path=str(balanced_path),
                    output_path=str(randommix_path),
                    min_fg_ratio=data_config['min_fg_ratio'],
                    prob=train_config.get('randommix_prob', 1.0),
                    seed=seed,
                )
            randommix_source = str(randommix_path)
            logger.info(
                "RandomMix ativo: treino usará apenas positivos de balanced_path + amostras de randommix_path")

        self.dataset_builder = CPICDatasetBuilder(
            dataset_path=str(balanced_path),
            image_size=model_config['img_size'],
            seed=seed,
            max_train_samples=max_samples,
            positive_only_from_dataset=randommix_enabled,
            min_fg_ratio=data_config['min_fg_ratio'],
            extra_train_source=randommix_source,
        )

        # 'valid' (valid_images/valid_masks) é o conjunto de teste final e permanece
        # isolado do treinamento; a validação usada por early stopping/checkpoint vem
        # de uma fração separada do próprio conjunto de treino (ver post_training_analysis
        # para a avaliação sobre 'valid').
        internal_val_fraction = data_config.get('internal_val_fraction', 0.1)
        train_pairs, internal_val_pairs = self.dataset_builder.load_pairs_train_val_split(
            val_fraction=internal_val_fraction)
        test_pairs = self.dataset_builder.load_pairs('valid')
        logger.info(
            f"Train pairs: {len(train_pairs)}, Val pairs (internos, {internal_val_fraction:.0%}): "
            f"{len(internal_val_pairs)}, Test pairs (reservado): {len(test_pairs)}")

        self.val_filenames = [Path(p[0]).stem for p in internal_val_pairs]

        self.train_ds = self.dataset_builder.build_dataset_from_pairs(
            train_pairs,
            batch_size=self.config['training']['batch_size'],
            training=True
        )
        self.val_ds = self.dataset_builder.build_dataset_from_pairs(
            internal_val_pairs,
            batch_size=self.config['training']['batch_size'],
            training=False
        )
        self.sample_ds = self.val_ds.take(1)

    def build_and_compile_model(self):
        model_config = self.config['model']
        self.model = build_unet(
            input_shape=(
                model_config['img_size'], model_config['img_size'], model_config['input_channels']),
            base_filters=model_config['unet_base_filters'],
            output_channels=model_config.get('output_channels', 1)
        )
        logger.info(get_model_summary(self.model))

        train_config = self.config['training']
        loss_fn = build_loss(train_config['loss'])

        metrics = [
            iou_score(),
            dice_score(),
            precision_score(),
            recall_score()
        ]

        if train_config['optimizer'] == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=train_config['learning_rate'])
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=train_config['learning_rate'])

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    def load_resume_model(self):
        if not self.resume_from or not os.path.exists(self.resume_from):
            raise FileNotFoundError(
                f"Resume path not found: {self.resume_from}")

        logger.info(f"Resuming training from {self.resume_from}")
        custom_objects = {
            **build_loss_custom_objects(self.config['training']['loss']),
            'iou_score': iou_score(),
            'dice_score': dice_score(),
            'precision_score': precision_score(),
            'recall_score': recall_score(),
        }
        self.model = tf.keras.models.load_model(
            self.resume_from, custom_objects=custom_objects)

        self.initial_epoch = self._detect_resume_epoch(Path(self.resume_from))

    def _detect_resume_epoch(self, resume_path: Path) -> int:
        """best_model.keras/last_model.keras não carregam a época no nome; a fonte é
        a última linha de training_log.csv no mesmo diretório. Mantém suporte ao
        padrão antigo 'epoch_NNN' por compatibilidade com checkpoints já existentes."""
        import re
        match = re.search(r'epoch_(\d+)', str(resume_path))
        if match:
            epoch = int(match.group(1))
            logger.info(f"Initial epoch {epoch} detectado pelo nome do arquivo")
            return epoch

        log_path = resume_path.parent / 'training_log.csv'
        if log_path.exists():
            with open(log_path, 'r') as f:
                lines = [line for line in f.read().splitlines() if line.strip()]
            if len(lines) > 1:
                last_epoch = int(lines[-1].split(',')[0])
                initial_epoch = last_epoch + 1  # CSVLogger grava a época em base 0
                logger.info(
                    f"Initial epoch {initial_epoch} detectado em {log_path}")
                return initial_epoch

        fallback = self.config['training'].get('resume_epoch', 0)
        logger.warning(
            f"Não foi possível detectar a época do checkpoint; usando resume_epoch={fallback} do config")
        return fallback

    def setup_callbacks(self):
        train_config = self.config['training']

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(self.output_dir / 'best_model.keras'),
                monitor='val_iou_score',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            # Checkpoint "last": sobrescreve o mesmo arquivo a cada época, sem acumular
            # espaço em disco. Permite retomar de onde parou independente do best_model.
            tf.keras.callbacks.ModelCheckpoint(
                str(self.output_dir / 'last_model.keras'),
                save_best_only=False,
                save_freq='epoch',
                verbose=0
            ),
            # Histórico incremental por época; history.json só é gravado ao final do fit().
            tf.keras.callbacks.CSVLogger(
                str(self.output_dir / 'training_log.csv'), append=True),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_iou_score',
                mode='max',
                patience=train_config['early_stopping_patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_iou_score',
                mode='max',
                factor=0.5,
                patience=train_config['reduce_lr_patience'],
                min_lr=1e-7
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.output_dir / 'tensorboard'),
            PredictionSaver(self.sample_ds, str(
                self.output_dir / 'predictions'), max_samples=4),
            EpochVisualizationCallback(
                validation_ds=self.val_ds,
                output_dir=str(self.output_dir / 'epoch_vis_fixed'),
                val_filenames=self.val_filenames,
                num_samples=5,
                sample_strategy='fixed',
                random_seed=self.config['project']['seed']
            ),
            EpochVisualizationCallback(
                validation_ds=self.val_ds,
                output_dir=str(self.output_dir / 'epoch_vis_random_each'),
                val_filenames=self.val_filenames,
                num_samples=5,
                sample_strategy='random_each_epoch',
                random_seed=self.config['project']['seed']
            )
        ]
        self.callbacks = callbacks

    def train(self):
        train_config = self.config['training']

        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        logger.info("Starting training")
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=train_config['epochs'],
            initial_epoch=self.initial_epoch,
            callbacks=self.callbacks,
            verbose=1
        )

        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history.history, f, indent=2)

        self.post_training_analysis(history.history)

        logger.info(f"Training completed. Model saved in {self.output_dir}")

    def post_training_analysis(self, history: dict):
        logger.info("Running post-training analysis...")
        analysis_dir = self.output_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)

        self.plot_metrics_history(history, analysis_dir)

        best_model_path = self.output_dir / 'best_model.keras'
        if best_model_path.exists():
            logger.info(f"Loading best model from {best_model_path}")
            custom_objects = {
                **build_loss_custom_objects(self.config['training']['loss']),
                'iou_score': iou_score(),
                'dice_score': dice_score(),
                'precision_score': precision_score(),
                'recall_score': recall_score(),
            }
            best_model = tf.keras.models.load_model(
                best_model_path, custom_objects=custom_objects)
        else:
            logger.warning("Best model not found, using current model")
            best_model = self.model

        img_metrics = self.evaluate_per_image(best_model, analysis_dir)
        self.save_ranking_and_visualizations(img_metrics, analysis_dir)

    def plot_metrics_history(self, history: dict, output_dir: Path):
        metrics = ['loss', 'iou_score', 'dice_score',
                   'precision_score', 'recall_score']
        epochs = len(history.get('loss', []))
        if epochs == 0:
            return

        step = max(5, epochs // 20)
        indices = list(range(0, epochs, step))
        if indices[-1] != epochs - 1:
            indices.append(epochs - 1)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break
            ax = axes[idx]
            train_vals = [history[metric][i] for i in indices]
            ax.plot(indices, train_vals, 'b-o', label='Train', markersize=4)
            val_key = f'val_{metric}'
            if val_key in history:
                val_vals = [history[val_key][i] for i in indices]
                ax.plot(indices, val_vals, 'r-s',
                        label='Validation', markersize=4)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_evolution.png', dpi=150)
        plt.close()
        logger.info(
            f"Metrics plot saved to {output_dir / 'metrics_evolution.png'}")

    def evaluate_per_image(self, model: tf.keras.Model, output_dir: Path) -> List[dict]:
        val_pairs = self.dataset_builder.load_pairs('valid')
        if not val_pairs:
            logger.warning(
                "No validation pairs found for per-image evaluation")
            return []

        img_size = self.config['model']['img_size']
        results = []

        for img_path, mask_path in val_pairs:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.expand_dims(img, axis=0)

            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.image.resize(mask, [img_size, img_size])
            mask = tf.cast(mask, tf.float32) / 255.0
            mask_np = mask.numpy().squeeze()

            pred = model.predict(img, verbose=0)[0, :, :, 0]
            pred_bin = (pred >= 0.5).astype(np.float32)

            intersection = np.sum((pred_bin == 1) & (mask_np == 1))
            union = np.sum((pred_bin == 1) | (mask_np == 1))
            iou = intersection / (union + 1e-8)
            dice = 2 * intersection / \
                (np.sum(pred_bin) + np.sum(mask_np) + 1e-8)
            precision = intersection / (np.sum(pred_bin) + 1e-8)
            recall = intersection / (np.sum(mask_np) + 1e-8)

            results.append({
                'img_name': Path(img_path).stem,
                'img_path': str(img_path),
                'mask_path': str(mask_path),
                'iou': float(iou),
                'dice': float(dice),
                'precision': float(precision),
                'recall': float(recall),
                'prediction': pred,
                'mask': mask_np,
                'image': img.numpy().squeeze()
            })
        return results

    def save_ranking_and_visualizations(self, img_metrics: List[dict], output_dir: Path):
        if not img_metrics:
            return

        sorted_by_iou = sorted(img_metrics, key=lambda x: x['iou'])
        worst_5 = sorted_by_iou[:5]
        best_5 = sorted_by_iou[-5:]

        ranking = {
            'best_5': [
                {k: v for k, v in m.items() if k not in [
                    'prediction', 'mask', 'image']}
                for m in best_5
            ],
            'worst_5': [
                {k: v for k, v in m.items() if k not in [
                    'prediction', 'mask', 'image']}
                for m in worst_5
            ]
        }
        with open(output_dir / 'ranking_top5_bottom5.json', 'w') as f:
            json.dump(ranking, f, indent=2)
        logger.info(
            f"Ranking saved to {output_dir / 'ranking_top5_bottom5.json'}")

        targets = [0.0, 0.25, 0.5, 0.75, 1.0]
        samples_for_vis = best_5 + worst_5
        for target in targets:
            closest = min(img_metrics, key=lambda x: abs(x['iou'] - target))
            if closest not in samples_for_vis:
                samples_for_vis.append(closest)

        seen = set()
        unique_samples = []
        for m in samples_for_vis:
            if m['img_name'] not in seen:
                seen.add(m['img_name'])
                unique_samples.append(m)

        vis_dir = output_dir / 'sample_visualizations'
        vis_dir.mkdir(exist_ok=True)
        for sample in unique_samples:
            self.visualize_overlay(sample, vis_dir)
        logger.info(f"Visualizations saved to {vis_dir}")

    def visualize_overlay(self, sample: dict, output_dir: Path):
        img = sample['image']
        mask = sample['mask']
        pred_bin = (sample['prediction'] >= 0.5).astype(np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title(f'Ground Truth (IoU: {sample["iou"]:.3f})')
        axes[0, 1].axis('off')
        axes[1, 0].imshow(pred_bin, cmap='gray')
        axes[1, 0].set_title('Prediction (Binary)')
        axes[1, 0].axis('off')
        overlay = np.zeros((*img.shape[:2], 3), dtype=np.float32)
        tp = (mask == 1) & (pred_bin == 1)
        fp = (mask == 0) & (pred_bin == 1)
        fn = (mask == 1) & (pred_bin == 0)
        overlay[tp, 1] = 1.0
        overlay[fp, 0] = 1.0
        overlay[fn, 2] = 1.0
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (Green=TP, Red=FP, Blue=FN)')
        axes[1, 1].axis('off')
        plt.suptitle(
            f"{sample['img_name']} | IoU={sample['iou']:.3f} | Dice={sample['dice']:.3f}")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{sample['img_name']}_iou_{sample['iou']:.3f}.png", dpi=100)
        plt.close()

    def run(self):
        self.prepare_data()
        if self.resume_from:
            self.load_resume_model()
        else:
            self.build_and_compile_model()
        self.setup_callbacks()
        self.train()
