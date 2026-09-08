#!/usr/bin/env python3
import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
        'cpic_pipeline.log', mode='a')]
)
logger = logging.getLogger(__name__)


def setup_paths() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def get_trainer(config_path: str):
    try:
        from src.training.train import Trainer
        return Trainer(config_path)
    except ImportError as e:
        logger.error(f"Failed to import Trainer: {e}")
        sys.exit(1)


def get_predictor(model_path: str, config_path: Optional[str] = None):
    try:
        from src.inference.predict import CPICPredictor
        return CPICPredictor(model_path, config_path)
    except ImportError as e:
        logger.error(f"Failed to import CPICPredictor: {e}")
        sys.exit(1)


def get_exporter(model_path: str, output_path: str, config_path: Optional[str] = None):
    try:
        from src.export.model_exporter import CPICExporter
        return CPICExporter(model_path, output_path, config_path)
    except ImportError as e:
        logger.error(f"Failed to import CPICExporter: {e}")
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='CPIC Brazil - Center Pivot Irrigation Cropland Mapping')
    parser.add_argument(
        '--mode', type=str, choices=['train', 'predict', 'export', 'validate'], default='train')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--test_split', type=str, default='valid', choices=['valid'],
                        help="Dataset split to evaluate. 'valid' maps to the "
                             "valid_images/valid_masks folder, which is the paper's "
                             "held-out TEST set (never used during training)")
    parser.add_argument('--device', type=str,
                        choices=['cpu', 'gpu', 'auto'], default='auto')
    return parser.parse_args()


def run_train(args: argparse.Namespace) -> None:
    logger.info(f"Training mode. Config: {args.config}")
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    trainer = get_trainer(args.config)
    if args.epochs:
        trainer.config['training']['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['training']['batch_size'] = args.batch_size
    if args.resume:
        trainer.resume_from = args.resume
    if args.device == 'cpu':
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        trainer.run()
        logger.info("Training completed")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=args.verbose)
        sys.exit(1)


def run_predict(args: argparse.Namespace) -> None:
    if not args.model or not args.input:
        logger.error("--model and --input are required for prediction mode")
        sys.exit(1)
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    predictor = get_predictor(str(model_path), args.config)
    predictor.threshold = args.threshold

    try:
        if input_path.is_file():
            result = predictor.predict_file(input_path, args.output)
        else:
            result = predictor.predict_directory(input_path, args.output)
        logger.info(f"Prediction completed. Output: {result}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=args.verbose)
        sys.exit(1)


def run_export(args: argparse.Namespace) -> None:
    if not args.model:
        logger.error("--model is required for export mode")
        sys.exit(1)
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    output_path = args.output or str(model_path.parent / 'exported')
    exporter = get_exporter(str(model_path), output_path, args.config)
    try:
        exported_paths = exporter.export()
        logger.info("Model exported successfully")
        for path in exported_paths:
            logger.info(f"  {path}")
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=args.verbose)
        sys.exit(1)


def run_validate(args: argparse.Namespace) -> None:
    if not args.model:
        logger.error("--model is required for validation mode")
        sys.exit(1)
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    predictor = get_predictor(str(model_path), args.config)
    try:
        results = predictor.validate(split=args.test_split)
        logger.info("Validation Results:")
        for key in ['iou', 'dice', 'precision', 'recall', 'f1']:
            logger.info(f"  {key}: {results.get(key, 0):.4f}")
        if 'confusion_matrix' in results:
            logger.info(f"  Confusion Matrix: {results['confusion_matrix']}")
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=args.verbose)
        sys.exit(1)


def main() -> None:
    setup_paths()
    args = parse_arguments()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Mode: {args.mode.upper()}")
    mode_handlers = {
        'train': run_train,
        'predict': run_predict,
        'export': run_export,
        'validate': run_validate
    }
    handler = mode_handlers.get(args.mode)
    if handler:
        handler(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
