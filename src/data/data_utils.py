import logging
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_zenodo_dataset(zip_path: Optional[Path] = None, extract_to: Optional[Path] = None,
                           remove_zip: bool = False) -> Path:
    if zip_path is None:
        raw_dataset_dir = Path("data/raw/dataset")
        zip_files = list(raw_dataset_dir.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError("No zip file found in data/raw/dataset/")
        zip_path = zip_files[0]
        logger.info(f"Found zip file: {zip_path.name}")
    if extract_to is None:
        extract_to = Path("data/dataset")
    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting to: {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Successfully extracted {zip_path.name}")
        if remove_zip:
            zip_path.unlink()
            logger.info(f"Removed zip file: {zip_path}")
    except zipfile.BadZipFile:
        raise RuntimeError(f"File is not a valid zip: {zip_path}")
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}")
    return extract_to


def verify_dataset_structure(dataset_path: Path = Path("data/dataset")) -> bool:
    required_dirs = ['train_images', 'train_masks',
                     'valid_data/valid_images', 'valid_data/valid_masks']
    logger.info("Verifying dataset structure:")
    all_exist = True
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        exists = dir_path.exists()
        status = "[OK]" if exists else "[MISSING]"
        logger.info(f"  {status} {dir_name}/")
        if not exists:
            all_exist = False
    if all_exist:
        logger.info("Dataset structure is correct.")
        train_images = len(list((dataset_path / 'train_images').glob('*.png')))
        train_masks = len(list((dataset_path / 'train_masks').glob('*.png')))
        valid_images = len(list((dataset_path / 'valid_data/valid_images').glob('*.png')))
        valid_masks = len(list((dataset_path / 'valid_data/valid_masks').glob('*.png')))
        logger.info(
            f"Sample counts: train_images={train_images}, train_masks={train_masks}, "
            f"valid_images={valid_images}, valid_masks={valid_masks}")
    else:
        logger.warning(
            "Dataset structure is incomplete. Run extract_zenodo_dataset() first.")
    return all_exist


def get_dataset_info() -> dict:
    dataset_path = Path("data/dataset")
    balanced_path = Path("data/dataset_balanced")
    info = {
        'raw_exists': Path("data/raw/dataset").exists(),
        'dataset_exists': dataset_path.exists(),
        'balanced_exists': balanced_path.exists(),
    }
    if dataset_path.exists():
        info['dataset'] = {
            'train_images': len(list(dataset_path.glob('train_images/*.png'))),
            'train_masks': len(list(dataset_path.glob('train_masks/*.png'))),
            'valid_images': len(list(dataset_path.glob('valid_data/valid_images/*.png'))),
            'valid_masks': len(list(dataset_path.glob('valid_data/valid_masks/*.png'))),
        }
    if balanced_path.exists():
        info['balanced'] = {
            'train_images': len(list(balanced_path.glob('train_images/*.png'))),
            'train_masks': len(list(balanced_path.glob('train_masks/*.png'))),
            'valid_images': len(list(balanced_path.glob('valid_images/*.png'))),
            'valid_masks': len(list(balanced_path.glob('valid_masks/*.png'))),
        }
    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    import argparse
    parser = argparse.ArgumentParser(description="CPIC Dataset utilities")
    parser.add_argument('--extract', action='store_true',
                        help='Extract dataset from raw folder')
    parser.add_argument('--verify', action='store_true',
                        help='Verify dataset structure')
    parser.add_argument('--info', action='store_true',
                        help='Show dataset information')
    parser.add_argument('--remove_zip', action='store_true',
                        help='Remove zip after extraction')
    args = parser.parse_args()

    if args.extract:
        extract_zenodo_dataset(remove_zip=args.remove_zip)
    if args.verify:
        verify_dataset_structure()
    if args.info:
        info = get_dataset_info()
        logger.info("Dataset Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
    if not any([args.extract, args.verify, args.info]):
        logger.info("Usage: python data_utils.py --extract --verify --info")
