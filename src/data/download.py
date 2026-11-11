"""Download the Food-101 dataset into data/food-101.

Food-101 (Bossard et al., 2014) is about 5 GB and is not stored in the repository.
This script downloads the official archive from ETH Zurich and unpacks it. It does
nothing if the dataset is already present.

Usage (from the repository root):
    python -m src.data.download
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

from src.paths import DATA_DIR

FOOD101_URL = 'https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'


def is_downloaded(data_dir):
    """Returns True if data_dir already contains the unpacked dataset."""
    food101 = data_dir / 'food-101'
    return (food101 / 'images').is_dir() and (food101 / 'meta' / 'train.txt').is_file()


def _report_progress(blocks, block_size, total_size):
    """Prints download progress on a single line."""
    if total_size > 0:
        done = min(blocks * block_size, total_size)
        print(f'\rDownloading: {done / 1e9:.2f} / {total_size / 1e9:.2f} GB', end='', flush=True)


def download(data_dir=DATA_DIR, url=FOOD101_URL):
    """Downloads and unpacks Food-101 so the images end up in data_dir/food-101/images."""
    data_dir = Path(data_dir)
    if is_downloaded(data_dir):
        print(f'Food-101 is already in {data_dir / "food-101"}; nothing to do.')
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / 'food-101.tar.gz'
    try:
        urllib.request.urlretrieve(url, archive, reporthook=_report_progress)
        print('\nUnpacking (this takes a few minutes)...')
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir, filter='data')
    finally:
        archive.unlink(missing_ok=True)

    if not is_downloaded(data_dir):
        raise RuntimeError(f'Unpacked archive, but {data_dir / "food-101"} does not look like Food-101.')
    print(f'Done: Food-101 is in {data_dir / "food-101"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR, help='folder to put food-101/ in')
    parser.add_argument('--url', default=FOOD101_URL, help='archive to download')
    args = parser.parse_args()
    download(args.data_dir, args.url)
