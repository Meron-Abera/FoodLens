"""Shared fixtures: a tiny fake Food-101 dataset, so tests don't need the real 5 GB download."""
import pandas as pd
import pytest
from PIL import Image

LABELS = ['pizza', 'apple_pie', 'ramen']  # deliberately not in alphabetical order


@pytest.fixture
def tiny_dataset(tmp_path):
    """Creates 3 classes x 4 images (2 train, 1 val, 1 test) and a matching index CSV.

    Returns:
        (data_dir, index_csv)
    """
    data_dir = tmp_path / 'food-101'
    rows = []
    for label in LABELS:
        (data_dir / 'images' / label).mkdir(parents=True)
        for i, split in enumerate(['train', 'train', 'val', 'test']):
            Image.new('RGB', (300, 200), color=(40 * i, 100, 150)).save(data_dir / 'images' / label / f'{i}.jpg')
            rows.append({'id': f'{label}/{i}', 'path': f'images/{label}/{i}.jpg', 'label': label,
                         'name': label.replace('_', ' ').capitalize(), 'split': split})
    index_csv = tmp_path / 'food101_clean.csv'
    pd.DataFrame(rows).to_csv(index_csv, index=False)
    return data_dir, index_csv
