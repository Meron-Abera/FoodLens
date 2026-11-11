"""Tests for src/data/food101_data.py and src/data/transforms.py."""
import torch
from PIL import Image

from src.data.food101_data import FoodDataset, load_splits, make_loader
from src.data.transforms import IMAGE_SIZE, eval_transform, train_transform


def test_transforms_produce_normalized_224_tensors():
    image = Image.new('RGB', (512, 341), color=(200, 120, 40))
    for transform in (train_transform(), eval_transform()):
        x = transform(image)
        assert x.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert x.dtype == torch.float32


def test_eval_transform_is_deterministic():
    image = Image.new('RGB', (400, 300), color=(10, 200, 90))
    assert torch.equal(eval_transform()(image), eval_transform()(image))


def test_load_splits_uses_alphabetical_class_order(tiny_dataset):
    data_dir, index_csv = tiny_dataset
    train_df, val_df, test_df, class_names = load_splits(index_csv, data_dir)
    assert class_names == ['apple_pie', 'pizza', 'ramen']
    assert (len(train_df), len(val_df), len(test_df)) == (6, 3, 3)
    for df in (train_df, val_df, test_df):
        assert (df['target'] == df['label'].map(class_names.index)).all()


def test_load_splits_resolves_paths(tiny_dataset):
    data_dir, index_csv = tiny_dataset
    train_df, *_ = load_splits(index_csv, data_dir)
    assert all(str(data_dir) in p for p in train_df['path'])


def test_load_splits_can_limit_classes(tiny_dataset):
    data_dir, index_csv = tiny_dataset
    train_df, val_df, test_df, class_names = load_splits(index_csv, data_dir, num_classes=2)
    assert class_names == ['apple_pie', 'pizza']
    assert set(train_df['target']) == {0, 1}
    assert 'ramen' not in set(test_df['label'])


def test_dataset_and_loader_yield_batches(tiny_dataset):
    data_dir, index_csv = tiny_dataset
    train_df, *_ = load_splits(index_csv, data_dir)
    dataset = FoodDataset(train_df, eval_transform())
    x, y = dataset[0]
    assert x.shape == (3, IMAGE_SIZE, IMAGE_SIZE) and y == train_df['target'][0]

    images, targets = next(iter(make_loader(train_df, eval_transform(), batch_size=4, shuffle=False)))
    assert images.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert targets.tolist() == train_df['target'][:4].tolist()
