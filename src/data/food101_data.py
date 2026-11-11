"""Loading the cleaned Food-101 index and turning it into PyTorch data loaders."""
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.paths import CLEAN_INDEX, FOOD101_DIR


class FoodDataset(Dataset):
    """Reads Food-101 images listed in a DataFrame with 'path' and 'target' columns.

    Args:
        df: One row per image; 'path' is the image file and 'target' its class index.
        transform: Applied to each PIL image before it is returned.
    """

    def __init__(self, df, transform):
        self.paths = df['path'].tolist()
        self.targets = df['target'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert('RGB')
        return self.transform(image), self.targets[i]


def load_splits(index_csv=CLEAN_INDEX, data_dir=FOOD101_DIR, num_classes=None):
    """Loads the cleaned index written by notebooks/01_eda.ipynb and splits it.

    Class indices follow the alphabetical order of the labels; the Core ML export
    relies on the same order.

    Args:
        index_csv: The cleaned index CSV, with image paths relative to `data_dir`.
        data_dir: Folder containing the Food-101 `images/` directory.
        num_classes: If given, keep only the first `num_classes` labels alphabetically
            (used for quick tests).

    Returns:
        (train_df, val_df, test_df, class_names). Each DataFrame has an absolute 'path'
        and an integer 'target' column.
    """
    df = pd.read_csv(index_csv)
    df['path'] = df['path'].map(lambda p: str(data_dir / p))

    class_names = sorted(df['label'].unique())
    if num_classes is not None:
        class_names = class_names[:num_classes]
        df = df[df['label'].isin(class_names)]
    df = df.assign(target=df['label'].map({c: i for i, c in enumerate(class_names)}))

    train_df, val_df, test_df = (df[df['split'] == s].reset_index(drop=True) for s in ['train', 'val', 'test'])
    return train_df, val_df, test_df, class_names


def make_loader(df, transform, batch_size, shuffle, num_workers=0):
    """Wraps a split in a DataLoader that reads images in `num_workers` background processes."""
    return DataLoader(FoodDataset(df, transform), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, persistent_workers=num_workers > 0)
