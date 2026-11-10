"""Dataset class used by notebooks/02_models.ipynb.
"""
from PIL import Image
from torch.utils.data import Dataset


class FoodDataset(Dataset):
    """Reads Food-101 images listed in a DataFrame with 'path' and 'target' columns."""

    def __init__(self, df, transform):
        self.paths = df['path'].tolist()
        self.targets = df['target'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert('RGB')
        return self.transform(image), self.targets[i]
