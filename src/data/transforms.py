"""Image preprocessing shared by training, evaluation and the Core ML export."""
from torchvision import transforms

IMAGE_SIZE = 224   # input resolution the ImageNet-pretrained backbone expects
RESIZE_SIZE = 256  # shorter side is resized to this before the center crop
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform(image_size=IMAGE_SIZE):
    """Returns the training transform: random crop, flip and color jitter, then normalization.

    The augmentation shows the model a slightly different version of each photo every epoch,
    which reduces overfitting and makes it less sensitive to the over-saturated photos found in the EDA.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size=IMAGE_SIZE):
    """Returns the evaluation transform: resize the shorter side, center-crop, normalize."""
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
