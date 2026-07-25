"""
Dataset loading & preprocessing for the Chest X-Ray (Pneumonia) classification task.

Expected folder layout (this is exactly how the Kaggle
"Chest X-Ray Images (Pneumonia)" dataset is structured once unzipped):

data/chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD,
)


def get_transforms():
    """Returns (train_transform, eval_transform).

    Training uses light augmentation (flip / rotation / color jitter) which
    helps generalization on the relatively small pneumonia dataset.
    Validation/Test use only resize + normalize so evaluation is deterministic.
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # X-rays are gray-scale
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    return train_transform, eval_transform


def get_dataloaders(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR,
                     batch_size=BATCH_SIZE, num_workers=2):
    """Builds train/val/test DataLoaders using torchvision's ImageFolder.

    ImageFolder automatically assigns class index 0/1 based on alphabetical
    folder name order -> NORMAL = 0, PNEUMONIA = 1 (matches src/config.LABELS).
    """
    train_transform, eval_transform = get_transforms()

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=eval_transform)
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes
