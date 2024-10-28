import os
import random
from typing import List, Tuple
import numpy as np
import scipy.io
from torch.utils.data import Dataset, Subset
from PIL import Image

class OCTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.split_type = split_type
        self.images = []
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        self._process_all_files()

    def _process_all_files(self):
        for file_name in self.files:
            file_path = os.path.join(self.data_dir, file_name)
            mat = scipy.io.loadmat(file_path)

            images = mat['images']

            x, y, nimages = images.shape
            step = 4
            ini = int(y / step)
            fin = int(ini * (step - 1))

            for i in range(nimages):
                curr_im = images[:, ini:fin, i]
                
                # Convert numpy array to PIL Image
                image = Image.fromarray(curr_im.astype(np.uint8)).convert('RGB')
                
                self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            if self.split_type == 'train':
                # For training, return two augmented views
                aug_1 = self.transform(image)
                aug_2 = self.transform(image)
                return aug_1, aug_2
            else:
                # For validation, return single transformed image
                return self.transform(image)
        return image


def get_oct_dataset(config, train_transform, val_transform):
    """
    Initialize and return OCT datasets for training and validation.
    
    Args:
    config (object): Configuration object containing dataset parameters.
    train_transform (callable): Transformations to apply to training data.
    val_transform (callable): Transformations to apply to validation data.
    
    Returns:
    tuple: (train_dataset, val_dataset)
    """
    
    full_dataset = OCTDataset(data_dir=config.data.dataset_root)
    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size

    # Create train dataset with train transform
    train_dataset = OCTDataset(
        data_dir=config.data.dataset_root,
        transform=train_transform,
        split_type='train'
    )
    train_indices = range(train_size)
    train_dataset = Subset(train_dataset, train_indices)

    # Create val dataset with val transform
    val_dataset = OCTDataset(
        data_dir=config.data.dataset_root,
        transform=val_transform,
        split_type='val'
    )
    val_indices = range(train_size, dataset_size)
    val_dataset = Subset(val_dataset, val_indices)

    print(f"Initialized OCT dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    train_sample = train_dataset[0]
    print(f"Training sample is tuple: {isinstance(train_sample, tuple)}")  # Should be True
    print(f"Training sample has 2 augmentations: {len(train_sample) == 2}")  # Should be True

    # Check validation sample
    val_sample = val_dataset[0]
    print(f"Validation sample is single tensor: {not isinstance(val_sample, tuple)}")  # Sho
    return train_dataset, val_dataset