import os
import random
from typing import List, Tuple
import numpy as np
import scipy.io
from torch.utils.data import Dataset, Subset
from PIL import Image

class OCTDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_training=False):
        self.data_dir = data_dir
        self.transform = transform
        
        self.images = []
        self.is_training = is_training
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
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            if self.is_training:
                aug_1 = self.transform(image)
                aug_2 = self.transform(image)
                return aug_1, aug_2
            else:
                return self.transform(image)
        return image


def get_oct_dataset(config, train_transform, val_transform):
    """
    Get train and validation datasets.
    
    Args:
        config: Configuration object with data paths
        train_transform: Transformations for training data
        val_transform: Transformations for validation data
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    train_dir = os.path.join(config.data.dataset_root, 'train_data')
    val_dir = os.path.join(config.data.dataset_root, 'val_data')
    
    train_dataset = OCTDataset(
        data_dir=train_dir,
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = OCTDataset(
        data_dir=val_dir,
        transform=val_transform,
        is_training=False
    )
    
    print(f"Datasets initialized:\n"
          f"Train: {len(train_dataset)} images\n"
          f"Val: {len(val_dataset)} images")
    
    return train_dataset, val_dataset