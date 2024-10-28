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
            aug_1 = self.transform(image)
            aug_2 = self.transform(image)
            return aug_1, aug_2
        else:
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
    
    # Initialize the full dataset with no transforms
    full_dataset = OCTDataset(data_dir=config.data.dataset_root)
    
    # Calculate the split
    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)  # 20% for validation
    train_size = dataset_size - val_size
    
    # Create train/val splits
    train_indices = range(train_size)
    val_indices = range(train_size, dataset_size)
    # indices = list(range(dataset_size))
    # train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    
    # Create Subset datasets with  different transforms
    train_dataset = Subset(OCTDataset(config.data.dataset_root, transform= train_transform), train_indices)
    val_dataset = Subset(OCTDataset(config.data.dataset_root, transform= val_transform), val_indices)
    

    print(f"Initialized OCT dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")

    return train_dataset, val_dataset