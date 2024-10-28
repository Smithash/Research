import random
import numpy as np
import os
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class SimCLROCTDataset(Dataset):
    def __init__(self, data_dir, num_files=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.valid_data = []
        self._process_all_files()

    def _process_all_files(self):
        for file_name in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
            mat = scipy.io.loadmat(file_path)

            images = mat['images']
            layers = mat['layerMaps']

            x, y, nimages = images.shape
            step = 4
            ini = int(y / step)
            fin = int(ini * (step - 1))
            thr = fin - ini - 1

            for i in range(nimages):
                curr_im = images[:, ini:fin, i]
                num_layers = layers.shape[2]
                if num_layers < 3:
                    continue

                curr_l1_0 = layers[i, ini:fin, 0]
                curr_l1_1 = layers[i, ini:fin, 1]
                curr_l1_2 = layers[i, ini:fin, 2]

                cn0 = np.count_nonzero(~np.isnan(curr_l1_0))
                cn1 = np.count_nonzero(~np.isnan(curr_l1_1))
                cn2 = np.count_nonzero(~np.isnan(curr_l1_2))

                flag = (cn0 > thr) and (cn1 > thr) and (cn2 > thr)

                if flag:
                    # Use original image without resizing
                    image = curr_im.astype(np.float32)
                    self.valid_data.append(image)

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        image = self.valid_data[idx]

        # Apply two different augmentations to the same image for SimCLR
        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)
        else:
            view1, view2 = image, image

        return view1, view2


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
        transform=train_transform
    )
    
    val_dataset = OCTDataset(
        data_dir=val_dir,
        transform=val_transform,
    )
    
    print(f"Datasets initialized:\n"
          f"Train: {len(train_dataset)} images\n"
          f"Val: {len(val_dataset)} images")
    
    return train_dataset, val_dataset