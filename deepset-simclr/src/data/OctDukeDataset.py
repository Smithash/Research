import os
import random
from typing import List, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class OCTDataset(Dataset):
    def __init__(self, root_dir: str, scan_folders: List[str], transform=None):
        self.root_dir = root_dir
        self.scan_folders = scan_folders
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self) -> List[Tuple[str, int]]:
        image_paths = []
        for folder in self.scan_folders:
            folder_path = os.path.join(self.root_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith('.jpg'):
                    try:
                        # Extract the index from the filename
                        index = int(file.split('-')[-1].split('.')[0])
                        image_paths.append((os.path.join(folder_path, file), index))
                    except ValueError:
                        print(f"Skipping file {file} as it doesn't match the expected format.")
        return sorted(image_paths, key=lambda x: (x[0], x[1]))  # Sort by folder and then by index

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, _ = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            aug1 = self.transform(image)
            aug2 = self.transform(image)
        else:
            aug1 = transforms.ToTensor()(image)
            aug2 = transforms.ToTensor()(image)

        return aug1, aug2, img_path

def get_oct_dataset(config, train_transform, val_transform):
    dataset_root = config.data.dataset_root
    all_scans = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    
    # Shuffle and split the scans for train and validation
    random.shuffle(all_scans)
    split_index = int(0.8 * len(all_scans))  # 80% for training, 20% for validation
    train_scans, val_scans = all_scans[:split_index], all_scans[split_index:]

    train_dataset = OCTDataset(dataset_root, train_scans, transform=train_transform)
    val_dataset = OCTDataset(dataset_root, val_scans, transform=val_transform)

    return train_dataset, val_dataset