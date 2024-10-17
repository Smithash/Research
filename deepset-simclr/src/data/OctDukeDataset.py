import os
import random
import numpy as np
import scipy.io
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AdaptedOCTDataset(Dataset):
    def __init__(self, scan_paths, image_paths, transform=None, config=None):
        self.scan_paths = scan_paths
        self.image_paths = image_paths
        self.transform = transform
        self.config = config
        self.valid_data = []

        self._process_all_files()

    def _process_all_files(self):
        for file_path in self.image_paths:
            mat = scipy.io.loadmat(file_path)

            images = mat['images']
            x, y, nimages = images.shape
            step = 4
            ini = int(y / step)
            fin = int(ini * (step - 1))

            for i in range(nimages):
                curr_im = images[:, ini:fin, i]
                image = curr_im.astype(np.float32)
                self.valid_data.append((image, file_path))

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        image, file_path = self.valid_data[idx]

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)

        return image1, image2, file_path

def get_oct_dataset(config, train_transform, val_transform):
    train_data_dir = config.data.dataset_root
    val_data_dir = config.data.dataset_root  # Assuming validation data is in the same directory

    train_scans = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.mat')]
    val_scans = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.mat')]

    # Shuffle and split the scans for train and validation
    random.shuffle(train_scans)
    split_index = int(0.8 * len(train_scans))  # 80% for training, 20% for validation
    train_scans, val_scans = train_scans[:split_index], train_scans[split_index:]

    # train_dataset = AdaptedOCTDataset(
    #     train_scans, train_scans, transform=train_transform, config=config
    # )

    # val_dataset = AdaptedOCTDataset(
    #     val_scans, val_scans, transform=val_transform, config=config
    # )
    
    if config.data.dataset_type == 'per_scan':
        logging.info('Initialising per-scan dataset')
        train_dataset = SimCLRNLSTPerScanAugDataset(
            dummy_train_paths, dummy_train_scans, train_transform, config
        )

        val_dataset = SimCLRNLSTPerScanAugDataset(
            dummy_val_paths, dummy_val_scans, val_transform, config
        )
    elif config.data.dataset_type == 'deepset':
        logging.info('Initialising Deep Set dataset')
        train_dataset = SimCLRNLSTDeepSetDataset(
            dummy_train_paths, dummy_train_scans, config
        )

        val_dataset = SimCLRNLSTDeepSetDataset(
            dummy_val_paths, dummy_val_scans, config
        )
    else:
        raise Exception(f'Dataset type not supported: {config.data.dataset_type}')

    logging.info(
        'Initialised dummy dataset: Train=%s, Val=%s',
        len(train_dataset), len(val_dataset)
    )

    return train_dataset, val_dataset