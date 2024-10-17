from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
import os
import os.path as osp
from torch.utils.data import DataLoader
from src.data.augmentations import SimCLRTransform
from src.configuration import Config

class OCTDataset2D(Dataset):
    def __init__(self, data_dir, phase, transform):
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform
        self.image_list = None
        self.read_lists()

    def __getitem__(self, index):
        image_path = self.image_list[index]

        image = Image.open(image_path).convert('RGB')

        image_1 = self.transform(image)
        image_2 = self.transform(image)

        return image_1, image_2

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        self.image_list, _ = get_matching_files(self.data_dir, self.phase)
               

def get_matching_files(data_dir, phase):
    img_dir = osp.join(data_dir, phase, 'img')
    img_files = sorted(os.listdir(img_dir))
    image_list =[]

    for img_file in img_files:
        image_list.append(osp.join(img_dir, img_file))

    return image_list, None




def get_oct_dataset(config:Config, train_transform, val_transform):
    """
    Initialize and return OCT datasets and dataloaders for training and validation.
    
    Args:
    config (object): Configuration object containing dataset parameters.
    
    Returns:
    tuple: (train_dataset, val_dataset)
    """
    

    # Initialize datasets
    train_dataset = OCTDataset2D(
        data_dir=config.data.dataset_root,
        phase='train',
        transform=train_transform
    )

    val_dataset = OCTDataset2D(
        data_dir=config.data.dataset_root,
        phase='eval',
        transform=val_transform
    )

    

    print(f"Initialized OCT dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")

    return train_dataset, val_dataset