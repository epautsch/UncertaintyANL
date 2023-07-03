import os
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform

        self.image_files = [f for f in os.listdir(dir_path) if f.endswith('.npz')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(seof, idx):
        image_file = self.image_files[idx]
        data = np.load(os.path.join(self.dir_path, image_file))
        image = data['image']

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(data['label'], dtype=torch.int32)
        
        return image, label


def get_custom_dataset():
    data_dir = '/home/epautsch/experiments_input'
    #data_dir = os.path.join(os.getcwd(), 'custom_dataset_dir')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: torch.as_tensor(x, dtype=torch.float16)),
    ])
    return CustomDataset(data_dir, transform=transform)


def input_fn_train(batch_size=4, drop_last=False):
    train_dataset = get_custom_dataset(train=True)
    return DataLoader(train_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)


def input_fn_eval(batch_size=4, drop_last=False):
    eval_dataset = get_custom_dataset(train=False)
    return DataLoader(eval_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=False)


