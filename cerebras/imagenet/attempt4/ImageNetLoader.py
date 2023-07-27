import sys
sys.path.insert(0, '/home/epautsch/R_1.8.0/modelzoo/')
from modelzoo.transformers.pytorch.input_utils import num_tasks, task_id

from torchvision import transforms
from torchvision.datasets import ImageNet

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F


class ImageNetDataset(ImageNet):
    def __init__(self, root, split='train', transform=None):
        super().__init__(root=root, split=split, transform=transform)

        # sharding for cerebras
        num_samples = len(self.samples)
        self.samples = self.samples[task_id()::num_tasks()]

    def __get__item(self, index):
        img, target = super().__getitem__(index)
        target = torch.tensor(target, dtype=torch.int32)
        return img, target


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.int32)


class ImageNetLoader(DataLoader):
    def __init__(self, root_dir='/srv/projects/UncertaintyDL/datasets/imagenet/', split='train', transform=None, batch_size=1):
        if transform is None:
            # default
            transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        dataset = ImageNetDataset(root=root_dir, split=split, transform=transform)

        super().__init__(dataset, batch_size=batch_size, shuffle=(split=='train'), drop_last=True, collate_fn=custom_collate_fn)

    
