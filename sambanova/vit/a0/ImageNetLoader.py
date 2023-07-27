from torchvision import transforms
from torchvision.datasets import ImageNet

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F



class ImageNetOneHot(ImageNet):
    def __init__(self, root, split='train', transform=None):
        super().__init__(root=root, split=split, transform=transform)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # one-hot encode to work w/ samba code
        target_tensor = torch.tensor(target)
        target_one_hot = F.one_hot(target_tensor, num_classes=1000)

        return img, target_one_hot


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

        dataset = ImageNet(root=root_dir, split=split, transform=transform)
        #dataset = ImageNetOneHot(root=root_dir, split=split, transform=transform)

        super().__init__(dataset, batch_size=batch_size, shuffle=(split=='train'), drop_last=True)
