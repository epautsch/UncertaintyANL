import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_imagenet_dataset(root_dir, train=True):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    return datasets.ImageFolder(os.path.join(root_dir, 'train' if train else 'val'), transform)


def input_fn_train(batch_size=4, drop_last=False):
    root_dir = '/srv/projects/UncertaintyDL/imagenet'
    train_dataset = get_imagenet_dataset(root_dir, train=True)
    return DataLoader(train_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)


def input_fn_eval(batch_size=4, drop_last=False):
    root_dir = '/srv/projects/UncertaintyDL/imagenet'
    eval_dataset = get_imagenet_dataset(root_dir, train=False)
    return DataLoader(eval_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=False)


