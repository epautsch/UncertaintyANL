import torch
from datasets import load_dataset, Value, load_from_disk
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.transform import resize
import numpy as np


def imagenet_transforms(image):
    image = resize(image, (224, 224), mode='reflect')

    image = torch.from_numpy(image).float()

    # check num of dims and add color channel if needed
    if image.dim() == 2:
        image = image.unsqueeze(0).repeat(3, 1, 1)
    else:
    # permute axes to (c, h, w) format
        image = image.permute(2, 0, 1)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    image = normalize(image)
    return image


def get_imagenet_dataset(split, slice_percentage=1.0):
    dataset = load_dataset('imagenet-1k', split=split, cache_dir='/srv/projects/UncertaintyDL/datasets/huggingface')
    #dataset = load_from_disk('/home/epautsch/huggingface_datasets/imagenet-1k')
    dataset.set_format(type='torch', columns=['image', 'label'])
    
    # get random slice of dataset
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)[:int(num_samples * slice_percentage)]
    dataset = dataset.select(indices)
    
    new_features = dataset.features.copy()
    new_features['label'] = Value('int32')
    dataset = dataset.cast(new_features)

    dataset = dataset.map(lambda x: {'image': imagenet_transforms(x['image']), 'label': x['label']})
    
    return dataset


def input_fn_train(batch_size=4, drop_last=False, slice_percentage=1.0):#slice_percentage=input_params['split_percentage']):
    train_dataset = get_imagenet_dataset('train', slice_percentage=slice_percentage)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)
    return data_loader


def input_fn_eval(batch_size=4, drop_last=False, slice_percentage=1.0):#slice_percentage=input_params['split_percentage']):
    eval_dataset = get_imagenet_dataset('validation', slice_percentage=slice_percentage)
    data_loader = DataLoader(eval_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=False)
    return data_loader
