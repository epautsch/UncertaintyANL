import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.transform import resize


def imagenet_transforms(image):
    image = resize(image, (224, 224), mode='reflect')

    image = torch.from_numpy(image).float()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)
    return image


def get_imagenet_dataset(split):
    dataset = load_dataset('imagenet-1k', split=split, cache_dir='/srv/projects/UncertaintyDL/datasets/huggingface')
    dataset.set_format(type='torch', columns=['image', 'label'])
    dataset = dataset.map(lambda x: {'image': imagenet_transforms(x['image']), 'label': x['label']})
    return dataset


def input_fn_train(batch_size=4, drop_last=False):
    train_dataset = get_imagenet_dataset('train')
    return DataLoader(train_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)


def input_fn_eval(batch_size=4, drop_last=False):
    eval_dataset = get_imagenet_dataset('validation')
    return DataLoader(eval_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=False)
