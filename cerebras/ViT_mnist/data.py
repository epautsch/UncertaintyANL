import torch
from torchvision import datasets, transforms

def get_train_dataloader(params):
    input_params = params["train_input"]

    batch_size = input_params.get("batch_size")
    dtype = torch.float16 if input_params["to_float16"] else torch.float32
    shuffle = input_params["shuffle"]

    train_dataset = datasets.MNIST(
        input_params["data_dir"],
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(
                    lambda x: torch.as_tensor(x, dtype=dtype)
                ),
            ]
        ),
        target_transform=transforms.Lambda(
            lambda x: torch.as_tensor(x, dtype=torch.int32)
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=input_params["drop_last_batch"],
        shuffle=shuffle,
        num_workers=input_params.get("num_workers", 0),
    )
    return train_loader

def get_eval_dataloader(params):
    input_params = params["eval_input"]

    batch_size = input_params.get("batch_size")
    dtype = torch.float16 if input_params["to_float16"] else torch.float32

    eval_dataset = datasets.MNIST(
        input_params["data_dir"],
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(
                    lambda x: torch.as_tensor(x, dtype=dtype)
                ),
            ]
        ),
        target_transform=transforms.Lambda(
            lambda x: torch.as_tensor(x, dtype=torch.int32)
        ),
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        drop_last=input_params["drop_last_batch"],
        shuffle=False,
        num_workers=input_params.get("num_workers", 0),
    )
    return eval_loader
