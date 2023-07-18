import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.optim as optim
import sambaflow.samba.utils as utils

from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.common import common_app_driver
from sambaflow.samba.sambaloader import SambaLoader

import math
from transformers import ViTImageProcessor, ViTForImageClassification

import torch
import torch.nn as nn
from functools import partial

from timm.models import create_model
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.transform import resize
import numpy as np
from PIL import Image

import sys
import argparse
import random
from typing import Tuple

from ViT_Custom import VisionTransformer, _cfg
from ImageNetLoader import ImageNetLoader


class VanillaVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        # interpolate patch embeddings
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        class_pos_embed = self.pos_embed[:, 0]
        N = self.pos_embed.shape[1] - 1
        patch_pos_embed = self.pos_embed[:, 1:]
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bilinear',
        )
        if w0 != patch_pos_embed.shape[-2]:
            helper = torch.zeros(h0)[None, None, None, :].repeat(1, dim, w0 - patch_pos_embed.shape[-2], 1).to(x.device)
            patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-2)
        if h0 != patch_pos_embed.shape[-1]:
            helper = torch.zeros(w0)[None, None, :, None].repeat(1, dim, 1, h0 - patch_pos_embed.shape[-1]).to(x.device)
            patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-1)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        # interpolate patch embeddings finish

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        layer_wise_tokens = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            layer_wise_tokens.append(x)

        layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]

        return [x[:, 0] for x in layer_wise_tokens], [x for x in layer_wise_tokens]

    def forward(self, x, labels, patches=False, only_last=True):
        list_out, patch_out = self.forward_features(x)
        if only_last:
            output = self.head(list_out[-1])
            loss = self.criterion(output, labels)
            return loss, output
        x = [self.head(x) for x in list_out]
        if patches:
            return x, patch_out
        else:
            return x


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    # 'pretrained_cfg' in kwargs causing issues. Since it's type None, remove
    if 'pretrained_cfg' in kwargs:
        del kwargs['pretrained_cfg']
    if 'pretrained_cfg_overlay' in kwargs:
        del kwargs['pretrained_cfg_overlay']
    model = VanillaVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


def add_user_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-bs",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        metavar="N",
        help="number of classes in dataset (default: 1000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Download location for Imagenet data",
    )
    parser.add_argument(
        "--model-path", type=str, default="model", help="Save location for model"
    )


def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor]:
    dummy_image = (
        samba.randn(args.bs, 3, 224, 224, name="image", batch_dim=0),
        samba.randint(args.num_classes, (args.bs,), name="label", batch_dim=0),
    )

    return dummy_image


def prepare_dataloader(args: argparse.Namespace) -> Tuple[sambaflow.samba.sambaloader.SambaLoader, sambaflow.samba.sambaloader.SambaLoader]:        
    train_loader = ImageNetLoader(split='train', batch_size=args.bs) 
    val_loader = ImageNetLoader(split='val', batch_size=args.bs)

    sn_train_loader = SambaLoader(train_loader, ['image', 'label'])
    sn_val_loader = SambaLoader(val_loader, ['image', 'label'])

    return sn_train_loader, sn_val_loader


def train(args: argparse.Namespace, model: nn.Module) -> None:
    sn_train_loader, _ = prepare_dataloader(args)
    hyperparam_dict = {'lr': args.learning_rate}

    total_step = len(sn_train_loader)
    loss_list = []
    acc_list = []
    
    for epoch in range(args.num_epochs):
        for i, (images, labels) in enumerate(sn_train_loader):
            loss, outputs = samba.session.run(
                    input_tensors=(images, labels),
                    output_tensors=model.output_tensors,
                    hyperparam_dict=hyperparam_dict
            )

            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            loss_list.append(loss.tolist())

            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print(
                        'Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}, Acc: {:.2f}%'.format(
                            epoch + 1,
                            args.num_epochs,
                            i + 1,
                            total_step,
                            torch.mean(loss),
                            (correct / total) * 100,
                        )
                )


def main(argv):

    args = parse_app_args(argv=argv, common_parser_fn=add_user_args)

    model = deit_base_patch16_224(pretrained=True)

    samba.from_torch_model_(model)

    optimizer = samba.optim.AdamW(model.parameters(), lr=args.learning_rate)

    inputs = get_inputs(args)

    if args.command == 'run':
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        train(args, model)
    else:
        common_app_driver(args=args,
                        model=model,
                        inputs=inputs,
                        optim=optimizer,
                        name=model.__class__.__name__,
                        init_output_grads=not args.inference,
                        app_dir=utils.get_file_dir(__file__))


if __name__ == '__main__':
    main(sys.argv[1:])
