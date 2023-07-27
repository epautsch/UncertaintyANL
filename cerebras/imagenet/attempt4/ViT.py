import math

import torch
import torch.nn as nn
from functools import partial

from timm.models import create_model
from timm.models.registry import register_model

import numpy as np

from ViT_Custom import VisionTransformer, _cfg


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
            mode='bicubic',
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

    def forward(self, batch, patches=False, only_last=True):
        x, labels = batch
        list_out, patch_out = self.forward_features(x)
        if only_last:
            output = self.head(list_out[-1])
            loss = self.criterion(output, labels)
            return loss
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
