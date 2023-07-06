import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class PatchEmbedding(nn.Module):
    def __init__(self,
            in_channels:int=model_params['in_channels'],
            patch_size:int=model_params['patch_size'],
            emb_size:int=model_params['emb_size']):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            Transpose(1, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x


class ViT(nn.Module):
    def __init__(self,
            in_channels:int=model_params['in_channels'],
            patch_size:int=model_params['patch_size'],
            emb_size:int=model_params['emb_size'],
            img_size:int=model_params['img_size'],
            depth:int=model_params['depth'],
            num_heads:int=model_params['num_heads'],
            num_classes:int=model_params['num_classes']):
        super().__init__()
        self.embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.positional_emb = nn.Parameter(torch.zeros(1, img_size // patch_size * img_size // patch_size + 1, emb_size))
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(emb_size, num_heads),
            depth
        )
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        B, N, _ = emb.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        emb = torch.cat([cls_tokens, emb], dim=1)
        emb = emb + self.positional_emb
        x = self.transformer(emb)
        x = x[:, 0]
        return self.classifier(x)
