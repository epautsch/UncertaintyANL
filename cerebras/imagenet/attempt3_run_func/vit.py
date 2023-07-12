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

#class PrintShapeAndType(nn.Module):
 #   def forward(self, x):
  #      print(f'Shape of x: {x.shape}')
   #     print(f'Type of x: {type(x)}')
    #    return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
     #       PrintShapeAndType(),
            nn.Flatten(2),
            Transpose(1, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f'Shape of x b4 projection: {x.shape}')
        print(f'Type of x b4 projection: {type(x)}')

        x = self.projection(x)

      #  print(f'Shape of x after projection: {x.shape}')
       # print(f'Type of x after projection: {type(x)}')

        return x


class ViT(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.in_channels = model_params['in_channels']
        self.patch_size = model_params['patch_size']
        self.emb_size = model_params['emb_size']
        self.img_size = model_params['img_size']
        self.depth = model_params['depth']
        self.num_heads = model_params['num_heads']
        self.num_classes = model_params['num_classes']

        self.embedding = PatchEmbedding(self.in_channels, self.patch_size, self.emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_size))
        self.positional_emb = nn.Parameter(torch.zeros(1, self.img_size // self.patch_size * self.img_size // self.patch_size + 1, self.emb_size))
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(self.emb_size, self.num_heads),
            self.depth
        )
        self.classifier = nn.Linear(self.emb_size, self.num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        B, N, _ = emb.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        emb = torch.cat([cls_tokens, emb], dim=1)
        emb = emb + self.positional_emb
        x = self.transformer(emb)
        x = x[:, 0]
        return self.classifier(x)
