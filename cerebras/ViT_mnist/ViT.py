import torch
from torch import nn


class VisionTransformer(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.patch_size = model_params['patch_size']
        self.hidden_dim = model_params['hidden_dim']
        self.num_heads = model_params['num_heads']
        self.num_layers = model_params['num_layers']

        self.embedding = nn.Linear(self.patch_size * self.patch_size * 1, self.hidden_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim)) # classification token

        encoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, self.num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.classifier = nn.Linear(self.hidden_dim, 10)

    def forward(self, x):
        # shape is batchSize, channels, h, w
        b, c, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, "Image dimensions notdivisible by patchsize"

        # reshape into patches
        patches = x.view(b, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 4, 3, 5, 1).contiguous()
        patches = patches.view(b, h // self.patch_size * w // self.patch_size, -1)
        
        x = self.embedding(patches)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer(x)
        x = x[:, 0]
        x = self.classifier(x)

        return x
