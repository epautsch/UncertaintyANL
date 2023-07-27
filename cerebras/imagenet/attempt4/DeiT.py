import torch.nn as nn
import torch.nn.functional as F
import timm

from modelzoo.common.pytorch.layers import CrossEntropyLoss

class Cerebras_DeiT(nn.Module):
    def __init__(self, params, num_classes=1000):
        super().__init__()
        self.model=timm.create_model('deit_base_patch16_224.fb_in1k', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        self.criterion = CrossEntropyLoss(use_autogen=True)

    def forward(self, batch):
        images, labels = batch[0], batch[1]

        outputs = self.model(images)

        loss = self.criterion(outputs, labels)

        return loss



