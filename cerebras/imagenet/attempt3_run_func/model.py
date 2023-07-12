import torch
import torch.nn as nn
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.common.pytorch.layers import CrossEntropyLoss
from vit import ViT


class ViTModel(PyTorchBaseModel):
    def __init__(self, params, device=None):
        self.params = params
        model_params = params["model"].copy()
        self.model = self.build_model(model_params)
        self.loss_fn = CrossEntropyLoss(use_autogen=True)

        super().__init__(params=params, model_fn=self.model, device=device)

    def build_model(self, model_params):
        dtype = torch.float32
        model = ViT(model_params)
        model.to(dtype)
        return model

    def __call__(self, data):
        inputs = data['image']
        labels = data['label']
        print(labels.dtype)
        labels = labels.to(torch.int32)
        print(labels.dtype)
        outputs = self.model(inputs)
        print(outputs.dtype)
        loss = self.loss_fn(outputs, labels)
        return loss
