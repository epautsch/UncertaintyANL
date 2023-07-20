import torch
import torch.nn as nn
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.common.pytorch.layers import CrossEntropyLoss
from vit import VisionTransformer


class ViTModel(PyTorchBaseModel):
    def __init__(self, params, device=None):
        self.params = params
        model_params = params["model"].copy()
        self.model = self.build_model(model_params)
        self.loss_fn = CrossEntropyLoss(use_autogen=True)

        super().__init__(params=params, model_fn=self.model, device=device)

    def build_model(self, model_params):
        dtype = torch.float16
        model = VisionTransformer(model_params=model_params)
        model.to(dtype)
        return model

    def __call__(self, data):
        inputs = data[0]
        labels = data[1]
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        return loss
