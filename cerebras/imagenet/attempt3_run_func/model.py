import torch
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from vit import ViT


class ViTModel(PyTorchBaseModel):
    def __init__(self, params, device=None):
        self.params = params
        model_params = params["model"].copy()
        self.model = self.build_model(model_params)
        self.loss_fn = nn.CrossEntropyLoss()

        super().__init__(params=params, model=self.model, device=device)

    def build_model(self, model_params):
        dtype = torch.float32
        model = ViT(model_params)
        model.to(dtype)
        return model

    def __call__(self, data):
        inputs, labels = data
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        return loss
