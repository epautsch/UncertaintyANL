import logging
import os

import cerebras_pytorch.experimental as cstorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader import input_fn_train, input_fn_eval


# import timm model
class CustomViTModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit,head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224))
        x = self.vit(x)
        return x


# Params
MODEL_DIR = "./"
COMPILE_ONLY = False
VALIDATE_ONLY = False

CKPT_STEPS = 5

CHECKPOINT_STEPS = 5
CHECKPOINT_PATH_EVAL = None

# eval loop
def main_eval_loop():
    model = CustomViTModel()
    compiled_model = cstorch.compile(model, backend="WSE_WS")


    def load_checkpoint(checkpoint_path):
        state_dict = cstorch.load(checkpoint_path)
        model.load_state_dict(state_dict["model"])

        global_step = state_dict.get("global_step", 0)
        return global_step


    global_step = 0
    
    if CHECKPOINT_PATH_EVAL is not None:
        global_step = load_checkpoint(CHECKPOINT_PATH_EVAL)
    else:
        logging.info(
                f"No checkpoint was provided, model parameters will be "
                f"initialized randomly"
        )

    writer = SummaryWriter(log_dir=os.path.join(MODEL_DIR, "eval"))

    accuracy = cstorch.metrics.AccuracyMetric(
            "accuracy", compute_on_system=True
    )




























