import logging
import os

import cerebras_pytorch.experimental as cstorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import timm

from dataloader import input_fn_train, input_fn_eval


# import timm model
class CustomViTModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x


# Params
MODEL_DIR = "./"
COMPILE_ONLY = True
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

    loss_fn = torch.nn.CrossEntropyLoss()

    @cstorch.compile_step
    def eval_step(batch):
        inputs, targets = batch
        outputs = compiled_model(inputs).to(torch.float16)
        loss = loss_fn(outputs, targets)

        accuracy(
                labels=targets.clone(), predictions=outputs.argmax(-1).int(),
        )
        
        return loss

    total_loss = 0
    total_steps = 0

    @cstorch.step_closure
    def post_eval_step(loss: torch.tensor):
        nonlocal total_loss
        nonlocal total_steps

        logging.info(
                f"| Eval: {compiled_model.backend.name} "
                f"Step={global_step}, "
                f"Loss={loss.item():.5f}"
        )

        if torch.isnan(loss).any().item():
            raise ValueError("NaN loss detected.")
        if torch.isinf(loss).any().item():
            raise ValueError("inf loss detected.")

        total_loss += loss.item()
        total_steps += 1

        cstorch.scalar_summary("loss", loss)

    batch_size = 4
    dataloader = cstorch.utils.data.DataLoader(
            input_fn_eval, batch_size, num_steps=10
    )

    for i, batch in enumerate(dataloader):
        loss = eval_step(batch)

        global_step += 1

        post_eval_step(loss)

    writer.add_scalar(f"Eval Accuracy", float(accuracy), global_step)
    cstorch.save_summaries(writer, global_step)


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    cstorch.configure(
            model_dir=MODEL_DIR,
            compile_dir="./compile_dir",
            mount_dirs=[os.getcwd()],
            python_paths=[os.getcwd()],
            compile_only=COMPILE_ONLY,
            validate_only=VALIDATE_ONLY,
            checkpoint_steps=CKPT_STEPS,
            # CSConfig params
            max_wgt_servers=1,
            num_workers_per_csx=1,
            max_act_per_csx=1,
    )

    main_eval_loop()
































