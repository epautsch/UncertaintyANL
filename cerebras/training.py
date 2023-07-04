import logging
import os

import sys
sys.path.insert(0, '/home/epautsch/R_1.8.0/modelzoo/')
from modelzoo.common.pytorch.layers import CrossEntropyLoss

import cerebras_pytorch.experimental as cstorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import timm

from dataloader import input_fn_train, input_fn_eval


class CustomViTModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x

# CONFIGURABLE VARIABLES FOR THIS SCRIPT
# Can optionally move these arguments to a params file and configure from there.
MODEL_DIR = "./"
COMPILE_ONLY = False
VALIDATE_ONLY = False

TRAINING_STEPS = 10
CKPT_STEPS = 5
LOG_STEPS = 5

# Checkpoint-related configurations
CHECKPOINT_STEPS = 5
IS_PRETRAINED_CHECKPOINT = False

def main_training_loop():
    torch.manual_seed(2023)

    model = CustomViTModel()
    compiled_model = cstorch.compile(model, backend="WSE_WS")

    loss_fn = CrossEntropyLoss(use_autogen=True)

    # Define the optimizer used for training.
    # This example will be using SGD from cerebras_pytorch.experimental.optim.Optimizer
    # For a complete list of optimizers available in the experimental API, please see
    # https://docs.cerebras.net/en/latest/wsc/port/porting-pytorch-to-cs/cstorch-api.html#initializing-the-optimizer
    optimizer = cstorch.optim.configure_optimizer(
        optimizer_type="SGD",
        params=model.parameters(),
        lr=0.01,
        momentum=0.0,
    )

    # Optionally define the learning rate scheduler
    # This example will be using LinearLR from cerebras_pytorch.experimental.optim.lr_scheduler
    # For a complete list of lr schedulers available in the experimental API, please see
    # https://docs.cerebras.net/en/latest/wsc/port/porting-pytorch-to-cs/cstorch-api.html#initializing-the-learning-rate-scheduler
    lr_params = {
        "scheduler": "Linear",
        "initial_learning_rate": 0.01,
        "end_learning_rate": 0.001,
        "total_iters": 5,
    }
    lr_scheduler = cstorch.optim.configure_lr_scheduler(optimizer, lr_params)

    # Define gradient scaling parameters.
    grad_scaler = cstorch.amp.GradScaler(loss_scale="dynamic")

    loss_values = []
    total_steps = 0

    @cstorch.step_closure
    def accumulate_loss(loss):
        nonlocal loss_values
        nonlocal total_steps

        loss_values.append(loss.item())
        total_steps += 1

    lr_values = []

    @cstorch.step_closure
    def save_learning_rate():
        lr_values.append(lr_scheduler.get_last_lr())

    # DEFINE METHOD FOR SAVING CKPTS
    @cstorch.step_closure
    def save_checkpoint(step):
        logging.info(f"Saving checkpoint at step {step}")

        checkpoint_file = os.path.join(MODEL_DIR, f"checkpoint_{step}.mdl")

        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "grad_scalar": grad_scaler.state_dict(),
        }

        state_dict["global_step"] = step

        cstorch.save(state_dict, checkpoint_file)
        logging.info(f"Saved checkpoint {checkpoint_file}")

    global_step = 0

    # DEFINE THE ACTUAL TRAINING LOOP
    @cstorch.compile_step
    def training_step(batch):
        inputs, targets = batch
        outputs = compiled_model(inputs)

        loss = loss_fn(outputs, targets)

        cstorch.amp.optimizer_step(
            loss, optimizer, grad_scaler,
        )

        lr_scheduler.step()

        save_learning_rate()

        accumulate_loss(loss)

        return loss

    # DEFINE POST-TRAINING LOOP IF YOU ARE INTERESTED IN TRACKING SUMMARIES, ETC.
    writer = SummaryWriter(log_dir=os.path.join(MODEL_DIR, "train"))

    @cstorch.step_closure
    def post_training_step(loss):
        if LOG_STEPS and global_step % LOG_STEPS == 0:
            # Define the logging any way desired.
            logging.info(
                f"| Train: {compiled_model.backend.name} "
                f"Step={global_step}, "
                f"Loss={loss.item():.5f}"
            )

        # Add handling for NaN values
        if torch.isnan(loss).any().item():
            raise ValueError(
                "NaN loss detected. "
                "Please try different hyperparameters "
                "such as the learning rate, batch size, etc."
            )
        if torch.isinf(loss).any().item():
            raise ValueError("inf loss detected.")

        for group, lr in enumerate(lr_scheduler.get_last_lr()):
            writer.add_scalar(f"lr.{group}", lr, global_step)

            cstorch.save_summaries(writer, global_step)

    # PERFORM TRAINING LOOPS
    batch_size = 4
    dataloader = cstorch.utils.data.DataLoader(
        input_fn_train, batch_size, num_steps=TRAINING_STEPS
    )
    
    print(f'number of samples in dataset: {len(dataloader.dataset)}')
    print(f'Batch size: {dataloader.batch_size}')
    print(f'Num workers: {dataloader.num_workers}')
    print(f'first batch: {next(iter(dataloader))}')

    for i, batch in enumerate(dataloader):
        loss = training_step(batch)

        global_step += 1

        post_training_step(loss)

        # Save the loss value to be able to plot the loss curve
        cstorch.scalar_summary("loss", loss)

        if CHECKPOINT_STEPS and global_step % CHECKPOINT_STEPS == 0:
            save_checkpoint(global_step)

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    os.makedirs(os.path.join(os.getcwd(),'mnist_dataset'), exist_ok=True)

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

    main_training_loop()
