import logging
import os

import cerebras_pytorch.experimental as cstorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader import input_fn_train, input_fn_eval
from vit_custom import VisionTransformer


# configs
MODEL_DIR = './'
COMPILE_ONLY = False
VALIDATE_ONLY = False

TRAINING_STEPS = 10
CKPT_STEPS = 5
LOG_STEPS = 5

CHECKPOINT_STEPS = 5
IS_PRETRAINED_CHECKPOINT = False


def main_training_loop(cs_config):
    torch.manual_seed(2023)
    
    backend = cstorch.backend(
            BACKEND,
            artifact_dir=MODEL_DIR,
            compile_dir="./compile_dir",
            compile_only=COMPILE_ONLY,
            validate_only=VALIDATE_ONLY,
    )

    with backend.device:
        model = VisionTransformer()

    model = cstorch.compile(model, backend)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = cstorch.optim.configure_optimizer(
            optimizer_type='SGD',
            params=model.parameters(),
            lr=0.01,
            momentum=0.0
    )

    lr_params = {
            'scheduler': 'Linear',
            'initial_learning_rate': 0.01,
            'end_learning_rate': 0.001,
            'total_iters': 5,
    }
    lr_scheduler = cstorch.optim.configure_lr_scheduler(optimizer, lr_params)

    grad_scaler = cstorch.amp.GradScaler(loss_scale='dynamic')

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

    @cstorch.step_closure
    def save_checkpoint(step):
        logging.info(f'Saving chkpnt at step {step}')

        checkpoint_file = os.path.join(MODEL_DIR, f'checkpoint_{step}.mdl')

        state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'grad_scalar': grad_scaler.state_dict()
        }

        state_dict['global_step'] = step

        cstorch.save(state_dict, checkpoint_file)
        logging.info(f'Saved chkpnt {checkpoint_file}')

    global_step = 0

    @cstorch.compile_step
    def training_step(batch):
        inputs, targets = batch
        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        cstorch.amp.optimizer_step(
                loss, optimizer, grad_scaler,
        )

        lr_scheduler.step()

        save_learning_rate()

        accumulate_loss(loss)

        cstorch.summarize_scalar('loss', loss)

        return loss

    writer = cstorch.utils.tensorboard.SummaryWriter(log_dir=os.path.join(MODEL_DIR, 'train'))

    @cstorch.step_closure
    def post_training_step(loss):
        if LOG_STEPS and global_step % LOG_STEPS == 0:
            logging.info(
                    f'| Train: {model.device} '
                    f'Step={global_step}, '
                    f'Loss={loss.item():.5f}'
            )

        if torch.isnan(loss).any().item():
            raise ValueError(
                    'NaN loss detected. '
                    'Please try diff hyperparams '
                    'such as the lr, batch size, etc.'
            )
        if torch.isinf(loss).any().item():
            raise ValueError('inf loss detected.')

        for group, lr in enumerate(lr_scheduler.get_last_lr()):
            writer.add_scalar(f'lr.{group}', lr, global_step)

            cstorch.save_summaries(writer, global_step)

    batch_size = 4
    dataloader = cstorch.utils.data.DataLoader(input_fn_train, batch_size)
    executor = cstorch.utils.dat.DataExecutor(
            dataloader,
            num_steps=TRAINING_STEPS,
            checkpoint_steps=CHECKPOINT_STEPS,
            writer=writer,
            cs_config=cs_config,
    )

    for _, batch in enumerate(executor):
        loss = training_step(batch)

        global_step += 1
        
        post_training_step(loss)

        if CHECKPOINT_STEPS and global_step % CHECKPOINT_STEPS == 0:
            save_checkpoint(global_step)

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)

    os.makedirs(os.path.join(os.getcwd(), 'imagenet_dataset'), exist_ok=True)

    cs_config = cstorch.utils.CSConfig(
            model_dirs=[os.getcwd()],
            python_paths=[os.getcwd()],
            max_wgt_servers=1,
            num_workers_per_csx=1,
            max_act_per_csx=1,
    )

    main_training_loop(cs_config)








