import logging
import os

import cerebras_pytorch.experimental as cstorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader import input_fn_train, input_fn_eval
from vit_custom import VisionTransformer

# autogen policy workaround 1/3
from cerebras_appliance.pb.workflow.appliance.common.common_config_pb2 import (
        DebugArgs,
)
from cerebras_appliance.run_utils import (
        update_debug_args_with_autogen_policy,
)
# end workaround

#import sys
#sys.path.insert(0, '/home/epautsch/R_1.8.0/modelzoo/')
#from modelzoo.common.pytorch.layers import CrossEntropyLoss


# configs
MODEL_DIR = './'
COMPILE_ONLY = False
VALIDATE_ONLY = False

TRAINING_STEPS = 10
CKPT_STEPS = 5
LOG_STEPS = 5

CHECKPOINT_STEPS = 5
IS_PRETRAINED_CHECKPOINT = False


def main_training_loop():
    torch.manual_seed(2023)
    
    model = VisionTransformer()
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(f'AParam #{idx}, Is leaf: {param.is_leaf}, Name: {name}, Device: {param.device.type}')
        print(param.requires_grad)

    compiled_model = cstorch.compile(model, backend='WSE_WS')
    
  #  for idx, (name, param) in enumerate(model.named_parameters()):
   #     print(f'BParam #{idx}, Is leaf: {param.is_leaf}, Size: {param.size()}, Name: {name}')

    loss_fn = nn.CrossEntropyLoss()

    optimizer = cstorch.optim.configure_optimizer(
            optimizer_type='Adam',
            params=compiled_model.parameters(),
            lr=0.01,
    )

    lr_params = {
            'scheduler': 'Linear',
            'initial_learning_rate': 0.01,
            'end_learning_rate': 0.001,
            'total_iters': 5,
    }

    lr_scheduler = cstorch.optim.configure_lr_scheduler(optimizer, lr_params)
    
    #for idx, (name, param) in enumerate(compiled_model.named_parameters()):
     #   print(f'CParam #{idx}, Is leaf: {param.is_leaf}, Name: {name}, Device: {param.device.type}')

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
        outputs = compiled_model(inputs)

        loss = loss_fn(outputs, targets)
        print(outputs.shape) 
        cstorch.amp.optimizer_step(
                loss, optimizer, grad_scaler,
        )

        lr_scheduler.step()

        save_learning_rate()

        accumulate_loss(loss)

        return loss

    writer = SummaryWriter(log_dir=os.path.join(MODEL_DIR, 'train'))

    @cstorch.step_closure
    def post_training_step(loss):
        if LOG_STEPS and global_step % LOG_STEPS == 0:
            logging.info(
                    f'| Train: {compiled_model.backend.name} '
                    f'Step={global_step}, '
                    f'Loss={loss.item():.5f}'
            )

        if torch.isnan(loss).any().item():
            raise ValueError('NaN loss detected. ')
        if torch.isinf(loss).any().item():
            raise ValueError('inf loss detected.')

        for group, lr in enumerate(lr_scheduler.get_last_lr()):
            writer.add_scalar(f'lr.{group}', lr, global_step)

            cstorch.save_summaries(writer, global_step)

    batch_size = 4
    dataloader = cstorch.utils.data.DataLoader(
            input_fn_train, batch_size, num_steps=TRAINING_STEPS
    )

    for i, batch in enumerate(dataloader):
        loss = training_step(batch)

        global_step += 1
        
        post_training_step(loss)

        cstorch.scalar_summary('loss', loss)

        if CHECKPOINT_STEPS and global_step % CHECKPOINT_STEPS == 0:
            save_checkpoint(global_step)

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)

    os.makedirs(os.path.join(os.getcwd(), 'imagenet_dataset'), exist_ok=True)
    
    # autogen policy workaround 2/3
    debug_args = DebugArgs()
    update_debug_args_with_autogen_policy(debug_args, 'medium')
    # end workaround

    cstorch.configure(
            model_dir=MODEL_DIR,
            compile_dir='./compile_dir',
            mount_dirs=[os.getcwd()],
            python_paths=[os.getcwd()],
            compile_only=COMPILE_ONLY,
            validate_only=VALIDATE_ONLY,
            checkpoint_steps=CKPT_STEPS,
            max_wgt_servers=1,
            num_workers_per_csx=1,
            max_act_per_csx=1,
            debug_args=debug_args, # autogen policy workaround 3/3
    )

    main_training_loop()








