import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.optim as optim
import sambaflow.samba.utils as utils

from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.common import common_app_driver
from sambaflow.samba.sambaloader import SambaLoader

import torch
import torch.nn as nn
import timm
from timm import create_model
import numpy as np

import os
import sys
import argparse
from typing import Tuple

from dataloader import input_fn_train, input_fn_eval


def add_user_args(parser: argparse.ArgumentParser) -> None:
   parser.add_argument(
       "-bs",
       type=int,
       default=512,
       metavar="N",
       help="input batch size for training (default: 512)",
   )
   parser.add_argument(
       "--num-epochs",
       type=int,
       default=1,
       metavar="N",
       help="number of epochs to train (default: 1)",
   )
   parser.add_argument(
       "--num-classes",
       type=int,
       default=1000,
       metavar="N",
       help="number of classes in dataset (default: 1000)",
   )
   parser.add_argument(
       "--learning-rate",
       type=float,
       default=0.,
       metavar="LR",
       help="learning rate (default: 0.0)",
   )
   parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Download location for Imagenet data",
    )
   parser.add_argument(
       "--model-path", type=str, default="model", help="Save location for model"
   )  # From MODEL_STORE_PATH
   parser.add_argument('-dx', type=bool, default=False, help='Used for outputting data')


# dummy image for compiliation tracing
def get_inputs(args):
    return samba.randn(args.bs, 3, 224, 224, name='image', batch_dim=0)


def prepare_dataloaders(args):
    # torch dataloaders
    train_loader = input_fn_train(batch_size=args.bs)
    eval_loader = input_fn_eval(batch_size=args.bs)
    # convert to sambaloaders
    sn_train_loader = SambaLoader(train_loader, ['image'], function_hook=lambda t: [t[0]], return_original_batch=True)
    sn_eval_loader = SambaLoader(eval_loader, ['image'], function_hook=lambda t: [t[0]], return_original_batch=True)

    return sn_train_loader, sn_eval_loader


def val(args, model, data_extract=False, save_dir='./'):
    _, sn_val_loader = prepare_dataloaders(args)
    hyperparam_dict = {"lr": 0.}

    total_step = len(sn_val_loader)
    acc_list = []
    results_list = [] # logits, image_name, label -> as tuple

    for epoch in range(args.num_epochs):
        for i, (images, original_batch) in enumerate(sn_val_loader):
            print(f'Processing batch {i+1} of {total_step}')

            labels = original_batch[1]

            outputs = samba.session.run(
                input_tensors=(images,),
                output_tensors=model.output_tensors,
                hyperparam_dict=hyperparam_dict,
                section_types=['FWD']
            )[0]

            # convert sambatensors back to torch tensors to calculate accuracy
            outputs = samba.to_torch(outputs)
            
            # img paths and labels for batch
            if data_extract:
                paths_and_labels = sn_val_loader.dataloader.dataset.imgs[i*args.bs:(i+1)*args.bs]
            
            # logits, img_name, label -> results_list
            if data_extract:
                for logits, (path, label) in zip(outputs.tolist(), paths_and_labels):
                    image_name = os.path.basename(path)
                    results_list.append((logits, image_name, label))

            # track accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_value = correct / total
            acc_list.append(acc_value)

            if (i + 1) % 5 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Accuracy: {:.2f}%".format(
                        epoch + 1,
                        args.num_epochs,
                        i + 1,
                        total_step,
                        acc_value * 100,
                    )
                )

    avg_acc = sum(acc_list) / len(acc_list)
    print(f'Final average accuracy: {avg_acc*100:.2f}%')

    if data_extract:
        model_name = model.pretrained_cfg['architecture']
        np.save(os.path.join(save_dir, f'{model_name}_results_list.npy'), np.array(results_list, dtype=object))

    

def main(argv):

    args = parse_app_args(argv=argv, common_parser_fn=add_user_args)

    model = create_model('deit_base_distilled_patch16_224', pretrained=True)

    samba.from_torch_model_(model)
    optimizer = samba.optim.AdamW(model.parameters(), lr=args.learning_rate)
    inputs = get_inputs(args)

    if args.command == 'run':
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        val(args, model, data_extract=args.dx, save_dir='/home/epautsch/sambanova/vit/deit_ensembles/saved_run_data')
    else:
        common_app_driver(args=args,
                        model=model,
                        inputs=inputs,
                        optim=optimizer,
                        name=model.__class__.__name__,
                        init_output_grads=not args.inference,
                        app_dir=utils.get_file_dir(__file__))


if __name__ == '__main__':
    main(sys.argv[1:])




