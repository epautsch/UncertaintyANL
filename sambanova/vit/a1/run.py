import sambaflow
import sambaflow.samba as samba
import sambaflow.samba.optim as optim
import sambaflow.samba.utils as utils

from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.common import common_app_driver
from sambaflow.samba.sambaloader import SambaLoader

import torch

import sys
import argparse
from typing import Tuple

from vit_samba import VisionTransformer
from dataloader import input_fn_train, input_fn_eval


def add_user_args(parser: argparse.ArgumentParser) -> None:
   parser.add_argument(
       "-bs",
       type=int,
       default=100,
       metavar="N",
       help="input batch size for training (default: 100)",
   )
   parser.add_argument(
       "--num-epochs",
       type=int,
       default=6,
       metavar="N",
       help="number of epochs to train (default: 6)",
   )
   parser.add_argument(
       "--num-classes",
       type=int,
       default=1000,
       metavar="N",
       help="number of classes in dataset (default: 10)",
   )
   parser.add_argument(
       "--learning-rate",
       type=float,
       default=0.00005,
       metavar="LR",
       help="learning rate (default: 0.001)",
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


# dummy image for compiliation tracing
def get_inputs(args):
    dummy_image = (
            samba.randn(args.bs, 3, 224, 224, name='image', batch_dim=0),
            samba.randint(args.num_classes, (args.bs,), name='label', batch_dim=0),
    )
    print(args.bs)
    return dummy_image


def prepare_dataloaders(args):
    # torch dataloaders
    train_loader = input_fn_train(batch_size=args.bs)
    eval_loader = input_fn_eval(batch_size=args.bs)
    # convert to sambaloaders
    sn_train_loader = SambaLoader(train_loader, ['image', 'label'])
    sn_eval_loader = SambaLoader(eval_loader, ['image', 'label'])

    return sn_train_loader, sn_eval_loader


def train(args, model):
    sn_train_loader, _ = prepare_dataloaders(args)
    hyperparam_dict = {"lr": args.learning_rate}

    total_step = len(sn_train_loader)
    loss_list = []
    acc_list = []

    for epoch in range(args.num_epochs):
        for i, (images, labels) in enumerate(sn_train_loader):
            print(f'EPOCH: {epoch}/{args.num_epochs}', f'ITER: {i}/{len(sn_train_loader)}')
            
            loss, outputs = samba.session.run(
                input_tensors=(images, labels),
                output_tensors=model.output_tensors,
                hyperparam_dict=hyperparam_dict
            )

            # Convert SambaTensors back to Torch Tensors to calculate accuracy
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            loss_list.append(loss.tolist())

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(
                        epoch + 1,
                        args.num_epochs,
                        i + 1,
                        total_step,
                        torch.mean(loss),
                        (correct / total) * 100,
                    )
                )


def main(argv):

    args = parse_app_args(argv=argv, common_parser_fn=add_user_args)
    model = VisionTransformer()
    samba.from_torch_model_(model)
    optimizer = samba.optim.AdamW(model.parameters(), lr=args.learning_rate)
    inputs = get_inputs(args)

    if args.command == 'run':
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        train(args, model)
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




