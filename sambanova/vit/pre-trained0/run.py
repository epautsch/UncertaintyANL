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

import sys
import argparse
from typing import Tuple

from dataloader import input_fn_train, input_fn_eval


#timm_model = create_model('vit_base_patch16_224', pretrained=True)


class ViT_pretrained(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        out = super().forward(x)
        loss = self.criterion(out, labels)
        return loss, out


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


# dummy image for compiliation tracing
def get_inputs(args):
    dummy_image = (
            samba.randn(args.bs, 3, 224, 224, name='image', batch_dim=0),
            samba.randint(args.num_classes, (args.bs,), name='label', batch_dim=0),
    )
    return dummy_image


def prepare_dataloaders(args):
    # torch dataloaders
    train_loader = input_fn_train(batch_size=args.bs)
    eval_loader = input_fn_eval(batch_size=args.bs)
    # convert to sambaloaders
    sn_train_loader = SambaLoader(train_loader, ['image', 'label'])
    sn_eval_loader = SambaLoader(eval_loader, ['image', 'label'])

    return sn_train_loader, sn_eval_loader


def val(args, model):
    _, sn_val_loader = prepare_dataloaders(args)
    hyperparam_dict = {"lr": 0.}

    total_step = len(sn_val_loader)
    loss_list = []
    acc_list = []

    for epoch in range(args.num_epochs):
        for i, (images, labels) in enumerate(sn_val_loader):

            loss, outputs = samba.session.run(
                input_tensors=(images, labels),
                output_tensors=model.output_tensors,
                hyperparam_dict=hyperparam_dict
            )

            # Convert SambaTensors back to Torch Tensors to calculate accuracy
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            loss_value = loss.mean().item()
            loss_list.append(loss_value)

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_value = correct / total
            acc_list.append(acc_value)

            if (i + 1) % 5 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(
                        epoch + 1,
                        args.num_epochs,
                        i + 1,
                        total_step,
                        loss_value,
                        acc_value * 100,
                    )
                )
    avg_loss = sum(loss_list) / len(loss_list)
    avg_acc = sum(acc_list) / len(acc_list)

    print(f'Final average loss: {avg_loss:.4f}, Final average accuracy: {avg_acc*100:.2f}%')
    

def main(argv):

    args = parse_app_args(argv=argv, common_parser_fn=add_user_args)

    pretrained = create_model('vit_base_patch16_224', pretrained=True)
    model = ViT_pretrained()
    model.load_state_dict(pretrained.state_dict())

    samba.from_torch_model_(model)
    optimizer = samba.optim.AdamW(model.parameters(), lr=args.learning_rate)
    inputs = get_inputs(args)

    if args.command == 'run':
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        val(args, model)
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




