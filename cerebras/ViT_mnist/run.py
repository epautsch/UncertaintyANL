import os
import sys

#Append path to parent directory of Cerebras ModelZoo Repository
sys.path.insert(0, '/home/epautsch/R_1.8.0/modelzoo/')
from modelzoo.common.pytorch.run_utils import run

from data import (
    get_train_dataloader,
    get_eval_dataloader,
)
from model import MNISTModel

def main():
    run(MNISTModel, get_train_dataloader, get_eval_dataloader)

if __name__ == '__main__':
    main()
