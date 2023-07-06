import os
import sys

#Append path to parent directory of Cerebras ModelZoo Repository
sys.path.insert(0, '/home/epautsch/R_1.8.0/modelzoo/')
from modelzoo.common.pytorch.run_utils import run

from data import (
    input_fn_train,
    input_fn_eval
)
from model import ViTModel

def main():
    run(ViTModel, input_fn_train, input_fn_eval)

if __name__ == '__main__':
    main()
