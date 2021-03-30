from collections import namedtuple
# neptune
import os
import numpy as np
from time import sleep
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.logger import NeptuneLoggr
from pytorch_lightning.callbacks import Model


from torch.nn import functional as F
from transformers import  T5ForConditionalGeneration, T5Config, get_linear_schedule_with_warmup, AdamW
from project.data.data import *

def main(args):
    seed_everything(argsseed)


if __name__ == '__main__':
    __spec__ = None
    