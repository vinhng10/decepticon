import os
import numpy as np
import argparse

from time import sleep
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.nn import functional as F
from models.t5 import *
from data.data import *

def main(hparams):
    seed_everything(hparams.seed)
    
    data = RaceDataModule(hparams)
    #hparams.tokenizer_len = len(data_module.tokenizer)
    
    model = T5FinetuneForRACE(hparams.to_dict())
    
    trainer = Trainer(accumulate_grad_batches=hparams.accumulate_grad_batches)
    trainer.fit(model, data_module)
    
    print("Fine Tuning: Finilised")

if __name__ == '__main__':
    __spec__ = None
    parser = argparse.ArgumentParser(description="T5 Model for FineTuning on RACE")

    # DATA
    parse.add_argument("--data_path", default = "Processed_New/", type=str)
    parse.add_argument("--batch_size", default = 16, type=int)
    parse.add_argument("--num_workers", default = 2, type=int)
    parse.add_argument("--padding_id", default = 0, type=int) #do not change that
    
    # MODEL
    parse.add_argument("--pretrained_model", default = "t5-small", type=str)
    
    # TRAINING
    parse.add_argument("--weight_decay", default = 1e-5, type=float)
    parse.add_argument("--learning_rate", default = 1e-5, type=float)



        self.weight_decay = 1e-5
        self.learning_rate = 1e-5
        self.padding_id = 0
    
    
    parser.add_argument("--accumulate_grad_batches", default=10, type=int)
    
    
    main(parser)
    
    