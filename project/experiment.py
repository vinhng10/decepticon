import os
import numpy as np
import argparse

from time import sleep
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn import functional as F
from models.t5 import *
from data.data import *

def main(hparams):
    seed_everything(hparams.seed)
    
    data = RaceDataModule(hparams, customed_collate_fn = RaceDataModule.t5_collate_fn)
    #hparams.tokenizer_len = len(data_module.tokenizer)
    early_stop_callback = EarlyStopping(
        monitor='val_perplexity',
        min_delta=0.5,
        patience=3,
        verbose=False,
        mode="min")
    
    
    model = T5FinetuneForRACE(hparams)
    logger = NeptuneLogger(project_name="carlomarxdk/T5-for-RACE",
                           params = vars(hparams),
                           experiment_name = "T5 finetuning to race: %s" %str(hparams.version),            api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTY1YzBlY2QtOTFlMS00Yzg2LWJiYzItNjQ2NDlhOGRhN2M5In0=')
    
    checkpoint_callback = ModelCheckpoint(monitor = None,
                                          dirpath = "checkpoints/",
                                          verbose = True,
                                          filename = str(hparams.version).replace(".", "_"))
    
    trainer = Trainer(accumulate_grad_batches=hparams.accumulate_grad_batches,
                      checkpoint_callback = checkpoint_callback,
                      callbacks=[early_stop_callback, LearningRateMonitor()],
                      logger = logger,
                      terminate_on_nan = hparams.terminate_on_nan,
                      benchmark = False,
                      precision = 16,
                      #log_gpy_memory = True,
                      enable_pl_optimizer=False,
                      track_grad_norm = 2,
                      max_epochs = 20,
                      log_every_n_steps = 150,
                      gradient_clip_val = 5,
                      stochastic_weight_avg = False,
                      gpus=-1)
    trainer.fit(model, data)
    
    print("Fine Tuning: Finilised")

if __name__ == '__main__':
    __spec__ = None
    parser = argparse.ArgumentParser(description="T5 Model for FineTuning on RACE")
    parser.add_argument("--version", type=float) ## you need to specify it on a form X.XX 

    # DATA
    parser.add_argument("--data_path", default = "D:\Github/decepticon/Processed_New", type=str)
    parser.add_argument("--batch_size", default = 16, type=int)
    parser.add_argument("--num_workers", default = 0, type=int)
    parser.add_argument("--padding_token", default = 0, type=int) # do not change that
    
    # LOGGER
    parser.add_argument("--log_path", default = "logs/", type=str)

    
    # MODEL
    parser.add_argument("--pretrained_model", default = "t5-small", type=str)
    parser.add_argument("--tokenizer_len", default=32102, type=int) #do nto touch it
    
    # TRAINING
    parser.add_argument("--seed", default = 2020, type=float)
    parser.add_argument("--weight_decay", default = 1e-4, type=float)
    parser.add_argument("--learning_rate", default = 1e-4, type=float)
    parser.add_argument("--accumulate_grad_batches", default = 10, type=int)
    parser.add_argument("--terminate_on_nan", default = True, type=int)
    parser.add_argument("--special_tokens", default = ["<answer>", "<context>"])
    
    args = parser.parse_args()
    
    
    main(args)
    
    