# Python Import:
import yaml
import torch
import numpy as np
from argparse import ArgumentParser
import gc
from ray import tune
from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
import os

# Pytorch Lightning Import:
import pytorch_lightning as pl

# Internal Import:
from data.data import RaceDataModule
from utils.utils import (
    t5_collate_fn, t5_dis_collate_fn,
    transformer_collate_fn,
    rnn_batch_fn, rnn_dis_batch_fn,
    display_result_as_string,
    serialize_config
)


if __name__ == "__main__":
    ##for RAY
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    # Choose the model
    # from models.transformer import RaceModule
    # from models.rnn import RaceModule
    from models.t5 import RaceModule

    batch_fn = None
    collate_fn = t5_collate_fn

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = RaceDataModule.add_model_specific_args(parser)
    parser = RaceModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    config = yaml.load(open("configs/t5.yaml"), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    fx_dm = RaceDataModule(args, collate_fn)

    trainer = pl.Trainer.from_argparse_args(args)
    
    fx_dm.setup()
    
    fx_model = RaceModule.load_from_checkpoint("D:/Github/decepticon/project/models/ckpts/t5_que.ckpt")
    fx_model.setup_tune(top_p = 0.8, top_k = 40, no_repeat_ngram_size = 4)

    fx_model.eval()
    f = open('sample_generation.txt','w')
    for x,y in fx_dm.test_dataloader():
        output = fx_model.generate(x)
        display_result_as_string(fx_dm.tokenizer, None, output, y["input_ids"])


