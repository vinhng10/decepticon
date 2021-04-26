# Python Import:
import yaml
import torch
import numpy as np
from argparse import ArgumentParser

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
    fx_dm.setup()

            # Trainer:
    trainer = pl.Trainer.from_argparse_args(args)
        
    
    fx_model = RaceModule.load_from_checkpoint("models/ckpts/t5.ckpt")
    fx_model.setup_tune(top_p = 0.95, top_k = 50, no_repeat_ngram_size = 2)
    
    
    
    
    
    trainer.test(fx_model, test_dataloaders=fx_dm.test_dataloader())