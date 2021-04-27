# Python Import:
import yaml
import torch
import numpy as np
from argparse import ArgumentParser

# Pytorch Lightning Import:
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
    from models.rnn import RaceModule
    # from models.t5 import RaceModule

    batch_fn = None
    collate_fn = None

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = RaceDataModule.add_model_specific_args(parser)
    parser = RaceModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    config = yaml.load(open("configs/rnn.yaml"), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    fx_dm = RaceDataModule(args, collate_fn)
    fx_model = RaceModule(args, batch_fn)

    # Callbacks:
    checkpoint = ModelCheckpoint(
        dirpath='models/ckpts/',
        filename="./fx-{epoch:02d}-{val_loss:.7f}",
        monitor="val_loss"
    )
    earlystopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0.1,
                                  patience=3,
                                  verbose=False,
                                  mode="min")

    # Logger:
    logger = TensorBoardLogger('models/logs/')
    # logger = NeptuneLogger(project_name="haonan-liu/RACE",
    #                        params=vars(args),
    #                        api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5OTI4YmE4My0wYmNmLTQwZWYtYTM4YS05ZTNiNjJkNDc2MTUifQ==')

    # Trainer:
    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint,
        callbacks=earlystopping,
        logger=logger
    )
    # trainer.fit(fx_model, fx_dm)
    # trainer.test(fx_model, test_dataloaders=fx_dm.test_dataloader())

    fx_dm.setup()
    fx_infer = RaceModule.load_from_checkpoint("models/ckpts/fx-epoch=00-val_loss=5.6027322.ckpt")
    fx_infer.eval()

    test_batch_fn = lambda batch: (batch['answers']['input_ids'], torch.cat([batch['answers']['input_ids'],batch['articles']['input_ids']], dim=1), batch['questions']['input_ids'])
    display_result_as_string(fx_dm.tokenizer, fx_dm.test_dataloader(),
                             fx_infer, test_batch_fn)
