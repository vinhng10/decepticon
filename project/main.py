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
from pytorch_lightning.callbacks import LearningRateMonitor

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
    fx_model = RaceModule(args, batch_fn)

    # Callbacks:
    checkpoint = ModelCheckpoint(
        dirpath='models/ckpts/',
        filename="./fx-{epoch:02d}-{val_loss:.7f}",
        monitor="val_loss"
    )
    earlystopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0.05,
                                  patience=3,
                                  verbose=False,
                                  mode="min")

    # Logger:
    # logger = TensorBoardLogger('models/logs/')
    logger = NeptuneLogger(project_name="carlomarxdk/T5-for-RACE",
                           params=vars(args),
                           experiment_name="T5 finetuning to race: %s" % str(args.version),
                           api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTY1YzBlY2QtOTFlMS00Yzg2LWJiYzItNjQ2NDlhOGRhN2M5In0=')

    # Trainer:
    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint,
        callbacks=[earlystopping, LearningRateMonitor()],
        logger=logger
    )
    trainer.fit(fx_model, fx_dm)
    trainer.test(fx_model, test_dataloaders=fx_dm.test_dataloader())

    # fx_infer = RaceModule.load_from_checkpoint(checkpoint.best_model_path)
    # fx_infer.eval()
    # fx_dm.setup()
    #
    # with torch.no_grad():
    #     for batch in fx_dm.test_dataloader():
    #         # ans = batch['inputs']['input_ids']
    #         ans = batch['answers']['input_ids']
    #         if batch_fn:
    #             x, y = batch_fn(batch)
    #         else:
    #             x, y = fx_infer.batch_fn(batch)
    #         # out = fx_infer(x, pred_len=50)
    #         out, _ = fx_infer(x)
    #         # translate(fx_dm.tokenizer, ans, out, y['input_ids'])
    #         translate(fx_dm.tokenizer, ans, out, y)
