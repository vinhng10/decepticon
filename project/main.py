# Python Import:
import yaml
import torch
import numpy as np
from argparse import ArgumentParser
import os

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
    serialize_config,
    t5_collate_fn, t5_dis_collate_fn,
    transformer_collate_fn,default_collate_fn, transformer_test_batch_fn,
    rnn_collate_fn, rnn_dis_collate_fn,  rnn_test_batch_fn,
    display_result_as_string,
)


if __name__ == "__main__":

    # Choose the model
    # from models.transformer import RaceModule
    # from models.rnn import RaceModule
    from models.t5 import RaceModule

    batch_fn = rnn_dis_batch_fn
    collate_fn = rnn_dis_collate_fn

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = RaceDataModule.add_model_specific_args(parser)
    parser = RaceModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    config = yaml.load(open("configs/transformer.yaml"), Loader=yaml.FullLoader)
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
                                  min_delta=0.01,
                                  patience=5,
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
        callbacks=[earlystopping, LearningRateMonitor()],
        logger=logger
    )
    trainer.fit(fx_model, fx_dm)
    trainer.test(fx_model, test_dataloaders=fx_dm.test_dataloader())

    fx_dm.setup()
    fx_infer = RaceModule.load_from_checkpoint("models/ckpts/t5_que.ckpt")
    fx_infer.eval()

    test_batch_fn = rnn_test_batch_fn
    display_result_as_string(fx_dm.tokenizer, fx_dm.test_dataloader(),
                             fx_infer, test_batch_fn)

    # Single test case
    article = "No one knows for certain why people dream , but some dreams mi ##sh ##t be connected to the mental processes that help us learn . In a recent study , scientists found a connection between nap - time dreams and better memory in people who were learning a new skill . \" I was astonished by this finding , \" Robert Stick ##gold told Science News . He is a cognitive ne ##uro ##s ##cie ##nti ##st at Harvard Medical School who worked on the study of - how the brain and nervous system work , and cognitive studies look at how people learn and reason . So a cognitive ne ##uro ##s ##cie ##nti ##st may study the brain processes that help people learn . In the study , 99 college students between the ages of 18 and 30 each spent an hour on a computer , trying to get through a virtual maze . The maze was difficult , and the study participants had to start in a different place each time they tried - making it even more difficult . They were also told to find a particular picture of a tree and remember where it was . For the first 90 minutes of a five - hour break , half of the particular ##ity stayed awake and half were told to take a short nap . Part ##ici ##pants who stayed awake were asked to describe their thoughts . Part ##ici ##pants who took a nap were asked about their dreams before sleep and after steep - and they were awakened within a minute of sleep to describe their dreams . About a dozen of the 50 people who slept said their dreams were connected to the maze . Some dreamed about the music that had been playing when they were working ; others said they dreamed about seeing people in the maze . When these people tried the computer maze again , they were generally able to find the tree faster than before their nap ##s . However , people who had other dreams , or people who didn ' t take a nap , didn ' t show the same improvement . Stick ##gold suggests the dream itself doesn ' t help a person learn - it ' s the other way around ."
    answer = 'see how dreams and learning are connected'
    questions = fx_infer.generate_question(article, answer, pred_len=64)

    fx_infer = RaceModule.load_from_checkpoint("models/ckpts/t5_dis.ckpt")
    fx_infer.eval()
    print("ANS", answer)
    print("QUE", questions)
    print("DIS", fx_infer.generate_distractor(article, answer, questions, pred_len=20))
