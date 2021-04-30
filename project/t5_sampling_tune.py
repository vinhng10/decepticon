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
from pytorch_lightning.loggers import TensorBoardLogger
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
    ##for RAY
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
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

    fx_dm.setup()

    def training_function(config):
        # Hyperparameters
        args.top_p, args.hidden_size = config["top_p"], config["hidden_size"]
        fx_model = RaceModule(args, batch_fn)
        trainer.fit(fx_model, fx_dm)
        result = trainer.test(fx_model, test_dataloaders=fx_dm.val_dataloader())
        result = result[0]
        score = sum([result["bleu_1"], result["bleu_2"], result["bleu_3"], result["bleu_4"], result["meteor"], result["rouge_l"]])
        # Feed the score back back to Tune.
        del fx_model
        gc.collect()
        tune.report(total_score=score)
        
        
    config={
            "top_p": tune.uniform(0.80, 0.999),
            "hidden_size": tune.randint(128, 512)
            }
    
    
    bohb_hyperband = HyperBandForBOHB(time_attr="training_iteration", max_t=100, reduction_factor=2)

    bohb_search = TuneBOHB(max_concurrent=2)

    analysis = tune.run(training_function,
                        name="bohb_test",
                        config=config,
                        scheduler=bohb_hyperband,
                        search_alg=bohb_search,
                        num_samples=12,
                        metric="total_score",
                        mode="max", 
                        resources_per_trial={'gpu': 1},
                        callbacks = [CSVLoggerCallback(), JsonLoggerCallback()])

    print("Best hyperparameters found were: ", analysis.best_config)
    
#     analysis = tune.run(
#         training_function, resources_per_trial={'gpu': 1},
#         callbacks = [CSVLoggerCallback(), JsonLoggerCallback()],
#         config=config)
    
#     print("Best config: ", analysis.get_best_config(metric="total_score", mode="max"))
    
    
#     trainer.test(fx_model, test_dataloaders=fx_dm.test_dataloader())




