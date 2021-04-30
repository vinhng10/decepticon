# Python Import:
import yaml
import torch
import numpy as np
from argparse import ArgumentParser
import os
import gc

# Pytorch Lightning Import:
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

# Ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback


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
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    from models.rnn import RaceModule
    #from models.t5 import RaceModule

    batch_fn = rnn_batch_fn
    collate_fn = None

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = RaceDataModule.add_model_specific_args(parser)
    parser = RaceModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    config_start = yaml.load(open("configs/rnn.yaml"), Loader=yaml.FullLoader)
    
    def fn_tune(config, config_init, num_epochs=10, num_gpus=1):
        gc.collect()
        config_init["batch_size"] = config["batch_size"]
        #config_init["learning_rate"] = config["learning_rate"]
        config_init["hidden_size"] = config["hidden_size"]
        config_init["bidirectional"] = config["bidirectional"]
        config_init["embed_dim"] = config["embed_dim"]
        config_init["num_layers"] = config["num_layers"]
        args = parser.parse_args(serialize_config(config_init))

        fx_dm = RaceDataModule(args, collate_fn)
        fx_dm.setup()
        fx_model = RaceModule(args, batch_fn)
        metrics = {"loss": "val_loss"}
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            limit_train_batches=0.25,
            precision = 16,
            gpus=num_gpus,
            progress_bar_refresh_rate=100,
            callbacks=[TuneReportCallback(metrics, on="validation_end")])
        
        trainer.fit(fx_model, fx_dm)
    
    current_best_params = [{"hidden_size": 320,
        "bidirectional": "True",
        "batch_size": 8,
        "embed_dim": 578,
        "num_layers": 2}]
    
    tune_config = {
        "hidden_size": tune.choice([128, 256, 320, 512, 780 , 1024]),
        "bidirectional": tune.choice(["True", "False"]),
        #"learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([8, 16, 32, 64, 96]),
        "embed_dim": tune.choice([128, 256, 320, 512, 578]),
        "num_layers": tune.choice([1,2])
        }
    
    scheduler = ASHAScheduler(time_attr='training_iteration', grace_period=2)
    
    hyper = HyperOptSearch(metric="loss", mode="min", points_to_evaluate=current_best_params)
    
    reporter = CLIReporter(
        parameter_columns=["hidden_size", "bidirectional", "learning_rate", "batch_size", "embed_dim", "num_layers"],
        metric_columns=["loss", "training_iteration"])
    

    analysis = tune.run(tune.with_parameters(fn_tune, config_init=config_start, num_epochs = 7, num_gpus=1),
                        name="rnn_tune",
                        config=tune_config,
                        scheduler=scheduler,
                        num_samples=10,
                        metric="loss",
                        mode="min", 
                        resources_per_trial={'gpu': 1},
                        progress_reporter=reporter,
                        search_alg=hyper,
                        callbacks = [CSVLoggerCallback(), JsonLoggerCallback()])

    print("Best hyperparameters found were: ", analysis.best_config)


    
    
    
    
    
    
    
    