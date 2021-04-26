# Python Import:
import yaml
import torch
import numpy as np
from argparse import ArgumentParser
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
#     fx_model = RaceModule.load_from_checkpoint("models/ckpts/t5.ckpt")
    #fx_model.setup_tune(top_p = 0.95, top_k = 50, no_repeat_ngram_size = 2)
#     trainer.test(fx_model, test_dataloaders=fx_dm.test_dataloader())

    
        # Trainer:
    trainer = pl.Trainer.from_argparse_args(args)
    
    fx_dm.setup()
    
    ## I had to put it here 
    def fn_objective(b1, b2, b3, b4, m, r):
        return b1 + b2 + b3 + b4 + m + r

    def training_function(config):
    # Hyperparameters
        top_p, top_k, no_repeat_ngram_size = config["top_p"], config["top_k"], config["no_repeat_ngram_size"]
        fx_model = RaceModule.load_from_checkpoint("D:/Github/decepticon/project/models/ckpts/t5.ckpt")
        fx_model.setup_tune(top_p = top_p, top_k = top_k, no_repeat_ngram_size = no_repeat_ngram_size)
        result = trainer.test(fx_model, test_dataloaders=fx_dm.val_dataloader())
        result = result[0]
        
        score = fn_objective(result["bleu_1"], result["bleu_2"], result["bleu_3"], result["bleu_4"], result["meteor"], result["rouge_l"])
        # Feed the score back back to Tune.
        tune.report(total_score=score)
        
        
    config={
            "top_p": tune.uniform(0.75, 0.999),
            "top_k": tune.randint(25, 125),
            "no_repeat_ngram_size": tune.randint(0,4)
            }
    
    
    bohb_hyperband = HyperBandForBOHB(time_attr="training_iteration", max_t=100, reduction_factor=4, stop_last_trials=False)

    bohb_search = TuneBOHB(max_concurrent=4)

    analysis = tune.run(training_function,
                        name="bohb_test",
                        config=config,
                        scheduler=bohb_hyperband,
                        search_alg=bohb_search,
                        num_samples=10,
                        stop={"training_iteration": 100},
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




