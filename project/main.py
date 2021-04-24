# Python Import:
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
    rnn_batch_fn, rnn_dis_batch_fn
)


def translate(tokenizer, ans, output, tgt):
    """
    Args:
        vocab dictionary of [index, word]
        ans (bsz, seq_len) Tensor
        tgt (bsz, seq_len) Tensor
        output (bsz, seq_len, vocab_size) OR (bsz, seq_len) Tensor
    """
    ans = ans[0, :].long().numpy()
    tgt = tgt[0, :].long().numpy()
    if len(output.shape) == 3:
        output = output[0, :, :].numpy()
        output = np.argmax(output, axis=1)
    else:
        output = output[0, :].long().numpy()
    ans_str = ' '.join(tokenizer.convert_ids_to_tokens(ans, True))
    tgt_str = ' '.join(tokenizer.convert_ids_to_tokens(tgt, True))
    out_str = ' '.join(tokenizer.convert_ids_to_tokens(output, True))
    print("\n============================")
    print("ANS:", ans_str)
    print("TGT:", tgt_str)
    print("OUT:", out_str)


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
    args = parser.parse_args("--data_path data/RACE_processed \
                              --num_workers 0 \
                              --batch_size 2 \
                              --version 8.8 \
                              --pretrained_model t5-small \
                              --auto_lr_find False \
                              --terminate_on_nan True \
                              --benchmark  True \
                              --track_grad_norm 2 \
                              --precision 16 \
                              --accumulate_grad_batches 10 \
                              --gradient_clip_val 5 \
                              --stochastic_weight_avg True\
                              --enable_pl_optimizer True \
                              --learning_rate 1e-5 \
                              --special_tokens [CON] [QUE] [ANS] [DIS] \
                              --gpus 1 \
                              --max_epochs 1 \
                              --log_every_n_steps 150 \
                              --check_val_every_n_epoch 1".split())

    # Rnn args:
    """
    "--data_path data/RACE_processed \
                              --batch_size 32 \
                              --num_workers 0 \
                              --top_p 0.3 \
                              --hidden_size 128 \
                              --learning_rate 1e-5 \
                              --special_tokens [CON] [QUE] [ANS] [DIS] \
                              --gpus 1 \
                              --max_epochs 1 \
                              --check_val_every_n_epoch 1".split()
    """
    # Transformer args:
    """
    "--data_path data/RACE_processed \
                              --d_model 128 \
                              --nhead 8 \
                              --batch_size 1 \
                              --num_workers 0 \
                              --num_layers 1 \
                              --top_p 0.5 \
                              --learning_rate 1e-5 \
                              --special_tokens [CON] [QUE] [ANS] [DIS] \
                              --pretrained_model prajjwal1/bert-tiny \
                              --gpus 1 \
                              --max_epochs 1 \
                              --check_val_every_n_epoch 1".split()
    """
    # T5 args:
    """

    "--data_path data/RACE_processed \
                              --batch_size 1 \
                              --version 0.0 \
                              --pretrained_model t5-small \
                              --top_p 0.5 \
                              --auto_lr_find False \
                              --terminate_on_nan True \
                              --benchmark  True \
                              --track_grad_norm 2 \
                              --precision 16 \
                              --gradient_clip_val 5 \
                              --stochastic_weight_avg True\
                              --pl_optimizer True \
                              --learning_rate 1e-5 \
                              --special_tokens [CON] [QUE] [ANS] [DIS] \
                              --gpus 1 \
                              --max_epochs 1 \
                              --log_every_n_steps 150 \
                              --check_val_every_n_epoch 1".split()
    """

    fx_dm = RaceDataModule(args, collate_fn)
    fx_model = RaceModule(args, batch_fn)

    # Callbacks:
    checkpoint = ModelCheckpoint(
        dirpath='models/ckpts/',
        filename="./fx-{epoch:02d}-{val_loss:.7f}",
        # filename = str(hparams.version).replace(".", "_"))
        monitor="val_loss"
    )
    earlystopping = EarlyStopping(monitor='val_perplexity',
                                  min_delta=0.1,
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
        callbacks=earlystopping,
        # early_stopping_callback=earlystopping, # TODO no idea how to add early_stopping here
        logger=logger
    )
    fx_dm.setup()
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
