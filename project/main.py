# Python Import:
import torch
import numpy as np
from argparse import ArgumentParser

# Pytorch Lightning Import:
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from data.data import RaceDataModule
from models.model import RaceRNNModule


def translate(vocab, ans, output, tgt):
    """
    Args:
        vocab dictionary of [index, word]
        ans (seq_len) np.array
        tgt (seq_len) np.array
        output (seq_len, vocab_size) np.array
    """
    output = np.argmax(output, axis=1)
    ans_str = ' '.join([vocab[i] for i in ans if i != 0])
    tgt_str = ' '.join([vocab[i] for i in tgt if i != 0])
    out_str = ' '.join([vocab[i] for i in output if i != 0])
    print("\n============================")
    print("ANS:", ans_str)
    print("TGT:", tgt_str)
    print("OUT:", out_str)


if __name__ == "__main__":
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = RaceDataModule.add_model_specific_args(parser)
    parser = RaceRNNModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args("--data_path data/RACE_processed \
                              --batch_size 16 \
                              --num_workers 0 \
                              --hidden_size 128 \
                              --learning_rate 1e-5 \
                              --special_tokens [CON] [QUE] [ANS] [DIS] \
                              --gpus 1 \
                              --max_epochs 5 \
                              --check_val_every_n_epoch 1".split())

    fx_dm = RaceDataModule(args)
    fx_model = RaceRNNModule(args)

    # Callbacks:
    checkpoint = ModelCheckpoint(
        dirpath='models/ckpts/',
        filename="./fx-{epoch:02d}-{val_loss:.7f}",
        monitor="val_loss"
    )

    # Logger:
    logger = TensorBoardLogger('models/logs/')

    # Trainer:
    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint,
        logger=logger
    )
    trainer.fit(fx_model, fx_dm)

    fx_infer = RaceRNNModule.load_from_checkpoint(checkpoint.best_model_path)
    vocab = {v:k for k,v in fx_dm.tokenizer.get_vocab().items()}
    fx_infer.eval()
    fx_dm.setup()
    with torch.no_grad():
      for batch in fx_dm.test_dataloader():
        art, que, ans = batch['articles']['input_ids'], batch['questions']['input_ids'], batch['answers']['input_ids']
        x, y = torch.cat([ans, art], dim=1).long(), que.long()
        out, _ = fx_infer(x)
        translate(vocab,ans, out, y)
