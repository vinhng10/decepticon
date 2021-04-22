# Python Import:
import torch
import numpy as np
from argparse import ArgumentParser

# Pytorch Lightning Import:
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from data.data import RaceDataModule


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


def transformer_collate_fn(batch, tokenizer):
    con_token, que_token, ans_token, dis_token = tokenizer.additional_special_tokens

    inputs = []
    targets = []

    for item in batch:
        inputs.append(" ".join([con_token, item["article"], ans_token, item["answer"]]))
        targets.append(" ".join([que_token, item["question"], dis_token, dis_token.join(item["distractors"])]))

    return {
        "inputs": tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"),
        "targets": tokenizer(targets, padding=True, truncation=True, return_tensors="pt"),
    }


def transfomer_batch_fn(batch):
    pass


def rnn_batch_fn(batch):
    """
    Description: from batch to x, y
    """
    art, que, ans = batch['articles']['input_ids'], batch['questions']['input_ids'], batch['answers']['input_ids']
    x, y = torch.cat([ans, art], dim=1).long(), que.long()
    return x, y


def rnn_dis_batch_fn(batch):
    art, que, ans, dis = batch['articles']['input_ids'], batch['questions']['input_ids'], batch['answers']['input_ids'], batch["distractors"]['input_ids']
    x, y = torch.cat([que, ans, art], dim=1).long(), dis.long()
    return x, y


if __name__ == "__main__":
    # TODO Merge t5

    # Choose the model
    # from models.transformer import RaceModule
    from models.rnn import RaceModule

    batch_fn = None
    collate_fn = None

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = RaceDataModule.add_model_specific_args(parser)
    parser = RaceModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args("--data_path data/RACE_processed \
                              --batch_size 32 \
                              --num_workers 0 \
                              --top_p 0.3 \
                              --hidden_size 128 \
                              --learning_rate 1e-5 \
                              --special_tokens [CON] [QUE] [ANS] [DIS] \
                              --gpus 1 \
                              --max_epochs 1 \
                              --check_val_every_n_epoch 1".split())
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

    fx_dm = RaceDataModule(args, collate_fn)
    fx_model = RaceModule(args, batch_fn)

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

    fx_infer = RaceModule.load_from_checkpoint(checkpoint.best_model_path)
    fx_infer.eval()
    fx_dm.setup()

    with torch.no_grad():
        for batch in fx_dm.test_dataloader():
            # ans = batch['inputs']['input_ids']
            ans = batch['answers']['input_ids']
            if batch_fn:
                x, y = batch_fn(batch)
            else:
                x, y = fx_infer.batch_fn(batch)
            # out = fx_infer(x, pred_len=50)
            out, _ = fx_infer(x)
            # translate(fx_dm.tokenizer, ans, out, y['input_ids'])
            translate(fx_dm.tokenizer, ans, out, y)
