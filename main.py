import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from project.data.data import RaceDataModule
from project.models.models import RaceModule


"""--data_path LON --batch_size 16 --num_workers 6 --d_model 128 --nhead 8 --num_layers 1 --learning_rate 1e-5 --special_tokens [CON] [QUE] [ANS] [DIS] --dm_pretrained_model prajjwal1/bert-tiny --m_pretrained_model prajjwal1/bert-tiny --gpus 1 --max_epochs 5 --check_val_every_n_epoch 1"""

########################################################################################

# Parse arguments:
parser = ArgumentParser()
parser = RaceDataModule.add_model_specific_args(parser)
parser = RaceModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()


# Set random seed:
pl.seed_everything(1234)


# Module and data module:
def customed_collate_fn(batch, tokenizer):
    con_token, que_token, ans_token, dis_token = tokenizer.additional_special_tokens

    inputs = []
    targets = []

    for item in batch:
        inputs.append(" ".join([con_token, item["article"], ans_token, item["answer"]]))
        targets.append(" ".join([que_token, item["question"], dis_token, dis_token.join(item["distractors"])]))

    print(tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")["input_ids"])
    return {
        "inputs": tokenizer(inputs, padding=True, truncation=True, return_tensors="pt"),
        "targets": tokenizer(targets, padding=True, truncation=True, return_tensors="pt"),
    }


data_module = RaceDataModule(args, customed_collate_fn)
module = RaceModule(args)


# Callbacks:
checkpoint = ModelCheckpoint(
    dirpath="./checkpoint/fx-{epoch:02d}-{val_loss:.7f}",
    monitor="val_loss"
)


# Trainer:
trainer = pl.Trainer.from_argparse_args(
    args,
    checkpoint_callback=checkpoint
)


trainer.fit(module, data_module)
