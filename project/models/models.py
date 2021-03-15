# Python Import:
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from argparse import ArgumentParser

# Pytorch Import:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

# Pytorch Lightning Import:
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# HuggingFace Import:
from transformers import AutoTokenizer, AutoModel

class RaceDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return None


class RaceDataModule(pl.LightningDataModule):
    """"""
    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Args:
                data_dir (str): Data directory.
                batch_size (int): Bacth size.
                source_len (int): Length of input sequence.
                target_len (int): Length of target sequence.
                step (int): Window size.
                test_size (float): Percentage of test dataset.
                val_size (float): Percentage of valid dataset (exclude test dataset).
                num_workers (int): Number of workers for data loading.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--source_len", type=int, default=192)
        parser.add_argument("--target_len", type=int, default=32)
        parser.add_argument("--step", type=int, default=1)
        parser.add_argument("--test_size", type=float, default=0.3)
        parser.add_argument("--val_size", type=float, default=0.25)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--pretrained_model", type=str, default="prajjwal1/bert-tiny")
        return parser

    def __init__(self, hparams):
        super(RaceDataModule, self).__init__()
        self.hparams = hparams
        self.tokenizer = AutoTokenizer(hparams.pretrained_model)
        self.backbone = AutoModel(hparams.pretrained_model)

    def prepare_data(self):

    def setup(self, stage=None):

        # Prepare datasets
        self.trainset = RaceDataset()
        self.valset = RaceDataset()
        self.testset = RaceDataset()

    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_loader = DataLoader(
            self.valset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        return self.test_loader


class RaceModule(pl.LightningModule):
    """"""
    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Args:
                source_size (int): The expected number of features in the input.
                target_size (int): The expected number of sequence features.
                hidden_size (int): The number of features in the hidden state.
                num_layers (int): Number of recurrent layers.
                bidirectional (boolean): whether to use bidirectional model.
                dropout (float): dropout probability.
                lr (float): Learning rate.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--source_size", type=int, default=4)
        parser.add_argument("--target_size", type=int, default=4)
        parser.add_argument("--hidden_size", type=int, default=256)
        parser.add_argument("--num_layers", type=int, default=1)
        parser.add_argument("--bidirectional", type=bool, default=False)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser

    def __init__(self, hparams):
        super(RaceModule, self).__init__()
        self.hparams = hparams
        # Encoder:
        num_directions = 2 if self.hparams.bidirectional else 1
        self.en_gru = nn.GRU(
            self.hparams.source_size,
            self.hparams.hidden_size,
            self.hparams.num_layers,
            dropout=self.hparams.dropout if self.hparams.num_layers else 0,
            bidirectional=self.hparams.bidirectional,
            batch_first=True
        )
        self.en_fc = nn.Linear(
            self.hparams.num_layers*num_directions,
            self.hparams.num_layers
        )

        # Decoder:
        self.de_gru = nn.GRU(
            self.hparams.target_size,
            self.hparams.hidden_size,
            self.hparams.num_layers,
            dropout=self.hparams.dropout if self.hparams.num_layers else 0,
            batch_first=True
        )
        self.de_fc = nn.Linear(
            self.hparams.hidden_size,
            self.hparams.target_size
        )

    def encode(self, input, hidden=None):
        """ Args:
                input (batch, seq_len, source_size): Input sequence.
                hidden (num_layers*num_directions, batch, hidden_size): Initial states.

            Returns:
                output (batch, seq_len, num_directions*hidden_size): Outputs at every step.
                hidden (num_layers, batch, hidden_size): Final state.
        """
        # Feed source sequences into GRU:
        outputs, hidden = self.en_gru(input, hidden)
        # Compress bidirection to one direction for decoder:
        hidden = hidden.permute(1, 2, 0)
        hidden = self.en_fc(hidden)
        hidden = hidden.permute(2, 0, 1)
        return outputs, hidden.contiguous()

    def forward(self, hidden, pred_len=32, target=None, teacher_forcing=0.0):
        """ Args:
                hidden (num_layers, batch, hidden_size): States of the GRU.
                pred_len (int): Length of predicted sequence.
                target (batch, seq_len, target_size): Target sequence. If None,
                    the output sequence is generated by feeding the output
                    of the previous timestep (teacher_forcing has to be False).
                teacher_forcing (float): Probability to apply teacher forcing.

            Returns:
                outputs (batch, seq_len, target_size): Tensor of log-probabilities
                    of words in the target language.
                hidden of shape (1, batch_size, hidden_size): New states of the GRU.
        """
        if target is None:
            assert not teacher_forcing, 'Cannot use teacher forcing without a target sequence.'

        # Determine constants:
        batch = hidden.shape[1]
        # Initial value to feed to the GRU:
        val = torch.zeros((batch, 1, self.hparams.target_size), device=hidden.device)
        if target is not None:
            target = torch.cat([val, target[:, :-1, :]], dim=1)
            pred_len = target.shape[1]
        # Sequence to record the predicted values:
        outputs = list()
        for i in range(pred_len):
            # Embed the value at ith time step:
            # If teacher_forcing then use the target value at current step
            # Else use the predicted value at previous step:
            val = target[:, i:i+1, :] if (np.random.rand() < teacher_forcing) else val
            # Feed the previous value and the hidden to the network:
            output, hidden = self.de_gru(val, hidden)
            # Predict new output:
            val = self.de_fc(output.relu()).sigmoid()
            # Record the predicted value:
            outputs.append(val)
        # Concatenate predicted values:
        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=1e-1, patience=2, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, h = self.encode(x)
        y_hat, _ = self(h, None, y, 1.0)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", loss, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, h = self.encode(x)
        y_hat, _ = self(h, None, y, 0.0)
        val_loss = F.mse_loss(y_hat, y)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True, logger=True)


if __name__ == "__main__":
    # Argument parser:
    parser = ArgumentParser()
    parser = RaceDataModule.add_model_specific_args(parser)
    parser = RaceModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Model & data module:
    fx_dm = RaceDataModule(args)
    fx_model = RaceModule(args)

    # Callbacks:
    checkpoint = ModelCheckpoint(
        filepath="./checkpoint/fx-{epoch:02d}-{val_loss:.7f}",
        monitor="val_loss"
    )

    # Logger:
    logger = TensorBoardLogger('logs/')

    # Trainer:
    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint,
        logger=logger
    )
    trainer.fit(fx_model, fx_dm)

    fx_infer = RaceModule.load_from_checkpoint(checkpoint.best_model_path)
    fx_infer.eval()
    print(fx_infer)

