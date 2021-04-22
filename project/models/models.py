# Python Import:
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
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup

# Internal Import:
from project.metrics.metrics import Input, Metrics


class RaceModule(pl.LightningModule):
    """ Race Module """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--d_model", type=int, default=256,
                            help="Number of expected features in the decoder input.")
        parser.add_argument("--nhead", type=int, default=1,
                            help="Number of heads in multihead attention model.")
        parser.add_argument("--num_layers", type=bool, default=False,
                            help="Number of sub-layers in the decoder.")
        parser.add_argument("--learning_rate", type=float, default=1e-3,
                            help="Learning rate.")
        parser.add_argument("--m_pretrained_model", type=str, default="prajjwal1/bert-tiny",
                            help="Pretrained model.")
        parser.add_argument("--num_warmup_steps", type=int, default=0,
                            help="The number of steps for the warmup phase.")
        parser.add_argument("--num_training_steps", type=int, default=1000,
                            help="The total number of training steps.")
        return parser

    @staticmethod
    def generate_tgt_mask(size):
        """"""
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Tokenizer:
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.m_pretrained_model)

        # Metrics:
        self.metrics = Metrics()

        # Encoder:
        self.encoder = AutoModel.from_pretrained(hparams.m_pretrained_model)
        vocab_size = self.encoder.config.vocab_size + len(hparams.special_tokens)
        self.encoder.resize_token_embeddings(vocab_size)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder:
        self.embedding = self.encoder.embeddings
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hparams.d_model, nhead=hparams.nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, hparams.num_layers, nn.LayerNorm(hparams.d_model, 1e-12))

        # Head:
        self.head = nn.Linear(hparams.d_model, vocab_size)

    def encode(self, input):
        """"""
        # Encode:
        memory = self.encoder(**input).last_hidden_state.permute((1, 0, 2))

        return memory

    def forward(self, target, memory, input_key_padding_mask=None, memory_key_padding_mask=None):
        """"""
        # Decode:
        decode = self.decoder(
            tgt=self.embedding(target).permute((1, 0, 2)),
            memory=memory,
            tgt_mask=self.generate_tgt_mask(target.shape[1]).to(target.device),
            tgt_key_padding_mask=input_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Head:
        output = self.head(decode).permute((1, 2, 0))

        return output

    def configure_optimizers(self):
        """"""
        # Optimizer:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        # Scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer, self.hparams.num_warmup_steps, self.hparams.num_training_steps)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        """"""
        # Prepare data:
        inputs, targets = batch["inputs"], batch["targets"]

        # Forward pass:
        generated = self(
            target=targets["input_ids"][:, :-1],
            memory=self.encode(inputs),
            input_key_padding_mask=targets["attention_mask"][:, :-1] == 0,
            memory_key_padding_mask=inputs["attention_mask"] == 0
        )

        # Compute loss:
        loss = F.cross_entropy(generated, targets["input_ids"][:, 1:])

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        """"""
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", loss, logger=True)

    def validation_step(self, batch, batch_idx):
        """"""
        val_loss = self.training_step(batch, batch_idx)["loss"]
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        """"""
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # Prepare data:
        inputs, targets = batch["inputs"], batch["targets"]

        # Forward pass:
        generated = self(
            target=targets["input_ids"][:, :-1],
            memory=self.encode(inputs),
            input_key_padding_mask=targets["attention_mask"][:, :-1] == 0,
            memory_key_padding_mask=inputs["attention_mask"] == 0
        )

        predictions = [
            generated[for this]
        ]

        references = [
            self.tokenizer.decode(target, skip_special_tokens=True)
            for target in targets["input_ids"][:, 1:]
        ]

        # Compute metrics:
        inputs = Input(predictions=predictions, references=references)
        metrics = self.metrics.compute_metrics(inputs)

        return metrics







