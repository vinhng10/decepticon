# Python Import:
from argparse import ArgumentParser
from transformers import AutoTokenizer

# Pytorch Import:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Pytorch Lightning Import:
import pytorch_lightning as pl

# Internal Import:
from project.metrics.metrics import Input, Metrics


class RaceModule(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embed_dim", type=int, default=256)
        parser.add_argument("--bidirectional", type=bool, default=False)
        parser.add_argument("--dropout", type=float, default=0)
        parser.add_argument("--top_p", type=float, default=0.5)
        parser.add_argument("--hidden_size", type=int, default=256,
                            help="hidden_sz of the GRU")
        parser.add_argument("--num_layers", type=int, default=1,
                            help="Number of layers in the GRU.")

        parser.add_argument("--learning_rate", type=float, default=1e-3,
                            help="Learning rate.")
        parser.add_argument("--num_training_steps", type=int, default=1000,
                            help="The total number of training steps.")
        return parser

    @staticmethod
    def default_batch_fn(batch):
        """
        Description: from batch to x, y
        """
        art, que, ans = batch['articles']['input_ids'], batch['questions']['input_ids'], batch['answers']['input_ids']
        x, y = torch.cat([ans, art], dim=1).long(), que.long()
        return x, y

    @staticmethod
    def top_p_filtering(score, top_p):
        """ Args:
                score (bsz, vocab_size): output of the last layer
                top_p float value: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Returns:
                score (bsz, vocab_size): output after redistributing the prob with top-p
        """
        sorted_logits, sorted_indices = torch.sort(score, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=sorted_indices_to_remove.dtype).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        score[indices_to_remove] = -float('Inf')
        return score

    def __init__(self, hparams, batch_fn=None):
        """
        :param batch_fn: function to process batch
        """
        super(RaceModule, self).__init__()

        if batch_fn:
            self.batch_fn = batch_fn
        else:
            self.batch_fn = self.default_batch_fn

        self.hparams = hparams
        self.save_hyperparameters(hparams)

        # Tokenizer:
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pretrained_model)

        # Metrics:
        self.metrics = Metrics()

        # Encoder:
        num_directions = 2 if self.hparams.bidirectional else 1
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.hparams.special_tokens})
        vocab_size = self.tokenizer.vocab_size + len(self.hparams.special_tokens)
        self.embedding = nn.Embedding(vocab_size + 1, self.hparams.embed_dim)
        self.en_gru = nn.GRU(
            self.hparams.embed_dim,
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
            self.hparams.embed_dim,
            self.hparams.hidden_size,
            self.hparams.num_layers,
            dropout=self.hparams.dropout if self.hparams.num_layers else 0,
            batch_first=True
        )
        self.de_fc = nn.Linear(
            self.hparams.hidden_size,
            vocab_size
        )

    def encode(self, input, hidden=None):
        """ Args:
                input (batch, seq_len): Input sequence.
                hidden (num_layers*num_directions, batch, hidden_size): Initial states.

            Returns:
                output (batch, seq_len, num_directions*hidden_size): Outputs at every step.
                hidden (num_layers, batch, hidden_size): Final state.
        """
        x = self.embedding(input)
        outputs, hidden = self.en_gru(x, hidden)
        hidden = hidden.permute(1, 2, 0)
        hidden = self.en_fc(hidden)
        hidden = hidden.permute(2, 0, 1)
        return outputs, hidden.contiguous()

    def generate(self, x, pred_len):
        """ Args:
                x (batch, seq_len): Input sequence.
                pred_len (int): Length of predicted sequence.

            Returns:
                outputs (batch, pred_len)
        """
        return self(x, pred_len)

    def forward(self, x, pred_len=32, target=None, teacher_forcing=False):
        """ Args:
                x (batch, seq_len): Input sequence.
                pred_len (int): Length of predicted sequence.
                target (batch, seq_len, embed_dim): Target sequence. If None,
                    the output sequence is generated by feeding the output
                    of the previous timestep (teacher_forcing has to be False).
                teacher_forcing (bool): Boolean to apply teacher forcing.

            Returns:
                outputs (batch, seq_len, vocab_size): Tensor of log-probabilities
                    of words in the target language.
                hidden of shape (1, batch_size, hidden_size): New states of the GRU.
        """
        _, hidden = self.encode(x)

        if target is None:
            assert not teacher_forcing, 'Cannot use teacher forcing without a target sequence.'
        else:
            pred_len = target.shape[1]

        # Determine constants:
        batch = hidden.shape[1]
        # Initial value to feed to the GRU:
        x = torch.zeros((batch, 1), device=hidden.device).long()
        # Sequence to record the predicted values:
        outputs = list()
        ids = list()
        for i in range(pred_len):
            # Embed the value at ith time step:
            x = self.embedding(x)
            output, hidden = self.de_gru(x, hidden)
            output = self.de_fc(output)
            output_ls = F.log_softmax(output,dim=2)
            outputs.append(output_ls)
            # If teacher_forcing then use the target value at current step
            # Else use the predicted value at previous step:
            if teacher_forcing:
                x = target[:, i:i+1]
            else:
                output = self.top_p_filtering(output, top_p=self.hparams.top_p)
                prob = F.softmax(output, dim=2).squeeze(1)
                x = torch.multinomial(prob, 1)
                ids.append(x)
        if not teacher_forcing:
            return torch.cat(ids, dim=1)
        else:
            return torch.cat(outputs, dim=1), hidden

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=1e-1, patience=2, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = self.batch_fn(batch)
        y_hat, _ = self(x, None, y, True)
        y_hat = y_hat.reshape(-1, y_hat.shape[-1])
        y = y.reshape(-1)
        loss = F.nll_loss(y_hat, y, ignore_index=0)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", loss, logger=True)

    def validation_step(self, batch, batch_idx):
        val_loss = self.training_step(batch, batch_idx)["loss"]
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # Prepare data:
        inputs, targets = self.batch_fn(batch)

        # Generations:
        generations = self.generate(inputs, pred_len=64)

        # Compute metrics:
        predictions = [
            self.tokenizer.decode(generation, skip_special_tokens=True)
            for generation in generations
        ]

        references = [
            self.tokenizer.decode(target, skip_special_tokens=True)
            for target in targets
        ]

        inputs = Input(predictions=predictions, references=references)
        metrics = self.metrics.compute_metrics(inputs)

        return metrics
