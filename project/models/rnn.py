# Python Import:
from argparse import ArgumentParser
from transformers import AutoTokenizer

# Pytorch Import:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Pytorch Lightning Import:
import pytorch_lightning as pl

# Internal Import:
from metrics.metrics import Input, Metrics


class RaceModule(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--version", type=float,
                            help="specify it in a form X.XX")
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)

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

    def encode(self, inputs, hidden=None):
        """ Args:
                input (batch, seq_len): Input sequence.
                hidden (num_layers*num_directions, batch, hidden_size): Initial states.

            Returns:
                output (batch, seq_len, num_directions*hidden_size): Outputs at every step.
                hidden (num_layers, batch, hidden_size): Final state.
        """
        x = self.embedding(inputs)
        outputs, hidden = self.en_gru(x, hidden)
        hidden = hidden.permute(1, 2, 0)
        hidden = self.en_fc(hidden)
        hidden = hidden.permute(2, 0, 1)
        return outputs, hidden.contiguous()

    def decode(self, inputs, context):
        inputs = self.embedding(inputs)
        output, hidden = self.de_gru(inputs, context)
        output = self.de_fc(output)
        return output

    def generate(self, enc_input, pred_len, sample_num=1, top_p=0.95):
        """ Args:
                enc_input (batch, seq_len): Encoder Input seqs.
                pred_len (int): Length of predicted sequence.
                sample_num (int): Number of generation.
                top_p (float): top_p

            Returns:
                outputs list of (batch, pred_len)
        """
        def generate_token(dec_inputs, hidden, sample_num, top_p):
            output = self.decode(dec_inputs, hidden)
            output = self.top_p_filtering(output, top_p=top_p)
            prob = F.softmax(output, dim=2).squeeze(1)
            res = torch.multinomial(prob, sample_num)
            return res

        _, hidden = self.encode(enc_input)
        bsz = hidden.shape[1]
        dec_init = torch.zeros((bsz, 1), device=hidden.device).long()
        samples = torch.unbind(generate_token(dec_init, hidden, sample_num, top_p), dim=1)
        outputs = []
        for i in range(sample_num):
            token = samples[i].reshape(bsz, 1) # (bsz, 1)
            tmp = [dec_init, token]
            for _ in range(pred_len-2):
                token = generate_token(token, hidden, 1, top_p) # (bsz, 1)
                if all(token == self.tokenizer.pad_token_id):
                    break
                tmp.append(token)
            outputs.append(torch.cat(tmp, dim=1))

        return outputs

    def forward(self, enc_input, target):
        """ Args:
                enc_input (batch, seq_len): Encoder Input seqs.
                target (batch, seq_len, embed_dim): Target sequence. If None,
                    the output sequence is generated by feeding the output
                    of the previous timestep.

            Returns:
                outputs (batch, seq_len, vocab_size): Tensor of log-probabilities
                    of words in the target language.
                hidden of shape (1, batch_size, hidden_size): New states of the GRU.
        """
        _, hidden = self.encode(enc_input)
        dec_input = torch.zeros((hidden.shape[1], 1), device=hidden.device).long()
        outputs = list()

        for i in range(target.shape[1]):
            output = self.decode(dec_input, hidden)
            output = F.log_softmax(output,dim=2)
            outputs.append(output)
            dec_input = target[:, i:i+1]

        return torch.cat(outputs, dim=1), hidden

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=1e-1, patience=2, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = self.batch_fn(batch)
        y_hat, _ = self(x, y)
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
        generations_list = self.generate(inputs, pred_len=64, sample_num=2)

        # Compute metrics:

        references = [
            self.tokenizer.decode(target, skip_special_tokens=True)
            for target in targets
        ]

        # Multiple generations:

        metrics = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge_l']
        final_metrics = dict(zip(metrics, [0]*len(metrics)))

        for generations in generations_list:

            predictions = [
                self.tokenizer.decode(generation, skip_special_tokens=True)
                for generation in generations
            ]

            inputs = Input(predictions=predictions, references=references)
            metrics = self.metrics.compute_metrics(inputs)

            for k in metrics:
                final_metrics[k] += metrics[k]

        for k in metrics:
            final_metrics[k] /= len(generations_list)

        # Log:
        self.log_dict(final_metrics)

        return final_metrics

    def generate_question(self, article, answer, pred_len=64, sample_num=1, top_p=0.95):
        context = " ".join([answer,self.tokenizer.sep_token, article])
        inputs = self.tokenizer([context], padding=True, truncation=True, max_length=512, return_tensors="pt")['input_ids']
        questions = self.generate(inputs, pred_len, sample_num=sample_num, top_p=top_p)

        return [self.tokenizer.decode(question.squeeze(), True) for question in questions]

    def generate_distractor(self, article, answer, question,  pred_len=64, sample_num=1, top_p=0.95):
        context = " ".join([answer, self.tokenizer.sep_token, article, self.tokenizer.sep_token, question])
        inputs = self.tokenizer([context], padding=True, truncation=True, max_length=512, return_tensors="pt")[
            'input_ids']
        distractors = self.generate(inputs, pred_len, sample_num=sample_num, top_p=top_p)

        return [self.tokenizer.decode(distractor.squeeze(), False) for distractor in distractors]
