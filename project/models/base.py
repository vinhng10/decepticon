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


class RaceBaseModel(pl.LightningModule):

    @staticmethod
    def default_batch_fn(batch):
        x, y = batch['inputs'], batch['targets'],
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
        super(RaceBaseModel, self).__init__()
        if batch_fn:
            self.batch_fn = batch_fn
        else:
            self.batch_fn = self.default_batch_fn

        self.hparams = hparams
        self.save_hyperparameters(hparams)

        # Tokenizer:
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.hparams.special_tokens})

        # Metrics:
        self.metrics = Metrics()

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

    def generate_sentence(self, article, answer, question=None, pred_len=64, sample_num=1, top_p=0.95, skip_special_tokens=True):
        """Args:
            article (str)
            answer (str)
            question (str): if not none, generating distractors
            pred_len (int):  Length of predicted sequence.
            sample_num (int): number of sample
            top_p (float): top_p for generation
            skip_special_tokens (bool): skip special_tokens while decoding
        :return:
            list of generated sentences, len(list) = sample_num
        """
        if not question:
            context = " ".join([answer, self.tokenizer.sep_token, article, self.tokenizer.sep_token, question])
        else:
            context = " ".join([answer,self.tokenizer.sep_token, article])
        inputs = self.tokenizer([context], padding=True, truncation=True, max_length=512, return_tensors="pt")
        questions = self.generate(inputs, pred_len, sample_num=sample_num, top_p=top_p)

        return [self.tokenizer.decode(question.squeeze(), skip_special_tokens) for question in questions]
