# Python Import:
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import (
    T5ForConditionalGeneration, T5Config, AutoTokenizer, AdamW, BertForSequenceClassification

)

from metrics.metrics import Input, Metrics


class RaceModule(pl.LightningModule):
    """ T5 Model """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--version", type=float,
                            help="specify it in a form X.XX")
        parser.add_argument("--padding_token", type=int, default=0,
                            help="don't change it")
        parser.add_argument("--tokenizer_len", type=int, default=32104,
                            help="don't touch it")
        parser.add_argument("--seed", default=1234, type=float)
        parser.add_argument("--weight_decay", default=5e-5, type=float)
        parser.add_argument("--learning_rate", default=1e-4, type=float)

        return parser

    @staticmethod
    def default_batch_fn(batch):
        x, y = batch
        return x, y

    def __init__(self, hparams, batch_fn=None):
        super(RaceModule, self).__init__()
        self.hparams = hparams
        self.save_hyperparameters(hparams)

        if batch_fn:
            self.batch_fn = batch_fn
        else:
            self.batch_fn = self.default_batch_fn

        if self.hparams.pretrained_model in ["t5-base","t5-small"]:
            # Model:
            config = T5Config(decoder_start_token_id = self.hparams.padding_token)
            self.model = T5ForConditionalGeneration(config).from_pretrained(self.hparams.pretrained_model)
            # Tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
            self.tokenizer_.add_special_tokens({"additional_special_tokens": ["[CON]","[QUE]","[ANS]","[DIS]"]})

            # Metrics:
            self.metrics = Metrics()
            try:
                self.model.resize_token_embeddings(self.hparams.tokenizer_len)
            except:
                self.model.resize_token_embeddings(32104)
        else:
            raise NotImplementedError
            
            
    def setup_tune(self, top_k: int = 50, top_p: float = 0.95, no_repeat_ngram_size: int = 2, num_samples = 5):
        """"""
        self.top_k = top_k ##1 75
        self.top_p = top_p ##2 0.9
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.num_samples = num_samples

    def mask_label_padding(self, labels):
        """"""
        MASK_ID = -100
        labels[labels == self.hparams.padding_token] = MASK_ID
        return labels

    def generate(self, inputs, use_beam=False, use_sample=True, **kwargs):
        """ Args:
            inputs dict: dict of input
            kwargs: for generation

            Returns:
                id_seqs (bsz, pred_len)
        """
        assert use_beam or use_sample, 'Must use one method for generation'
        if use_beam:
            return self.generate_with_beam(inputs, **kwargs)
        if use_sample:
            return self.generate_with_sampling(inputs, **kwargs)

    def forward(self, ids, mask, labels):
        """"""
        return self.model(input_ids=ids, attention_mask=mask, labels=labels)

    def configure_optimizers(self):
        """"""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        fn_lambda = lambda epoch: 0.95 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [fn_lambda, fn_lambda])

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        """"""
        x, y = self.batch_fn(batch)
        output = self(x["input_ids"], x["attention_mask"], self.mask_label_padding(y["input_ids"]))
        loss = output.loss

        self.logger.experiment.log_metric('train_loss', loss.detach())
        self.logger.experiment.log_metric('train_perplexity', torch.exp(loss.detach()))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """"""
        val_loss = self.training_step(batch, batch_idx)['loss']
        # logger
        self.logger.experiment.log_metric('val_loss', val_loss.detach())
        self.logger.experiment.log_metric('val_perplexity', torch.exp(val_loss.detach()))
        self.log('val_perplexity', torch.exp(val_loss.detach()))
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        """"""
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """"""
        # Prepare data:
        x, y = batch

        # Generations:
        generations = self.generate(
            inputs=x,
            use_beam = False,
            use_sample=True,
            max_length=64,
        )
        try:
            if self.num_samples > 1:
                y["input_ids"] = torch.repeat_interleave(y["input_ids"], expand_size, dim = 0)
        except:
            pass

        predictions = [
            self.tokenizer.decode(generation, skip_special_tokens=True)
            for generation in generations
        ]

        references = [
            self.tokenizer.decode(target, skip_special_tokens=True)
            for target in y["input_ids"]
        ]

        # Compute metrics:
        inputs = Input(predictions=predictions, references=references)
        metrics = self.metrics.compute_metrics(inputs)

        # Log:
        self.log_dict(metrics)

        return metrics

    def generate_with_beam(self, inputs,
                           num_beams: int = 6,
                           no_repeat_ngram_size: int = 2,
                           max_length: int = 64,
                           early_stopping: bool = True,
                           num_beam_groups: int = 2):
        """"""

        generated = self.model.generate(**inputs, # context -> answer + article
                                        num_beams=num_beams,
                                        num_beam_groups=num_beam_groups,
                                        max_length=max_length,
                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                        early_stopping=early_stopping)

        return generated

    def generate_with_sampling(self, inputs,
                               top_k: int = 50, ##1 75
                               top_p: float = 0.95, ##2 0.9
                               max_length: int = 64,
                               do_sample: bool = True,
                               no_repeat_ngram_size: int = 2,
                               num_samples = 1):
        """"""
        # [bsz, pred_len]
        try: ## for tuning
#             "Tunning is ON"
            top_k = self.top_k ##1 75
            top_p = self.top_p ##2 0.9
            no_repeat_ngram_size = self.no_repeat_ngram_size
            num_samples = self.num_samples
        except:
            pass
        
        generated = self.model.generate(**inputs, # context -> answer + article
                                        max_length=max_length,
                                        do_sample=do_sample,
                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                        num_return_sequences=num_samples,
                                        top_k=top_k,
                                        top_p=top_p)
        return generated


    def generate_question(self, article, answer, pred_len=64):
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        tokenizer.add_special_tokens({"additional_special_tokens": self.hparams.special_tokens})
        context = " ".join(['[ANS]', answer, '[CON]', article])
        inputs = tokenizer([context], padding=True, truncation=True, max_length=512, return_tensors="pt")
        question = self.generate(inputs, use_sample=True)

        return tokenizer.decode(question.squeeze(), True)

    def generate_distractor(self, article, answer, question, pred_len=64):
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        tokenizer.add_special_tokens({"additional_special_tokens": self.hparams.special_tokens})
        context = " ".join(['[ANS]', answer, '[QUE]', question, '[CON]', article])
        inputs = tokenizer([context], padding=True, truncation=True, max_length=512, return_tensors="pt")
        question = self.generate(inputs, use_sample=True)

        return tokenizer.decode(question.squeeze(), True)