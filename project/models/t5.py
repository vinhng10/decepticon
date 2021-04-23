import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import (
    T5ForConditionalGeneration, T5Config, AutoTokenizer,
    get_linear_schedule_with_warmup, AdamW
)

from project.metrics.metrics import Input, Metrics


class T5FinetuneForRACE(pl.LightningModule):
    """ T5 Model """

    def __init__(self, hparams):
        super(T5FinetuneForRACE, self).__init__()
        # self.save_hyperparameters()
        self.hparams = hparams
        self.learning_rate = self.hparams.learning_rate

        if self.hparams.pretrained_model in ["t5-base", "t5-small"]:
            # Model:
            config = T5Config(decoder_start_token_id=self.hparams.padding_token)
            self.model = T5ForConditionalGeneration(config).from_pretrained(self.hparams.pretrained_model)

            # Tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(hparams.pretrained_model)

            # Metrics:
            self.metrics = Metrics()

            try:
                self.model.resize_token_embeddings(hparams.tokenizer_len)
            except:
                self.model.resize_token_embeddings(32102)
        else:
            raise NotImplementedError

    def mask_label_padding(self, labels):
        MASK_ID = -100
        labels[labels == self.hparams.padding_token] = MASK_ID
        return labels

    def decode(self, sequence):
        """"""
        return [self.tokenizer.decode(token) for token in sequence]

    def forward(self, ids, mask, labels):
        """"""
        return self.model(input_ids=ids, attention_mask=mask, labels=labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x["input_ids"], x["attention_mask"], self.mask_label_padding(y["input_ids"]))
        loss = output.loss

        # logger
        # if self.trainer.global_step % 50 == 0:
        self.logger.experiment.log_metric('train_loss', loss.detach())
        self.logger.experiment.log_metric('train_perplexity', torch.exp(loss.detach()))
        return loss

    def on_test_epoch_end(self):
        """"""
        self.logger.experiment.log_artifact('checkpoints/')

    def validation_step(self, batch, batch_idx):
        """"""
        x, y = batch
        output = self(x["input_ids"], x["attention_mask"], self.mask_label_padding(y["input_ids"]))
        loss = output[0]
        # logger
        self.logger.experiment.log_metric('val_loss', loss.detach())
        self.logger.experiment.log_metric('val_perplexity', torch.exp(loss.detach()))
        self.log('val_perplexity', torch.exp(loss.detach()))
        return loss

    def test_step(self, batch, batch_idx):
        """"""
        # Prepare data:
        x, y = batch

        # Generations:
        generations = self.generate(
            inputs=inputs,
            pred_len=64,
            memory_key_padding_mask=inputs["attention_mask"] == 0
        )

        predictions = [
            self.tokenizer.decode(generation, skip_special_tokens=True)
            for generation in generations
        ]

        references = [
            self.tokenizer.decode(target, skip_special_tokens=True)
            for target in targets["input_ids"]
        ]

        # Compute metrics:
        inputs = Input(predictions=predictions, references=references)
        metrics = self.metrics.compute_metrics(inputs)

        # Log:
        self.log_dict(metrics)

        return metrics

    def configure_optimizers(self):
        """"""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = ({'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                                         'weight_decay': self.hparams.weight_decay},
                                        {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                                         'weight_decay': 0.0})

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=min(1e-4, self.learning_rate))
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=100,
#             # Default value in run_glue.py
#             num_training_steps=50000)

        return optimizer#, scheduler

    def generate_questions(self, context, decode: bool = True, num_return_sequences: int = 6,
                           num_beams: int = 6, no_repeat_ngram_size: int = 2):
        """"""
        assert num_return_sequences <= num_beams, "Value is too large"
        NUM_RETURN_SEQ = num_return_sequences

        generated = self.model.generate(input_ids = context, # context -> answer + article
                                        num_beams = num_beams,
                                        num_beam_groups = 2,
                                        num_return_sequences = NUM_RETURN_SEQ,
                                        max_length = 30,
                                        no_repeat_ngram_size =  no_repeat_ngram_size,
                                        early_stopping = False)

        generated = generated # returnx [batch * NUM_RETURN_SEQ x MAX_LENGTH]

        if decode:
            output = self.datamodule.tokenizer.batch_decode(generated,
                                                            skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=True)
            chunked_list = [output[i * NUM_RETURN_SEQ:(i + 1) * NUM_RETURN_SEQ] \
                            for i in range((len(output) + NUM_RETURN_SEQ - 1) // NUM_RETURN_SEQ )] # chunk into [batch x NUM_RETURN]
            return chunked_list
        else:
            return generated.view(context.shape[0], NUM_RETURN_SEQ, -1)

    def generate_top(self, context, decode: bool = True, num_return_sequences=5):
        NUM_RETURN_SEQ = num_return_sequences

        generated = self.model.generate(input_ids = context, # context -> answer + article
                                        num_beams = None,
                                        num_return_sequences = NUM_RETURN_SEQ,
                                        do_sample = True,
                                        no_repeat_ngram_size = 2,
                                        top_k = 75,
                                        top_p = 0.9, ## the more - the less diverse
                                        early_stopping = True)

        generated = generated # returnx [batch * NUM_RETURN_SEQ x MAX_LENGTH]

        if decode:
            output = self.datamodule.tokenizer.batch_decode(generated,
                                                            skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=True)
            chunked_list = [output[i * NUM_RETURN_SEQ:(i + 1) * NUM_RETURN_SEQ] \
                            for i in range((len(output) + NUM_RETURN_SEQ - 1) // NUM_RETURN_SEQ )] # chunk into [batch x NUM_RETURN]
            return chunked_list
        else:
            return generated.view(context.shape[0], NUM_RETURN_SEQ, -1)

