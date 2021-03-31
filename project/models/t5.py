import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import  T5ForConditionalGeneration, T5Config, get_linear_schedule_with_warmup, AdamW



class T5FinetuneForRACE(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FinetuneForRACE, self).__init__()
        self.save_hyperparameters()
        self.hparams = hparams
        if self.hparams.pretrained_model in ["t5-base","t5-small"]:
            config = T5Config(decoder_start_token_id = self.hparams.padding_token)
            self.model = T5ForConditionalGeneration(config).from_pretrained(self.hparams.pretrained_model)
            self.model.resize_token_embeddings(self.hparams.tokenizer_len)
        else:
            raise NotImplementedError
            
            
    def mask_label_padding(self, labels):
        MASK_ID = -100
        labels[labels == self.hparams.padding_id] = MASK_ID 
        return labels
 
    def forward(self, ids, mask, labels):
        return self.model(input_ids = ids, attention_mask = mask, labels = labels)


    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x["input_ids"], x["attention_mask"], self.mask_label_padding(y["input_ids"]))
        loss = output.loss
        
        ## logger
        self.logger.experiment.log_metric('train_loss', loss.detach())
        self.logger.experiment.log_metric('train_perplexity', torch.exp(loss.detach()))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x["input_ids"], x["attention_mask"], self.mask_label_padding(y["input_ids"]))
        loss = output[0]
        ### logger
        self.logger.experiment.log_metric('val_loss', loss.detach())
        self.logger.experiment.log_metric('val_perplexity', torch.exp(loss.detach()))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x["input_ids"], x["attention_mask"], self.mask_label_padding(y["input_ids"]))
        loss = output[0]
        return loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                                         'weight_decay': self.hparams.weight_decay}, 
                                        {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                                         'weight_decay': 0.0}]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr = self.hparams.learning_rate)
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=0,
#             # Default value in run_glue.py
#             num_training_steps=self.hparams.num_training_steps)
 
        return [optimizer]#, [scheduler]