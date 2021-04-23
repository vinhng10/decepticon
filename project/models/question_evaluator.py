import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import  BertForSequenceClassification, 

class QA_PairEvaluation(pl.LightningModule):
    def __init__(self, hparams):
        super(QuestionEvaluator, self).__init__()
        self.save_hyperparameters()
        self.hparams = hparams
        assert self.hparams.pretrained_eval_model == "iarfmoose/bert-base-cased-qa-evaluator", NotImplemented()
        self.evaluator = BertForSequenceClassification.from_pretrained(hparams.pretrained_eval_model)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pretrained_eval_model)
        
    def forward(self, x):
        
        
    def encode_pair(self, x):
        
        