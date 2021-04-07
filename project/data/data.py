import os
import sys
import re
import glob
import json
from pathlib import Path
from tqdm import tqdm
from collections import *
from functools import partial
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch
from pytorch_lightning import LightningDataModule

from transformers import AutoTokenizer


class RaceDataset(Dataset):
    """ Race Dataset """

    def __init__(self, data_paths):
        """"""
        super().__init__()
        self.data_paths = data_paths
        self.dataset = []
        for path in data_paths:
            with open(path) as f:
                data = json.load(f)
            for i in range(len(data["answers"])):
                answer_idx = ord(data["answers"][i]) - ord("A")
                answer = data["options"][i].pop(answer_idx)
                distractors = data["options"][i]
                self.dataset.append({
                    "article": data["article"],
                    "question": data["questions"][i],
                    "answer": answer,
                    "distractors": distractors,
                })

    def __len__(self):
        """"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """"""
        return self.dataset[idx]


class RaceDataModule(LightningDataModule):
    """ Race Data Module """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", type=str,
                            help="Path to data.")
        parser.add_argument("--batch_size", type=int, default=256,
                            help="Batch size.")
        parser.add_argument("--num_workers", type=int, default=8,
                            help="Number of workers for data loading.")
        parser.add_argument("--special_tokens", nargs="*", default=["[CON]", "[QUE]", "[ANS]", "[DIS]"],
                            help="Additional special tokens.")
        parser.add_argument("--pretrained_model", type=str, default="prajjwal1/bert-tiny",
                            help="Pretrained model.")
        return parser

    @staticmethod
    def default_collate_fn(batch, tokenizer):
        """"""
        articles = []
        questions = []
        answers = []
        distractors = []

        for item in batch:
            articles.append(item["article"])
            questions.append(item["question"])
            answers.append(item["answer"])
            distractors.append(tokenizer.additional_special_tokens[-1].join(item["distractors"]))

        return {
            "articles": tokenizer(articles, padding=True, truncation=True, return_tensors="pt"),
            "questions": tokenizer(questions, padding=True, truncation=True, return_tensors="pt"),
            "answers": tokenizer(answers, padding=True, truncation=True, return_tensors="pt"),
            "distractors": tokenizer(distractors, padding=True, truncation=True, return_tensors="pt"),
        }
    @staticmethod
    def t5_collate_fn(batch, tokenizer):
        """"""
        context = []
        questions = []
        for item in batch:
            context.append(" ".join(["<answer>", item["answer"], "<context>", item["article"]]))
            questions.append(item["question"])
        context = tokenizer(context, padding=True, 
                                               truncation=True, 
                                               return_tensors="pt", 
                                               pad_to_max_length=True, 
                                               max_length=512)
        questions = tokenizer(questions, padding=True, 
                                               truncation=True, 
                                               return_tensors="pt", 
                                               pad_to_max_length=True, 
                                               max_length=512)
        context['input_ids'] = torch.squeeze(context['input_ids'])
        context['attention_mask'] = torch.squeeze(context['attention_mask'])
        questions['input_ids'] = torch.squeeze(questions['input_ids'])
        questions['attention_mask'] = torch.squeeze(questions['attention_mask'])
        return (context, questions)
        

    def __init__(self, hparams, custom_collate_fn=None):
        """"""
        super().__init__()
        self.hparams = hparams

        if custom_collate_fn:
            print("DataModule: Custom collate function is detected")
            self.collate_fn = custom_collate_fn
        else:
            self.collate_fn = self.default_collate_fn

        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pretrained_model)
        self.tokenizer.add_special_tokens({"additional_special_tokens": hparams.special_tokens})

    def prepare_data(self):
        """"""
        pass

    def setup(self, stage=None):
        """"""
        # Prepare data paths:
        train_paths = Path(self.hparams.data_path).glob("train/*/*.txt")
        val_paths = Path(self.hparams.data_path).glob("dev/*/*.txt")
        test_paths = Path(self.hparams.data_path).glob("test/*/*.txt")

        # Prepare datasets
        print("SETUP: Training Dataset")
        self.trainset = RaceDataset(train_paths)
        print("SETUP: Validation Dataset")
        self.valset = RaceDataset(val_paths)
        print("SETUP: Test Dataset")
        self.testset = RaceDataset(test_paths)

    def train_dataloader(self):
        """"""
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer),
        )
        return self.train_loader

    def val_dataloader(self):
        """"""
        self.val_loader = DataLoader(
            self.valset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer),
        )
        return self.val_loader

    def test_dataloader(self):
        """"""
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer),
        )
        return self.test_loader

