import os
import sys
import re
import glob
import json
from pathlib import Path
from tqdm import tqdm
from collections import *
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch
from pytorch_lightning import LightningDataModule

from transformers import AutoTokenizer

import nltk
from nltk import sent_tokenize, word_tokenize
nltk.download('punkt')


def tokenize(st):
    ans = []
    for sent in sent_tokenize(st):
        ans += word_tokenize(sent)
    return " ".join(ans).lower()


def count_question_type(questions):
    question_types = ["what", "who", "where", "when", "why",
                      "how", "which", "whose", "because", "mics"]
    records = []
    for question in tqdm(questions):
        tokens = question.lower().split()
        if any(set(tokens).intersection(set(question_types))):
            t = set(tokens).intersection(set(question_types))
            records.append(list(t)[0])
        elif "_" in question:
            records.append("cloze")
        else:
            records.append("mics")
    counter = dict(Counter(records).most_common())
    return counter


def prepare_data(race, tokenizer):
    paths = race["middle"] + race["high"]
    inputs = []
    outputs = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        article = data["article"]
        for question, answer, options in zip(data["questions"], data["answers"], data["options"]):
            correct_option = options.pop(ord(answer) - ord("A"))
            inputs.append(" [SEP] ".join([article, correct_option]))
            outputs.append(" [SEP] ".join([question, *options]))
    return inputs, outputs


class RaceDataProcessor:
    """ Race Data Processor """
    def process_common(self, text):
        """ Preprocess text.

        Parameters
        ----------
        text: str
            Text to be processed.

        Returns
        -------
        text: str
            Processed text.
        """
        # Replace "`" with "'":
        text = text.replace("`", "'")
        # Remove "s6t----":
        text = text.replace("s6t----", "")
        # Remove [...]:
        text = "".join(re.split(r"\[+.*\]*", text))
        return text

    def process_question(self, question):
        """ Preprocess question.

        Parameters
        ----------
        question: str
            Question to be processed.

        Returns
        -------
        question: str
            Processed question.
        """
        # Common process:
        question = self.process_common(question)
        # Remove question number and "." at the start
        question = "".join(re.split(r"\A\d*\. *", question))
        # Remove redundant duplicate of "-----":
        question = " ".join(re.split(r"--+", question))
        # Remove redundant whitespace:
        question = " ".join(question.split())
        return question

    def process_options(self, options):
        """ Preprocess options.

        Parameters
        ----------
        options: list of str
            Options to be processed.

        Returns
        -------
        options: list of str
            Processed options.
        """
        for i in range(len(options)):
            # Remove redundant whitespace:
            options[i] = self.process_common(options[i])
            # Remove redundant whitespace:
            options[i] = " ".join(options[i].split())
        return options

    def process_article(self, article):
        """ Preprocess article.

        Parameters
        ----------
        article: str
            Article to be processed.

        Returns
        -------
        article: str
            Processed article.
        """
        # Common process:
        article = self.process_common(article)
        # Replace redundant duplicate of "-----" with "--":
        article = re.sub(r"--+", "--", article)
        # Remove redundant whitespace:
        article = " ".join(article.split())
        return article

    def process_data(self, data_path, save_path):
        """ Preprocess data.

        The function will perform preprocessing on all files in the directory
        given in the "data_path". The processed data will be saved to the
        directory given in the "save_path". The directory structure of the
        processed files will be the same as that of the raw data files.

        Parameters
        ----------
        data_path: str
            Path to raw data directory.
        save_path: str
            Path to directory to save processed data.

        Returns
        -------
        None
        """
        # Prepare path for saving processed data:
        save_path = Path(save_path)

        # Glob data paths:
        paths = Path(data_path).glob("*/*/*.txt")

        # Process data:
        for path in tqdm(paths):
            # Read data:
            with open(path) as f:
                data = json.load(f)

            # Process article:
            data["article"] = self.process_article(data["article"])

            # Process questions and options:
            for i in range(len(data["questions"])):
                data["questions"][i] = self.process_question(data["questions"][i])
                data["options"][i] = self.process_options(data["options"][i])

            # Create new file to save processed data:
            dataset = path.parent.parent.stem
            level = path.parent.stem
            name = path.name
            parent_path = save_path / dataset / level
            parent_path.mkdir(parents=True, exist_ok=True)
            dump_path = parent_path / name
            dump_path.touch(exist_ok=True)

            # Dump processed data to save path:
            with open(dump_path, "w") as f:
                json.dump(data, f)

        print("Preprocess data successfully!")


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

    def __init__(self, hparams):
        """"""
        super().__init__()
        self.hparams = hparams
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        if self.hparams.pretrained_model in ["t5-base", "t5-small"]:
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<answer>', '<context>']})

    def collate_fn(self, batch):
        """"""
        articles = []
        questions = []
        answers = []
        distractors = []
        
        #print("COLLATE based of %s" %self.hparams.pretrained_model)
        if self.hparams.pretrained_model in ["t5-base", "t5-small"]:
            for item in batch:
                articles.append(" ".join(["<answer>", item["answer"], "<context>", item["article"]]))
                questions.append(item["question"])
            articles = self.tokenizer(articles, padding=True, 
                                               truncation=True, 
                                               return_tensors="pt", 
                                               pad_to_max_length=True, 
                                               max_length=512)
            questions = self.tokenizer(questions, padding=True, 
                                               truncation=True, 
                                               return_tensors="pt", 
                                               pad_to_max_length=True, 
                                               max_length=512)
            articles['input_ids'] = torch.squeeze(articles['input_ids'])
            articles['attention_mask'] = torch.squeeze(articles['attention_mask'])
            questions['input_ids'] = torch.squeeze(questions['input_ids'])
            questions['attention_mask'] = torch.squeeze(questions['attention_mask'])

            return (articles, questions)
        else:
            raise NotImplementedError


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
            collate_fn=self.collate_fn,
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
            collate_fn=self.collate_fn,
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
            collate_fn=self.collate_fn,
        )
        return self.test_loader


