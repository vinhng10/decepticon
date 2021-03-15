import os
import sys
import re
import glob
import json
from tqdm import tqdm
from collections import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def preprocess_common(text):
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


def preprocess_question(question):
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
    # Common preprocess:
    question = preprocess_common(question)
    # Remove question number and "." at the start
    question = "".join(re.split(r"\A\d*\. *", question))
    # Remove redundant duplicate of "-----":
    question = " ".join(re.split(r"--+", question))
    # Remove redundant whitespace:
    question = " ".join(question.split())
    return question


def preprocess_options(options):
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
        options[i] = preprocess_common(options[i])
        # Remove redundant whitespace:
        options[i] = " ".join(options[i].split())
    return options


def preprocess_article(article):
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
    # Common preprocess:
    article = preprocess_common(article)
    # Replace redundant duplicate of "-----" with "--":
    article = re.sub(r"--+", "--", article)
    # Remove redundant whitespace:
    article = " ".join(article.split())
    return article



