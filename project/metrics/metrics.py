import datasets
from typing import List, Dict


class Input:
    """ Prediction & Reference Input Class """

    def __init__(self, predictions: List[str], references: List[str]):
        self._predictions = predictions
        self._references = references

    @property
    def bleu_predictions(self) -> List[List[str]]:
        return [prediction.split() for prediction in self._predictions]

    @property
    def bleu_references(self) -> List[List[List[str]]]:
        return [[reference.split()] for reference in self._references]

    @property
    def meteor_predictions(self) -> List[str]:
        return self._predictions

    @property
    def meteor_references(self) -> List[str]:
        return self._references

    @property
    def rouge_predictions(self) -> List[str]:
        return self._predictions

    @property
    def rouge_references(self) -> List[str]:
        return self._references


class Metrics:
    """ Metrics Computation: BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L """

    def __init__(self):
        self.bleu = datasets.load_metric("bleu")
        self.meteor = datasets.load_metric("meteor")
        self.rouge = datasets.load_metric("rouge")

    def compute_bleu(self, input: Input, max_order: int) -> float:
        """"""
        result = self.bleu.compute(
            predictions=input.bleu_predictions,
            references=input.bleu_references,
            max_order=max_order,
        )
        return result["bleu"]

    def compute_meteor(self, input: Input) -> float:
        """"""
        result = self.meteor.compute(
            predictions=input.meteor_predictions,
            references=input.meteor_references,
        )
        return result["meteor"]

    def compute_rouge(self, input: Input) -> float:
        """"""
        result = self.rouge.compute(
            predictions=input.rouge_predictions,
            references=input.rouge_references,
        )
        return result["rougeL"].mid.recall

    def compute_metrics(self, input: Input) -> Dict[str, float]:
        """"""
        return {
            "bleu_1": self.compute_bleu(input, 1),
            "bleu_2": self.compute_bleu(input, 2),
            "bleu_3": self.compute_bleu(input, 3),
            "bleu_4": self.compute_bleu(input, 4),
            "meteor": self.compute_meteor(input),
            "rouge_l": self.compute_rouge(input),
        }








