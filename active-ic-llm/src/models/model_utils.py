from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math


class ModelUtils:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def compute_perplexity(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, labels=inputs["input_ids"])
        loss = out.loss.item()
        return math.exp(loss)

    def _score_completion(self, prompt: str, completion: str) -> float:
        text = prompt + completion
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, labels=inputs["input_ids"])
        return -out.loss.item() * inputs["input_ids"].size(1)

    def predict_classification(self, prompt: str, candidate_labels: List[str]) -> str:
        scores = {lab: self._score_completion(prompt, " " + lab) for lab in candidate_labels}
        return max(scores, key=scores.get)

    def predict_multichoice(self, prompt: str, choices: List[str]) -> str:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        scores = {}
        for i, choice in enumerate(choices):
            comp = f" {letters[i]}) {choice}"
            scores[letters[i]] = self._score_completion(prompt, comp)
        return max(scores, key=scores.get)
