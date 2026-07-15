from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class InferenceResult:
    confidence: float
    latency_ms: float
    is_correct: bool


class ModelClient:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, true_label: int | None = None) -> InferenceResult:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        latency = (time.perf_counter() - start) * 1000
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence_class1 = float(probs[0, 1].item())
        predicted = 1 if confidence_class1 > 0.5 else 0
        is_correct = true_label is not None and predicted == true_label
        return InferenceResult(confidence=confidence_class1, latency_ms=latency, is_correct=is_correct)


class DistilBERTEdge(ModelClient):
    def __init__(self, device: str = "cpu"):
        super().__init__("newsmediabias/DistilBert_Toxicity_Classification", device)


class BERTLargeCloud(ModelClient):
    def __init__(self, device: str = "cpu"):
        super().__init__("AnonymousCS/bert-large-uncased-Twitter-toxicity", device)
