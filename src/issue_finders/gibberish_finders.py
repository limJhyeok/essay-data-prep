import torch
import re
from . import base_finder
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from base_issue import BaseIssue


class DeepGibberishFinder(base_finder.TextDataIssueFinder):
    def __init__(self, gibberish_detector, noise_labels: list):
        super(DeepGibberishFinder, self).__init__()
        self.gibberish_detector = gibberish_detector
        self.noise_labels = noise_labels

    def identify_issues(self, datasets) -> list[BaseIssue]:
        issues = []
        for i, text in enumerate(tqdm(datasets, desc="Finding gibberish issues")):
            result = self.gibberish_detector(text)[0]
            if result["label"] in self.noise_labels:
                issue = BaseIssue.from_data(
                    id=i,
                    data=datasets[i],
                    meta={
                        "length": len(datasets[i]),
                        "gibberish_pred": result["label"],
                        "gibberish_score": result["score"],
                    },
                )
                issues.append(issue)
        return issues


def is_potentially_gibberish(text):
    text_clean = text.strip()
    if len(text_clean) < 10:
        return True

    alpha_ratio = sum(c.isalpha() for c in text_clean) / len(text_clean)
    if alpha_ratio < 0.7:
        return True

    nonword_ratio = len(re.findall(r"\W", text_clean)) / len(text_clean)
    if nonword_ratio > 0.4:
        return True

    if re.search(r"(.)\1{4,}", text_clean):
        return True

    if type_token_ratio(text) < 0.4:
        return True

    return False


def type_token_ratio(text):
    tokens = text.split()
    return len(set(tokens)) / len(tokens) if tokens else 0


def compute_perplexity(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity
