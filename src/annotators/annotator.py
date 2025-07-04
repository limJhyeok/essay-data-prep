import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import utils
from base_issue import BaseIssue
from annotators.base_annotators import Annotator


class LongestNgramAnnotator(Annotator):
    def __init__(self, callbacks=[]):
        super(LongestNgramAnnotator, self).__init__()
        self.callbacks = callbacks

    def annotate(self, issues: list[BaseIssue]) -> list[BaseIssue]:
        issue_list = []
        for _issue in tqdm(issues, desc="Annotating the longest N-gram"):
            if _issue.ref_data is None:
                raise ValueError(f"{_issue} has no ref_data attribute")
            longest_ngram = utils.get_longest_ngram(
                _issue.data,
                _issue.ref_data,
                # utils.normalize_string(_issue.data),
                # utils.normalize_string(_issue.ref_data)
            )
            _issue.meta["longest_ngram"] = longest_ngram
            _issue.meta["longest_ngram_length"] = len(longest_ngram)
            issue_list.append(_issue)
        return issue_list


class PerplexityAnnotator(Annotator):
    def __init__(self, ppl_model, ppl_tokenizer, device):
        super(PerplexityAnnotator, self).__init__()
        self.ppl_model = ppl_model
        self.ppl_tokenizer = ppl_tokenizer
        self.device = device

    def annotate(self, issues: BaseIssue) -> list[BaseIssue]:
        issue_list = []
        for _issue in tqdm(issues, desc="Annotating Perplexity"):
            ppl = compute_perplexity(
                _issue.data, self.ppl_model, self.ppl_tokenizer, self.device
            )
            _issue.meta["perplexity"] = ppl
            issue_list.append(_issue)
        return issue_list


class TypeTokenRatioAnnotator(Annotator):
    def __init__(self):
        super(TypeTokenRatioAnnotator, self).__init__()
        pass

    def annotate(self, issues: BaseIssue) -> list[BaseIssue]:
        issue_list = []
        for _issue in tqdm(issues, desc="Annotating Type Token Ratio"):
            ttr = type_token_ratio(_issue.data)
            _issue.meta["type_token_ratio"] = ttr
            issue_list.append(_issue)
        return issue_list


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


def type_token_ratio(text):
    tokens = text.split()
    return len(set(tokens)) / len(tokens) if tokens else 0
