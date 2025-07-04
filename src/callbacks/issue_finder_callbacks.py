from tqdm import tqdm
import re
import utils
from issue_finders.base_finder import TextDataIssueFinder
from . import base_callbacks


class GibberishFilterIssueFinderCallback(base_callbacks.IssueFinderCallback):
    def __init__(self):
        pass

    def on_identify_begin(self, issue_finder: TextDataIssueFinder):
        _datasets = []
        for data in tqdm(issue_finder.datasets, desc="Filtering Gibberish candidates"):
            if self.is_potentially_gibberish(data):
                _datasets.append(data)

        issue_finder.datasets = _datasets

    def is_potentially_gibberish(self, text):
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

        if utils.type_token_ratio(text) < 0.4:
            return True

        return False
