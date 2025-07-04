from typing import Callable
from tqdm import tqdm
import utils
from . import base_finder
from base_issue import BaseIssue


class NgramDuplicateFinder(base_finder.TextDataIssueFinder):
    def __init__(self, ngram_threshold: int, select_fn: Callable):
        super(NgramDuplicateFinder, self).__init__()
        self.ngram_threshold = ngram_threshold
        self.select_fn = select_fn

    def identify_issues(self, datasets) -> list[BaseIssue]:
        ngram_lookup = utils.build_ngram_lookup(datasets, self.ngram_threshold)
        candidates = []
        for doc_ids in tqdm(
            ngram_lookup.values(), desc="Finding the duplicated issues"
        ):
            if len(doc_ids) >= 2:
                is_in = False
                for can in candidates:
                    if any(doc_id in can for doc_id in doc_ids):
                        can.update(doc_ids)
                        is_in = True
                        break
                if not is_in:
                    candidates.append(set(doc_ids))

        issues = []
        for candidate in tqdm(candidates, desc="Processing the duplicated issues"):
            parent_id = self.select_fn(datasets, candidate)
            child_ids = candidate - {parent_id}
            for child_id in child_ids:
                issue = BaseIssue.from_data(
                    id=child_id,
                    data=datasets[child_id],
                    ref_id=parent_id,
                    ref_data=datasets[parent_id],
                    meta={
                        "length": len(datasets[child_id]),
                        "ref_length": len(datasets[parent_id]),
                    },
                )
                issues.append(issue)

        return issues
