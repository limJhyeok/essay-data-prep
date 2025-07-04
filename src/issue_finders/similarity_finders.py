from . import base_finder
from typing import Callable
from tqdm import tqdm
import operator
from base_issue import BaseIssue


class SimilarityFinder(base_finder.TextDataIssueFinder):
    def __init__(
        self, similarity_fn: Callable, similarity_threshold: float, compare_op: operator
    ):
        super(SimilarityFinder, self).__init__()
        self.similarity_fn = similarity_fn
        self.similarity_threshold = similarity_threshold
        self.compare_op = compare_op

    def identify_issues(self, datasets) -> list[BaseIssue]:
        issues = []
        for i, (text_1, text_2) in enumerate(
            tqdm(datasets, desc="Finding similarity issues")
        ):
            similarity = self.similarity_fn(text_1, text_2)

            if self.compare_op(similarity, self.similarity_threshold):
                issue = BaseIssue.from_data(
                    id=i,
                    data=text_1,
                    ref_id=i,
                    ref_data=text_2,
                    meta={
                        "length": len(text_1),
                        "ref_length": len(text_2),
                        "similarity": similarity,
                    },
                )
                issues.append(issue)
        return issues
