from abc import ABC
from annotators.base_annotators import Annotator
from issue_finders.base_finder import TextDataIssueFinder


class AnnotatorCallback(ABC):
    def on_annotate_begin(self, annotator: Annotator):
        pass

    def on_annotate_end(self, annotator: Annotator):
        pass


class IssueFinderCallback(ABC):
    def on_identify_begin(self, issue_finder: TextDataIssueFinder):
        pass

    def on_identify_end(self, issue_finder: TextDataIssueFinder):
        pass
