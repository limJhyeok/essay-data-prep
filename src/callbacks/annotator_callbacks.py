from . import base_callbacks

from annotators.base_annotators import Annotator


class NormalizeAnnotatorCallback(base_callbacks.AnnotatorCallback):
    def __init__(self, normaliser):
        self.normaliser = normaliser

    def on_annotate_begin(self, annotator: Annotator):
        for i, _issue in enumerate(annotator.issues):
            _data = self.normaliser(_issue.data)
            annotator.issues[i].data = _data
            if _issue.ref_data is not None:
                _ref_data = self.normaliser(_issue.ref_data)
                annotator.issues[i]._ref_data = _ref_data
