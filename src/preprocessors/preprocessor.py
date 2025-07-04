from . import base_preprocessor


class NormalizePreprocessor(base_preprocessor.Preprocessor):
    def __init__(self, normaliser):
        self.normaliser = normaliser

    def preprocess(self, datasets):
        return self._normalise_all(datasets)

    def _normalise_all(self, datasets):
        return [self.normaliser(data) for data in datasets]
