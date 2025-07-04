from base_issue import BaseIssue


class Preprocessor:
    def __init__(self):
        self.datasets = None
        self.callbacks = []
        self.outputs = None

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def call_on_preprocess_begin_callbacks(self):
        for callback in self.callbacks:
            callback.on_preprocess_begin(self)

    def call_on_preprocess_end_callbacks(self):
        for callback in self.callbacks:
            callback.on_preprocess_end(self)

    def preprocess(self, datasets) -> list[BaseIssue]:
        raise NotImplementedError("Subclasses must implement `preprocess`.")

    def __call__(self, datasets) -> list[BaseIssue]:
        self.datasets = datasets

        self.call_on_preprocess_begin_callbacks()
        outputs = self.preprocess(self.datasets)
        self.outputs = outputs
        self.call_on_preprocess_end_callbacks()
        return self.outputs
