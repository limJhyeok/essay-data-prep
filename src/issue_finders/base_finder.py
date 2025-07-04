from base_issue import BaseIssue


class TextDataIssueFinder:
    def __init__(self):
        self.datasets = None
        self.callbacks = []
        self.outputs = None

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def call_on_identify_begin_callbacks(self):
        for callback in self.callbacks:
            callback.on_identify_begin(self)

    def call_on_identify_end_callbacks(self):
        for callback in self.callbacks:
            callback.on_identify_end(self)

    def identify_issues(self, datasets) -> list[BaseIssue]:
        raise NotImplementedError("Subclasses must implement `identify_issues`.")

    def __call__(self, datasets) -> list[BaseIssue]:
        self.datasets = datasets

        self.call_on_identify_begin_callbacks()
        outputs = self.identify_issues(self.datasets)
        self.outputs = outputs
        self.call_on_identify_end_callbacks()
        return self.outputs
