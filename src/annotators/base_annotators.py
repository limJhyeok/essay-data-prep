from base_issue import BaseIssue


class Annotator:
    def __init__(self):
        self.issues = None
        self.callbacks = []
        self.outputs = None

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def call_on_annotate_begin_callbacks(self):
        for callback in self.callbacks:
            callback.on_annotate_begin(self)

    def call_on_annotate_end_callbacks(self):
        for callback in self.callbacks:
            callback.on_annotate_end(self)

    def annotate(self, issues: list[BaseIssue]) -> list[BaseIssue]:
        raise NotImplementedError("Subclasses must implement `identify_issues`.")

    def __call__(self, issues: list[BaseIssue]):
        self.issues = issues

        self.call_on_annotate_begin_callbacks()
        outputs = self.annotate(self.issues)
        self.outputs = outputs
        self.call_on_annotate_end_callbacks()
        return self.outputs
