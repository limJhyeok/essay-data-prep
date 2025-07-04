from typing import Any, Optional
from base_issue import BaseIssue
import pandas as pd


def flatten_issue_dict(issue: BaseIssue) -> dict[str, Any]:
    """
    Converts a BaseIssue into a flat dictionary for CSV output.
    Flattens `meta` into the top-level dictionary.
    """
    base_dict = issue.model_dump()
    meta = base_dict.pop("meta", {})

    # Avoid overwriting base keys with meta keys
    for k, v in meta.items():
        if k in base_dict:
            base_dict[f"meta_{k}"] = v
        else:
            base_dict[k] = v

    return base_dict


def save_issues_to_csv(
    issues: list[BaseIssue],
    filename: str,
    field_mapping: Optional[dict[str, str]] = None,
):
    if not issues:
        print("No issues to save.")
        return

    field_mapping = field_mapping or {}

    flattened_data = [flatten_issue_dict(issue) for issue in issues]

    df = pd.DataFrame(flattened_data)
    df.rename(columns=field_mapping, inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.to_csv(filename, index=False, encoding="utf-8")
