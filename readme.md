# Essay Data Prep

<p align="left">
    <a href="readme.ko.md">한국어</a>&nbsp ｜ &nbspEnglish&nbsp
</p>

## Description and Purpose

This project provides a suite of tools for preparing and cleaning essay data for training or evaluating AI models. It identifies and annotates various issues in essay datasets, such as duplicate essays, gibberish content, and essays that are off-prompt. The primary goal is to improve the quality of data used for NLP tasks, particularly in the context of essay scoring and evaluation.

-   **Duplicate Essays:** Finding and flagging essays that are identical or nearly identical.
-   **Gibberish Content:** Detecting essays that consist of random characters or nonsensical text.
-   **Similar Essays:** Identifying essays that are highly similar to each other or to the prompts.
-   **Off-the-Prompt Essays:** Finding essays that do not align with the given prompt.
-   **Suspect Scoring:** Identifying essays with scores that are inconsistent with their content (e.g., very short essays with high scores).

## Installation

To install the necessary dependencies, run the following command:

```bash
uv sync
```

## Usage

The main script for processing the data is located in `src/quality_check.py`. You can run it from the root of the project directory:

```bash
uv run src/quality_check.py
```

This will perform the following steps:
1.  Load the dataset from the specified source.
2.  Apply various quality checks to identify issues.
3.  Generate reports in the `issue_reports` directory, detailing the findings.
You can customize the behavior of the script by modifying the parameters in `src/quality_check.py`, such as the dataset name, file paths, and thresholds for various checks.
