import pandas as pd


def duplicate_issue_report_prompt(original_df: pd.DataFrame, issue_df: pd.DataFrame):
    sample_original = original_df.head(1)
    sample_issues = issue_df.head()

    return f"""
You are an expert in essay scoring data cleaning and analysis.

I have already written some candidates which I assume the duplication happens on essay data, such as:

The results of these checks are recorded in the following CSV files:

original data shape:
{original_df.shape}

Sample original data:
{sample_original.to_markdown(index=False)}

original data description:
{original_df.describe().to_markdown(index=False)}

Sample issue data:
{sample_issues.to_markdown(index=False)}

issue description:
{issue_df.describe().to_markdown(index=False)}

Your task is to:
1. Read the sample data I gave you
2. Summarise the issues present in the data
3. Provide a breakdown by issue type
4. Provide insights or patterns you observe
5. Suggest any further cleaning or review steps, if necessary

Write the output as a **step-by-step data issue report**, in clear and structured markdown format, suitable for internal QA and analysis.

Start your report with a title and short summary, followed by sections such as:

## 1. Overview  
## 2. Issue Breakdown  
## 3. Patterns & Observations  
## 4. Recommendations
"""
