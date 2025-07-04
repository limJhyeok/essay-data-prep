from datetime import datetime
import pandas as pd
import utils
import os
import prompts
from datasets import load_dataset

if __name__ == "__main__":
    dataset_name = "chillies/IELTS-writing-task-2-evaluation"
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset["train"])
    df["band"] = df["band"].apply(utils.clean_band)
    df.drop("evaluation", axis=1, inplace=True)

    issue_report_dir_name = "issue_reports"
    issue_report_dir = os.path.join(os.getcwd(), issue_report_dir_name)
    os.makedirs(issue_report_dir, exist_ok=True)

    filenames = os.listdir(issue_report_dir)

    file_locations = [os.path.join(issue_report_dir, fn) for fn in filenames]
    issue_df = pd.read_csv(file_locations[2])
    prompt = prompts.duplicate_issue_report_prompt(df, issue_df)

    issue_report = utils.call_gemini(prompt)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_fn = f"issue_report-{timestamp}.md"
    with open(os.path.join(issue_report_dir, output_fn), "w", encoding="utf-8") as f:
        f.write(issue_report)
