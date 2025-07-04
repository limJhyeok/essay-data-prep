import re
import os
import pandas as pd
from datasets import load_dataset, Dataset
import utils
from dotenv import load_dotenv


def remove_substr(tgt: str, src: str) -> str:
    """
    Removes a source substring (e.g. prompt) from the target text, allowing for flexible whitespace between words.

    Parameters:
        tgt (str): The target text (e.g. essay or document).
        src (str): The substring or phrase to remove, with flexibility in spacing. (e.g. prompt)

    Returns:
        str: The filtered text with the source string removed.
    """
    escaped_src = re.escape(src)
    # Allow flexible whitespace between words in the source string
    pattern = re.sub(r"\\ ", r"\\s+", escaped_src)
    filtered_tgt = re.sub(pattern, "", tgt, flags=re.IGNORECASE).strip()
    return filtered_tgt


if __name__ == "__main__":
    load_dotenv()

    hf_user_name = os.getenv("HF_USER_NAME")
    hf_data_name = os.getenv("HF_DATA_NAME")

    issue_report_dir_name = "issue_reports"
    similar_issue_csv_name = "prompt_similar_essays_issues.csv"
    issue_report_dir = os.path.join(os.getcwd(), issue_report_dir_name)
    dataset_name = "chillies/IELTS-writing-task-2-evaluation"
    split = "train"
    save_dir_name = "cleaned"
    save_dir = os.path.join(os.getcwd(), save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    save_fn = f"ielts-{split}"

    dataset = load_dataset(dataset_name)

    similar_issue_df = pd.read_csv(
        os.path.join(issue_report_dir, similar_issue_csv_name)
    )
    similar_threshold = 0.99

    dataset_df = pd.DataFrame(dataset[split])
    dataset_df["band"] = dataset_df["band"].apply(utils.clean_band)
    dataset_df.drop("evaluation", axis=1, inplace=True)

    dataset_dict = dataset_df.to_dict()

    filtered_df = similar_issue_df[similar_issue_df["similarity"] > similar_threshold]
    for i, _issue in filtered_df.iterrows():
        _id = _issue["id"]
        essay = dataset_dict["essay"][_id]
        prompt = dataset_dict["prompt"][_id]

        norm_prompt = utils.normalize_string(prompt)
        _essay = remove_substr(essay, prompt)
        dataset_dict["essay"][_id] = _essay

    df = pd.DataFrame.from_dict(dataset_dict)

    hf_dataset = Dataset.from_pandas(df, preserve_index=False)

    issue_csv_names = [
        "duplicate_issues.csv",
        "gibberish_issues.csv",
        "suspect_scoring_issues.csv",
    ]
    remove_ids = []
    for name in issue_csv_names:
        issue_df = pd.read_csv(os.path.join(issue_report_dir, name))
        remove_ids.extend(issue_df["id"].to_list())

    off_the_prompt_issue_df = pd.read_csv(
        os.path.join(issue_report_dir, "off_the_prompt_issues.csv")
    )

    off_the_prompt_df = df.iloc[off_the_prompt_issue_df["id"]]
    off_the_prompt_ids = off_the_prompt_df[
        off_the_prompt_df["band"] >= 4
    ].index.to_list()

    remove_ids.extend(off_the_prompt_ids)

    not_contaminated_ids = set(range(len(hf_dataset))) - set(remove_ids)
    filtered_dataset = hf_dataset.select(not_contaminated_ids)
    filtered_dataset.save_to_disk(os.path.join(save_dir, save_fn))

    if hf_user_name and hf_data_name:
        try:
            filtered_dataset.push_to_hub(f"{hf_user_name}/{hf_data_name}")
            print(f"\nSuccessfully uploaded dataset to {hf_user_name}/{hf_data_name}")
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")
