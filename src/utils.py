import subprocess
import numpy as np
import collections
from tqdm import tqdm
import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sentence_transformers
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from base_issue import BaseIssue


def normalize_string(text: str) -> str:
    text = text.lower().strip()
    text = " ".join(text.split())
    return text


def word_ngrams(text: str, n: int) -> list[str]:
    """Generate word-level n-grams from text."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def build_ngram_lookup(
    documents: list[str], ngram_size: int = 13
) -> dict[str, set[int]]:
    """Build ngram lookup for documents."""
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(
        tqdm(documents, desc=f"Building {ngram_size}-gram lookup...")
    ):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(doc_id)

    return lookup


def get_longest_ngram(text1: str, text2: str) -> str:
    tokens1 = text1.split()
    tokens2 = text2.split()
    len1, len2 = len(tokens1), len(tokens2)

    # Create a 2D DP table
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    max_len = 0
    end_pos = 0  # End position in tokens1

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i

    # Reconstruct the ngram
    max_ngram = tokens1[end_pos - max_len : end_pos]
    return " ".join(max_ngram)


def longest_ngram_ratio(source: str, compare: str) -> float:
    longest_ngram = get_longest_ngram(source, compare)
    return len(longest_ngram) / len(source)


def select_longest_id(datasets, ids: list[int]):
    return max(ids, key=lambda i: len(datasets[i]))


def upload_to_huggingface(
    selected_examples, repo_id="your-username/dataset-name", source="Omni-MATH"
):
    # Format the data for upload
    formatted_data = []
    for ex in tqdm(selected_examples, desc="Formatting examples"):
        try:
            if "domain" in ex or "difficulty" in ex:
                formatted_example = {
                    "problem": ex["problem"],
                    "solution": ex["solution"],
                    "domain": ex["domain"][0],
                    "difficulty": ex["difficulty"],
                    "subdomain": ex["domain"][0].split(" -> ")[2],
                    "source": source,
                }
            else:
                formatted_example = {
                    "problem": ex["problem"],
                    "solution": ex["solution"],
                    "source": source,
                    "messages": ex["messages"],
                }
            formatted_data.append(formatted_example)
        except Exception as e:
            print(f"Error formatting example: {e}")
            continue

    # Create the dataset
    dataset = datasets.Dataset.from_list(formatted_data)

    # Print dataset info
    print("\nDataset Statistics:")
    print(f"Total examples: {len(dataset)}")
    print("\nFeatures:", dataset.features)

    # Push to hub
    try:
        dataset.push_to_hub(repo_id)
        print(f"\nSuccessfully uploaded dataset to {repo_id}")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        return None

    return dataset


def type_token_ratio(text):
    tokens = text.split()
    return len(set(tokens)) / len(tokens) if tokens else 0


def compute_tf_idf_cosine_sim(
    text_1: str,
    text_2: str,
) -> float:
    """
    Compute cosine similarity between two text documents using a vectoriser.
    """
    vectors = TfidfVectorizer().fit_transform([text_1, text_2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0].item()


def compute_embedding_cosine_sim(
    text_1: str,
    text_2: str,
    model: SentenceTransformer,
) -> float:
    emb_1 = model.encode(text_1, convert_to_tensor=True)
    emb_2 = model.encode(text_2, convert_to_tensor=True)
    return sentence_transformers.util.pytorch_cos_sim(emb_1, emb_2).item()


def merge_issues(issues: list[BaseIssue]) -> list[BaseIssue]:
    """
    Merge multiple BaseIssue instances with the same `id` (i.e., same data unit)
    by combining their metadata. If ref_data differs, keep all unique.
    """
    grouped = defaultdict(list)
    for issue in issues:
        grouped[(issue.id, issue.ref_id, issue.data, issue.ref_data)].append(issue)

    merged_issues = []
    for key, issue_list in grouped.items():
        if len(issue_list) > 2:
            base = issue_list[0]
            for _issue in issue_list[1:]:
                base.meta.update(_issue.meta)
            merged_issues.append(base)
        else:
            merged_issues.append(issue_list[0])

    return merged_issues


def clean_band(band_value):
    # TODO: scoring the band score under the 4 using AI model or rule
    if isinstance(band_value, str):
        band_value = band_value.strip()
        if band_value == "<4":
            return 3
        try:
            return float(band_value)
        except ValueError:
            return np.nan
    return band_value


def call_gemini(prompt: str) -> str:
    """ref: https://github.com/google-gemini/gemini-cli"""
    try:
        result = subprocess.run(
            ["npx", "https://github.com/google-gemini/gemini-cli", "-p", prompt],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return ""
