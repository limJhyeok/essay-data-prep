import os
import pandas as pd
from datasets import load_dataset
from callbacks.issue_finder_callbacks import GibberishFilterIssueFinderCallback
import utils
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from functools import partial
from issue_finders import duplicate_finders, similarity_finders, gibberish_finders
from preprocessors.preprocessor import NormalizePreprocessor
from annotators.annotator import (
    LongestNgramAnnotator,
    PerplexityAnnotator,
    TypeTokenRatioAnnotator,
)
from callbacks.annotator_callbacks import NormalizeAnnotatorCallback
import operator
import writer


if __name__ == "__main__":
    issue_report_dir_name = "issue_reports"
    issue_report_dir = os.path.join(os.getcwd(), issue_report_dir_name)
    os.makedirs(issue_report_dir, exist_ok=True)

    dataset_name = "chillies/IELTS-writing-task-2-evaluation"
    split = "train"
    min_rating = 0
    max_rating = 9
    ngram_threshold = 30

    short_essay_len = 20
    band_threshold = 4

    ppl_ckpt = "HuggingFaceTB/SmolLM2-135M"  # ppl: perplexity
    device = "cpu"
    ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_ckpt)
    ppl_model = AutoModelForCausalLM.from_pretrained(ppl_ckpt).to(device)

    gibberish_detector_ckpt = "madhurjindal/autonlp-Gibberish-Detector-492513457"
    gibberish_detector = pipeline(
        "text-classification",
        model=gibberish_detector_ckpt,
        tokenizer=gibberish_detector_ckpt,
        truncation=True,
        max_length=512,
    )
    gibberish_threshold = 0.5
    gibberish_noise_labels = ["noise", "word_salad"]

    ngram_ratio_threshold = 0.5
    emb_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
    emb_model = SentenceTransformer(emb_ckpt)
    emb_similarity_threshold = 0

    dataset = load_dataset(dataset_name)
    df_train = pd.DataFrame(dataset[split])

    df_train["band"] = df_train["band"].apply(utils.clean_band)

    # # # duplicate candidates
    preprocessors = [NormalizePreprocessor(normaliser=utils.normalize_string)]

    datasets = dataset[split]["essay"]
    for processor in preprocessors:
        datasets = processor.preprocess(datasets)

    duplicate_issues = []
    duplicate_finder_list = [
        duplicate_finders.NgramDuplicateFinder(
            ngram_threshold=30, select_fn=utils.select_longest_id
        )
    ]
    for finder in duplicate_finder_list:
        _issue = finder(datasets)
        duplicate_issues.extend(_issue)

    duplicate_issues = utils.merge_issues(duplicate_issues)

    annotators = [
        LongestNgramAnnotator(
            callbacks=[NormalizeAnnotatorCallback(utils.normalize_string)]
        )
    ]
    for anno in annotators:
        duplicate_issues = anno(duplicate_issues)

    writer.save_issues_to_csv(
        issues=duplicate_issues,
        filename=os.path.join(issue_report_dir, "duplicate_issues.csv"),
        field_mapping={
            "ref_id": "parent_id",
            "data": "essay",
            "ref_data": "parent_essay",
            "ref_length": "parent_length",
        },
    )

    # null check
    assert (
        (df_train["essay"].str.strip() == "").sum() + df_train["essay"].isna().sum()
    ).item() == 0
    assert (
        (df_train["prompt"].str.strip() == "").sum() + df_train["prompt"].isna().sum()
    ).item() == 0

    # type check
    assert all(isinstance(p, str) for p in dataset[split]["prompt"]), (
        "All elements in prompt must be strings"
    )
    assert all(isinstance(p, str) for p in dataset[split]["essay"]), (
        "All elements in essay must be strings"
    )
    assert all(isinstance(x, float) for x in df_train["band"]), (
        "All band scores must be floats"
    )

    # range check
    count_invalid_band = (
        (df_train["band"] < min_rating) | (df_train["band"] > max_rating)
    ).sum()
    assert count_invalid_band.item() == 0

    # short length but moderate score(label-data inconsistency)

    suspect_scoring_essays = df_train[
        (df_train["essay"].str.strip().str.len() <= short_essay_len)
        & (df_train["band"] >= band_threshold)
    ]
    suspect_scoring_essays = suspect_scoring_essays.reset_index().rename(
        columns={"index": "id"}
    )
    suspect_scoring_essays.to_csv(
        os.path.join(issue_report_dir, "suspect_scoring_issues.csv"), index=False
    )

    # capture similar essay to prmopt(copy & paste)
    similarity_fn = utils.longest_ngram_ratio
    similarity_finder = similarity_finders.SimilarityFinder(
        similarity_fn=similarity_fn,
        similarity_threshold=ngram_ratio_threshold,
        compare_op=operator.gt,
    )
    preprocessors = []

    datasets = zip(dataset[split]["prompt"], dataset[split]["essay"])
    for processor in preprocessors:
        datasets = processor.preprocess(datasets)

    similar_issues = []
    similarity_finder_list = [similarity_finder]
    for finder in similarity_finder_list:
        _issue = finder(datasets)
        similar_issues.extend(_issue)

    similar_issues = utils.merge_issues(similar_issues)

    annotators = [
        LongestNgramAnnotator(
            callbacks=[NormalizeAnnotatorCallback(utils.normalize_string)]
        )
    ]
    for anno in annotators:
        similar_issues = anno(similar_issues)

    writer.save_issues_to_csv(
        issues=similar_issues,
        filename=os.path.join(issue_report_dir, "prompt_similar_essays_issues.csv"),
        field_mapping={
            "data": "prompt",
            "ref_data": "essay",
            "ref_length": "essay_length",
        },
    )

    # capture gibberish essay
    gibberish_finder = gibberish_finders.DeepGibberishFinder(
        gibberish_detector, gibberish_noise_labels
    )
    # TODO: Debug: id is changed because of applying filter
    gibberish_finder.add_callback(GibberishFilterIssueFinderCallback())

    preprocessors = []

    datasets = dataset[split]["essay"]
    for processor in preprocessors:
        datasets = processor.preprocess(datasets)

    gibberish_issues = []
    gibberish_finder_list = [gibberish_finder]
    for finder in gibberish_finder_list:
        _issue = finder(datasets)
        gibberish_issues.extend(_issue)

    gibberish_issues = utils.merge_issues(gibberish_issues)

    annotators = [
        PerplexityAnnotator(ppl_model, ppl_tokenizer, device),
        TypeTokenRatioAnnotator(),
    ]
    for anno in annotators:
        gibberish_issues = anno(gibberish_issues)

    writer.save_issues_to_csv(
        issues=gibberish_issues,
        filename=os.path.join(issue_report_dir, "gibberish_issues.csv"),
        field_mapping={
            "data": "essay",
        },
    )

    # Off the prompt candidates(Essay Prompt-Content Alignment)

    similarity_fn = partial(
        utils.compute_embedding_cosine_sim, model=emb_model
    )  # deep learning
    # similarity_fn = utils.compute_tf_idf_cosine_sim # tf-idf

    off_the_prompt_finder = similarity_finders.SimilarityFinder(
        similarity_fn=similarity_fn,
        similarity_threshold=emb_similarity_threshold,
        compare_op=operator.le,
    )

    datasets = zip(dataset[split]["prompt"], dataset[split]["essay"])
    preprocessors = []
    for processor in preprocessors:
        datasets = processor.preprocess(datasets)

    similar_issues = []
    sim_finder_list = [off_the_prompt_finder]
    for finder in sim_finder_list:
        _issue = finder(datasets)
        similar_issues.extend(_issue)

    similar_issues = utils.merge_issues(similar_issues)

    annotators = []
    for anno in annotators:
        similar_issues = anno(similar_issues)

    writer.save_issues_to_csv(
        issues=similar_issues,
        filename=os.path.join(issue_report_dir, "off_the_prompt_issues.csv"),
        field_mapping={
            "data": "prompt",
            "ref_data": "essay",
            "ref_length": "essay_length",
        },
    )
