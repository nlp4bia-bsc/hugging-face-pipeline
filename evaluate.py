# Import necessary libraries
import pandas as pd
import json
import re

# Define paths to the TSV files
PREDS_PATH = "hugging-face-pipeline/pred.tsv"
GS_PATH = "hugging-face-pipeline/test.tsv"

# Load the TSV files into DataFrames
preds_mentions = pd.read_csv(PREDS_PATH, sep="\t")
gs_mentions = pd.read_csv(GS_PATH, sep="\t")

# Add interval columns for relaxed matching
gs_mentions["interval"] = gs_mentions.apply(lambda row: pd.Interval(row["start_span"], row["end_span"], closed="both"), axis=1)
preds_mentions["interval"] = preds_mentions.apply(lambda row: pd.Interval(row["start_span"], row["end_span"], closed="both"), axis=1)

# Merge DataFrames for relaxed matching
df_merged = gs_mentions.merge(preds_mentions, on="filename", suffixes=["_gs", "_pred"])

# Calculate overlap and label match
df_merged["overlap"] = df_merged.apply(
    lambda row: row["interval_gs"].overlaps(row["interval_pred"]), axis=1
)
df_merged["label_match"] = df_merged["label_gs"] == df_merged["label_pred"]

# Simple Overlap: Count all overlapping matches
df_merged["simple_match"] = df_merged["overlap"] & df_merged["label_match"]
TP_simple = df_merged["simple_match"].sum()
FP_simple = len(preds_mentions) - TP_simple
FN_simple = len(gs_mentions) - TP_simple

# Unique Overlap: Ensure each ground truth mention matches only one prediction
used_gs_indices = set()
TP_unique = 0

for _, row in df_merged.iterrows():
    if row["overlap"] and row["label_match"] and row.name not in used_gs_indices:
        TP_unique += 1
        used_gs_indices.add(row.name)

FP_unique = len(preds_mentions) - TP_unique
FN_unique = len(gs_mentions) - TP_unique

# Strict Boundary: Ensure exact boundary and type match
df_merged["strict_match"] = (
    (df_merged["start_span_gs"] == df_merged["start_span_pred"]) &
    (df_merged["end_span_gs"] == df_merged["end_span_pred"]) &
    (df_merged["label_gs"] == df_merged["label_pred"])
)
TP_strict = df_merged["strict_match"].sum()
FP_strict = len(preds_mentions) - TP_strict
FN_strict = len(gs_mentions) - TP_strict


def compute_fscore_per_entity(gs_mentions, preds_mentions):
    """
    Calculate F-score per entity type for both strict and relaxed matching.
    :param gs_mentions: DataFrame containing ground truth mentions
    :param preds_mentions: DataFrame containing predicted mentions
    :return: Dictionary with precision, recall, and F-score per entity type
    """
    labels = gs_mentions["label"].unique()
    scores_final = {}
    strict_TP, strict_FP, strict_FN = 0, 0, 0
    relaxed_TP, relaxed_FP, relaxed_FN = 0, 0, 0

    for label in labels:
        strict_tp, strict_fp, strict_fn = 0, 0, 0
        relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0

        gs_filtered = gs_mentions[gs_mentions["label"] == label]
        preds_filtered = preds_mentions[preds_mentions["label"] == label]

        strict_gs_used = set()
        strict_preds_used = set()
        relaxed_gs_used = set()
        relaxed_preds_used = set()

        for _, gs_row in gs_filtered.iterrows():
            for _, pred_row in preds_filtered.iterrows():
                if pred_row.name in strict_preds_used:
                    continue

                # Strict matching: exact boundary and label match
                if (
                    gs_row["start_span"] == pred_row["start_span"]
                    and gs_row["end_span"] == pred_row["end_span"]
                    and gs_row["label"] == pred_row["label"]
                ):
                    strict_tp += 1
                    strict_gs_used.add(gs_row.name)
                    strict_preds_used.add(pred_row.name)
                    break

        for _, gs_row in gs_filtered.iterrows():
            for _, pred_row in preds_filtered.iterrows():
                if pred_row.name in relaxed_preds_used:
                    continue

                # Relaxed matching: overlap and label match
                if (
                    gs_row["interval"].overlaps(pred_row["interval"])
                    and gs_row["label"] == pred_row["label"]
                ):
                    relaxed_tp += 1
                    relaxed_gs_used.add(gs_row.name)
                    relaxed_preds_used.add(pred_row.name)
                    break

        strict_fp = len(preds_filtered) - len(strict_preds_used)
        strict_fn = len(gs_filtered) - len(strict_gs_used)
        relaxed_fp = len(preds_filtered) - len(relaxed_preds_used)
        relaxed_fn = len(gs_filtered) - len(relaxed_gs_used)

        try:
            strict_precision = strict_tp / (strict_tp + strict_fp)
        except ZeroDivisionError:
            strict_precision = 0
        try:
            strict_recall = strict_tp / (strict_tp + strict_fn)
        except ZeroDivisionError:
            strict_recall = 0
        strict_fscore = (
            2 * strict_precision * strict_recall / (strict_precision + strict_recall)
            if strict_precision + strict_recall > 0
            else 0
        )

        try:
            relaxed_precision = relaxed_tp / (relaxed_tp + relaxed_fp)
        except ZeroDivisionError:
            relaxed_precision = 0
        try:
            relaxed_recall = relaxed_tp / (relaxed_tp + relaxed_fn)
        except ZeroDivisionError:
            relaxed_recall = 0
        relaxed_fscore = (
            2 * relaxed_precision * relaxed_recall / (relaxed_precision + relaxed_recall)
            if relaxed_precision + relaxed_recall > 0
            else 0
        )

        scores_final[f"{label}_strict_precision"] = strict_precision
        scores_final[f"{label}_strict_recall"] = strict_recall
        scores_final[f"{label}_strict_fscore"] = strict_fscore
        scores_final[f"{label}_relaxed_precision"] = relaxed_precision
        scores_final[f"{label}_relaxed_recall"] = relaxed_recall
        scores_final[f"{label}_relaxed_fscore"] = relaxed_fscore

        strict_TP += strict_tp
        strict_FP += strict_fp
        strict_FN += strict_fn
        relaxed_TP += relaxed_tp
        relaxed_FP += relaxed_fp
        relaxed_FN += relaxed_fn

    try:
        strict_micro_precision = strict_TP / (strict_TP + strict_FP)
    except ZeroDivisionError:
        strict_micro_precision = 0
    try:
        strict_micro_recall = strict_TP / (strict_TP + strict_FN)
    except ZeroDivisionError:
        strict_micro_recall = 0
    strict_micro_fscore = (
        2 * strict_micro_precision * strict_micro_recall
        / (strict_micro_precision + strict_micro_recall)
        if strict_micro_precision + strict_micro_recall > 0
        else 0
    )

    try:
        relaxed_micro_precision = relaxed_TP / (relaxed_TP + relaxed_FP)
    except ZeroDivisionError:
        relaxed_micro_precision = 0
    try:
        relaxed_micro_recall = relaxed_TP / (relaxed_TP + relaxed_FN)
    except ZeroDivisionError:
        relaxed_micro_recall = 0
    relaxed_micro_fscore = (
        2 * relaxed_micro_precision * relaxed_micro_recall
        / (relaxed_micro_precision + relaxed_micro_recall)
        if relaxed_micro_precision + relaxed_micro_recall > 0
        else 0
    )

    scores_final["strict_micro_precision"] = strict_micro_precision
    scores_final["strict_micro_recall"] = strict_micro_recall
    scores_final["strict_micro_fscore"] = strict_micro_fscore
    scores_final["relaxed_micro_precision"] = relaxed_micro_precision
    scores_final["relaxed_micro_recall"] = relaxed_micro_recall
    scores_final["relaxed_micro_fscore"] = relaxed_micro_fscore

    return scores_final


# Calculate F-scores per entity
scores = compute_fscore_per_entity(gs_mentions, preds_mentions)

# Calculate macro-average precision, recall, and F1-score
macro_precision = sum(
    scores[f"{label}_strict_precision"] for label in gs_mentions["label"].unique()
) / len(gs_mentions["label"].unique())

macro_recall = sum(
    scores[f"{label}_strict_recall"] for label in gs_mentions["label"].unique()
) / len(gs_mentions["label"].unique())

macro_f1 = sum(
    scores[f"{label}_strict_fscore"] for label in gs_mentions["label"].unique()
) / len(gs_mentions["label"].unique())

# Calculate macro-average relaxed precision, recall, and F1-score
relaxed_macro_precision = sum(
    scores[f"{label}_relaxed_precision"] for label in gs_mentions["label"].unique()
) / len(gs_mentions["label"].unique())

relaxed_macro_recall = sum(
    scores[f"{label}_relaxed_recall"] for label in gs_mentions["label"].unique()
) / len(gs_mentions["label"].unique())

relaxed_macro_f1 = sum(
    scores[f"{label}_relaxed_fscore"] for label in gs_mentions["label"].unique()
) / len(gs_mentions["label"].unique())

# Group scores by label and include overall metrics
grouped_scores = {
    "metrics_by_label": {},
    "summary": {
        "strict_micro": {
            "precision": round(scores["strict_micro_precision"], 4),
            "recall": round(scores["strict_micro_recall"], 4),
            "f1": round(scores["strict_micro_fscore"], 4)
        },
        "strict_macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4)
        },
        "relaxed_micro": {
            "precision": round(scores["relaxed_micro_precision"], 4),
            "recall": round(scores["relaxed_micro_recall"], 4),
            "f1": round(scores["relaxed_micro_fscore"], 4)
        },
        "relaxed_macro": {
            "precision": round(relaxed_macro_precision, 4),
            "recall": round(relaxed_macro_recall, 4),
            "f1": round(relaxed_macro_f1, 4)
        }
    }
}

# Regular expression to extract label, metric type, and metric name
pattern = re.compile(r"^(.*?)_(strict|relaxed)_(precision|recall|fscore)$")

for key, value in scores.items():
    match = pattern.match(key)
    if match:
        label, metric_type, metric = match.groups()
        if label not in grouped_scores["metrics_by_label"]:
            grouped_scores["metrics_by_label"][label] = {"strict": {}, "relaxed": {}}
        grouped_scores["metrics_by_label"][label][metric_type][metric] = round(value, 4)

# Convert the grouped scores dictionary to JSON
scores_json = json.dumps(grouped_scores, indent=4)

# Print the JSON
print(scores_json)

# Return the JSON object
def get_scores_as_json():
    return scores_json