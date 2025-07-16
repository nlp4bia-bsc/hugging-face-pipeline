# Import necessary libraries
import pandas as pd
import json
import re
import csv
import argparse
import sys

def compute_fscore_per_entity(gs_mentions, preds_mentions, output_json=None):
    """
    Calculate F-score per entity type for both strict and relaxed matching, comparing only mentions within the same file.
    Optionally, process and output the results as JSON to a file or stdout.
    :param gs_mentions: DataFrame containing ground truth mentions
    :param preds_mentions: DataFrame containing predicted mentions
    :param output_json: Path to output JSON file (optional)
    :return: Dictionary with precision, recall, and F-score per entity type
    """
    # Add interval columns for relaxed matching
    gs_mentions["interval"] = gs_mentions.apply(lambda row: pd.Interval(row["start_span"], row["end_span"], closed="both"), axis=1)
    preds_mentions["interval"] = preds_mentions.apply(lambda row: pd.Interval(row["start_span"], row["end_span"], closed="both"), axis=1)

    labels = gs_mentions["label"].unique()
    scores_final = {}
    strict_TP, strict_FP, strict_FN = 0, 0, 0
    relaxed_TP, relaxed_FP, relaxed_FN = 0, 0, 0

    for label in labels:
        strict_tp, strict_fp, strict_fn = 0, 0, 0
        relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0

        # Get all unique filenames for this label in either gs or preds
        filenames = set(gs_mentions[gs_mentions["label"] == label]["filename"]).union(
            set(preds_mentions[preds_mentions["label"] == label]["filename"]))

        for filename in filenames:
            gs_filtered = gs_mentions[(gs_mentions["label"] == label) & (gs_mentions["filename"] == filename)]
            preds_filtered = preds_mentions[(preds_mentions["label"] == label) & (preds_mentions["filename"] == filename)]

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

            strict_fp += len(preds_filtered) - len(strict_preds_used)
            strict_fn += len(gs_filtered) - len(strict_gs_used)
            relaxed_fp += len(preds_filtered) - len(relaxed_preds_used)
            relaxed_fn += len(gs_filtered) - len(relaxed_gs_used)

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

    # --- Output processing and JSON ---
    labels = list(gs_mentions["label"].unique())
    def macro_avg(metric):
        return sum(scores_final[f"{label}_{metric}"] for label in labels) / len(labels)

    grouped_scores = {
        "metrics_by_label": {},
        "summary": {
            "strict_micro": {k: round(scores_final[f"strict_micro_{k}"], 4) for k in ("precision", "recall", "fscore")},
            "strict_macro": {k: round(macro_avg(f"strict_{k}"), 4) for k in ("precision", "recall", "fscore")},
            "relaxed_micro": {k: round(scores_final[f"relaxed_micro_{k}"], 4) for k in ("precision", "recall", "fscore")},
            "relaxed_macro": {k: round(macro_avg(f"relaxed_{k}"), 4) for k in ("precision", "recall", "fscore")},
        }
    }

    # Group metrics by label
    pattern = re.compile(r"^(.*?)_(strict|relaxed)_(precision|recall|fscore)$")
    for key, value in scores_final.items():
        match = pattern.match(key)
        if match:
            label, metric_type, metric = match.groups()
            grouped_scores["metrics_by_label"].setdefault(label, {"strict": {}, "relaxed": {}})[metric_type][metric] = round(value, 4)

    scores_json = json.dumps(grouped_scores, indent=4)

    if output_json:
        with open(output_json, "w") as f:
            f.write(scores_json)
    else:
        print(scores_json)

    return scores_final


def parse_args():
    """
    Parse command-line arguments for the evaluation script.
    Returns:
        argparse.Namespace: Parsed arguments with attributes preds, gs, and output_json.
    """
    parser = argparse.ArgumentParser(description="Evaluate NER predictions against gold standard.")
    parser.add_argument("--preds", required=True, help="Path to predictions TSV file")
    parser.add_argument("--gs", required=True, help="Path to gold standard TSV file")
    parser.add_argument("-o", "--output_json", required=False, help="Path to output JSON file (optional)")
    return parser

def main(argv=None):
    """
    Main entry point for evaluation. Loads prediction and gold standard TSVs, computes metrics, and prints or saves JSON output.
    
    Expected TSV format (columns):
    filename: Name of the document/file (e.g., S0716-10182006000400006-1.ann)
    ann_id:   Annotation ID (e.g., T1, T2, ...)
    label:    Entity label (e.g., AGE, AGE_PERSON, ...)
    start_span: Start character offset of the entity
    end_span:   End character offset of the entity
    text:    The entity text span
    """
    args = parse_args().parse_args(argv[1:]) if argv is not None else parse_args()
    preds_mentions = pd.read_csv(args.preds, sep="\t", quoting=csv.QUOTE_NONE, na_filter=False)
    gs_mentions = pd.read_csv(args.gs, sep="\t", quoting=csv.QUOTE_NONE, na_filter=False)
    compute_fscore_per_entity(gs_mentions, preds_mentions, args.output_json)


if __name__ == "__main__":
    main(sys.argv)