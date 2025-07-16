"""
simple_inference.py

This script performs token classification (e.g., NER) inference on a set of text files using a HuggingFace Transformers model.
It reads .txt files, runs the model, and writes .ann annotation files in BRAT format.

Usage example:
  python simple_inference.py -i <input_txt_dir> -o <output_ann_dir> -m <model_path> [--overwrite] [--agg_strat <strategy>]

Author: Jan Rodríguez Miret
"""

import os
import re
import argparse
import sys
import torch
from glob import glob
from transformers import RobertaForTokenClassification, AutoTokenizer, pipeline
from spacy.lang.es import Spanish


def parse_args():
    parser = argparse.ArgumentParser(description="Specify variables for the Model Inference")
    parser.add_argument('-i', "--txts_path", required=True, type=str, help="Input directory containing the .txt text files")
    parser.add_argument('-m', "--model_path", required=True, type=str, help="Path to the model")
    parser.add_argument("-o", "--anns_path", required=False, type=str, help="Output directory for .ann annotation files")
    parser.add_argument("-ow", "--overwrite", action='store_true', default=False, help="Overwrite current .ann files in the output directory with new generated ones")
    parser.add_argument("-agg", "--agg_strat", default="first", type=str, help="Aggregation strategy. One of ('simple', 'first', 'max', or 'average'")
    return parser

def get_added_spaces(sentence, sentence_pretokenized):
    """
    Given an original sentence and its pretokenized version (with added spaces),
    return a list of indices where spaces were added in the pretokenized string.
    This is used to align model predictions back to the original text offsets.
    """
    # 'i' contains the current character index of 'sentence'
    # 'j' contains the current character index of 'sentence_pretokenized' (which has added_spaces)
    i = j = 0
    added_spaces = []
    while j < len(sentence_pretokenized):
        if sentence[i] == sentence_pretokenized[j]:
            i += 1
            j += 1
        elif sentence[i] == sentence_pretokenized[j+1] and sentence_pretokenized[j] == ' ':
            added_spaces.append(j)
            j += 1
        else:
            raise AssertionError("This should never be called.")
    return added_spaces

def align_results(results_pre, added_spaces, start_sent_offset):
    """
    Adjusts the entity offsets in the model's output to match the original text,
    correcting for any added spaces during pretokenization.
    Returns a list of aligned entity dictionaries.
    """
    aligned_results = []
    for entity in results_pre:
        aligned_entity = entity.copy()
        num_added_spaces_before = len(list(filter(lambda offset: offset < entity['start'], added_spaces)))
        num_added_spaces_after = len(list(filter(lambda offset: offset < entity['end'], added_spaces)))
        added_spaces_between = list(filter(lambda offset: (offset > entity['start']) & (offset < entity['end']), added_spaces))
        aligned_entity['word'] = entity['word'].strip()
        aligned_entity['word'] = ''.join([char for i, char in enumerate(aligned_entity['word']) if i + aligned_entity['start'] not in added_spaces_between])
        aligned_entity['start'] = start_sent_offset + entity['start'] - num_added_spaces_before
        aligned_entity['end'] = start_sent_offset + entity['end'] - num_added_spaces_after
        aligned_results.append(aligned_entity)
    return aligned_results

def write_to_ann(ann_path, results):
    """
    Write the model's entity predictions to a .ann file in BRAT format.
    Each entity is written as a line: T<ID>\t<LABEL> <START> <END>\t<TEXT>
    """
    results_ann_str = "\n".join([f"T{tid+1}\t{result['entity_group']} {result['start']} {result['end']}\t{result['word']}" for tid, result in enumerate(results)])
    with open(ann_path, "w+") as file:
        file.write(results_ann_str)

def make_inference(args):
    """
    Main inference function. Loads the model and tokenizer, processes all .txt files in the input directory,
    runs the model, aligns results, and writes .ann files to the output directory.
    Skips files that already have .ann files unless --overwrite is specified.
    """
    # Parse arguments
    TXTS_PATH, ANNS_PATH, MODEL_PATH, OVERWRITE, AGG_STRAT = args.txts_path, args.anns_path, args.model_path, args.overwrite, args.agg_strat
    
    # Load the HuggingFace model and tokenizer
    model = RobertaForTokenClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Regex for pretokenization (splitting words and punctuation)
    PRETOKENIZATION_REGEX = re.compile(
        r'([0-9A-Za-zÀ-ÖØ-öø-ÿ]+|[^0-9A-Za-zÀ-ÖØ-öø-ÿ])')
    
    # Load spaCy Spanish pipeline for sentence splitting
    nlp = Spanish()
    nlp.add_pipe("sentencizer")
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("token-classification", model=MODEL_PATH, aggregation_strategy=AGG_STRAT, device=device) # "simple" allows for different tags in a word, otherwise "first", "average", or "max".
    
    # Get all .txt files recursively from input directory
    txts_paths = glob(f"{TXTS_PATH}/**/*.txt", recursive=True)
    
    # If not overwriting, skip .txt files that already have .ann files in output
    if OVERWRITE == False:
        existing_anns = glob(f"{ANNS_PATH}/**/*.ann", recursive=True)
        existing_anns = list(map(lambda ann: ann[:-4] + ".txt", existing_anns))
        txts_paths = list(set(txts_paths).difference(existing_anns))
        if len(existing_anns):
            print(f"Warning: there are {len(existing_anns)} omitted .txt files, as they already have .ann files in the output directory. Please use --overwrite (-ow) to overwrite them.")
    
    # Process each .txt file
    for i_txt, txt_path in enumerate(txts_paths):
        # Build output .ann filename
        ann_name = os.path.basename(txt_path)[:-4] + ".ann"
        ann_path = os.path.join(ANNS_PATH, ann_name)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(ann_path), exist_ok=True)
        
        # Read lines from the .txt file
        lines = open(txt_path, "r").readlines()
        results_file = []
        start_sent_offset = 0
        
        # Track the offset of the start of each line in the file
        line_start_offset = 0
        for line in lines:
            doc = nlp(line)
            sents = list(doc.sents)
            for sentence in sents:
                # Pretokenize sentence for model compatibility
                pretokens = [t for t in PRETOKENIZATION_REGEX.split(sentence.text) if t]
                # Add space between two non-space pretokens (we cannot join by whitespace directly,
                # because of double whitespaces, tabs).
                # This is necessary because the model expects tokens to be separated by spaces.
                # The loop checks each pair of consecutive pretokens, and if both are not whitespace,
                # it inserts a space between them. After inserting, it updates the length and index
                # to account for the new space. This ensures that the pretokenized sentence matches
                # the expected input format for the model, and that the mapping between original and
                # pretokenized text can be reconstructed for offset alignment.
                i_pret = 1
                len_pretokens = len(pretokens)
                while i_pret < len_pretokens:
                    if (not pretokens[i_pret-1].isspace() and not pretokens[i_pret].isspace()):
                        pretokens.insert(i_pret, " ")
                        len_pretokens = len(pretokens)
                        i_pret += 1 # Move one more because we added one before
                    i_pret += 1
                sentence_pretokenized = ''.join(pretokens)
                # Find where spaces were added
                added_spaces = get_added_spaces(sentence.text, sentence_pretokenized)
                # Run model inference
                results_pre = pipe(sentence_pretokenized)
                # Align model results to original text offsets
                # Use sentence.start_char + line_start_offset for robust offset alignment
                results_sent = align_results(results_pre, added_spaces, sentence.start_char + line_start_offset)
                results_file.extend(results_sent)
            line_start_offset += len(line)
        # Write all entity results to .ann file
        write_to_ann(ann_path, results_file)
        print(f"Finished {i_txt+1}/{len(txts_paths)} ({round((i_txt+1)*100/len(txts_paths), 3)}%)")


def main(argv):
    """
    Entry point for the script. Parses arguments, checks input directory, and calls make_inference().
    """
    args = parse_args().parse_args(argv[1:])
    if not args.anns_path:
        args.anns_path = args.txts_path
    if not os.path.isdir(args.txts_path):
        raise NotADirectoryError("Input directory (-i) does not exist")
    # Generate the .conll files recursively
    make_inference(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
