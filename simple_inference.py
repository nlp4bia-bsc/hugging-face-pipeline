import os
import re
import argparse
import sys
import time
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
    results_ann_str = "\n".join([f"T{tid+1}\t{result['entity_group']} {result['start']} {result['end']}\t{result['word']}" for tid, result in enumerate(results)]) + "\n"
    with open(ann_path, "w+") as file:
        file.write(results_ann_str)

def make_inference(args):
    TXTS_PATH, ANNS_PATH, MODEL_PATH, OVERWRITE, AGG_STRAT = args.txts_path, args.anns_path, args.model_path, args.overwrite, args.agg_strat
    # Initialize global variables
    model = RobertaForTokenClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    PRETOKENIZATION_REGEX = re.compile(
        r'([0-9A-Za-zÀ-ÖØ-öø-ÿ]+|[^0-9A-Za-zÀ-ÖØ-öø-ÿ])')
    nlp = Spanish()
    nlp.add_pipe("sentencizer")
    pipe = pipeline("token-classification", model=MODEL_PATH, aggregation_strategy=AGG_STRAT, device=0) # "simple" allows for different tags in a word, otherwise "first", "average", or "max".
    
    txts_paths = glob(f"{TXTS_PATH}/**/*.txt", recursive=True)
    if OVERWRITE == False:
        existing_anns = glob(f"{ANNS_PATH}/**/*.ann", recursive=True)
        existing_anns = list(map(lambda ann: ann[:-4] + ".txt", existing_anns))
        txts_paths = list(set(txts_paths).difference(existing_anns))
        if len(existing_anns):
            print(f"Warning: there are {len(existing_anns)} omitted .txt files, as they already have .ann files in the output directory. Please use --overwrite (-ow) to overwrite them.")
    
    
    for i_txt, txt_path in enumerate(txts_paths[:1]):
        ann_name = txt_path[:-4] + ".ann"
        ann_path = os.path.join([ANN_PATH, ann_name])
        lines = open(txt_path, "r").readlines()
        results_file = []
        start_sent_offset = 0
        for line in lines:
            doc = nlp(line)
            sents = list(doc.sents)
            for sentence in sents:
                pretokens = [t for t in PRETOKENIZATION_REGEX.split(sentence.text) if t]
                # Add space between two non-space pretokens
                i_pret = 1
                len_pretokens = len(pretokens)
                while i_pret < len_pretokens:
                    if (not pretokens[i_pret-1].isspace() and not pretokens[i_pret].isspace()):
                        pretokens.insert(i_pret, " ")
                        len_pretokens = len(pretokens)
                        i_pret += 1 # We have to move one more because we added one before
                    i_pret += 1
                sentence_pretokenized = ''.join(pretokens)
                added_spaces = get_added_spaces(sentence.text, sentence_pretokenized)
                results_pre = pipe(sentence_pretokenized)
                results_sent = align_results(results_pre, added_spaces, start_sent_offset)
                results_file.extend(results_sent)
                start_sent_offset += len(sentence.text)
        write_to_ann(ann_path, results_file)
        print(f"Finished {i_txt+1}/{len(txts_paths)} ({(i_txt+1)*100/len(txts_paths)}%)")


def main(argv):
    args = parse_args().parse_args(argv[1:])
    if not args.anns_path:
        args.anns_path = args.txts_path
    if not os.path.isdir(args.txts_path):
        raise NotADirectoryError("Input directory (-i) does not exist")
    # Generate the .conll files recursively
    make_inference(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
