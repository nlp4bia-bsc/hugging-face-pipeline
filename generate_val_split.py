#!/usr/bin/env python3
"""
Creates a new directory with a new split for a brat annotated dataset.
Example usage:
python generate_val_split.py \
    --source corpus_carmen_ner/train \
    --destination corpus_carmen_ner/test \
    --test_size 125 
"""
import os
import sys
import argparse
from sklearn.model_selection import train_test_split


def argparser():
    ap = argparse.ArgumentParser(
        description='Create a new directory containing a new split for a brat annotated dataset')
    ap.add_argument('-s', '--source',
                    help='Source (train) directory containing .txt and .ann files', required=True, default="train")
    ap.add_argument('-d', '--destination',
                    help='Destination (validation) directory that will contain the new split (.txt and .ann files)', required=True, default="valid")
    ap.add_argument('--test_size',
                    help='Size (number of .ann files) of the new split (e.g. 2500, 0.33). Use decimal to indicate percentage', default=1/3, type=lambda x: int(x) if x.isdigit() else float(x))
    return ap


def generate_split(source, destination, test_size):
    # Get all text filenames
    text_files = [filename for filename in os.listdir(
        source) if filename.endswith(".txt")]

    # Generate the splits
    train, valid = train_test_split(
        text_files, test_size=test_size, random_state=42)

    # Create the new valid split
    os.makedirs(destination)

    # Move (rename) all selected files for validation (both txt and ann files)
    for text_file in valid:
        ann_file = text_file.replace(".txt", ".ann")
        os.rename(f"{source}/{text_file}", f"{destination}/{text_file}")
        os.rename(f"{source}/{ann_file}", f"{destination}/{ann_file}")


def main(argv):
    options = argparser().parse_args(argv[1:])
    # Generate the .conll files recursively
    generate_split(options.source, options.destination, options.test_size)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
