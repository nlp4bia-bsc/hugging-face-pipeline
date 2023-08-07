#!/usr/bin/env python3
"""
Generate the .conll files recursively of a given directory containing texts (.txt) and annotations (.ann)
"""
import sys
import os
import argparse
from brat.tools import anntoconll


def argparser():
    ap = argparse.ArgumentParser(description='Convert text and standoff ' +
                                 'annotations into CoNLL format.')
    ap.add_argument('-r', '--root-dir',
                    help='Root directory containing .txt and .ann files', required=True)
    return ap


def brat2conll_recursive(dir):
    if os.path.isfile(dir):
        anntoconll.main([dir])
    else:
        # Call the brat2conll recursively to all subdirectories in the directory
        for child in [f.path for f in os.scandir(dir) if f.is_dir()]:
            brat2conll_recursive(child)
        # Get all (.txt) files in the directory and generate .conll files
        # using the anntoconll from brat original repository
        files = [f.path for f in os.scandir(
            dir) if f.name.endswith('.txt')]
        if files:
            anntoconll.main(files)


def main(argv):
    options = argparser().parse_args(argv[1:])
    # Generate the .conll files recursively
    brat2conll_recursive(options.root_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
