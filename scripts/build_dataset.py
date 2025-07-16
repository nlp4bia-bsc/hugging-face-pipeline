#!/usr/bin/env python3
"""
Author: Jan Rodríguez
Date: 10/04/2024
"""
# TODO: improve robustness (do not call bash processes)

import subprocess
import sys
import argparse
import os
from shutil import copy2
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Specify variables for the Model Inference")
    parser.add_argument('-d', "--dir", required=True, type=str, help="Path to the ann+txt dataset")
    parser.add_argument('-n', "--name", required=True, type=str, help="Name of the created HF dataset (directory)")

    return parser

def to_camel_case(text):
    s = text.replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0] + ''.join(i.capitalize() for i in s[1:])

def build_dataset(args):
    # Execute the brat2conll
    subprocess.run(f'python hugging-face-pipeline/scripts/brat2conll.py -r {args.dir}', shell=True)
    # Join all conll files
    subprocess.run(f'bash hugging-face-pipeline/scripts/join_all_conlls.sh {args.dir}', shell=True)
    # Create the HF dataset dir
    os.makedirs(args.name)
    loader_path = f"{args.name}/{os.path.basename(args.name)}.py"
    copy2(src="hugging-face-pipeline/templates/meddoplace-ner.py", dst=loader_path)
    name_camel = to_camel_case(os.path.basename(args.name))
    # Change names in the HF dataset loading script
    subprocess.run(f'sed -i "s/\meddoplace/{name_camel}/g;s/Meddoplace/{name_camel[0].upper()+name_camel[1:]}/g;s/MEDDOPLACE/{name_camel.upper()}/g" {loader_path}', shell=True)
    # Get all classes that appear in the dataset (ordered) and substitute them in the HF dataset loading script
    command = "find {}/train {}/valid {}/test -name '*.ann' -type f -exec grep -Hr '^T' {{}} + | cut -f2 | cut -d' ' -f1 | sort | uniq".format(args.dir, args.dir, args.dir)
    print(f"{command=}")
    class_names = subprocess.run(command, shell=True, text=True, capture_output=True).stdout.splitlines()
    print(f"{class_names=}")
    class_names_str = "CLASS_NAMES = [" + ', '.join([f"'{name}'" for name in class_names]) + "]"
    print(f"{class_names_str=}")
    subprocess.run(f'sed -i "s/CLASS_NAMES = \[\'EXAMPLE_CLASS\'\]/{class_names_str}/" {loader_path}', shell=True)
    # Copy generated files from brat2conll
    copy2(f"{args.dir}/train.conll", args.name)
    copy2(f"{args.dir}/validation.conll", args.name)
    copy2(f"{args.dir}/test.conll", args.name)
    # Make sure the dataset can load correctly
    dataset = load_dataset(args.name, download_mode='force_redownload', trust_remote_code=True)
    print(dataset)
    print(dataset['train'][0])


def main(argv):
    args = parse_args().parse_args(argv[1:])
    # Generate the .conll files recursively
    build_dataset(args)
    

if __name__ == "__main__":
    sys.exit(main(sys.argv))
