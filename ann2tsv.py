#!/usr/bin/env python
# coding: utf-8

# ## .ann to .tsv (with filename)
# Merge annotation files (.ann) into one file and include filename
# 
# After merging all resulting .ann files to one using:
# ```bash
# find . -name '*.ann' -type f -exec grep "^T" {} + > ../all_merged.ann
# ```
# 
# We then adapt the columns of the .ann file with the format required by the subtask's evaluation.

# In[13]:
import pandas as pd
import csv
import argparse
import subprocess
import sys
import os

def check_output_filename(filename):
    if not filename.endswith('.tsv'):
        raise ValueError("Outputfile should end with .tsv")
    return filename

def parse_args(args):
    parser = argparse.ArgumentParser(description="Training configuration for machine learning model")
    parser.add_argument('--input_dir', '-i', type=str, required=True, help="Directory containing .ann files to merge.")
    parser.add_argument('--output_file', '-o', type=check_output_filename, required=True, help="Path to resulting .tsv output file.")

    args = parser.parse_args()
    return args

def main(args):
    options = parse_args(args)
    tmp_merged_ann = options.output_file.replace('.tsv', '.ann')
    # Merge all .ann files into one, with the filename included
    command = f'find {options.input_dir} -name "*.ann" -type f -exec grep "^T" {{}} + > {tmp_merged_ann}'
    subprocess.run(command, shell=True)
    
    df = pd.read_csv(tmp_merged_ann, quoting=csv.QUOTE_NONE,
                     sep="\t", usecols=[0, 1, 2], names=['id', "label", 'text'], header=None)
    
    df['start_span'] = df['label'].apply(lambda elem: elem.split()[1])
    df['end_span'] = df['label'].apply(lambda elem: elem.split()[2])
    df['label'] = df['label'].apply(lambda elem: elem.split()[0])
    df['filename'] = df['id'].apply(lambda elem: elem.split(':')[0])
    df['filename'] = df['filename'].apply(lambda filename: filename.split(os.sep)[-1].replace('.ann',''))
    df['ann_id'] = df['id'].apply(lambda elem: elem.split(':')[1])
    df = df.drop(columns='id')
    # Reorder columns
    df = df[['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text']]
    df.to_csv(options.output_file, quoting=csv.QUOTE_NONE, sep="\t", index=False, header=True)
    os.remove(tmp_merged_ann)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
