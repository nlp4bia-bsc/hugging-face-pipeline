# %% [markdown]
# # Model inference
# Author: Jan Rodríguez Miret
# 
# It makes inference on the test set of the given dataset, creating the ann files.
# We need to create the CoNLL during the process.

# %%
import os
import pandas as pd
import torch
from datasets import load_dataset
import csv
import argparse
import sys

# %%
# General

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Specify variables for the Model Inference")
    parser.add_argument('-ds', "--dataset", required=True, type=str, help="Path to the HF dataset")
    parser.add_argument('-m', "--model", required=True, type=str, help="Path to the model")
    parser.add_argument("--merged_conll", required=True, type=str, help="Path to the merged CoNLL file")
    parser.add_argument("--original_conlls_dir", required=True, type=str, help="Directory containing original CoNLL files")
    parser.add_argument("--original_txts_dir", required=True, type=str, help="Directory containing original TXT files (defaults to original_conlls_dir)")
    parser.add_argument("--output_anns_dir", required=True, type=str, help="Output directory for annotations")
    parser.add_argument("--output_conlls_dir", required=True, type=str, help="Output directory for CoNLL files")
    return parser


def model_inference(args):
    # Environment variables
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:4096" # otherwise we get HIP Error for memory fragmentation
    #os.environ["WANDB_NOTEBOOK_NAME"] = f"{PROJECT_NAME.replace('-','_')}.ipynb"

    # %%
    HF_DATASET = args.dataset
    MODEL_PATH = args.model
    MERGED_CONLL = args.merged_conll
    ORIGINAL_CONLLS_DIR = args.original_conlls_dir
    ORIGINAL_TXTS_DIR = args.original_txts_dir if args.original_txts_dir else ORIGINAL_CONLLS_DIR
    OUTPUT_ANNS_DIR = args.output_anns_dir
    OUTPUT_CONLLS_DIR = args.output_conlls_dir

    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device

    # %%
    dataset = load_dataset(HF_DATASET)
    dataset

    # %% [markdown]
    # ## Training

    # %% [markdown]
    # ### Prepare data & class weights

    # %%
    classes = dataset["train"].features["ner_tags"].feature
    id2label = {idx: tag for idx, tag in enumerate(classes.names)}
    label2id = {tag: idx for idx, tag in enumerate(classes.names)}

    # %%
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer

    # %%
    def tokenize_and_align_labels(samples):
        tokenized_inputs = tokenizer(samples["tokens"], truncation=True, is_split_into_words=True)

        labs = []
        for i, label in enumerate(samples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to the current label
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labs.append(label_ids)

        tokenized_inputs["labels"] = labs
        return tokenized_inputs

    # %%
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset

    # %% [markdown]
    # ### Evaluation metrics

    # %%
    from transformers import RobertaForTokenClassification

    model = RobertaForTokenClassification.from_pretrained(MODEL_PATH).to(device)

    # %%
    from transformers import DataCollatorForTokenClassification

    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')

    # %% [markdown]
    # ## Evaluate model

    # %%
    def forward_pass_with_label(batch):
        # Convertimos los datos en una lista de diccionarios para que puedan ser procesados por el
        # data collator.
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        # Padding de las entradas y etiquetas para obtener las predicciones.
        new_batch = data_collator(features)
        input_ids = new_batch["input_ids"].to(device)
        attention_mask = new_batch["attention_mask"].to(device)
        labels = new_batch["labels"].to(device)
        with torch.no_grad():
            # Pasa los datos a través del modelo
            output = model(input_ids, attention_mask)
            # Logit.size: [batch_size, sequence_length, classes]
            # Predecimos la clase más probable como aquella que tenga el logit más alto.
            predicted_label = torch.argmax(output.logits, axis=-1).cpu().numpy()
        # Calculamos la loss por token. La los en NER está siendo cross_entroy
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(output.logits.view(-1, classes.num_classes),
                            labels.view(-1))
        # Hacemos el unflatten para ponerlo en formato de salida
        loss = loss.view(len(input_ids), -1).cpu().numpy()

        return {"loss":loss, "predicted_label": predicted_label, 'logits': output.logits}

    # %%
    # Make inference
    test_subset = tokenized_dataset["test"].map(batched=True, batch_size=32, remove_columns=["id","ner_tags","tokens"])
    test_subset = test_subset.map(forward_pass_with_label, batched=True, batch_size=32)
    test_df = test_subset.to_pandas()

    # %%
    df = test_df.copy()

    # %%
    df_tokens = df.apply(lambda x: x.apply(pd.Series).stack())
    # NaN comes from padding-added tokens. For ignored tokens (special characters and not-first subtokens of a word), label is -100
    df_tokens = df_tokens.dropna()
    df_tokens

    # %%
    # Add label in string format (int to string)
    df_tokens['labels_str'] = df_tokens['labels'].apply(lambda x: 'IGN' if x not in id2label else id2label[x])
    df_tokens['predicted_label_str'] = df_tokens['predicted_label'].apply(lambda x: 'IGN' if x not in id2label else id2label[x])

    # %%
    # Filter out predictions that should be ignored
    df_filtered = df_tokens[df_tokens['labels'] != -100.]

    # %%
    # Load the reference CoNLL (whole split)
    df_conll = pd.read_csv(MERGED_CONLL, sep='\t', quoting=csv.QUOTE_NONE, header=None, na_filter=False)
    df_conll.columns = ['label', 'start', 'end', 'text']
    df_conll

    # %% [markdown]
    # ## Correct missing tokens due to truncation

    # %%
    df['token_length'] = df['labels'].apply(len)
    too_long_level_0 = df[df['token_length'] >= 512].index
    df[df['token_length'] >= 512] # max input RobertaModel

    # %%
    df_flat = df_filtered.reset_index()
    too_long_level_1 = df_flat[df_flat['level_0'].isin(too_long_level_0)].groupby('level_0')['level_1'].count().values
    too_long_level_1

    # %%
    labels_list = df_filtered['labels_str'].to_list()
    predicted_labels_list = df_filtered['predicted_label_str'].to_list()

    # %%
    for level_0, level_1 in zip(too_long_level_0, too_long_level_1):
        too_long_idx_flat = df_flat[(df_flat['level_0'] == level_0) & (df_flat['level_1'] == level_1)].index[0]
        tokenized_too_long = tokenizer(dataset['test'][4349]['tokens'], is_split_into_words=True, return_length=True)
        num_words = len(dataset['test'][level_0]['ner_tags'])
        print(f"{num_words - level_1 = }")
        for i in range(num_words - level_1):
            labels_list.insert(too_long_idx_flat, 'O')
            predicted_labels_list.insert(too_long_idx_flat, 'O')

    # %%
    print(f"{len(labels_list) = }")
    print(f"{len(df_conll) = }")
    print(f"{len(predicted_labels_list) = }")

    # %%
    # Make sure that both true labels from the dataset and CoNLL are the same
    assert labels_list ==  df_conll['label'].to_list()

    # %%
    # Replace true labels with predicted labels
    df_conll['label'] = predicted_labels_list

    # %%
    # Get the filenames of CoNLLs (important that they are sorted)
    original_conlls = sorted([filename for filename in os.listdir(ORIGINAL_CONLLS_DIR) if filename.endswith('.conll')])

    # %%
    os.makedirs(OUTPUT_CONLLS_DIR)

    # %%
    # Generate the .conll files by using offset
    current_offset = 0
    file_idx = 0 # Position of file within all retrieved with listdir
    start_token_idx = 0 # Index within the dataframe that marks the start of a file
    for idx, line in df_conll.iterrows():
        # If we reach the end of a file
        if line['start'] < current_offset:
            df_conll.loc[start_token_idx:idx-1].to_csv(os.path.join(OUTPUT_CONLLS_DIR, original_conlls[file_idx]), sep='\t', quoting=csv.QUOTE_NONE, header=None, index=False)
            file_idx += 1
            current_offset = 0
            start_token_idx = idx
        current_offset = line['end']
    # Add last document
    df_conll.loc[start_token_idx:idx].to_csv(os.path.join(OUTPUT_CONLLS_DIR, original_conlls[file_idx]), sep='\t', quoting=csv.QUOTE_NONE, header=None, index=False)

    # %%
    from brat.tools import BIOtoStandoff

    # %%
    os.makedirs(OUTPUT_ANNS_DIR)

    # %%
    # Write an .ann file for each .conll file by calling BIOtoStandoff.py
    conll_files = [file for file in os.listdir(OUTPUT_CONLLS_DIR) if file.endswith(".conll")]

    for conll_file in conll_files:
        txt_file = conll_file.replace('.conll', '.txt')
        argv = ["brat/tools/BIOtoStandoff.py", os.path.join(ORIGINAL_TXTS_DIR, txt_file), os.path.join(OUTPUT_CONLLS_DIR, conll_file), "-1", "0"]
        res = BIOtoStandoff.main(argv)
        ann_file = conll_file.replace('.conll', '.ann')
        with open(os.path.join(OUTPUT_ANNS_DIR, ann_file), 'w') as file:
            ann_content = map(lambda line: str(line)+'\n', res)
            file.writelines(ann_content)


def main(argv):
    args = parse_args().parse_args(argv[1:])
    # Generate the .conll files recursively
    model_inference(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))


