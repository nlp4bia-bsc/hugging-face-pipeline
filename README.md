# Hugging Face Pipeline

Automation tools for different processes regarding the training and deployment of models and datasets to Hugging Face.

NLP for Biomedical Information Analysis (NLP4BIA).

## Training Pipeline

The general pipeline for NER is the following:
1. Annotate using Brat, download the corpus. An example can be found in the [zip](https://zenodo.org/records/8224056/files/medprocner_gs_train+test+gazz+multilingual+crossmap_230808.zip?download=1) file of the [MedProcNER](https://zenodo.org/records/8224056) Shared Task page, in the train and test directories.
2. Make the splits, only **if not already split**, with the `generate_val_split.py` script. Indicate the source (-s), destination (-d), and test_size (number of txts if type integer; percentage if float between 0 and 1)
    - `python hugging-face-pipeline/generate_val_split.py -s <my-corpus>/train -d <my-corpus>/test --test_size 125`
3. Use the `build_dataset.py` script to create a Hugging Face Dataset with the pre-tokenized input. Indicate the source directory of the corpus (-d) and the output name/path of the Hugging Face dataset (-n).
    - `python hugging-face-pipeline/build_dataset.py -d <my-corpus>/ -n <my-corpus-ner>`
4. Train the model using the `train.py` on MareNostrum 4 (CTE-AMD) or Google Colab (with GPU runtime). TODO: training without *Weights & Biases*. Ask Jan for a simpler version.
    1. Save models and select best model.

## Inference Pipeline

We will reuse the Training Pipeline and make inference to the **test** set of the dataset.

1. Generate empty .ann files with: `find /path/to/your/directory -type f -name '*.txt' -exec bash -c 'touch "${1%.txt}.ann"' _ {} \;`. The .ann files are needed for the pre-tokenization that takes place in the `brat2conll` subscript of `build_dataset.py`.
2. Create the same directory structure of training (train/valid/test with all .txts and .anns). We will generate the predictions for the test split only, so train and valid can be just a copy of test.
    - `cd <my-inference-corpus>`
    - `mkdir test`
    - `mv * test`
    - `cp -r test train`
    - `cp -r test valid`
2. Use the `build_dataset.py` script to create a Hugging Face Dataset with the pre-tokenized input. Indicate the source directory of the corpus (-d) and the output name/path of the Hugging Face dataset (-n).
    - `python hugging-face-pipeline/build_dataset.py -d <my-corpus>/ -n <my-corpus-ner>`
    
    Make sure that the generated HF dataset loader script (e.g. my-corpus-ner.py) has the right tags and in the same order as the one used during training (usually alphabetic)! Otherwise, you will get swapped entities.
3. Run `model_inference.py` specifying all the arguments. It will run on GPU if available. Make sure that both output directories do not exist (this is to prevent unwanted overwriting). Example usage:
    - `python hugging-face-pipeline/model_inference.py -ds <my-corpus-ner> -m <my-model> -ocd <my-corpus>/test -o <my-corpus-predictions>`


## Directory structure

- requirements.txt -> list of necessary
- brat -> adapted version of original [brat repository](https://github.com/nlplab/brat) files. Necessary tools for some scripts.
- templates -> files used in different scripts as templates.
- scripts:
    - build_dataset.py -> generate a HF NER dataset (CoNLL + loader script)
    - brat2conll.py -> Brat (.txt & .ann) files to CoNLL (.conll)
    - model_inference.py -> process a HF NER dataset with a specific model)
    - multiple_model_inference.py -> call model_inference.py for a list of corpus and models.
    - simple_inference.py -> use the HF NER pipeline directly, which is limited
    - train.py -> script called within Weights & Biases Sweep for training (with hyperparameter optimization)
    - join_all_conlls.sh -> generate a {train|validation|test}.conll files by concatenating in order all conll files in each of the 3 subdirectories
    - generate_val_split.py -> randomnly generate a split
    - generate_brat_colors.py -> generate colors for brat (to be included in visual.conf) for a given set of classes/entities


- notebooks: directory containing useful tools for debugging, partial pipelines and work not polished yet (paths might need an update).
    - ann2tsv.ipynb -> Create a tsv (with filename) from .ann files
    - load_predictions.ipynb -> snippet to load .ann files into a pandas dataframe
    - model_inference_pipeline -> tests to integrate within HF NER pipeline (not working)
    - conll2ann.ipynb -> .conll & .txt to Standoff (.ann)
    - predictions2conll.ipynb -> model predictions (.json & original .conll) to CoNLL (.conll) (not used anymore, as predictions are directly written to CoNLL files)

### brat2conll

For the brat to CoNLL converter, we used an adapted version from the original [brat repository](https://github.com/nlplab/brat) (version Oct 4, 2021)[^1]

You can find the modifications noted in comments starting with "NOTE".

The script will generate (.conll) files in the same directory as the original (.txt) and (.ann) files. It can be called from the command line with:

```bash
$ python brat2conll.py -r path/to/root_directory
```

or you can call it on just one (.txt) file. As always, the (.ann) file should be in the same directory as (.txt):

```bash
$ python brat2conll.py -r path/to/file.txt
```

The script will look recursively for (.txt) files and their corresponding (.ann) files. This enables this same script to work for many different dataset structures like the following:

```
root_directory
│   file001.ann
│   file001.txt
|   file002.ann
|   file002.txt
└   ...
```

```
root_directory
│
└───train
│   │   file001.ann
│   |   file001.txt
|   |   file002.ann
|   |   file002.txt
|   └   ...
│
└───test
    │   file011.ann
    │   file011.txt
    │   file012.ann
    |   file012.txt
    └   ...
```

or even

```
root_directory
│
└───train
│   └───procedures
|   |   |    file001.ann
│   |   |    file001.txt
|   |   |    file002.ann
|   |   |    file002.txt
|   |   └    ...
|   |
|   └───diseases
|       |    file011.ann
│       |    file011.txt
|       |    file012.ann
|       |    file012.txt
|       └    ...
│
└───test
│   └───procedures
|   |   |    file021.ann
│   |   |    file021.txt
|   |   |    file022.ann
|   |   |    file022.txt
|   |   └    ...
|   |
|   └───diseases
|       |    file031.ann
│       |    file031.txt
|       |    file032.ann
|       |    file032.txt
|       └    ...
```

In the case of overlapping annotations, only the longest one will be considered. When this happens, the script displays a message like the following:

```
Eliminate Textbound(start=707, end=723, type='PROCEDIMIENTO', text='cultivo de orina') due to overlap with Textbound(start=692, end=723, type='PROCEDIMIENTO', text='sedimento y el cultivo de orina')`
```

**NOTES**

- Make sure to **NOT** include any other (.txt) files in the root directory rather than the raw text for annotation.

- The **note annotations** (comments) in the (.ann) files will be **ignored** and won't be included in the CoNLL resulting file. These note annotations take the following format:

```
T1	PROCEDIMIENTO 775 794	ecografía abdominal
#1	AnnotatorNotes T1	This is a free-text note annotation associated with T1
T2	PROCEDIMIENTO 1229 1292	TAC abdomino-pélvico realizado con contraste oral e intravenoso
```

## Contact

Please, for any comment or doubt, you can contact me on: janrodmir \[at] gmail \[dot] com

## References

[^1]: Stenetorp, Pontus and Pyysalo, Sampo and Topi\'{c}, Goran and Ohta, Tomoko and Ananiadou, Sophia and Tsujii, Jun'ichi. brat: a Web-based Tool for {NLP}-Assisted Text Annotation. Proceedings of the Demonstrations Session at EACL 2012, April 2012, Association for Computational Linguistics.
