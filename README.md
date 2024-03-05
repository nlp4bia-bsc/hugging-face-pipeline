# Hugging Face Pipeline

Automation tools for different processes regarding the training and deployment of models and datasets to Hugging Face.

NLP for Biomedical Information Analysis (NLP4BIA).

## General Pipeline

The general pipeline for NER is the following:
1. Annotate using Brat, download the corpus.
2. Make the splits, only **if not already split**, with the `generate_val_split.py` script by having all 
2. Transform standoff (.ann) format to CoNLL (IOB scheme) using `brat2conll` script (check the `--multi-label` option).
3. Join all `.conll` files into one (one `.conll` per split) using the `join_all_conlls.sh` script (needs execution permission `chmod +x join_all_conlls.sh`).
    - `./join_all_conlls.sh <dataset_directory>`
4. Generate a **Hugging Face Dataset**:
    1. Create a (private) dataset using the web. License cc-by-4.0 is okay.
    2. Clone it locally. Using SSH: `git clone git@hf.co:datasets/<your-username>/<dataset-name>`. Alternatively, if not configured, you can also use HTTPS: `git clone https://huggingface.co/<your-username>/<dataset-name>`.
    3. Copy the uploader template (single or multi-label) to the cloned directory and modify it accordingly (change name and label classes).
        - You can get all (present) labels by going to the directory containing the ann files and executing: `find . -name '*.ann' -type f -exec grep -Hr '^T' {} + | cut -f2 | cut -d' ' -f1 | sort | uniq`
        - Make sure that the uploader file has the **same name** as the cloned dataset repository (e.g. `meddoplace-ner`) with the `.py` extension (e.g. `meddoplace-ner.py`).
    4. If you have larger files (>10MB), setup the [Git LFS](https://huggingface.co/docs/hub/repositories-getting-started).
    5. Copy the joint CoNLL files (i.e. \[train|validation|test\].conll)
    6. Push the changes:
        - `git add -A`
        - `git commit -m "Initial commit"`
        - `git push`
5. Train the model using the notebook on Google Colab (with GPU runtime) or AMD-CTE (GPU) in MareNostrum.
    1. Keep one notebook per experiment.
    2. Save models and select best model.
    3. Save prediction results (JSON).
6. Use `predictions2conll` script to generate CoNLLs with predictions (JSON).
7. Use `conll2ann` script to generate Standoff (.ann) files with predictions.
    1. Join all anotation files and give the correct format for the subtask evaluation.


## Directory structure

- brat -> adapted version of original [brat repository](https://github.com/nlplab/brat) files. Necessary tools for some scripts.
- brat2conll.py -> Brat (.txt & .ann) files to CoNLL (.conll)
- predictions2conll.ipynb -> model predictions (.json & original .conll) to CoNLL (.conll)
- conll2ann.ipynb -> .conll & .txt to Standoff (.ann)



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
