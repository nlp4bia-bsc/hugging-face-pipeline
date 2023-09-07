# Hugging Face Pipeline

Automation tools for different processes regarding the training and deployment of models and datasets to Hugging Face.

NLP for Biomedical Information Analysis (NLP4BIA).

## General Pipeline

The general pipeline for NER is the following:
1. Annotate using Brat, download the corpus
2. Use `brat2conll` script
3. Join all `.conll` files into one (one `.conll` per split)
4. Generate `Hugging Face Dataset` and upload to Hub (private) or Google Drive
5. Train model using the notebook on Google Colab (with GPU runtime)
    1. Keep one notebook per experiment
    2. Save models and select best model
    3. Save prediction results (JSON)
6. Use `predictions2conll` script to generate CoNLLs with predictions (JSON)
7. Use `conll2ann` script to generate Standoff (.ann) files with predictions
    1. Join all anotation files and give the correct format for the subtask evaluation


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
