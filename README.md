# Hugging Face Pipeline

Automation tools for different processes regarding the training and deployment of models and datasets to Hugging Face.

NLP for Biomedical Information Analysis.

---

For the brat to CoNLL converter, we used an adapted version from the original [brat repository](https://github.com/nlplab/brat) (version Oct 4, 2021)[^1]

You can find the modifications noted in comments starting with "NOTE".

## Usage examples:

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

## References

[^1]: Stenetorp, Pontus and Pyysalo, Sampo and Topi\'{c}, Goran and Ohta, Tomoko and Ananiadou, Sophia and Tsujii, Jun'ichi. brat: a Web-based Tool for {NLP}-Assisted Text Annotation. Proceedings of the Demonstrations Session at EACL 2012, April 2012, Association for Computational Linguistics.
