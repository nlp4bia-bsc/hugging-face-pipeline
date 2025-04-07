#!/usr/bin/env python3
"""
Generate the .conll files recursively of a given directory containing texts (.txt) and annotations (.ann)
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
    ap.add_argument('--multi-label', action='store_true',
                    help='Indicate if the generated CoNLLs should contain flat or nested (multi-label) entities.', default=False)
    return ap


def brat2conll_recursive(dir, multi_label):
    if os.path.isfile(dir):
        anntoconll.main([dir])
    else:
        # Call the brat2conll recursively to all subdirectories in the directory
        for child in [f.path for f in os.scandir(dir) if f.is_dir()]:
            brat2conll_recursive(child, multi_label)
        # Get all (.txt) files in the directory and generate .conll files
        # using the anntoconll from brat original repository
        files = [f.path for f in os.scandir(
            dir) if f.name.endswith('.txt')]
        if files:
            if multi_label:
                files.insert(0, "--multi-label")
            anntoconll.main(files)


def main(argv):
    options = argparser().parse_args(argv[1:])
    # Generate the .conll files recursively
    brat2conll_recursive(options.root_dir, options.multi_label)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
