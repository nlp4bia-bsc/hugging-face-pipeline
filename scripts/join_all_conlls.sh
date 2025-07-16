#!/bin/bash
# We need to add the LC_ALL=C because Python sorting does not work the same way as Bash 
find "$1/train" -maxdepth 1 -name '*.conll' -type f | LC_ALL=C sort | xargs sed -s -e '$a\' > "$1/train.conll"
find "$1/valid" -maxdepth 1 -name '*.conll' -type f | LC_ALL=C sort | xargs sed -s -e '$a\' > "$1/validation.conll"
find "$1/test" -maxdepth 1 -name "*.conll" -type f | LC_ALL=C sort | xargs sed -s -e '$a\' > "$1/test.conll"
