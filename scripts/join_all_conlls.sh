#!/bin/bash

find "$1/train" -maxdepth 1 -name '*.conll' -type f | sort | xargs sed -s -e '$a\' > "$1/train.conll"
find "$1/valid" -maxdepth 1 -name '*.conll' -type f | sort | xargs sed -s -e '$a\' > "$1/validation.conll"
find "$1/test" -maxdepth 1 -name "*.conll" -type f | sort | xargs sed -s -e '$a\' > "$1/test.conll"
