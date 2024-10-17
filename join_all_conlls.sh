#!/bin/bash

find "$1/train" -name '*.conll' -type f | sort | xargs sed -s -e '$a\' > "$1/train.conll"
find "$1/valid" -name '*.conll' -type f | sort | xargs sed -s -e '$a\' > "$1/validation.conll"
find "$1/test" -name "*.conll" -type f | sort | xargs sed -s -e '$a\' > "$1/test.conll"
