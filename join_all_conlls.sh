#!/bin/bash

sed -s -e $'$a\\\n' $1/train/*.conll > $1/train.conll
sed -s -e $'$a\\\n' $1/valid/*.conll > $1/validation.conll
sed -s -e $'$a\\\n' $1/test/*.conll > $1/test.conll
