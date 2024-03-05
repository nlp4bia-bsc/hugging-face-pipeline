#!/usr/bin/env python

# Adapted version from https://github.com/nlplab/brat.
# The modifications are noted with comments starting with 'NOTE'.

# Convert text and standoff annotations into CoNLL format.

from __future__ import print_function

import os
import re
import sys
from collections import namedtuple
from io import StringIO
from os import path

# NOTE: Modify import to be called from outside
from .sentencesplit import sentencebreaks_to_newlines

# assume script in brat tools/ directory, extend path to find sentencesplit.py
sys.path.append(os.path.join(os.path.dirname(__file__), '../server/src'))
sys.path.append('.')

options = None
missing_ann_files = 0

EMPTY_LINE_RE = re.compile(r'^\s*$')
CONLL_LINE_RE = re.compile(r'^\S+\t\d+\t\d+.')


class FormatError(Exception):
    pass


def argparser():
    import argparse

    ap = argparse.ArgumentParser(description='Convert text and standoff ' +
                                 'annotations into CoNLL format.')
    ap.add_argument('-a', '--annsuffix', default="ann",
                    help='Standoff annotation file suffix (default "ann")')
    # ap.add_argument('-c', '--singleclass', default=None,
    #                 help='Use given single class for annotations')
    ap.add_argument('-n', '--nosplit', default=False, action='store_true',
                    help='No sentence splitting')
    ap.add_argument('-o', '--outsuffix', default="conll",
                    help='Suffix to add to output files (default "conll")')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Verbose output')
    ap.add_argument('text', metavar='TEXT', nargs='+',
                    help='Text files ("-" for STDIN)')
    ap.add_argument('--multi-label', action=argparse.BooleanOptionalAction,
                    help='Indicate if the generated CoNLLs should contain flat or nested (multi-label) entities.', default=False)
    return ap


def read_sentence(f):
    """Return lines for one sentence from the CoNLL-formatted file.

    Sentences are delimited by empty lines.
    """

    lines = []
    for l in f:
        lines.append(l)
        if EMPTY_LINE_RE.match(l):
            break
        if not CONLL_LINE_RE.search(l):
            raise FormatError(
                'Line not in CoNLL format: "%s"' %
                l.rstrip('\n'))
    return lines


def strip_labels(lines):
    """Given CoNLL-format lines, strip the label (first TAB-separated field)
    from each non-empty line.

    Return list of labels and list of lines without labels. Returned
    list of labels contains None for each empty line in the input.
    """

    labels, stripped = [], []

    labels = []
    for l in lines:
        if EMPTY_LINE_RE.match(l):
            labels.append(None)
            stripped.append(l)
        else:
            fields = l.split('\t')
            labels.append(fields[0])
            stripped.append('\t'.join(fields[1:]))

    return labels, stripped


def attach_labels(labels, lines):
    """Given a list of labels and CoNLL-format lines, affix TAB-separated label
    to each non-empty line.

    Returns list of lines with attached labels.
    """

    assert len(labels) == len(
        lines), "Number of labels (%d) does not match number of lines (%d)" % (len(labels), len(lines))

    attached = []
    for label, line in zip(labels, lines):
        empty = EMPTY_LINE_RE.match(line)
        assert (label is None and empty) or (label is not None and not empty)

        if empty:
            attached.append(line)
        else:
            attached.append('%s\t%s' % (label, line))

    return attached


# NERsuite tokenization: any alnum sequence is preserved as a single
# token, while any non-alnum character is separated into a
# single-character token. TODO: non-ASCII alnum.
# NOTE: Modified to support accents and other common characters.
TOKENIZATION_REGEX = re.compile(
    r'([0-9A-Za-zÀ-ÖØ-öø-ÿ]+|[^0-9A-Za-zÀ-ÖØ-öø-ÿ])')

NEWLINE_TERM_REGEX = re.compile(r'(.*?\n)')


def text_to_conll(f):
    """Convert plain text into CoNLL format."""
    global options

    if options.nosplit:
        sentences = f.readlines()
    else:
        sentences = []
        for l in f:
            # NOTE: replace 'zero width no-break space' chars with normal whitespaces
            # This is to avoid strange chars in the resulting CoNLL files, which cause boundary missmatches.
            l = l.replace('\ufeff', ' ')
            l = sentencebreaks_to_newlines(l)
            sentences.extend([s for s in NEWLINE_TERM_REGEX.split(l) if s])

    lines = []

    offset = 0
    for s in sentences:
        nonspace_token_seen = False

        tokens = [t for t in TOKENIZATION_REGEX.split(s) if t]

        for t in tokens:
            if not t.isspace():
                # NOTE: empty tags instead of 'O'.
                lines.append([[], offset, offset + len(t), t])
                nonspace_token_seen = True
            offset += len(t)

        # sentences delimited by empty lines
        if nonspace_token_seen:
            lines.append([])

    # add labels (other than 'O') from standoff annotation if specified
    if options.annsuffix:
        lines = relabel(lines, get_annotations(f.name))

    # NOTE: reorder with
    lines = [[l[0], str(l[1]), str(l[2]), l[3]] if l else l for l in lines]
    return StringIO('\n'.join(('\t'.join(l) for l in lines)))


def relabel(lines, annotations):
    global options

    # # TODO: this could be done more neatly/efficiently
    # offset_labels = {}

    # for tb in annotations:
    #     for i in range(tb.start, tb.end):
    #         # NOTE: do not print warning if multi-label
    #         if i in offset_labels:
    #             offset_labels[i].append(tb)
    #             if not options.multi_label:
    #                 print("Warning: overlapping annotations", file=sys.stderr)
    #         else:
    #             offset_labels[i] = [tb]
            
    for tb in annotations:
        # Get word that starts entity. One line_or_row = One word (CoNLL).
        starting_line_idx = None
        for i, line in enumerate(lines):
            # If new sentence, do nothing
            if not len(line):
                continue
            # Entity starts exactly in this word
            elif line[1] == tb.start:
                starting_line_idx = i
                break
            # Entity started at middle of the word
            elif line[1] < tb.start < line[2]:
                print('Warning: annotation-token boundary mismatch: "%s" --- "%s"' % (
                            line[3], tb.text), file=sys.stderr)
                starting_line_idx = i
                break
        line[0].append(f"B-{tb.type}")
        # If it's last one (single-word entity), go to next iteration
        # This is to avoid e.g. the '.' in "Chile." to be tagged as I-GPE
        if line[2] >= tb.end:
            continue
        # Add I-tags to subsequent words of entity
        for i, line in enumerate(lines[starting_line_idx+1:]):
            # If new sentence, do nothing
            if not len(line):
                continue
            # Entity exactly finishes here
            elif line[2] == tb.end:
                line[0].append(f"I-{tb.type}")
                break
            # Entity finishes in the middle of the word
            elif line[1] < tb.start < line[2]:
                print('Warning: annotation-token boundary mismatch: "%s" --- "%s"' % (
                            line[3], tb.text), file=sys.stderr)
                line[0].append(f"I-{tb.type}")
                break
            # We are already outside of entity (just to ensure)
            elif line[1] > tb.end:
                break
            # Word belongs to entity and is not the end. 
            else:
                line[0].append(f"I-{tb.type}")

    for line in lines:
        # If new sentence, do nothing
        if not len(line):
            continue
        # If no class, assign O
        elif not len(line[0]):
            line[0] = 'O'
        # Join all labels
        else:
            # NOTE: make the set to avoid duplicated classes for the same word, which might happen
            # in both single and multi-label 
            # (e.g. adriamicinaciclofosfamida -> adriamicina + ciclofosfamida -> B-FARMACO x2)
            if len(set(line[0])) != len(line[0]):
                print('Warning: multiple of the same label for the same word: "%s" --- %s' % (
                            line[3], line[0]), file=sys.stderr)
            line[0] = '|'.join(set(line[0]))

    # # optional single-classing
    # if options.singleclass:
    #     for l in lines:
    #         if l and l[0] != 'O':
    #             l[0] = l[0][:2] + options.singleclass

    return lines


def process(f):
    return text_to_conll(f)


def process_files(files):
    global options

    nersuite_proc = []

    try:
        for fn in files:
            try:
                if fn == '-':
                    lines = process(sys.stdin)
                else:
                    with open(fn, 'rU') as f:
                        lines = process(f)

                # TODO: better error handling
                if lines is None:
                    raise FormatError

                if fn == '-' or not options.outsuffix:
                    sys.stdout.write(''.join(lines))
                else:
                    ofn = path.splitext(fn)[0] + options.outsuffix
                    with open(ofn, 'wt') as of:
                        of.write(''.join(lines))

            except BaseException:
                # TODO: error processing
                raise
    except Exception as e:
        for p in nersuite_proc:
            p.kill()
        if not isinstance(e, FormatError):
            raise

# start standoff processing


TEXTBOUND_LINE_RE = re.compile(r'^T\d+\t')

Textbound = namedtuple('Textbound', 'start end type text')


def parse_textbounds(f):
    """Parse textbound annotations in input, returning a list of Textbound."""

    textbounds = []

    for l in f:
        l = l.rstrip('\n')

        if not TEXTBOUND_LINE_RE.search(l):
            continue

        id_, type_offsets, text = l.split('\t')
        type_, start, end = type_offsets.split()
        start, end = int(start), int(end)

        textbounds.append(Textbound(start, end, type_, text))

    return textbounds


def eliminate_overlaps(textbounds):
    global options
    eliminate = {}

    # TODO: avoid O(n^2) overlap check
    for t1 in textbounds:
        for t2 in textbounds:
            if t1 is t2:
                continue
            if t2.start >= t1.end or t2.end <= t1.start:
                continue
            # NOTE: Eliminate only if flat NER (not nested) or if nested and types match
            if (not options.multi_label) or (t1.type == t2.type):
                # eliminate shorter
                if t1.end - t1.start > t2.end - t2.start:
                    print("Eliminate %s due to overlap with %s" % (
                        t2, t1), file=sys.stderr)
                    eliminate[t2] = True
                else:
                    print("Eliminate %s due to overlap with %s" % (
                        t1, t2), file=sys.stderr)
                    eliminate[t1] = True

    return [t for t in textbounds if t not in eliminate]


def get_annotations(fn):
    global options
    global missing_ann_files

    annfn = path.splitext(fn)[0] + options.annsuffix
    # NOTE: wrap open within try-except so that if no ann
    try:
        with open(annfn, 'rU') as f:
            textbounds = parse_textbounds(f)
    except FileNotFoundError:
        missing_ann_files += 1
        print(f"Warning: annotation file {annfn} does not exist. No entities will appear in the resulting CoNLL file.", file=sys.stderr)
        textbounds = []

    textbounds = eliminate_overlaps(textbounds)
    # NOTE: sort by (1) earlier, and if tied, (2) longer entities.
    textbounds = sorted(textbounds, key=lambda tb: (tb.start, -tb.end))

    return textbounds

# end standoff processing


def main(argv=None):
    if argv is None:
        argv = sys.argv

    global options
    # NOTE: Modified so that we can call main directly
    options = argparser().parse_args(argv)

    # make sure we have a dot in the suffixes, if any
    if options.outsuffix and options.outsuffix[0] != '.':
        options.outsuffix = '.' + options.outsuffix
    if options.annsuffix and options.annsuffix[0] != '.':
        options.annsuffix = '.' + options.annsuffix

    process_files(options.text)

    # NOTE: count number of missing ann files
    global missing_ann_files
    if missing_ann_files:
        print(f"A total of {missing_ann_files} annotation files were missing. No entities will appear in their resulting CoNLL file.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
