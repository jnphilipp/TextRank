#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
"""TextRank.

Script to evaluate corpa for keyword extraction using TextRank.
Based on https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0.
"""

import json
import logging
import numpy as np
import spacy
import sys

from argparse import ArgumentParser, FileType, RawTextHelpFormatter
from collections import OrderedDict
from csv import DictReader
from spacy.language import Language
from typing import Dict, List, Optional, Set, Tuple


__author__ = "J. Nathanael Philipp (jnphilipp)"
__copyright__ = "Copyright 2021 J. Nathanael Philipp (jnphilipp)"
__email__ = "nathanael@philipp.land"
__license__ = "GPLv3"
__version__ = "0.1.0"
__github__ = "https://github.com/jnphilipp/TextRank"
VERSION = (
    f"%(prog)s v{__version__}\n{__copyright__}\n"
    + "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>."
    + "\nThis is free software: you are free to change and redistribute it.\n"
    + "There is NO WARRANTY, to the extent permitted by law.\n\n"
    + f"Report bugs to {__github__}/issues."
    + f"\nWritten by {__author__} <{__email__}>"
)


def textrank(
    model: Language,
    text: str,
    candidate_pos: Optional[List[str]] = ["NOUN", "PROPN"],
    window_size: int = 4,
    lower: bool = False,
    lemma: bool = True,
    steps: int = 10,
    d: float = 0.85,
    min_diff: float = 1e-5,
) -> List[Tuple[str, float]]:
    """Perform TextRank and return keywords.

    Args:
     * text: text to analyze
     * candidate_pos: part of speach tags to filter for
     * window_size: window size
     * lower: convert to lower case
     * lemma: use lemmatized forms
     * d: damping coefficient, usually is .85
     * min_diff: convergence threshold
     * steps: iteration steps
    """

    doc = model(text)

    sentences = []
    for sent in doc.sents:
        words = []
        entity: List[str] = []
        for token in sent:
            word = None
            if lower is True:
                if lemma:
                    word = token.lemma_.lower()
                else:
                    word = token.text.lower()
            else:
                if lemma:
                    word = token.lemma_
                else:
                    word = token.text

            if token.ent_iob == 0 or token.ent_iob == 2 and entity:
                if len(entity) > 1:
                    words.append(" ".join(entity))
                entity = []
            elif token.ent_iob == 3:
                if len(entity) > 1:
                    words.append(" ".join(entity))
                entity = [word]
            elif token.ent_iob == 1 and word:
                entity.append(word)
            if candidate_pos and token.pos_ in candidate_pos and token.is_stop is False:
                words.append(word)
            elif not candidate_pos and token.is_stop is False:
                words.append(word)

        sentences.append(words)

    vocab: Dict[str, int] = dict()
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)

    token_pairs = list()
    for sentence in sentences:
        for i, word in enumerate(sentence):
            for j in range(i + 1, i + window_size):
                if j >= len(sentence):
                    break
                pair = (word, sentence[j])
                if pair not in token_pairs:
                    token_pairs.append(pair)

    # Get normalized matrix
    g = np.zeros((len(vocab), len(vocab)), dtype="float")
    for word1, word2 in token_pairs:
        g[vocab[word1]][vocab[word2]] = 1
    # Get Symmeric matrix
    g += g.T - np.diag(g.diagonal())

    # Normalize matrix by column
    norm = np.sum(g, axis=0)
    g = np.divide(g, norm, where=norm != 0)

    # Initalization for weights (pagerank value)
    weights = np.array([1] * len(vocab))

    # Iteration
    previous_weights = 0
    for epoch in range(steps):
        weights = (1 - d) + d * np.dot(g, weights)
        if abs(previous_weights - sum(weights)) < min_diff:
            break
        else:
            previous_weights = sum(weights)

    # Get weight for each node
    node_weights = dict()
    for word, index in vocab.items():
        node_weights[word] = weights[index]

    node_weights = OrderedDict(
        sorted(node_weights.items(), key=lambda t: t[1], reverse=True)
    )

    keywords = list()
    for i, (word, weight) in enumerate(node_weights.items()):
        keywords.append((word, weight))
    return keywords


def acc_k(expected: Set[str], found: Set[str], k: int) -> float:
    """Calculate top k-accuracy of two keyword sets.

    Args:
     * expected: set of expected keywords
     * found: set of found keywords
    """
    if len(expected.intersection(found)) >= min(k, len(expected)):
        return 1.0
    else:
        return 0.0


def precision(expected: Set[str], found: Set[str]) -> float:
    """Calculate precision of two keyword sets.

    Args:
     * expected: set of expected keywords
     * found: set of found keywords
    """
    if len(found) == 0:
        return 0.0
    return len(expected.intersection(found)) / len(found)


def recall(expected: Set[str], found: Set[str]) -> float:
    """Calculate recall of two keyword sets.

    Args:
     * expected: set of expected keywords
     * found: set of found keywords
    """
    if len(expected) == 0:
        return 0.0
    return len(expected.intersection(found)) / len(expected)


def f1(precision: float, recall: float) -> float:
    """Calculate F1.

    Args:
     * precision: precision
     * recall: recall
    """
    if precision + recall == 0:
        return 0.0
    return 2 * ((precision * recall) / (precision + recall))


if __name__ == "__main__":
    parser = ArgumentParser(prog="TextRank", formatter_class=RawTextHelpFormatter)
    parser.add_argument("-V", "--version", action="version", version=VERSION)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "-f",
        "--log-format",
        help="Logging format. (default: %%(asctime)s %%(levelname)s %%(message)s)",
        default="%(asctime)s %(levelname)s %(message)s",
    )
    parser.add_argument(
        "-n",
        "--num-acc",
        default=5,
        help="Number of top k-accuracies. (default: 5)",
    )
    parser.add_argument(
        "-w",
        "--window-size",
        default=4,
        help="Window size of TextRank. (default: 4)",
    )
    parser.add_argument(
        "-p",
        "--pos",
        nargs="*",
        default=["NOUN", "PROPN"],
        help="Part of speach tags to filter for.",
    )
    parser.add_argument(
        "-o",
        "--lower",
        action="store_true",
        help="Convert all to lowercase.",
    )
    parser.add_argument(
        "-k",
        "--num-keywords",
        default=10,
        action="store_true",
        help="Convert all to lowercase.",
    )
    parser.add_argument(
        "-l",
        "--language-model",
        required=True,
        type=str,
        help="spaCy language model to use.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--csv",
        type=FileType("r", encoding="utf8"),
        help="Read texts from a CSV file. Assumes the text is in a column named text "
        + "and the keywords are in a column named keywords.",
    )
    group.add_argument(
        "--json",
        type=FileType("r", encoding="utf8"),
        help="Read texts from a JSON file. Assumes the text is in a field named text"
        + " and the keywords are in a field names keywords.",
    )

    args = parser.parse_args()
    if args.verbose == 0:
        logging.basicConfig(format=args.log_format, level=logging.WARN)
    elif args.verbose == 1:
        logging.basicConfig(format=args.log_format, level=logging.INFO)
    else:
        logging.basicConfig(format=args.log_format, level=logging.DEBUG)

    model = spacy.load(args.language_model)

    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    for i in range(1, args.num_acc + 1):
        metrics[f"acc{i}"] = 0.0
    num_texts = 0

    if args.csv:
        reader = DictReader(args.csv)

        for row in reader:
            if "text" not in row:
                logging.error("Column text not found in CSV file.")
                sys.exit(1)
            elif "keywords" not in row:
                logging.error("Column keywords not found in CSV file.")
                sys.exit(1)

            keywords = textrank(
                model,
                row["text"],
                candidate_pos=args.pos,
                window_size=args.window_size,
                lower=args.lower,
            )

            expected = set(
                (w.lower() if args.lower else w) for w in row["keywords"].split(",")
            )
            found = set([k[0] for i, k in enumerate(keywords) if i < args.num_keywords])

            p = precision(expected, found)
            r = recall(expected, found)
            metrics["precision"] += p
            metrics["recall"] += r
            metrics["f1"] += f1(p, r)
            for j in range(1, args.num_acc + 1):
                metrics[f"acc{j}"] += acc_k(expected, found, j)

            num_texts += 1
    elif args.json:
        data = json.loads(args.json.read())

        for text in data:
            if "text" not in text:
                logging.error("Column text not found in CSV file.")
                sys.exit(1)
            elif "keywords" not in text:
                logging.error("Column keywords not found in CSV file.")
                sys.exit(1)

            keywords = textrank(
                model,
                text["text"],
                candidate_pos=args.pos,
                window_size=args.window_size,
                lower=args.lower,
            )

            expected = set((w.lower() if args.lower else w) for w in text["keywords"])
            found = set([k[0] for i, k in enumerate(keywords) if i < args.num_keywords])

            p = precision(expected, found)
            r = recall(expected, found)
            metrics["precision"] += p
            metrics["recall"] += r
            metrics["f1"] += f1(p, r)
            for j in range(1, args.num_acc + 1):
                metrics[f"acc{j}"] += acc_k(expected, found, j)

            num_texts += 1
            if num_texts > 100:
                break

    metrics["precision"] /= num_texts
    metrics["recall"] /= num_texts
    metrics["f1"] /= num_texts
    for j in range(1, args.num_acc + 1):
        metrics[f"acc{j}"] /= num_texts

    for k in metrics.keys():
        print(f"{k}:", metrics[k])
