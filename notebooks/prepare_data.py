import os
from pathlib import Path
import string
import sys
import warnings

import numpy as np
import pandas as pd

import datasets
from datasets import (
    Dataset,
    DatasetDict,
)

import spacy_alignments as alignments
from transformers import AutoTokenizer

warnings.simplefilter("ignore")

INPUT_DIR = "."
MODEL_PATH = "microsoft/deberta-v3-base"
MAX_TRAIN_LENGTH = 3072  # (number of tokens)

LABELS = (
    "O",
    "B-EMAIL",
    "B-ID_NUM",
    "B-NAME_STUDENT",
    "B-PHONE_NUM",
    "B-STREET_ADDRESS",
    "B-URL_PERSONAL",
    "B-USERNAME",
    "I-ID_NUM",
    "I-NAME_STUDENT",
    "I-PHONE_NUM",
    "I-STREET_ADDRESS",
    "I-URL_PERSONAL",
)
LABEL2ID = dict((k, i) for (i, k) in enumerate(LABELS))
ID2LABEL = dict((i, k) for (i, k) in enumerate(LABELS))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def load_train_data(downsample_frac=0.4):
    print("Loading training data...")
    input_dir = Path(INPUT_DIR) / "datasets"

    kaggle = load(input_dir / "train.json", 0, check=False)  # pre-checked, no issues
    kaggle = downsample(kaggle, downsample_frac)

    mixtral1 = load(input_dir / "Fake_data_1850_218.json", 10_000, check=True)
    mixtral2 = load(
        input_dir / "mpware_mixtral8x7b_v1.1-no-i-username.json", 20_000
    )  # pre-checked
    private = load(input_dir / "gemini-20240303-train.json", 30_000, check=True)
    extra = pd.concat([mixtral2, mixtral1, private])

    kaggle_ds = Dataset.from_pandas(kaggle).class_encode_column("pii")
    extra_ds = Dataset.from_pandas(extra).class_encode_column("pii")

    return DatasetDict(kaggle=kaggle_ds, extra=extra_ds)


def load(path, n=0, do_normalize=True, check=False):
    if check:
        print(f"Loading and checking {path} ...")
        df = clean_data(path)
    else:
        print(f"Loading {path}...")
        df = pd.read_json(path)

    df.document = range(n, n + len(df))  # training only
    df.rename({"full_text": "text"}, axis=1, inplace=True)
    if do_normalize:
        df.tokens = df.tokens.apply(normalize_spacy_tokens)
        df.text = df.apply(reconstruct_text, axis=1)
    df["pii"] = df.labels.apply(has_pii).astype(int)
    avg_ntokens = df.tokens.apply(len).mean()
    print(
        f"Loaded {path} ({len(df)} docs; pii {sum(df.pii)} docs; avg {avg_ntokens:.1f} tokens/doc)"
    )
    return df


def clean_data(obj):
    df = check_data(obj)
    n = len(df)
    df = df[
        (df.nincorrect_labels == 0)
        & (df.mismatched_ntokens_nlabels == 0)
        & (df.mismatched_ntokens_nws == 0)
    ]
    df.drop(
        [
            "incorrect_labels",
            "nincorrect_labels",
            "mismatched_ntokens_nlabels",
            "mismatched_ntokens_nws",
        ],
        axis=1,
        inplace=True,
    )
    m = len(df)
    if m != n:
        print(f"Removed {n - m} doc(s) with incorrect labels")
    return df


def check_data(obj):
    if isinstance(obj, (str, Path)):
        df = pd.read_json(obj)
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        raise ValueError(f"Expected a str of a dataframe, but got {type(obj)}")

    nlabels = df.labels.apply(len)
    ntokens = df.tokens.apply(len)
    df["mismatched_ntokens_nlabels"] = (nlabels != ntokens).astype(int)

    if "trailing_whitespace" in df.columns:
        nws = df.trailing_whitespace.apply(len)
        df["mismatched_ntokens_nws"] = (nws != ntokens).astype(int)
        if sum(df.mismatched_ntokens_nws):
            print("ERROR: num tokens doesn't always match num trailing_whitespace")

    if sum(df.mismatched_ntokens_nlabels):
        print("ERROR: num tokens doesn't always match num labels")

    df["incorrect_labels"] = df.apply(check_labels, axis=1)
    df["nincorrect_labels"] = df.incorrect_labels.apply(len)

    if sum(df.nincorrect_labels):
        n = len(df[df.nincorrect_labels > 0])
        if n == 1:
            print(f"ERROR: {n} doc has incorrect labels")
        else:
            print(f"ERROR: {n} docs have incorrect labels")
    else:
        print("Found no labeling errors")

    df.sort_values("nincorrect_labels", inplace=True, ascending=False)
    return df


def check_labels(row):
    ZWS = chr(0x200B)

    res = []

    tokens = [x.strip().strip(ZWS).strip() for x in row.tokens]
    labels = row.labels

    for i, (x, y) in enumerate(zip(row.labels, tokens)):
        if x not in labels:  # (0) we only accept known BIO-labels
            res.append(i)
            continue

        prevy = tokens[i - 1] if i > 0 else None
        prevx = labels[i - 1] if i > 0 else None
        nexty = tokens[i + 1] if i + 1 < len(row.tokens) else None
        nextx = labels[i + 1] if i + 1 < len(row.labels) else None

        if x[0] == "B":
            if not y:  # (1) no entity should start with white space
                res.append(i)
            elif y == "@":  # (2) only username and perhaps id_num can start with '@'
                if x not in ("B-USERNAME", "B-ID_NUM"):
                    res.append(i)
            elif y == "(":  # (3) only phone numbers can start with '('
                if x != "B-PHONE_NUM":
                    res.append(i)
            elif y in string.punctuation:
                # (4) otherwise, nothing can start with punctuation
                # might be too strict in general, but still a good precaution
                res.append(i)

        elif x[0] == "I":
            if i == 0:  # (5) I-label can not be first in doc
                res.append(i)
            elif prevx == "O":  # (6) I-label can not start en entity span
                res.append(i)
            elif (
                i == len(tokens) - 1 and not y
            ):  # (7) very last I label should not be for whitespace
                res.append(i)
            elif y in string.punctuation:  # (also True for empty y)
                # (7) white-space and punctuation should be preceded by either B or I of the same label
                #     and should be followed by I of the same label; in other words,
                #     white-space and punctuation can only be labeled as I, if this is in the middle
                #     of an entity span
                if x[1:] != prevx[1:] or x != nextx:
                    res.append(i)

        elif x == "O":
            # (8) if preceded by B-X or I-X and followed by I-X, then this O is _likely_ incorrect;
            # we also indicate an error for the following I-X in this case (rule 6) and it's a bit
            # ambiguous which one is actually incorrect (probably only the O-label)
            if 0 < i < len(tokens) - 1 and prevx[1:] == nextx[1:] and nextx[0] == "I":
                res.append(i)

        else:
            assert False  # we already checked for known labels at top

    return res


def has_pii(labels):
    return any(x != "O" for x in labels)


def downsample(df, frac=None):
    if frac is None:
        return df

    print(f"Downsampling negative samples (frac={frac:.3f})")
    n = len(df)
    if frac == 0.0:
        df = df[df.pii == 1].copy()
        print(f"Removed all negative samples ({n} -> {len(df)} docs)")
    else:
        df_pos = df[df.pii == 1]
        df_neg = df[df.pii == 0].sample(frac=frac)
        df = pd.concat([df_pos, df_neg])
        print(f"Downsampled negative samples ({n} -> {len(df)} docs)")

    return df


def normalize_spacy_tokens(
    spacy_tokens,
):  # if we do this in training, we also need to do so in inference!
    return [normalize_spacy_token(x) for x in spacy_tokens]


def normalize_spacy_token(token, max_length=21):
    ZERO_WIDTH_SPACE = chr(0x200B)  # not stripped off by python str.strip!
    stripped = token.strip().strip(ZERO_WIDTH_SPACE).strip()
    if not stripped:
        if "\n" in token or "\r" in token:
            return "\n"
        return " "
    token = stripped
    if len(token) >= max_length and len(set(token)) <= 3:
        token = "\n\n"
    return token


def reconstruct_text(row):
    tokens = [
        token + " " * ws for (token, ws) in zip(row.tokens, row.trailing_whitespace)
    ]
    return "".join(tokens)


def inspect(df, i=None, doc=None, showall=False):  # interactive debugging
    if doc is not None:
        sample = df[df.document == doc].iloc[0]
    else:
        sample = df.iloc[i]
    if "incorrect_labels" in df.columns:
        incorrect = set(sample.incorrect_labels)
    else:
        incorrect = None

    for j, (x, y) in enumerate(zip(sample.labels, sample.tokens)):
        if incorrect is None:
            flag = "-"
        elif j in incorrect:
            flag = "X"
        else:
            flag = "OK"
        if showall:
            print(j, x, repr(y), flag)
        elif flag != "OK" or x != "O":
            print(j, x, repr(y), flag)

    return sample
